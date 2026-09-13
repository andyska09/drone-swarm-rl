"""Fly to a stationary goal and hold position using throttle and body-rate commands."""

import flax.struct
import jax
import jax.numpy as jnp

from swarm.envs import Obs
from swarm.sim import control, dynamics

NUM_ACTIONS = 4
NEIGHBOR_FEATURES = 7


@flax.struct.dataclass
class RewardConfig:
    distance: float = 1.0
    crash: float = 10.0
    spin: float = 0.1


@flax.struct.dataclass
class EnvParams:
    model: dynamics.Params = flax.struct.field(default_factory=dynamics.default_params)
    reward: RewardConfig = flax.struct.field(default_factory=RewardConfig)

    roles: tuple = flax.struct.field(pytree_node=False, default=("drone",))
    n_neighbors: int = flax.struct.field(pytree_node=False, default=6)

    sim_dt: float = 0.005
    steps_per_action: int = flax.struct.field(pytree_node=False, default=2)
    max_steps: int = flax.struct.field(pytree_node=False, default=500)

    max_rate_rp: float = 4.0
    max_rate_yaw: float = 2.0

    # Start, goal box and arena all sit around this point.
    center: tuple = flax.struct.field(pytree_node=False, default=(0.0, 0.0, 3.0))
    start_pos_range: float = 0.5
    start_vel_range: float = 0.5
    start_tilt: float = 0.2
    goal_range: tuple = flax.struct.field(pytree_node=False, default=(0.0, 0.0, 0.0))
    arena: float = 10.0

    @property
    def n_drones(self):
        return len(self.roles)

    @property
    def n_visible(self):
        return min(self.n_neighbors, self.n_drones - 1)

    @property
    def policy_dt(self):
        return self.sim_dt * self.steps_per_action


@flax.struct.dataclass
class EnvState:
    drone: dynamics.State
    goal: jnp.ndarray
    rate_pid: control.PIDState
    alive: jnp.ndarray
    time: jnp.ndarray


def _norm(v):
    return jnp.sqrt(jnp.sum(v**2, axis=-1) + 1e-12)


def _to_body(R, v):
    return jnp.einsum("nji,nj->ni", R, v)


def _rate_scale(params):
    return jnp.array([params.max_rate_rp, params.max_rate_rp, params.max_rate_yaw])


def action_to_command(action, params):
    """CTBR action in [-1,1] to a throttle in [0,1] and body rates in rad/s."""

    return 0.5 * (action[..., 0] + 1.0), action[..., 1:] * _rate_scale(params)


def command_to_action(throttle, rate_ref, params):
    """Convert throttle and body rates to an action clipped to [-1, 1]."""

    action = jnp.concatenate(
        [jnp.atleast_1d(2.0 * throttle - 1.0), rate_ref / _rate_scale(params)]
    )
    return jnp.clip(action, -1.0, 1.0)


def get_obs(state, params):
    d, n = state.drone, params.n_drones
    return Obs(
        # Height is a scalar, so it carries no frame and keeps the body-frame symmetry.
        own=jnp.concatenate(
            [_to_body(d.R, d.v), d.R.reshape(n, 9), d.omega, d.x[:, 2:3]], axis=-1
        ),
        neighbors=jnp.zeros((n, params.n_visible, NEIGHBOR_FEATURES)),
        neighbor_mask=jnp.zeros((n, params.n_visible)),
        target=jnp.concatenate(
            [_to_body(d.R, state.goal - d.x), _to_body(d.R, -d.v)], axis=-1
        ),
        target_mask=jnp.ones(n),
    )


def is_dead(drone, params):
    return (
        (drone.x[:, 2] < 0.0)
        | (_norm(drone.x - jnp.asarray(params.center)) > params.arena)
        | (drone.R[:, 2, 2] < 0.0)
        | ~jnp.all(jnp.isfinite(drone.x), axis=-1)
    )


def compute_reward(state, alive, died, params):
    cfg, d = params.reward, state.drone
    cost = cfg.distance * _norm(state.goal - d.x) + cfg.spin * _norm(d.omega)
    return jnp.where(alive, -params.policy_dt * cost, 0.0) - cfg.crash * died


def _random_tilt(key, max_angle):
    k_axis, k_angle = jax.random.split(key)
    axis = jax.random.normal(k_axis, (3,))
    K = dynamics.skew(axis / _norm(axis))
    a = jax.random.uniform(k_angle, (), minval=0.0, maxval=max_angle)
    return jnp.eye(3) + jnp.sin(a) * K + (1.0 - jnp.cos(a)) * (K @ K)


def reset(key, params):
    n = params.n_drones
    k_pos, k_vel, k_att, k_goal = jax.random.split(key, 4)
    center = jnp.asarray(params.center)

    hover_rpm = jnp.sqrt(
        params.model.mass * params.model.g / (params.model.n_motors * params.model.kf)
    )
    drone = dynamics.State(
        x=center
        + jax.random.uniform(
            k_pos, (n, 3), minval=-params.start_pos_range, maxval=params.start_pos_range
        ),
        v=jax.random.uniform(
            k_vel, (n, 3), minval=-params.start_vel_range, maxval=params.start_vel_range
        ),
        R=jax.vmap(_random_tilt, in_axes=(0, None))(
            jax.random.split(k_att, n), params.start_tilt
        ),
        omega=jnp.zeros((n, 3)),
        rpm=jnp.full((n, 4), hover_rpm),
    )

    goal = center + jnp.asarray(params.goal_range) * jax.random.uniform(
        k_goal, (n, 3), minval=-1.0, maxval=1.0
    )

    state = EnvState(
        drone=drone,
        goal=goal,
        rate_pid=control.pid_init((n, 3)),
        alive=jnp.ones(n, bool),
        time=jnp.int32(0),
    )
    return get_obs(state, params), state


def step(key, state, action, params):
    del key
    action = jnp.clip(action, -1.0, 1.0)
    throttle, rate_ref = action_to_command(action, params)
    gains = control.cascade_gains(params.model)
    rate_gains, allocation, dt = gains[3], gains[4], params.sim_dt

    def substep(carry, _):
        drone, pid = carry

        def one(d, r, t, p):
            group, p = control.rate_controller(
                d, r, t, p, rate_gains, dt, d_on_measurement=True
            )
            return (
                dynamics.step(d, control.mixer(group, allocation), params.model, dt),
                p,
            )

        return jax.vmap(one)(drone, rate_ref, throttle, pid), None

    (drone, rate_pid), _ = jax.lax.scan(
        substep, (state.drone, state.rate_pid), None, length=params.steps_per_action
    )

    died = state.alive & is_dead(drone, params)
    alive = state.alive & ~died

    new_state = EnvState(
        drone=drone,
        goal=state.goal,
        rate_pid=rate_pid,
        alive=alive,
        time=state.time + 1,
    )

    reward = compute_reward(new_state, state.alive, died, params)
    truncated = new_state.time >= params.max_steps
    done = truncated | ~jnp.any(alive)

    info = {
        "alive": alive,
        "died_this_step": died,
        "truncated": truncated,
        "distance": _norm(new_state.goal - drone.x),
    }
    return get_obs(new_state, params), new_state, reward, done, info


PRESETS = {
    "default": EnvParams(goal_range=(4.0, 4.0, 2.0)),
    "hover": EnvParams(),
}
