"""N drones on a circle, each flying to the opposite point; any contact ends the episode."""

import flax.struct
import jax
import jax.numpy as jnp

from swarm.envs import Obs
from swarm.sim import control, dynamics

NUM_ACTIONS = 4
NEIGHBOR_FEATURES = 6


@flax.struct.dataclass
class RewardConfig:
    distance: float = 1.0
    crash: float = 25.0  # per drone in the scene
    spin: float = 0.1
    close: float = 10.0
    orient: float = 1.0


@flax.struct.dataclass
class EnvParams:
    model: dynamics.Params = flax.struct.field(default_factory=dynamics.default_params)
    reward: RewardConfig = flax.struct.field(default_factory=RewardConfig)

    roles: tuple = flax.struct.field(pytree_node=False, default=("drone", "drone"))
    scripted: tuple = flax.struct.field(pytree_node=False, default=())

    sim_dt: float = 0.005
    steps_per_action: int = flax.struct.field(pytree_node=False, default=2)
    max_steps: int = flax.struct.field(pytree_node=False, default=700)

    max_rate_rp: float = 4.0
    max_rate_yaw: float = 2.0

    # The ring, the goals and the arena all sit around this point.
    center: tuple = flax.struct.field(pytree_node=False, default=(0.0, 0.0, 3.0))
    radius: float = 3.0
    start_vel_range: float = 0.5
    start_tilt: float = 0.2
    arena: float = 10.0

    # MRS kills on arm+prop+arm+prop; close_dist is where the soft cost starts.
    collision_dist: float = 0.8
    close_dist: float = 1.5

    @property
    def n_drones(self):
        return len(self.roles)

    @property
    def n_visible(self):
        return self.n_drones - 1

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


def _pair_dist(x):
    """Drone-to-drone distances, diagonal at infinity so nobody is their own neighbour."""

    d = _norm(x[None, :, :] - x[:, None, :])
    return jnp.where(jnp.eye(x.shape[0], dtype=bool), jnp.inf, d)


def _neighbors(drone, params):
    _, idx = jax.lax.top_k(-_pair_dist(drone.x), params.n_visible)
    take = lambda a: jnp.take_along_axis(a, idx[..., None], axis=1)
    rel = lambda a: jnp.einsum("nji,nkj->nki", drone.R, take(a))
    return jnp.concatenate(
        [
            rel(drone.x[None, :, :] - drone.x[:, None, :]),
            rel(drone.v[None, :, :] - drone.v[:, None, :]),
        ],
        axis=-1,
    )


def get_obs(state, params):
    d, n = state.drone, params.n_drones
    return Obs(
        # Height is a scalar, so it carries no frame and keeps the body-frame symmetry.
        own=jnp.concatenate(
            [_to_body(d.R, d.v), d.R.reshape(n, 9), d.omega, d.x[:, 2:3]], axis=-1
        ),
        neighbors=_neighbors(d, params),
        neighbor_mask=jnp.ones((n, params.n_visible)),
        target=jnp.concatenate(
            [_to_body(d.R, state.goal - d.x), _to_body(d.R, -d.v)], axis=-1
        ),
        target_mask=jnp.ones(n),
    )


def reference(state, params):
    """Where each drone is trying to fly: the cascade's input, and the viewer's marker."""

    del params
    return state.goal


def is_dead(drone, params):
    return (
        (drone.x[:, 2] < 0.0)
        | (_norm(drone.x - jnp.asarray(params.center)) > params.arena)
        | (drone.R[:, 2, 2] < 0.0)
        | jnp.any(_pair_dist(drone.x) < params.collision_dist, axis=-1)
        | ~jnp.all(jnp.isfinite(drone.x), axis=-1)
    )


def compute_reward(state, alive, died, params):
    cfg, d = params.reward, state.drone
    close = jnp.sum(
        jnp.maximum(1.0 - _pair_dist(d.x) / params.close_dist, 0.0), axis=-1
    )
    cost = (
        cfg.distance * _norm(state.goal - d.x)
        + cfg.spin * _norm(d.omega)
        + cfg.close * close
        - cfg.orient * d.R[:, 2, 2]
    )
    crash = cfg.crash * params.n_drones
    return jnp.where(alive, -params.policy_dt * cost, 0.0) - crash * died


def _start_attitude(key, yaw, max_angle):
    """Body x points along `yaw`, plus a random tilt of up to `max_angle`."""

    k_axis, k_angle = jax.random.split(key)
    axis = jax.random.normal(k_axis, (3,))
    K = dynamics.skew(axis / _norm(axis))
    a = jax.random.uniform(k_angle, (), minval=0.0, maxval=max_angle)
    tilt = jnp.eye(3) + jnp.sin(a) * K + (1.0 - jnp.cos(a)) * (K @ K)

    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ tilt


def reset(key, params):
    n = params.n_drones
    k_vel, k_att, k_yaw = jax.random.split(key, 3)
    center = jnp.asarray(params.center)

    angle = 2.0 * jnp.pi * jnp.arange(n) / n
    ring = params.radius * jnp.stack(
        [jnp.cos(angle), jnp.sin(angle), jnp.zeros(n)], axis=-1
    )

    hover_rpm = jnp.sqrt(
        params.model.mass * params.model.g / (params.model.n_motors * params.model.kf)
    )
    drone = dynamics.State(
        x=center + ring,
        v=jax.random.uniform(
            k_vel, (n, 3), minval=-params.start_vel_range, maxval=params.start_vel_range
        ),
        # Random heading, so the policy cannot key on one fixed R per drone.
        R=jax.vmap(_start_attitude, in_axes=(0, 0, None))(
            jax.random.split(k_att, n),
            jax.random.uniform(k_yaw, (n,), maxval=2.0 * jnp.pi),
            params.start_tilt,
        ),
        omega=jnp.zeros((n, 3)),
        rpm=jnp.full((n, 4), hover_rpm),
    )

    state = EnvState(
        drone=drone,
        goal=center - ring,
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
    # Shared fate: one death ends the episode, so nobody ever observes a dead drone.
    done = truncated | ~jnp.all(alive)

    distance = _norm(new_state.goal - drone.x)
    # Averaging over drones hides the one that failed, so the scene-wide worst
    # cases are their own scalars.
    info = {
        "alive": alive,
        "died_this_step": died,
        "truncated": truncated,
        "distance": distance,
        "worst_distance": jnp.max(distance),
        "min_dist": jnp.min(_pair_dist(drone.x)),
    }
    return get_obs(new_state, params), new_state, reward, done, info


PRESETS = {
    "pair": EnvParams(),
    "circle8": EnvParams(roles=("drone",) * 8),
}
