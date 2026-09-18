"""One drone chases a scripted evader and catches it with a net hung under its body.

Replicates Gavin & Bronz 2026 (arXiv:2607.05939) with the evader flown by the
cascade instead of a policy. Design and sources: research/notes/task_chase.md.
"""

import flax.struct
import jax
import jax.numpy as jnp

from swarm.envs import Obs
from swarm.sim import control, dynamics

NUM_ACTIONS = 4
NEIGHBOR_FEATURES = 6

PURSUER, EVADER = 0, 1


@flax.struct.dataclass
class RewardConfig:
    """Gavin Nv1 Table I. The per-step weights are his divided by policy_dt."""

    distance: float = 0.1  # λdist  = 0.001
    step: float = 4.0  # λstep  = 0.04
    cmd: float = 0.02  # λcmd   = 2e-4
    catch: float = 10.0  # λcatch = 10.0
    crash: float = 30.0  # λfail  = 30.0


@flax.struct.dataclass
class EnvParams:
    model: dynamics.Params = flax.struct.field(default_factory=dynamics.default_params)
    reward: RewardConfig = flax.struct.field(default_factory=RewardConfig)

    roles: tuple = flax.struct.field(pytree_node=False, default=("pursuer", "evader"))
    # A scripted role gets no network and never enters the PPO loss.
    scripted: tuple = flax.struct.field(pytree_node=False, default=("evader",))

    sim_dt: float = 0.005
    steps_per_action: int = flax.struct.field(pytree_node=False, default=2)
    max_steps: int = flax.struct.field(pytree_node=False, default=1000)

    max_rate_rp: float = 4.0
    max_rate_yaw: float = 2.0

    center: tuple = flax.struct.field(pytree_node=False, default=(0.0, 0.0, 8.0))
    arena: tuple = flax.struct.field(pytree_node=False, default=(32.0, 32.0, 16.0))
    margin: float = 2.0
    start_vel_range: float = 0.5
    start_tilt: float = 0.2

    # capture_dist is only the thickness that stops a fast target skipping the
    # plane between two steps: 15 m/s closing moves 0.15 m per 0.01 s step.
    net_radius: float = 0.5
    net_offset: float = 0.5
    capture_dist: float = 0.2

    waypoint_range: float = 3.0
    waypoint_reach: float = 0.2

    @property
    def n_drones(self):
        return len(self.roles)

    @property
    def n_visible(self):
        """Neighbours are same-role drones only; the other side is `target`."""

        return max(self.roles.count(r) for r in self.roles) - 1

    @property
    def policy_dt(self):
        return self.sim_dt * self.steps_per_action


@flax.struct.dataclass
class EnvState:
    drone: dynamics.State
    rate_pid: control.PIDState
    evader_pid: tuple
    waypoint: jnp.ndarray
    key: jnp.ndarray
    alive: jnp.ndarray
    time: jnp.ndarray


def _norm(v):
    return jnp.sqrt(jnp.sum(v**2, axis=-1) + 1e-12)


def _to_body(R, v):
    return jnp.einsum("nji,nj->ni", R, v)


def _rate_scale(params):
    return jnp.array([params.max_rate_rp, params.max_rate_rp, params.max_rate_yaw])


def _half(params):
    return 0.5 * jnp.asarray(params.arena)


def action_to_command(action, params):
    """CTBR action in [-1,1] to a throttle in [0,1] and body rates in rad/s."""

    return 0.5 * (action[..., 0] + 1.0), action[..., 1:] * _rate_scale(params)


def command_to_action(throttle, rate_ref, params):
    """Convert throttle and body rates to an action clipped to [-1, 1]."""

    action = jnp.concatenate(
        [jnp.atleast_1d(2.0 * throttle - 1.0), rate_ref / _rate_scale(params)]
    )
    return jnp.clip(action, -1.0, 1.0)


def net_centre(drone, params):
    """Where the net hangs: net_offset down the pursuer's body z axis."""

    return drone.x[PURSUER] - params.net_offset * drone.R[PURSUER, :, 2]


def _net_frame(drone, params):
    """The evader in the net's frame: distance across the disc, and along its axis.

    The net is a vertical disc hanging net_offset below the pursuer, its face
    pointing forward along body x. The pursuer drags it into the evader.
    """

    p = drone.R[PURSUER].T @ (drone.x[EVADER] - drone.x[PURSUER])
    across = jnp.array([p[1], p[2] + params.net_offset])
    return _norm(across), p[0]


def is_caught(drone, params):
    across, along = _net_frame(drone, params)
    return (across <= params.net_radius) & (jnp.abs(along) <= params.capture_dist)


def get_obs(state, params):
    d, n = state.drone, params.n_drones
    other = jnp.array([EVADER, PURSUER])
    return Obs(
        # Height is a scalar, so it carries no frame and keeps the body-frame symmetry.
        own=jnp.concatenate(
            [_to_body(d.R, d.v), d.R.reshape(n, 9), d.omega, d.x[:, 2:3]], axis=-1
        ),
        neighbors=jnp.zeros((n, params.n_visible, NEIGHBOR_FEATURES)),
        neighbor_mask=jnp.zeros((n, params.n_visible)),
        target=jnp.concatenate(
            [_to_body(d.R, d.x[other] - d.x), _to_body(d.R, d.v[other] - d.v)], axis=-1
        ),
        target_mask=jnp.ones(n),
    )


def reference(state, params):
    """Where each drone is trying to fly: the cascade's input, and the viewer's marker."""

    del params
    return jnp.stack([state.drone.x[EVADER], state.waypoint])


def is_dead(drone, params):
    off = drone.x - jnp.asarray(params.center)
    return jnp.any(jnp.abs(off) > _half(params), axis=-1) | ~jnp.all(
        jnp.isfinite(drone.x), axis=-1
    )


def compute_reward(state, rate_ref, alive, died, caught, params):
    cfg = params.reward
    effort = cfg.cmd * _norm(rate_ref)
    distance = _norm(state.drone.x[EVADER] - net_centre(state.drone, params))

    # The pursuer pays the clock and the evader earns it — Gavin's λstep.
    pursuer = -(cfg.distance * distance + cfg.step + effort[PURSUER])
    evader = cfg.step - effort[EVADER]

    return (
        jnp.where(alive, params.policy_dt * jnp.stack([pursuer, evader]), 0.0)
        + cfg.catch * jnp.array([1.0, -1.0]) * caught
        - cfg.crash * died
    )


def _waypoint(key, x, params):
    """A point within waypoint_range of x, clipped into the arena minus the margin."""

    k_dir, k_len = jax.random.split(key)
    direction = jax.random.normal(k_dir, (3,))
    reach = jax.random.uniform(k_len, (), maxval=params.waypoint_range)
    centre = jnp.asarray(params.center)
    limit = _half(params) - params.margin
    offset = x - centre + reach * direction / _norm(direction)
    return centre + jnp.clip(offset, -limit, limit)


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
    k_pos, k_vel, k_att, k_yaw, k_wp, k_state = jax.random.split(key, 6)
    centre = jnp.asarray(params.center)
    limit = _half(params) - params.margin

    hover_rpm = jnp.sqrt(
        params.model.mass * params.model.g / (params.model.n_motors * params.model.kf)
    )
    drone = dynamics.State(
        x=centre + jax.random.uniform(k_pos, (n, 3), minval=-limit, maxval=limit),
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
        rate_pid=control.pid_init((n, 3)),
        evader_pid=control.cascade_init(),
        waypoint=_waypoint(k_wp, drone.x[EVADER], params),
        key=k_state,
        alive=jnp.ones(n, bool),
        time=jnp.int32(0),
    )
    return get_obs(state, params), state


def step(key, state, action, params):
    # The waypoint draw uses the key carried in the state, so callers that reuse
    # one key per rollout still get a fresh draw every step.
    del key
    gains = control.cascade_gains(params.model)
    rate_gains, allocation, dt = gains[3], gains[4], params.sim_dt

    # The evader is scripted: the cascade flies it, the policy's row is discarded.
    (throttle, rate), evader_pid = control.cascade_outer(
        jax.tree.map(lambda a: a[EVADER], state.drone),
        state.waypoint,
        0.0,
        state.evader_pid,
        gains,
        params.model,
        params.policy_dt,
    )
    action = action.at[EVADER].set(command_to_action(throttle, rate, params))

    action = jnp.clip(action, -1.0, 1.0)
    throttle, rate_ref = action_to_command(action, params)

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

    caught = is_caught(drone, params) & jnp.all(state.alive)
    died = state.alive & is_dead(drone, params)
    alive = state.alive & ~died

    key, k_wp = jax.random.split(state.key)
    reached = _norm(state.waypoint - drone.x[EVADER]) < params.waypoint_reach
    waypoint = jnp.where(
        reached, _waypoint(k_wp, drone.x[EVADER], params), state.waypoint
    )

    new_state = EnvState(
        drone=drone,
        rate_pid=rate_pid,
        evader_pid=evader_pid,
        waypoint=waypoint,
        key=key,
        alive=alive,
        time=state.time + 1,
    )

    reward = compute_reward(new_state, rate_ref, state.alive, died, caught, params)
    truncated = new_state.time >= params.max_steps
    done = truncated | caught | ~jnp.all(alive)

    across, _ = _net_frame(drone, params)
    info = {
        "alive": alive,
        "died_this_step": died,
        "truncated": truncated,
        "caught": caught,
        "distance": _norm(drone.x[EVADER] - net_centre(drone, params)),
        # Pliska's interception accuracy: how far off the net centre the catch was.
        "accuracy": jnp.where(caught, across, 0.0),
        "evader_died": died[EVADER],
    }
    return get_obs(new_state, params), new_state, reward, done, info


PRESETS = {"default": EnvParams()}
