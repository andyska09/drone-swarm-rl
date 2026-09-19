"""N drones chase one evader and catch it with a net hung under the pursuer's body.

Replicates Gavin & Bronz 2026 (arXiv:2607.05939) with the evader flown by the
cascade instead of a policy. Design and sources: research/notes/task_chase.md.
"""

import flax.struct
import jax
import jax.numpy as jnp

from swarm.envs import Obs, core
from swarm.sim import control, dynamics

NUM_ACTIONS = 4


@flax.struct.dataclass
class RewardConfig:
    """Gavin Nv1 Table I. The per-step weights are his divided by policy_dt."""

    distance: float = 0.1  # λdist  = 0.001
    step: float = 4.0  # λstep  = 0.04
    cmd: float = 0.02  # λcmd   = 2e-4
    catch: float = 10.0  # λcatch = 10.0
    crash: float = 30.0  # λfail  = 30.0


@flax.struct.dataclass
class EnvParams(core.Common):
    reward: RewardConfig = flax.struct.field(default_factory=RewardConfig)

    roles: tuple = flax.struct.field(pytree_node=False, default=("pursuer", "evader"))
    scripted: tuple = flax.struct.field(pytree_node=False, default=("evader",))
    max_steps: int = flax.struct.field(pytree_node=False, default=1000)

    net_side: float = 1.0
    net_offset: float = 0.4

    waypoint_range: float = 3.0
    waypoint_reach: float = 0.2


@flax.struct.dataclass
class EnvState:
    drone: dynamics.State
    rate_pid: control.PIDState
    evader_pid: tuple
    waypoint: jnp.ndarray
    key: jnp.ndarray
    alive: jnp.ndarray
    time: jnp.ndarray


def _sides(params):
    """Drone rows: every pursuer, and the one evader. `roles` is static."""

    return jnp.array([i for i, r in enumerate(params.roles) if r == "pursuer"]), (
        params.roles.index("evader")
    )


def net_centre(drone, params):
    """One point per pursuer, net_offset down its own body z axis."""

    pursuers, _ = _sides(params)
    return drone.x[pursuers] - params.net_offset * drone.R[pursuers, :, 2]


def _net_frame(drone, params):
    """The evader across and along each net: a square hanging net_offset below a
    pursuer, face pointing forward along its body x."""

    pursuers, evader = _sides(params)
    p = jnp.einsum("pji,pj->pi", drone.R[pursuers], drone.x[evader] - drone.x[pursuers])
    return jnp.stack([p[:, 1], p[:, 2] + params.net_offset], axis=-1), p[:, 0]


def is_caught(drone, params):
    """Per pursuer: the evader's centre is inside the square and its body crosses it."""

    across, along = _net_frame(drone, params)
    return jnp.all(jnp.abs(across) <= 0.5 * params.net_side, axis=-1) & (
        jnp.abs(along) <= core.body_radius(params)
    )


def get_obs(state, params):
    others, mask = core.others(state.drone, params)
    return Obs(
        own=core.own_obs(state.drone),
        others=others,
        others_mask=mask,
        # Nobody here flies at a fixed point: a pursuer chases a drone and the
        # evader runs from one, and both sit in `others`.
        target=jnp.zeros((params.n_drones, 0)),
    )


def is_dead(drone, params):
    """Gavin ends an episode on out of bounds and on contact, not on attitude."""

    return core.out_of_arena(drone, params) | jnp.any(core.hits(drone, params), axis=-1)


def reference(state, params):
    pursuers, evader = _sides(params)
    return (
        jnp.broadcast_to(state.waypoint, (params.n_drones, 3))
        .at[pursuers]
        .set(state.drone.x[evader])
    )


def compute_reward(state, rate_ref, alive, died, caught, params):
    cfg, d = params.reward, state.drone
    pursuers, evader = _sides(params)
    distance = core.norm(d.x[evader] - net_centre(d, params))

    # The pursuers pay the clock and the evader earns it — Gavin's λstep.
    clock = jnp.full(params.n_drones, -cfg.step).at[evader].set(cfg.step)
    clock = clock.at[pursuers].add(-cfg.distance * distance)
    # One catch pays every pursuer and bills the evader once.
    catch = jnp.full(params.n_drones, cfg.catch).at[evader].set(-cfg.catch)

    return (
        jnp.where(alive, params.policy_dt * (clock - cfg.cmd * core.norm(rate_ref)), 0.0)
        + catch * caught
        - cfg.crash * died
    )


def _waypoint(key, x, params):
    k_dir, k_len = jax.random.split(key)
    direction = jax.random.normal(k_dir, (3,))
    reach = jax.random.uniform(k_len, (), maxval=params.waypoint_range)
    centre = jnp.asarray(params.center)
    limit = core.half(params) - params.margin
    offset = x - centre + reach * direction / core.norm(direction)
    return centre + jnp.clip(offset, -limit, limit)


def _fly_scripted(state, action, params):
    """The cascade flies the evader to its waypoint; its network row is dropped."""

    if "evader" not in params.scripted:
        return action, state.evader_pid

    _, evader = _sides(params)
    (throttle, rate), pid = control.cascade_outer(
        jax.tree.map(lambda a: a[evader], state.drone),
        state.waypoint,
        0.0,
        state.evader_pid,
        control.cascade_gains(params.model),
        params.model,
        params.policy_dt,
    )
    return action.at[evader].set(core.command_to_action(throttle, rate, params)), pid


def reset(key, params):
    n = params.n_drones
    _, evader = _sides(params)
    k_pos, k_spawn, k_wp, k_state = jax.random.split(key, 4)
    centre = jnp.asarray(params.center)
    limit = core.half(params) - params.margin

    x = centre + jax.random.uniform(k_pos, (n, 3), minval=-limit, maxval=limit)
    drone = core.spawn(k_spawn, x, params)

    state = EnvState(
        drone=drone,
        rate_pid=control.pid_init((n, 3)),
        evader_pid=control.cascade_init(),
        waypoint=_waypoint(k_wp, drone.x[evader], params),
        key=k_state,
        alive=jnp.ones(n, bool),
        time=jnp.int32(0),
    )
    return get_obs(state, params), state


def step(key, state, action, params):
    # The waypoint draw uses the key carried in the state, so callers that reuse
    # one key per rollout still get a fresh draw every step.
    del key
    _, evader = _sides(params)
    action, evader_pid = _fly_scripted(state, action, params)
    drone, rate_pid, rate_ref = core.fly(state.drone, state.rate_pid, action, params)

    caught = jnp.any(is_caught(drone, params)) & jnp.all(state.alive)
    # The net reaches inside the hitbox, so a catch is a catch, not a crash.
    died = state.alive & is_dead(drone, params) & ~caught
    alive = state.alive & ~died

    key, k_wp = jax.random.split(state.key)
    reached = core.norm(state.waypoint - drone.x[evader]) < params.waypoint_reach
    waypoint = jnp.where(
        reached, _waypoint(k_wp, drone.x[evader], params), state.waypoint
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
        "distance": jnp.min(core.norm(drone.x[evader] - net_centre(drone, params))),
        # Pliska's interception accuracy: how far off the net centre the catch was.
        "accuracy": jnp.where(caught, jnp.min(core.norm(across)), 0.0),
        "evader_died": died[evader],
    }
    return get_obs(new_state, params), new_state, reward, done, info


PRESETS = {
    "default": EnvParams(),
    "three": EnvParams(roles=("pursuer",) * 3 + ("evader",)),
}
