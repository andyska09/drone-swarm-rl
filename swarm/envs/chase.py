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

    distance: float = 0.1  # λdist   = 0.001
    step: float = 4.0  # λstep   = 0.04
    cmd: float = 0.02  # λcmd    = 2e-4
    catch: float = 10.0  # λcatch  = 10.0
    collPE: float = 10.0  # λcollPE = 0.1, per step, the episode goes on
    collPP: float = 10.0  # λcollPP = 10.0, ends the episode
    crash: float = 30.0  # λfail   = 30.0, a wall or the ground, ends the episode


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


def _drop(params):
    """The net hangs by net_offset"""

    return params.net_offset + 0.5 * params.net_side


def net_centre(drone, params):
    """Gavin's c_net: the middle of the square, straight down the body z axis."""

    pursuers, _ = _sides(params)
    return drone.x[pursuers] - _drop(params) * drone.R[pursuers, :, 2]


def net_frame(drone, params):
    """(P, N, 3): every drone seen from every pursuer's net. The net is a square in the body yz plane, so x is the distance across it and (y, z) the spot on it."""

    pursuers, _ = _sides(params)
    p = jnp.einsum(
        "pji,pkj->pki", drone.R[pursuers], drone.x - drone.x[pursuers][:, None]
    )
    return p.at[..., 2].add(_drop(params))


def in_net(drone, params):
    """(P, N): drone k touches pursuer p's net.

    The square has no thickness. The drone is a ball, so it touches the square when its centre is within one body radius of the plane and sits over the square.
    """

    p = net_frame(drone, params)
    return jnp.all(jnp.abs(p[..., 1:]) <= 0.5 * params.net_side, axis=-1) & (
        jnp.abs(p[..., 0]) <= core.body_radius(params)
    )


def is_caught(drone, params):
    _, evader = _sides(params)
    return in_net(drone, params)[:, evader]


def net_collision(drone, params):
    """(N,): a pursuer's net touches another pursuer. Both of them die."""

    pursuers, evader = _sides(params)
    hit = in_net(drone, params).at[:, evader].set(False)
    netter = jnp.zeros(params.n_drones, bool).at[pursuers].set(jnp.any(hit, axis=-1))
    return netter | jnp.any(hit, axis=0)


def get_obs(state, params):
    others, mask = core.others(state.drone, params)
    return Obs(
        own=core.own_obs(state.drone, params),
        others=others,
        others_mask=mask,
        # Nobody here flies at a fixed point: a pursuer chases a drone and the evader runs from one, and both sit in `others`.
        target=jnp.zeros((params.n_drones, 0)),
    )


def crashed(drone, params):
    """Hit the world: a wall, the ground, the ceiling, or a non-finite state."""

    return core.out_of_arena(drone, params)


def collPP(drone, params):
    """(N,): a pursuer against another pursuer, body or net. Ends the episode."""

    _, evader = _sides(params)
    hit = core.hits(drone, params).at[evader].set(False).at[:, evader].set(False)
    return jnp.any(hit, axis=-1) | net_collision(drone, params)


def collPE(drone, params):
    """(P,): a pursuer body against the evader body. Costs, but does not end it."""

    pursuers, evader = _sides(params)
    return core.hits(drone, params)[pursuers, evader]


def is_dead(drone, params):
    """Both end the episode, but they are paid at different rates, so `step` keeps them apart"""

    return crashed(drone, params) | collPP(drone, params)


def reference(state, params):
    """A pursuer flies its NET onto the evader, so it aims `_drop` above it. Flying
    its body there would hold the evader above the square and never catch."""

    pursuers, evader = _sides(params)
    d = state.drone
    aim = d.x[evader] + _drop(params) * d.R[pursuers, :, 2]
    return jnp.broadcast_to(state.waypoint, (params.n_drones, 3)).at[pursuers].set(aim)


def compute_reward(state, rate_ref, alive, crash, collide, caught, params):
    cfg, d = params.reward, state.drone
    pursuers, evader = _sides(params)
    distance = core.norm(d.x[evader] - net_centre(d, params))

    # Gavin charges both sides the same for a pursuer-evader collision.
    hit = collPE(d, params)
    both = jnp.zeros(params.n_drones).at[pursuers].set(hit).at[evader].set(jnp.any(hit))

    # The pursuers pay the clock and the evader earns it — Gavin's λstep.
    clock = jnp.full(params.n_drones, -cfg.step).at[evader].set(cfg.step)
    clock = clock.at[pursuers].add(-cfg.distance * distance) - cfg.collPE * both
    # One catch pays every pursuer and bills the evader once.
    catch = jnp.full(params.n_drones, cfg.catch).at[evader].set(-cfg.catch)

    return (
        jnp.where(
            alive, params.policy_dt * (clock - cfg.cmd * core.norm(rate_ref)), 0.0
        )
        + catch * caught
        - cfg.collPP * collide
        - cfg.crash * crash
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
    pursuers, evader = _sides(params)
    action, evader_pid = _fly_scripted(state, action, params)
    drone, rate_pid, rate_ref = core.fly(state.drone, state.rate_pid, action, params)

    caught = jnp.any(is_caught(drone, params)) & jnp.all(state.alive)
    # The net hangs through the hitbox, so a catch is a catch, not a collision.
    crash = state.alive & crashed(drone, params) & ~caught
    collide = state.alive & collPP(drone, params) & ~caught
    died = crash | collide
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

    reward = compute_reward(
        new_state, rate_ref, state.alive, crash, collide, caught, params
    )
    truncated = new_state.time >= params.max_steps
    done = truncated | caught | ~jnp.all(alive)

    # Gavin's score for the pursuers, read at the step the episode ends.
    rho = jnp.where(
        caught,
        1.0 - new_state.time / params.max_steps,
        (died[evader] & ~jnp.any(died[pursuers])).astype(float),
    )

    on_net = net_frame(drone, params)[:, evader, 1:]
    info = {
        "alive": alive,
        "died_this_step": died,
        "truncated": truncated,
        "caught": caught,
        "rho": rho,
        "distance": jnp.min(core.norm(drone.x[evader] - net_centre(drone, params))),
        # How far off the net centre the catch landed.
        "accuracy": jnp.where(caught, jnp.min(core.norm(on_net)), 0.0),
        "crashed": jnp.any(crash),
        "collPP": jnp.any(collide),
        "collPE": jnp.any(collPE(drone, params)),
        "evader_died": died[evader],
    }
    return get_obs(new_state, params), new_state, reward, done, info


PRESETS = {
    "default": EnvParams(),
    "duel": EnvParams(scripted=()),
}
