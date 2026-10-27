"""Two drones fly head-on through a closed square tube, each to the other's start.

The design: research/notes/task_avoidance.md.
"""

import flax.struct
import jax
import jax.numpy as jnp

from swarm.envs import Obs, core
from swarm.sim import control, dynamics

NUM_ACTIONS = 4
PROGRESS = ("ep_length", "death_rate")


@flax.struct.dataclass
class RewardConfig:
    arrive: float = 1.0
    crash: float = 1.0


@flax.struct.dataclass
class EnvParams(core.Common):
    reward: RewardConfig = flax.struct.field(default_factory=RewardConfig)

    roles: tuple = flax.struct.field(pytree_node=False, default=("drone", "drone"))
    max_steps: int = flax.struct.field(pytree_node=False, default=1100)

    width: float = 3.0
    length: float = 5.0
    length_step: float = 5.0
    length_max: float = 20.0
    grow_at: float = 0.8

    start_gap: float = 1.0
    start_box: float = 1.6
    arrive_radius: float = 1.0
    timeout: float = 10.0

    sight: float = float("inf")


@flax.struct.dataclass
class EnvState:
    drone: dynamics.State
    rate_pid: control.PIDState
    heading: jnp.ndarray
    length: jnp.ndarray
    goal: jnp.ndarray
    mirror: jnp.ndarray
    turn: jnp.ndarray
    alive: jnp.ndarray
    arrived: jnp.ndarray
    arrive_time: jnp.ndarray
    time: jnp.ndarray
    closest: jnp.ndarray


def middle(params):
    return jnp.array([params.center[0], params.center[1], 0.5 * params.width])


def half(state, params):
    return jnp.array([0.5 * state.length, 0.5 * params.width, 0.5 * params.width])


def to_tube(x, heading, params):
    return (x - middle(params)) @ core.yaw(heading)


def to_world(p, heading, params):
    return middle(params) + p @ core.yaw(heading).T


def sees(state, params):
    flying = state.alive & ~state.arrived
    near = core.pair_dist(state.drone.x) < params.sight
    return near & flying[:, None] & flying[None, :]


def get_obs(state, params):
    d = state.drone
    tube = core.yaw(state.heading)
    p = to_tube(d.x, state.heading, params)
    R = tube.T @ d.R
    drones = jnp.concatenate(
        [
            p,
            d.v @ tube,
            R.reshape(-1, 9),
            d.omega,
            to_tube(state.goal, state.heading, params),
            state.arrived[:, None].astype(p.dtype),
            state.mirror[:, None].astype(p.dtype),
            jnp.cos(state.turn)[:, None],
            jnp.sin(state.turn)[:, None],
        ],
        axis=-1,
    )
    clock = params.timeout - state.time * params.policy_dt

    # Each drone sees the world through F = yaw(turn) S, where S mirrors y if `mirror`.
    # Its body y axis flips with the mirror, so its R stays a rotation.
    flip = jnp.where(state.mirror[:, None], jnp.array([1.0, -1.0, 1.0]), 1.0)
    F = jax.vmap(core.yaw)(state.turn) * flip[:, None, :]
    on = lambda v: jnp.einsum("nij,nj->ni", F, v)
    seen = d.replace(
        x=on(d.x),
        v=on(d.v),
        R=F @ d.R * flip[:, None, :],
        # A rate is a pseudo-vector, so the mirror flips roll and yaw, not pitch.
        omega=flip[:, 1:2] * flip * d.omega,
    )
    walls = core.box_dist(
        p * flip, flip[:, :, None] * R * flip[:, None, :], half(state, params)
    )
    # Each drone measures the others in its own body frame, mirrored with its world.
    mirrored = d.replace(R=d.R * flip[:, None, :])
    rows = jnp.concatenate(
        [
            core.own_obs(seen, walls),
            core.others(mirrored, params, sees(state, params), core.same_role(params.roles)),
            core.target_obs(seen, on(state.goal)),
        ],
        axis=-1,
    )
    return Obs(
        actor=core.by_role(rows, params.roles),
        critic=jnp.concatenate([drones.reshape(-1), jnp.stack([state.length, clock])]),
    )


def reference(state, params):
    del params
    return state.goal


def scenery(state, params):
    corners = jnp.array(
        [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]]
    )
    return {
        "walls": to_world(corners * half(state, params), state.heading, params),
        "wall_height": jnp.asarray(params.width),
    }


def compute_reward(arrive, died, params):
    return params.reward.arrive * arrive - params.reward.crash * died


def action_signs(state, params):
    """A drone that sees the world in a mirror flies mirrored roll and yaw rates."""

    del params
    s = jnp.where(state.mirror, -1.0, 1.0)
    return jnp.stack([jnp.ones_like(s), s, jnp.ones_like(s), s], axis=-1)


def reset(key, params):
    n = params.n_drones
    k_heading, k_across, k_spawn, k_mirror, k_turn = jax.random.split(key, 5)
    heading = jax.random.uniform(k_heading, (), maxval=2.0 * jnp.pi)
    length = jnp.asarray(params.length, float)
    end = 0.5 * length - params.start_gap
    along = jnp.where(jnp.arange(n) % 2 == 0, -end, end)
    across = jax.random.uniform(
        k_across, (n, 2), minval=-0.5 * params.start_box, maxval=0.5 * params.start_box
    )
    start = to_world(jnp.concatenate([along[:, None], across], axis=-1), heading, params)
    drone = core.spawn(k_spawn, start, params)
    mapped = jnp.arange(n) % 2 == 1

    state = EnvState(
        drone=drone,
        rate_pid=control.pid_init((n, 3)),
        heading=heading,
        length=length,
        goal=start[::-1],
        mirror=mapped & jax.random.bernoulli(k_mirror, shape=(n,)),
        turn=jnp.where(mapped, jax.random.uniform(k_turn, (n,), maxval=2.0 * jnp.pi), 0.0),
        alive=jnp.ones(n, bool),
        arrived=jnp.zeros(n, bool),
        arrive_time=jnp.full(n, jnp.nan, float),
        time=jnp.int32(0),
        closest=jnp.min(core.pair_dist(drone.x)),
    )
    return get_obs(state, params), state


def step(key, state, action, params):
    del key
    flying = state.alive & ~state.arrived
    signed = action * action_signs(state, params)
    moved = core.fly(state.drone, state.rate_pid, signed, params)[:2]
    drone, rate_pid = jax.tree.map(
        lambda a, b: jnp.where(jnp.expand_dims(flying, tuple(range(1, a.ndim))), a, b),
        moved,
        (state.drone, state.rate_pid),
    )
    p = to_tube(drone.x, state.heading, params)

    room = half(state, params) - core.body_radius(params)
    wall = flying & (
        jnp.any(jnp.abs(p) > room, axis=-1) | ~jnp.all(jnp.isfinite(p), axis=-1)
    )
    both = flying[:, None] & flying[None, :]
    contact = jnp.any(core.hits(drone, params) & both, axis=-1)
    died = wall | contact
    alive = state.alive & ~died
    arrive = flying & ~died & (core.norm(drone.x - state.goal) < params.arrive_radius)
    arrived = state.arrived | arrive

    time = state.time + 1
    seconds = time * params.policy_dt
    new_state = state.replace(
        drone=drone,
        rate_pid=rate_pid,
        alive=alive,
        arrived=arrived,
        arrive_time=jnp.where(arrive, seconds, state.arrive_time),
        time=time,
        closest=jnp.minimum(
            state.closest, jnp.min(jnp.where(both, core.pair_dist(drone.x), jnp.inf))
        ),
    )

    reward = compute_reward(arrive, died, params)
    crashed = jnp.any(died)
    finished = jnp.all(arrived)
    out_of_time = seconds >= params.timeout
    over = crashed | finished | out_of_time

    info = {
        "alive": alive,
        "flying": alive & ~arrived,
        # Every end of the task is terminal.
        "bootstrap": jnp.zeros(params.n_drones, bool),
        "end": {
            "success": finished,
            "wall": jnp.any(wall),
            "contact": jnp.any(contact),
            "timeout": out_of_time & ~finished & ~crashed,
            "closest": new_state.closest,
            "time_to_arrive": new_state.arrive_time,
            "length": state.length,
        },
        "step": {},
    }
    return get_obs(new_state, params), new_state, reward, over, info


def eval_params(params):
    return params.replace(length=params.length_max)


def curriculum(params, metrics):
    settled = abs(metrics["length"] - params.length) < 1e-3
    if settled and metrics["success"] > params.grow_at and params.length < params.length_max:
        return params.replace(
            length=min(params.length + params.length_step, params.length_max)
        )
    return None


def cascade(params, rows):
    """The cascade flies each drone to its goal. It reads the true world, so it sends a
    mirrored drone mirrored commands, and `step` mirrors them back."""

    fly, start = core.cascade_pilot(params, rows, lambda state: state.goal[rows])

    def act(state, obs, pids):
        action, pids = fly(state, obs, pids)
        return action * action_signs(state, params)[rows], pids

    return act, start


PILOTS = {"cascade": cascade}


PRESETS = {
    "d_inf": EnvParams(),
    "d6": EnvParams(sight=6.0),
    "d5": EnvParams(sight=5.0),
    "d4": EnvParams(sight=4.0),
    "d3": EnvParams(sight=3.0),
    "d0": EnvParams(sight=0.0),
}
