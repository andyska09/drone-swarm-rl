"""Two drones fly head-on through a closed square tube, each to the other's start.

The design: research/notes/task_avoidance.md.
"""

import flax.struct
import jax
import jax.numpy as jnp

from swarm.envs import Obs, core
from swarm.sim import control, dynamics

NUM_ACTIONS = 4


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
    time_limit: float = 10.0

    sight: float = float("inf")


@flax.struct.dataclass
class EnvState:
    drone: dynamics.State
    rate_pid: control.PIDState
    heading: jnp.ndarray
    length: jnp.ndarray
    goal: jnp.ndarray
    alive: jnp.ndarray
    arrived: jnp.ndarray
    arrive_time: jnp.ndarray
    steps: jnp.ndarray
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
    turn = core.yaw(state.heading)
    p = to_tube(d.x, state.heading, params)
    R = turn.T @ d.R
    others, mask = core.others(d, params)
    _, idx = jax.lax.top_k(-core.pair_dist(d.x), params.n_visible)
    drones = jnp.concatenate(
        [
            p,
            d.v @ turn,
            R.reshape(-1, 9),
            d.omega,
            to_tube(state.goal, state.heading, params),
            state.arrived[:, None].astype(p.dtype),
        ],
        axis=-1,
    )
    clock = params.time_limit - state.steps * params.policy_dt
    return Obs(
        own=core.own_obs(d, core.box_dist(p, R, half(state, params))),
        others=others,
        others_mask=mask * jnp.take_along_axis(sees(state, params), idx, axis=1),
        target=core.target_obs(d, state.goal),
        scene=jnp.concatenate([drones.reshape(-1), jnp.stack([state.length, clock])]),
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


def reset(key, params):
    n = params.n_drones
    k_heading, k_across, k_spawn = jax.random.split(key, 3)
    heading = jax.random.uniform(k_heading, (), maxval=2.0 * jnp.pi)
    length = jnp.asarray(params.length, float)
    end = 0.5 * length - params.start_gap
    along = jnp.where(jnp.arange(n) % 2 == 0, -end, end)
    across = jax.random.uniform(
        k_across, (n, 2), minval=-0.5 * params.start_box, maxval=0.5 * params.start_box
    )
    start = to_world(jnp.concatenate([along[:, None], across], axis=-1), heading, params)
    drone = core.spawn(k_spawn, start, params)

    state = EnvState(
        drone=drone,
        rate_pid=control.pid_init((n, 3)),
        heading=heading,
        length=length,
        goal=start[::-1],
        alive=jnp.ones(n, bool),
        arrived=jnp.zeros(n, bool),
        arrive_time=jnp.full(n, jnp.nan, float),
        steps=jnp.int32(0),
        time=jnp.int32(0),
        closest=jnp.min(core.pair_dist(drone.x)),
    )
    return get_obs(state, params), state


def step(key, state, action, params):
    del key
    flying = state.alive & ~state.arrived
    moved = core.fly(state.drone, state.rate_pid, action, params)[:2]
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

    steps = state.steps + 1
    seconds = steps * params.policy_dt
    new_state = state.replace(
        drone=drone,
        rate_pid=rate_pid,
        alive=alive,
        arrived=arrived,
        arrive_time=jnp.where(arrive, seconds, state.arrive_time),
        steps=steps,
        time=state.time + 1,
        closest=jnp.minimum(
            state.closest, jnp.min(jnp.where(both, core.pair_dist(drone.x), jnp.inf))
        ),
    )

    reward = compute_reward(arrive, died, params)
    crashed = jnp.any(died)
    finished = jnp.all(arrived)
    out_of_time = seconds >= params.time_limit
    over = crashed | finished | out_of_time
    limit = new_state.time >= params.max_steps

    info = {
        "alive": alive,
        "died_this_step": died,
        "arrived": arrived,
        "arrived_this_step": arrive,
        "truncated": limit & ~over,
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
    return get_obs(new_state, params), new_state, reward, over | limit, info


def eval_params(params):
    return params.replace(length=params.length_max)


def curriculum(params, metrics):
    settled = abs(metrics["length"] - params.length) < 1e-3
    if settled and metrics["success"] > params.grow_at and params.length < params.length_max:
        return params.replace(
            length=min(params.length + params.length_step, params.length_max)
        )
    return None


PRESETS = {
    "d_inf": EnvParams(),
    "d6": EnvParams(sight=6.0),
    "d5": EnvParams(sight=5.0),
    "d4": EnvParams(sight=4.0),
    "d3": EnvParams(sight=3.0),
    "d0": EnvParams(sight=0.0),
}
