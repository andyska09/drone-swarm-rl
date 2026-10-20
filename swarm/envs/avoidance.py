"""Two drones fly straight lines that meet, each paid for staying near a point that
moves along its own line.

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
    scale: float = 1.0
    crash: float = 30.0


@flax.struct.dataclass
class EnvParams(core.Common):
    reward: RewardConfig = flax.struct.field(default_factory=RewardConfig)

    roles: tuple = flax.struct.field(pytree_node=False, default=("drone", "drone"))
    max_steps: int = flax.struct.field(pytree_node=False, default=600)

    cruise: float = 3.0
    angle_min: float = 90.0
    angle_max: float = 180.0
    meet_radius: float = 2.0
    miss: float = 1.0
    before_min: float = 6.0
    before_max: float = 12.0
    after: float = 4.0

    sight: float = float("inf")


@flax.struct.dataclass
class EnvState:
    drone: dynamics.State
    rate_pid: control.PIDState
    start: jnp.ndarray
    direction: jnp.ndarray
    length: jnp.ndarray
    travelled: jnp.ndarray
    alive: jnp.ndarray
    time: jnp.ndarray
    closest: jnp.ndarray
    unseen: jnp.ndarray
    unseen_distance: jnp.ndarray


def moving_point(state, params):
    point = state.start + state.travelled * state.direction
    return point, params.cruise * state.direction


def sees(x, params):
    return jnp.any(core.pair_dist(x) < params.sight, axis=1)


def lines(key, params):
    k_meet, k_reach, k_yaw, k_angle, k_side, k_miss, k_around, k_before = (
        jax.random.split(key, 8)
    )
    toward = jax.random.normal(k_meet, (3,))
    reach = params.meet_radius * jax.random.uniform(k_reach) ** (1.0 / 3.0)
    meet = jnp.asarray(params.center) + reach * toward / core.norm(toward)

    angle = jax.random.uniform(k_angle, minval=params.angle_min, maxval=params.angle_max)
    side = jnp.where(jax.random.bernoulli(k_side), 1.0, -1.0)
    yaw = jax.random.uniform(k_yaw, maxval=2.0 * jnp.pi) + jnp.array(
        [0.0, side * jnp.deg2rad(angle)]
    )
    direction = jnp.stack([jnp.cos(yaw), jnp.sin(yaw), jnp.zeros(2)], axis=-1)

    closing = direction[0] - direction[1]
    across = jnp.array([-closing[1], closing[0], 0.0]) / core.norm(closing)
    around = jax.random.uniform(k_around, maxval=2.0 * jnp.pi)
    up = jnp.array([0.0, 0.0, 1.0])
    miss = jax.random.uniform(k_miss, maxval=params.miss) * (
        jnp.cos(around) * across + jnp.sin(around) * up
    )

    before = jax.random.uniform(
        k_before, minval=params.before_min, maxval=params.before_max
    )
    start = meet + 0.5 * jnp.stack([miss, -miss]) - before * direction
    return start, direction, before + params.after


def get_obs(state, params):
    d = state.drone
    others, mask = core.others(d, params)
    point, velocity = moving_point(state, params)
    return Obs(
        own=core.own_obs(d, params),
        others=others,
        others_mask=mask * (core.norm(others[..., :3]) < params.sight),
        target=core.target_obs(d, point, velocity),
        scene=jnp.concatenate(
            [
                core.scene_obs(d),
                point.reshape(-1),
                velocity.reshape(-1),
                jnp.atleast_1d(state.length - state.travelled),
                sees(d.x, params).astype(point.dtype),
            ]
        ),
    )


def reference(state, params):
    return moving_point(state, params)[0]


def compute_reward(distance, alive, died, params):
    cfg = params.reward
    pay = params.policy_dt * jnp.exp(-distance / cfg.scale)
    return jnp.where(alive, pay, 0.0) - cfg.crash * died


def reset(key, params):
    n = params.n_drones
    k_lines, k_spawn = jax.random.split(key)
    start, direction, length = lines(k_lines, params)
    drone = core.spawn(k_spawn, start, params)
    drone = drone.replace(v=drone.v + params.cruise * direction)

    state = EnvState(
        drone=drone,
        rate_pid=control.pid_init((n, 3)),
        start=start,
        direction=direction,
        length=length,
        travelled=jnp.zeros(()),
        alive=jnp.ones(n, bool),
        time=jnp.int32(0),
        closest=jnp.min(core.pair_dist(drone.x)),
        unseen=jnp.zeros(n, jnp.int32),
        unseen_distance=jnp.zeros(n),
    )
    return get_obs(state, params), state


def step(key, state, action, params):
    del key
    drone, rate_pid, _ = core.fly(state.drone, state.rate_pid, action, params)

    out = core.out_of_arena(drone, params)
    contact = jnp.any(core.hits(drone, params), axis=-1)
    died = state.alive & (out | contact)
    alive = state.alive & ~died

    travelled = state.travelled + params.cruise * params.policy_dt
    point, _ = moving_point(state.replace(travelled=travelled), params)
    distance = core.norm(drone.x - point)
    unseen = ~sees(drone.x, params)

    new_state = state.replace(
        drone=drone,
        rate_pid=rate_pid,
        travelled=travelled,
        alive=alive,
        time=state.time + 1,
        closest=jnp.minimum(state.closest, jnp.min(core.pair_dist(drone.x))),
        unseen=state.unseen + unseen,
        unseen_distance=state.unseen_distance + unseen * distance,
    )

    reward = compute_reward(distance, state.alive, died, params)
    finished = travelled >= state.length
    crashed = ~jnp.all(alive)
    limit = new_state.time >= params.max_steps
    truncated = limit & ~finished & ~crashed
    done = finished | crashed | limit

    info = {
        "alive": alive,
        "died_this_step": died,
        "truncated": truncated,
        "end": {
            "success": finished & ~crashed,
            "contact": jnp.any(died & contact),
            "out": jnp.any(died & out),
            "closest": new_state.closest,
            "point_distance_unseen": jnp.where(
                new_state.unseen > 0,
                new_state.unseen_distance / jnp.maximum(new_state.unseen, 1),
                jnp.nan,
            ),
        },
        "step": {"point_distance": distance},
    }
    return get_obs(new_state, params), new_state, reward, done, info


PRESETS = {
    "d_inf": EnvParams(),
    "d6": EnvParams(sight=6.0),
    "d5": EnvParams(sight=5.0),
    "d4": EnvParams(sight=4.0),
    "d3": EnvParams(sight=3.0),
}
