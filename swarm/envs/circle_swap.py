"""N drones on a circle, each flying to the opposite point; any contact ends the episode."""

import flax.struct
import jax
import jax.numpy as jnp

from swarm.envs import Obs, core
from swarm.sim import control, dynamics

NUM_ACTIONS = 4


@flax.struct.dataclass
class RewardConfig:
    distance: float = 1.0
    crash: float = 25.0  # per drone in the scene
    spin: float = 0.1
    close: float = 10.0
    orient: float = 1.0


@flax.struct.dataclass
class EnvParams(core.Common):
    reward: RewardConfig = flax.struct.field(default_factory=RewardConfig)

    roles: tuple = flax.struct.field(pytree_node=False, default=("drone", "drone"))
    max_steps: int = flax.struct.field(pytree_node=False, default=700)

    radius: float = 3.0


@flax.struct.dataclass
class EnvState:
    drone: dynamics.State
    goal: jnp.ndarray
    rate_pid: control.PIDState
    alive: jnp.ndarray
    time: jnp.ndarray


def get_obs(state, params):
    others, mask = core.others(state.drone, params)
    return Obs(
        own=core.own_obs(state.drone, params),
        others=others,
        others_mask=mask,
        target=core.target_obs(state.drone, state.goal),
    )


def reference(state, params):
    del params
    return state.goal


def compute_reward(state, alive, died, params):
    cfg, d = params.reward, state.drone
    close = jnp.sum(
        jnp.maximum(1.0 - core.pair_dist(d.x) / params.close_dist, 0.0), axis=-1
    )
    cost = (
        cfg.distance * core.norm(state.goal - d.x)
        + cfg.spin * core.norm(d.omega)
        + cfg.close * close
        - cfg.orient * d.R[:, 2, 2]
    )
    crash = cfg.crash * params.n_drones
    return jnp.where(alive, -params.policy_dt * cost, 0.0) - crash * died


def reset(key, params):
    n = params.n_drones
    center = jnp.asarray(params.center)

    angle = 2.0 * jnp.pi * jnp.arange(n) / n
    ring = params.radius * jnp.stack(
        [jnp.cos(angle), jnp.sin(angle), jnp.zeros(n)], axis=-1
    )

    state = EnvState(
        drone=core.spawn(key, center + ring, params),
        goal=center - ring,
        rate_pid=control.pid_init((n, 3)),
        alive=jnp.ones(n, bool),
        time=jnp.int32(0),
    )
    return get_obs(state, params), state


def step(key, state, action, params):
    del key
    drone, rate_pid, _ = core.fly(state.drone, state.rate_pid, action, params)

    died = state.alive & core.is_dead(drone, params)
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

    distance = core.norm(new_state.goal - drone.x)
    # Averaging over drones hides the one that failed.
    info = {
        "alive": alive,
        "died_this_step": died,
        "truncated": truncated,
        "distance": distance,
        "worst_distance": jnp.max(distance),
        "min_dist": jnp.min(core.pair_dist(drone.x)),
    }
    return get_obs(new_state, params), new_state, reward, done, info


PRESETS = {
    "pair": EnvParams(),
    "circle8": EnvParams(roles=("drone",) * 8, n_neighbors=7),
}
