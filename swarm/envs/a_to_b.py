"""Fly to a stationary goal and hold position using throttle and body-rate commands."""

import flax.struct
import jax
import jax.numpy as jnp

from swarm.envs import Obs, core
from swarm.sim import control, dynamics

NUM_ACTIONS = 4


@flax.struct.dataclass
class RewardConfig:
    distance: float = 1.0
    crash: float = 10.0
    spin: float = 0.1


@flax.struct.dataclass
class EnvParams(core.Common):
    reward: RewardConfig = flax.struct.field(default_factory=RewardConfig)

    start_pos_range: float = 0.5
    goal_range: tuple = flax.struct.field(pytree_node=False, default=(0.0, 0.0, 0.0))

    # The cascade gate allows 5 s, which is not enough to fly out of a yaw slew.
    random_yaw: bool = flax.struct.field(pytree_node=False, default=False)


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
    cost = cfg.distance * core.norm(state.goal - d.x) + cfg.spin * core.norm(d.omega)
    return jnp.where(alive, -params.policy_dt * cost, 0.0) - cfg.crash * died


def reset(key, params):
    n = params.n_drones
    k_pos, k_goal, k_spawn = jax.random.split(key, 3)
    center = jnp.asarray(params.center)

    x = center + jax.random.uniform(
        k_pos, (n, 3), minval=-params.start_pos_range, maxval=params.start_pos_range
    )
    goal = center + jnp.asarray(params.goal_range) * jax.random.uniform(
        k_goal, (n, 3), minval=-1.0, maxval=1.0
    )

    state = EnvState(
        drone=core.spawn(k_spawn, x, params),
        goal=goal,
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
    done = truncated | ~jnp.all(alive)

    info = {
        "alive": alive,
        "died_this_step": died,
        "truncated": truncated,
        "distance": core.norm(new_state.goal - drone.x),
    }
    return get_obs(new_state, params), new_state, reward, done, info


PRESETS = {
    "default": EnvParams(goal_range=(4.0, 4.0, 2.0)),
    "hover": EnvParams(),
}
