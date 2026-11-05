"""N drones fly to random goals, one after another. A dead drone leaves, the others fly on."""

import flax.struct
import jax
import jax.numpy as jnp

from swarm.envs import Obs, core
from swarm.sim import control, dynamics

NUM_ACTIONS = 4
PROGRESS = ("ep_length", "death_rate", "goals", "contact")
# One draw of 10 drones keeps the spacing about 1 time in 4.
START_DRAWS = 64


@flax.struct.dataclass
class RewardConfig:
    goal: float = 1.0  # per metre closer to the goal
    crash: float = 30.0


@flax.struct.dataclass
class EnvParams(core.Common):
    reward: RewardConfig = flax.struct.field(default_factory=RewardConfig)

    max_steps: int = flax.struct.field(pytree_node=False, default=1000)

    # The starts and the goals lie in this box around `center`.
    goal_box: tuple = flax.struct.field(pytree_node=False, default=(20.0, 20.0, 1.0))
    goal_radius: float = 0.5
    spacing: float = 2.0

    blind: bool = flax.struct.field(pytree_node=False, default=False)
    sense: tuple = flax.struct.field(pytree_node=False, default=())


@flax.struct.dataclass
class EnvState:
    drone: dynamics.State
    rate_pid: control.PIDState
    key: jnp.ndarray  # eval gives `step` the same key each step, so goals draw from this
    goal: jnp.ndarray
    alive: jnp.ndarray
    reached: jnp.ndarray  # the goals that each drone reached
    contact: jnp.ndarray  # the drone died by touching another drone
    closest: jnp.ndarray


def in_box(key, n, params):
    half = 0.5 * jnp.asarray(params.goal_box)
    points = jax.random.uniform(key, (n, 3), minval=-half, maxval=half)
    return jnp.asarray(params.center) + points


def both_fly(alive):
    return alive[:, None] & alive[None, :]


def get_obs(state, params):
    d = state.drone
    seen = both_fly(state.alive) & (not params.blind)
    actor = jnp.concatenate(
        [
            core.own_obs(d, core.wall_dist(d, params)),
            core.others(d, params, seen, params.sense),
            core.target_obs(d, state.goal),
        ],
        axis=-1,
    )
    return Obs(
        actor=core.by_role(actor, params.roles),
        critic=jnp.concatenate(
            [
                core.scene_obs(d),
                state.goal.reshape(-1),
                d.rpm.reshape(-1) / params.model.max_rpm,
                state.alive.astype(d.x.dtype),
            ]
        ),
    )


def reference(state, params):
    del params
    return state.goal


def compute_reward(state, drone, died, params):
    cfg = params.reward
    progress = core.norm(state.goal - state.drone.x) - core.norm(state.goal - drone.x)
    return cfg.goal * progress - cfg.crash * died


def reset(key, params):
    n = params.n_drones
    k_start, k_spawn, k_goal, k_state = jax.random.split(key, 4)
    draws = jax.vmap(in_box, in_axes=(0, None, None))(
        jax.random.split(k_start, START_DRAWS), n, params
    )
    drone = core.spawn(k_spawn, core.first_apart(draws, params.spacing), params)

    state = EnvState(
        drone=drone,
        rate_pid=control.pid_init((n, 3)),
        key=k_state,
        goal=in_box(k_goal, n, params),
        alive=jnp.ones(n, bool),
        reached=jnp.zeros(n, jnp.int32),
        contact=jnp.zeros(n, bool),
        closest=jnp.min(core.pair_dist(drone.x)),
    )
    return get_obs(state, params), state


def step(key, state, action, params):
    del key
    flown, flown_pid, _ = core.fly(state.drone, state.rate_pid, action, params)

    both = both_fly(state.alive)
    contact = jnp.any(core.hits(flown, params) & both, axis=-1)
    died = contact | (state.alive & core.lost(flown, params))
    alive = state.alive & ~died
    # A dead drone stays where the step found it, so a broken state cannot reach the others.
    drone, rate_pid = core.select_rows(
        alive, (flown, flown_pid), (state.drone, state.rate_pid)
    )

    reach = alive & (core.norm(state.goal - drone.x) < params.goal_radius)
    k_state, k_goal = jax.random.split(state.key)
    new_state = state.replace(
        drone=drone,
        rate_pid=rate_pid,
        key=k_state,
        goal=jnp.where(
            reach[:, None], in_box(k_goal, params.n_drones, params), state.goal
        ),
        alive=alive,
        reached=state.reached + reach,
        contact=state.contact | contact,
        closest=jnp.minimum(
            state.closest, jnp.min(jnp.where(both, core.pair_dist(flown.x), jnp.inf))
        ),
    )

    info = {
        "alive": alive,
        "flying": alive,
        # Only the death of the last drone ends the episode, and a death is terminal.
        "bootstrap": jnp.zeros(params.n_drones, bool),
        "end": {
            "goals": new_state.reached,
            "contact": new_state.contact,
            "wall": ~alive & ~new_state.contact,
            "closest": new_state.closest,
        },
        "step": {},
    }
    reward = compute_reward(state, drone, died, params)
    return get_obs(new_state, params), new_state, reward, ~jnp.any(alive), info


def cascade(params, rows):
    """The cascade flies each drone to its goal."""

    return core.cascade_pilot(params, rows, lambda state: state.goal[rows])


PILOTS = {"cascade": cascade}


TRAFFIC = EnvParams(roles=("drone",) * 10, n_neighbors=9)

PRESETS = {
    "alone": EnvParams(),
    "blind": TRAFFIC.replace(blind=True),
    "base": TRAFFIC,
    "attitude": TRAFFIC.replace(sense=("attitude",)),
    "rpm": TRAFFIC.replace(sense=("rpm",)),
    "attitude_rpm": TRAFFIC.replace(sense=("attitude", "rpm")),
}
