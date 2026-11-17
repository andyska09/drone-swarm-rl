import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import pytest

from swarm.envs import chase, circle_swap, core, random_goals
from swarm.learn import config, evaluate, vecenv

SOLO = circle_swap.EnvParams(roles=("drone",))


def test_cascade_arrives_and_stays():
    params = SOLO
    act, start = circle_swap.PILOTS["cascade"](params, jnp.arange(1))
    keys = jax.random.split(jax.random.PRNGKey(0), 32)

    fly = lambda key: evaluate.rollout(key, circle_swap, params, act, start)
    drone, goal, _, _, _, _ = jax.vmap(fly)(keys)

    distance = core.norm(goal - drone.x)
    settled = distance[:, -int(1.0 / params.policy_dt) :]
    assert jnp.all(settled < 0.2), f"worst settled distance {settled.max():.3f} m"


def test_the_fatal_step_pays_the_crash_penalty():
    key = jax.random.PRNGKey(0)
    _, state = circle_swap.reset(key, SOLO)
    motors_off = jnp.array([[-1.0, 0.0, 0.0, 0.0]])

    def body(state, _):
        _, state, reward, _, info = circle_swap.step(key, state, motors_off, SOLO)
        return state, (reward[0], info["alive"][0])

    _, (reward, alive) = jax.lax.scan(body, state, None, length=SOLO.max_steps)
    assert not bool(alive[-1]), "motors off and the drone never died"
    death = int(jnp.argmin(alive))
    assert float(reward[death]) <= -SOLO.reward.crash, "no crash penalty on the fatal step"


def chase_scene(x, limit):
    params = chase.PRESETS["default"]
    cfg = config.TrainConfig(task="chase", preset="default", num_envs=1)
    _, vec = vecenv.reset(jax.random.PRNGKey(0), chase, params, cfg)
    drone = vec.env.drone.replace(x=jnp.array([x]), R=jnp.broadcast_to(jnp.eye(3), (1, 2, 3, 3)))
    vec = vec.replace(env=vec.env.replace(drone=drone), limit=jnp.full(1, limit, jnp.int32))
    action = jnp.zeros((1, params.n_drones, chase.NUM_ACTIONS))
    return lambda vec: vecenv.step(vec, action, chase, params, cfg), vec, params


def test_the_step_limit_bootstraps_unless_the_task_ends_the_episode():
    step, vec, params = chase_scene([[5.0, 0.0, 8.0], [-5.0, 0.0, 8.0]], 2)
    _, vec, _, done, info = step(vec)
    assert not bool(done[0])
    _, vec, _, done, info = step(vec)
    assert bool(done[0]) and bool(info["end"]["time_limit"][0])
    assert list(info["bootstrap"][0]) == [1.0, 1.0], "the step limit is not terminal"
    assert int(vec.limit[0]) == params.max_steps, "the next episode is not full length"

    step, vec, _ = chase_scene([[0.0, 0.0, 8.0], [0.2, 0.0, 7.1]], 1)
    _, _, _, done, info = step(vec)
    assert bool(done[0]) and bool(info["end"]["caught_e"][0])
    assert not bool(info["end"]["time_limit"][0])
    assert list(info["bootstrap"][0]) == [0.0, 0.0], "a catch is terminal"


@pytest.mark.parametrize("pursuers", (1, 3))
def test_chase_reads_its_drone_count_from_roles(pursuers):
    params = chase.EnvParams(roles=("pursuer",) * pursuers + ("evader",))
    n, k = params.n_drones, params.n_visible
    obs, state = chase.reset(jax.random.PRNGKey(0), params)

    action = jnp.zeros((n, chase.NUM_ACTIONS))
    _, _, reward, done, info = chase.step(jax.random.PRNGKey(1), state, action, params)

    assert obs.actor["pursuer"].shape == (pursuers, 20 + 8 * k)
    assert obs.actor["evader"].shape == (1, 20 + 8 * k)
    assert reward.shape == (n,) and done.shape == () and info["end"]["caught_e"].shape == ()
    assert chase.reference(state, params).shape == (n, 3)

    actor = jnp.concatenate([obs.actor["pursuer"], obs.actor["evader"]])
    slots, filled = actor[:, 20 : 20 + 7 * k].reshape(n, k, 7), actor[:, 20 + 7 * k :]
    role = jnp.array([params.roles[i] == "pursuer" for i in range(n)])
    assert jnp.all(filled == 1.0), "a drone in range lost its slot"
    for i in range(n):
        assert int(slots[i, :, -1].sum()) == int(jnp.sum(role == role[i])) - 1


THREE = chase.EnvParams(roles=("pursuer",) * 3 + ("evader",))


def chase_step(x):
    _, state = chase.reset(jax.random.PRNGKey(0), THREE)
    drone = state.drone.replace(x=jnp.array(x), R=jnp.stack([jnp.eye(3)] * THREE.n_drones))
    action = jnp.zeros((THREE.n_drones, chase.NUM_ACTIONS))
    return chase.step(jax.random.PRNGKey(1), state.replace(drone=drone), action, THREE)


def test_one_catch_pays_every_pursuer():
    x = [[-6.0, 0.0, 8.0], [0.0, 0.0, 8.0], [6.0, 0.0, 8.0], [0.2, 0.0, 7.1]]
    _, _, reward, done, info = chase_step(x)
    assert bool(done) and bool(info["end"]["caught_e"])
    assert jnp.all(reward[:3] > 0.9 * THREE.reward.catch), "a pursuer went unpaid"
    assert reward[3] < -0.9 * THREE.reward.catch, "the evader was not billed"


def test_a_net_on_another_pursuer_kills_both():
    x = [[0.3, 0.4, 7.1], [0.0, 0.0, 8.0], [6.0, 0.0, 8.0], [-6.0, 0.0, 8.0]]
    _, _, _, done, info = chase_step(x)
    assert list(info["alive"]) == [False, False, True, True]
    assert bool(done) and bool(info["end"]["coll_pp"])


def test_chase_starts_keep_the_drones_apart_and_inside_the_box():
    keys = jax.random.split(jax.random.PRNGKey(0), 256)
    _, state = jax.vmap(chase.reset, in_axes=(0, None))(keys, THREE)
    _, start = chase.PILOTS["paths"](THREE, jnp.array([3]))
    on_path, _ = jax.vmap(start)(state)

    room = 0.5 * jnp.asarray(THREE.arena) - THREE.margin + 1e-9
    eye = jnp.eye(THREE.n_drones, dtype=bool)
    for x in (state.drone.x, on_path.drone.x):
        assert jnp.all(jnp.abs(x - jnp.asarray(THREE.center)) <= room), "a start is too near a face"
        gap = jnp.linalg.norm(x[:, :, None] - x[:, None, :], axis=-1)
        assert jnp.all(jnp.where(eye, jnp.inf, gap) >= THREE.spacing), "two drones started too near"


def test_after_a_failure_only_the_survivor_bootstraps():
    step, vec, _ = chase_scene([[5.0, 0.0, 8.0], [-5.0, 0.0, -0.5]], 100)
    _, _, _, done, info = step(vec)
    assert bool(done[0]) and bool(info["end"]["crash_e"][0])
    assert list(info["bootstrap"][0]) == [1.0, 0.0], "pursuer, evader"


def test_a_dead_drone_keeps_its_state():
    params = random_goals.EnvParams(roles=("drone",) * 2)
    key = jax.random.PRNGKey(0)
    _, before = random_goals.reset(key, params)
    under_the_floor = before.drone.x.at[0, 2].set(-1.0)
    before = before.replace(drone=before.drone.replace(x=under_the_floor))

    action = jnp.zeros((2, random_goals.NUM_ACTIONS))
    _, after, _, _, info = random_goals.step(key, before, action, params)

    assert list(info["alive"]) == [False, True]
    same = jax.tree.map(lambda a, b: bool(jnp.all(a[0] == b[0])), after.drone, before.drone)
    assert all(jax.tree.leaves(same)), "the dead drone moved"
