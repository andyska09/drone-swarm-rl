"""Check the env contract of swarm/envs/__init__.py on every task and preset."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import pytest

TASKS = ("circle_swap", "chase", "avoidance")
ENVS = {task: importlib.import_module(f"swarm.envs.{task}") for task in TASKS}
EPISODES = 3
# The vmap and the loop round differently, and a tumbling drone grows the difference.
COMPARED = 50


def rollout(env, params, key, actions):
    _, state = env.reset(key, params)

    def body(state, action):
        obs, state, reward, done, info = env.step(key, state, action, params)
        return state, (obs, reward, done, info["alive"], info["flying"])

    return jax.lax.scan(body, state, actions)[1]


def same_shapes(a, b):
    return jax.tree.structure(a) == jax.tree.structure(b) and all(
        (x.shape, x.dtype) == (y.shape, y.dtype)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


@pytest.mark.parametrize("task, preset", [(t, p) for t in TASKS for p in ENVS[t].PRESETS])
def test_the_task_keeps_the_contract(task, preset):
    env = ENVS[task]
    params = env.PRESETS[preset]
    n, key = params.n_drones, jax.random.PRNGKey(0)

    obs, state = jax.eval_shape(lambda k: env.reset(k, params), key)
    action = jnp.zeros((n, env.NUM_ACTIONS))
    stepped = jax.eval_shape(lambda k, s: env.step(k, s, action, params), key, state)
    obs_after, state_after, reward, done, info = stepped
    assert same_shapes(obs, obs_after) and same_shapes(state, state_after), "a step changed a shape"
    assert reward.shape == (n,) and done.shape == ()
    assert all(info[f].shape == (n,) for f in ("alive", "flying", "bootstrap"))

    keys = jax.random.split(key, EPISODES)
    actions = jax.random.uniform(
        key, (EPISODES, params.max_steps, n, env.NUM_ACTIONS), minval=-1.0, maxval=1.0
    )
    batch = jax.jit(jax.vmap(lambda k, a: rollout(env, params, k, a)))(keys, actions)
    one = jax.jit(lambda k, a: rollout(env, params, k, a[:COMPARED]))
    for i in range(EPISODES):
        alone = one(keys[i], actions[i])
        first = jax.tree.map(lambda x: x[i, :COMPARED], batch)
        close = jax.tree.map(lambda a, b: bool(jnp.allclose(a, b, equal_nan=True)), first, alone)
        assert all(jax.tree.leaves(close)), "vmap and a Python loop disagree"

    obs, reward, done, alive, flying = batch
    assert bool(jnp.all(jnp.any(done, axis=1))), "random actions never ended an episode"
    for i, end in enumerate(jnp.argmax(done, axis=1)):
        upto = jax.tree.map(lambda x: x[i, : int(end) + 1], (obs, reward))
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(upto)), "not finite"

    before = jnp.concatenate([jnp.ones_like(flying[:, :1]), flying[:, :-1]], axis=1)
    assert not bool(jnp.any(flying & ~before)), "a drone that stopped flew again"
    assert not bool(jnp.any(alive[:, 1:] & ~alive[:, :-1])), "a dead drone came back"
    assert bool(jnp.all(before | (reward == 0.0))), "a drone was paid after it stopped"
