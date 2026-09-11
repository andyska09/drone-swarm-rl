"""Check cascade goal tracking, episode endings, and batched env behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import pytest

from swarm.envs import a_to_b, flat
from swarm.sim import control

PRESETS = ("hover", "default")


def cascade_policy(params):
    gains = control.cascade_gains(params.model)
    pids = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (params.n_drones,) + x.shape),
        control.cascade_init(),
    )

    def act(state, pids):
        def one(drone, goal, pid):
            (throttle, rate_ref), pid = control.cascade_outer(
                drone, goal, 0.0, pid, gains, params.model, params.policy_dt
            )
            return a_to_b.command_to_action(throttle, rate_ref, params), pid

        return jax.vmap(one)(state.drone, state.goal, pids)

    return act, pids


def replay(state, actions):
    return actions[state.time], actions


def rollout(params, key, act, carry):
    _, state = a_to_b.reset(key, params)

    def body(both, _):
        state, carry = both
        action, carry = act(state, carry)
        _, state, reward, done, info = a_to_b.step(key, state, action, params)
        return (state, carry), (reward, done, info["distance"], info["alive"])

    return jax.lax.scan(body, (state, carry), None, length=params.max_steps)


@pytest.mark.parametrize("preset", PRESETS)
def test_cascade_arrives_and_stays(preset):
    params = a_to_b.PRESETS[preset]
    act, pids = cascade_policy(params)
    keys = jax.random.split(jax.random.PRNGKey(0), 32)

    _, (_, _, distance, _) = jax.vmap(rollout, in_axes=(None, 0, None, None))(
        params, keys, act, pids
    )

    settled = distance[:, -int(1.0 / params.policy_dt) :]
    assert jnp.all(settled < 0.2), f"worst settled distance {settled.max():.3f} m"


def test_shapes_and_dtypes_survive_a_step():
    params = a_to_b.PRESETS["default"]
    n, k = params.n_drones, params.n_visible
    obs, state = a_to_b.reset(jax.random.PRNGKey(0), params)

    assert obs.own.shape == (n, 16)
    assert obs.neighbors.shape == (n, k, a_to_b.NEIGHBOR_FEATURES)
    assert obs.target.shape == (n, 6) and obs.target_mask.shape == (n,)
    assert flat(obs).shape == (n, 16 + k * a_to_b.NEIGHBOR_FEATURES + k + 7)

    action = jnp.zeros((n, a_to_b.NUM_ACTIONS))
    obs2, state2, reward, done, info = a_to_b.step(jax.random.PRNGKey(1), state, action, params)

    assert reward.shape == (n,) and done.shape == ()
    assert info["alive"].shape == (n,) and info["truncated"].shape == ()
    assert jax.tree.structure(obs) == jax.tree.structure(obs2)

    for field, before in jax.tree_util.tree_leaves_with_path(state):
        after = dict(jax.tree_util.tree_leaves_with_path(state2))[field]
        assert before.dtype == after.dtype, f"dtype drift in {field}"


def test_random_actions_stay_finite_and_terminate():
    params = a_to_b.PRESETS["hover"]
    key = jax.random.PRNGKey(0)
    actions = jax.random.uniform(
        key, (params.max_steps, params.n_drones, a_to_b.NUM_ACTIONS), minval=-1, maxval=1
    )
    _, (reward, done, distance, _) = rollout(params, key, replay, actions)

    assert jnp.all(jnp.isfinite(reward)) and jnp.all(jnp.isfinite(distance))
    assert jnp.any(done), "random actions never ended an episode"


def test_dead_drone_freezes():
    params = a_to_b.PRESETS["hover"]
    key = jax.random.PRNGKey(0)
    actions = jnp.tile(
        jnp.array([-1.0, 0.0, 0.0, 0.0]), (params.max_steps, params.n_drones, 1)
    )
    _, (_, _, distance, alive) = rollout(params, key, replay, actions)

    death = int(jnp.argmin(alive[:, 0]))
    assert 0 < death < params.max_steps, "motors off and the drone never died"
    assert not jnp.any(alive[death:]), "a dead drone came back"
    assert jnp.allclose(distance[death:], distance[death]), "a dead drone kept moving"


def test_timeout_truncates_without_terminating():
    params = a_to_b.PRESETS["hover"]
    _, state = a_to_b.reset(jax.random.PRNGKey(0), params)
    state = state.replace(time=jnp.int32(params.max_steps - 1))
    action = jnp.zeros((params.n_drones, a_to_b.NUM_ACTIONS))

    _, _, _, done, info = a_to_b.step(jax.random.PRNGKey(1), state, action, params)
    assert bool(done) and bool(info["truncated"]) and bool(jnp.all(info["alive"]))


def test_vmap_matches_python_loop():
    params = a_to_b.PRESETS["default"]
    act, pids = cascade_policy(params)
    keys = jax.random.split(jax.random.PRNGKey(3), 4)

    batched = jax.vmap(rollout, in_axes=(None, 0, None, None))(params, keys, act, pids)[1][2]
    looped = jnp.stack([rollout(params, k, act, pids)[1][2] for k in keys])
    assert jnp.allclose(batched, looped)
