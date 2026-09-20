"""Check that PPO updates are repeatable and that env metrics reach the caller."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from swarm import envs
from swarm.learn import config, ppo


def train(updates, seed=0):
    cfg = config.TrainConfig(
        seed=seed, num_envs=128, num_steps=32, total_timesteps=300_000, num_minibatches=4
    )
    env, env_params = envs.make("a_to_b", "hover")
    # Short episodes so some finish inside a rollout and the per-episode metrics exist.
    env_params = env_params.replace(max_steps=100)
    init, update = ppo.make(cfg, env, env_params)

    carry = init(jax.random.PRNGKey(cfg.seed))
    history = []
    for _ in range(updates):
        carry, metrics = update(*carry)
        history.append(metrics)
    return history


def test_same_seed_same_numbers():
    a, b = train(3), train(3)
    assert [float(m["policy_loss"]) for m in a] == [float(m["policy_loss"]) for m in b]


def test_metrics_carry_the_env_measurements():
    metrics = train(1)[0]
    assert "distance" in metrics, "an info scalar the env declared did not reach the metrics"
    assert set(metrics) >= {"return", "ep_length", "death_rate", "explained_var", "approx_kl"}


def test_a_frozen_role_does_not_move():
    """Both sides learn, but train_roles names one. Task 4 alternates like this."""

    cfg = config.TrainConfig(
        num_envs=64,
        num_steps=32,
        total_timesteps=100_000,
        num_minibatches=4,
        train_roles=("pursuer",),
    )
    env, env_params = envs.make("chase", "default")
    # Nothing scripted, so the evader gets its own weights and its own loss.
    env_params = env_params.replace(scripted=(), max_steps=100)

    init, update = ppo.make(cfg, env, env_params)
    carry = init(jax.random.PRNGKey(0))
    before = jax.tree.map(jnp.copy, {r: t.params for r, t in carry[0].items()})
    carry, _ = update(*carry)
    after = {r: t.params for r, t in carry[0].items()}

    assert set(before) == {"pursuer", "evader"}
    moved = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), before, after)
    assert not any(jax.tree.leaves(moved["evader"])), "a frozen role took a step"
    assert any(jax.tree.leaves(moved["pursuer"])), "the trained role did not move"

    # The swap passes the turn in as an argument, which overrides train_roles.
    before = jax.tree.map(jnp.copy, {r: t.params for r, t in carry[0].items()})
    carry, _ = update(*carry, {"pursuer": False, "evader": True})
    after = {r: t.params for r, t in carry[0].items()}

    moved = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), before, after)
    assert not any(jax.tree.leaves(moved["pursuer"])), "the frozen turn took a step"
    assert any(jax.tree.leaves(moved["evader"])), "the training turn did not move"
