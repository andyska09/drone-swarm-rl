"""Check that PPO updates are repeatable and that env metrics reach the caller."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

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
