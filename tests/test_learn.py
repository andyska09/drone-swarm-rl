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
        task="circle_swap", preset="pair", seed=seed, num_envs=128, num_steps=32,
        total_timesteps=300_000, num_minibatches=4,
    )
    env, env_params = envs.make(cfg.task, cfg.preset)
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


def test_metrics_split_by_role_and_leave_out_undefined_episodes():
    """The rules that training and eval share, on two episodes of two pursuers and an
    evader: a drone metric splits by role, a scene metric keeps its name, and an
    episode whose value is NaN is left out of the mean."""

    info = {
        "ep_length": jnp.array([10, 20]),
        "ep_return": jnp.array([[1.0, 3.0, -2.0], [5.0, 7.0, -6.0]]),
        "alive": jnp.ones((2, 3), bool),
        "end": {
            "caught": jnp.array([1.0, 0.0]),
            "offset": jnp.array([[0.5, 0.7, 0.0], [jnp.nan, jnp.nan, 0.0]]),
        },
        "step": {},
    }
    slices = envs.role_slices(("pursuer", "pursuer", "evader"))
    # One row per episode, read at its end.
    metrics = ppo.episode_metrics(info, jnp.ones(2), jnp.ones(2), slices, lead=1)

    assert set(metrics) == {
        "ep_length", "return_p", "return_e", "death_rate_p",
        "death_rate_e", "caught", "offset_p", "offset_e",
    }
    assert float(metrics["return_p"]) == 4.0 and float(metrics["return_e"]) == -4.0
    assert float(metrics["caught"]) == 0.5
    assert jnp.isclose(metrics["offset_p"], 0.6), "the NaN episode counted"
    assert float(metrics["offset_e"]) == 0.0


def test_a_frozen_role_does_not_move():
    """A pool learner trains one side, and its `train` freezes the other side."""

    cfg = config.TrainConfig(
        task="chase",
        preset="default",
        num_envs=64,
        num_steps=32,
        total_timesteps=100_000,
        num_minibatches=4,
    )
    env, env_params = envs.make("chase", "default")
    env_params = env_params.replace(max_steps=100)

    init, update = ppo.make(cfg, env, env_params)
    ts, learner = init(jax.random.PRNGKey(0))
    assert set(ts) == {"pursuer", "evader"}

    for side, rival in (("pursuer", "evader"), ("evader", "pursuer")):
        before = jax.tree.map(jnp.copy, {r: t.params for r, t in ts.items()})
        (ts, learner), _ = update(ts, learner.replace(train={side: True, rival: False}))
        after = {r: t.params for r, t in ts.items()}

        moved = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), before, after)
        assert not any(jax.tree.leaves(moved[rival])), f"the frozen {rival} took a step"
        assert any(jax.tree.leaves(moved[side])), f"the trained {side} did not move"
