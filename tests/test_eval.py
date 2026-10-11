"""Check what eval writes, and that hand-written seats score their known numbers."""

import json
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarm import envs
from swarm.learn import config, evaluate, ppo, runner

OUTCOMES = ("caught_evader", "crash_pursuer", "OOB_pursuer", "timeout",
            "crash_evader", "OOB_evader", "fail_both")


def fixed_match(seats, episodes=64):
    """Hand-written sources in both chase seats, so no run and no weights."""

    env, params = envs.make("chase", "default")
    cfg = config.TrainConfig(task="chase", preset="default")
    act, start, _ = evaluate.make_policy(cfg, env, params, None, "latest", seats)
    slices = ppo.role_slices(params.roles)
    keys = jax.random.split(jax.random.PRNGKey(evaluate.EVAL_SEED), episodes)

    def measure(key):
        _, _, _, reward, info, live = evaluate.rollout(key, env, params, act, start)
        return evaluate.summarize(reward, info, live, slices)

    return jax.jit(jax.vmap(measure))(keys)


def small(**overrides):
    return config.TrainConfig(
        task="circle_swap", preset="pair", num_envs=8, num_steps=8, total_timesteps=8 * 8 * 4,
        num_minibatches=2, checkpoint_every=2, log_every=100, **overrides
    )


def test_cascade_in_both_seats_scores_its_baseline(tmp_path):
    """A seat that names a source loads no weights, so the run needs only its config."""

    _, params = envs.make("chase", "default")
    path = runner.new_run(config.TrainConfig(task="chase", preset="default"), params, tmp_path)
    out, summary = evaluate.evaluate(
        path, episodes=32, seats=[("pursuer", "cascade"), ("evader", "cascade")]
    )

    caught = summary["caught_evader"]
    assert caught > 0.9, f"the cascade caught the cascade in only {caught:.2f} of episodes"
    assert out.name == "evader_cascade_pursuer_cascade"


def test_eval_writes_what_the_viewer_needs(tmp_path):
    path, _ = runner.train(small(), root=tmp_path)
    out, _ = evaluate.evaluate(path, episodes=16)

    episodes = np.load(out / "episodes.npz")
    assert episodes["return"].shape == (16,), "per-episode arrays, one row per episode"

    traj = np.load(out / "trajectory.npz")
    assert traj["x"].shape == (evaluate.TRAJECTORIES, 700, 2, 3)
    assert traj["R"].shape == (evaluate.TRAJECTORIES, 700, 2, 3, 3)
    # The goal is recorded per step, because a chase target moves.
    assert traj["goal"].shape == (evaluate.TRAJECTORIES, 700, 2, 3)

    header = json.loads((out / "header.json").read_text())
    assert header["roles"] == ["drone", "drone"] and header["arena"] == [32.0, 32.0, 16.0]

    spec = json.loads((out / "eval.json").read_text())
    assert spec["eval_seed"] == evaluate.EVAL_SEED and spec["episodes"] == 16


def test_a_dead_episode_stops_counting(tmp_path):
    """No auto-reset: a crash ends the episode and the held tail must not score."""

    path, _ = runner.train(small(), root=tmp_path)
    out, _ = evaluate.evaluate(path, episodes=64)

    # One death ends a circle_swap episode, so a crash shows as a death rate above 0.
    episodes = np.load(out / "episodes.npz")
    died = episodes["death_rate"] > 0.0
    assert died.any(), "an untrained policy never crashed"
    assert np.all(episodes["ep_length"][died] < 700), "a crashed episode ran the full length"


def test_every_episode_has_exactly_one_outcome():
    per = fixed_match({"pursuer": "frpn", "evader": "apf"})
    total = sum(np.asarray(per[k]) for k in OUTCOMES)
    assert np.allclose(total, 1.0), f"outcomes sum to {sorted(set(total.round(3)))}"


def test_frpn_catches_the_path_evader():
    caught = float(np.mean(fixed_match({"pursuer": "frpn", "evader": "paths"})["caught_evader"]))
    assert caught > 0.6, f"FRPN caught the path evader in only {caught:.2f} of episodes"
