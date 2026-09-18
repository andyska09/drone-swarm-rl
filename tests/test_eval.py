"""Check that eval measures what training measured, and that the cascade scores the M2 gate."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarm.learn import config, evaluate, runner


def small(**overrides):
    return config.TrainConfig(
        task="a_to_b", preset="hover", num_envs=8, num_steps=8, total_timesteps=8 * 8 * 4,
        num_minibatches=2, checkpoint_every=2, log_every=100, **overrides
    )


def test_cascade_in_the_seat_reproduces_the_gate(tmp_path):
    path, _ = runner.train(small(), root=tmp_path)
    out, summary = evaluate.evaluate(path, episodes=32, policy="cascade")

    assert summary["distance_final"] < 0.2, f"cascade settled at {summary['distance_final']:.3f} m"
    assert summary["length"] == 500.0, "the cascade did not fly the whole episode"
    assert summary["alive_final"] == 1.0
    assert out.name == "cascade"


def test_eval_writes_what_the_viewer_needs(tmp_path):
    path, _ = runner.train(small(), root=tmp_path)
    out, _ = evaluate.evaluate(path, episodes=16)

    episodes = np.load(out / "episodes.npz")
    assert episodes["return"].shape == (16,), "per-episode arrays, one row per episode"

    traj = np.load(out / "trajectory.npz")
    assert traj["x"].shape == (evaluate.TRAJECTORIES, 500, 1, 3)
    assert traj["R"].shape == (evaluate.TRAJECTORIES, 500, 1, 3, 3)
    # The goal is recorded per step, because a chase target moves.
    assert traj["goal"].shape == (evaluate.TRAJECTORIES, 500, 1, 3)

    header = json.loads((out / "header.json").read_text())
    assert header["roles"] == ["drone"] and header["arena"] == 10.0

    spec = json.loads((out / "eval.json").read_text())
    assert spec["eval_seed"] == evaluate.EVAL_SEED and spec["episodes"] == 16


def test_a_dead_episode_stops_counting(tmp_path):
    """No auto-reset: a crash ends the episode and the held tail must not score."""

    path, _ = runner.train(small(), root=tmp_path)
    out, _ = evaluate.evaluate(path, episodes=64, preset="default")

    episodes = np.load(out / "episodes.npz")
    died = episodes["alive_final"] == 0.0
    assert died.any(), "an untrained policy on the wide preset never crashed"
    assert np.all(episodes["length"][died] < 500), "a crashed episode ran the full length"
