"""Check that eval measures what training measured, and that the cascade scores the M2 gate."""

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
        task="a_to_b", preset="hover", num_envs=8, num_steps=8, total_timesteps=8 * 8 * 4,
        num_minibatches=2, checkpoint_every=2, log_every=100, **overrides
    )


def test_cascade_in_the_seat_reproduces_the_gate(tmp_path):
    path, _ = runner.train(small(), root=tmp_path)
    out, summary = evaluate.evaluate(path, episodes=32, seats=[("drone", "cascade")])

    assert summary["distance"] < 0.2, f"cascade settled at {summary['distance']:.3f} m"
    assert summary["ep_length"] == 500.0, "the cascade did not fly the whole episode"
    assert summary["death_rate"] == 0.0
    assert out.name == "drone_cascade"


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
    assert header["roles"] == ["drone"] and header["arena"] == [32.0, 32.0, 16.0]

    spec = json.loads((out / "eval.json").read_text())
    assert spec["eval_seed"] == evaluate.EVAL_SEED and spec["episodes"] == 16


def test_a_dead_episode_stops_counting(tmp_path):
    """No auto-reset: a crash ends the episode and the held tail must not score."""

    # The wide preset, so an untrained policy leaves the arena and the episode ends.
    cfg = small()
    cfg.preset = "default"
    path, _ = runner.train(cfg, root=tmp_path)
    out, _ = evaluate.evaluate(path, episodes=64)

    episodes = np.load(out / "episodes.npz")
    died = episodes["death_rate"] == 1.0
    assert died.any(), "an untrained policy on the wide preset never crashed"
    assert np.all(episodes["ep_length"][died] < 500), "a crashed episode ran the full length"


def test_every_episode_has_exactly_one_outcome():
    per = fixed_match({"pursuer": "frpn", "evader": "apf"})
    total = sum(np.asarray(per[k]) for k in OUTCOMES)
    assert np.allclose(total, 1.0), f"outcomes sum to {sorted(set(total.round(3)))}"


def test_frpn_catches_a_hovering_evader():
    caught = float(np.mean(fixed_match({"pursuer": "frpn", "evader": "hover"})["caught_evader"]))
    assert caught > 0.7, f"FRPN caught a hovering evader in only {caught:.2f} of episodes"
