import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarm import envs
from swarm.envs import avoidance, chase
from swarm.learn import config, evaluate, runner

OUTCOMES = ("caught_e", "fail_pe", "crash_p", "OOB_p", "coll_pp",
            "crash_e", "OOB_e", "time_limit")


def fixed_match(seats, params=chase.PRESETS["default"], episodes=64):
    env = chase
    cfg = config.TrainConfig(task="chase", preset="default")
    act, start, _ = evaluate.make_policy(cfg, env, params, None, "latest", seats)
    slices = envs.role_slices(params.roles)
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


def test_cascade_in_both_seats_needs_no_weights(tmp_path):
    _, params = envs.make("chase", "default")
    path = runner.new_run(config.TrainConfig(task="chase", preset="default"), params, tmp_path)
    out, summary = evaluate.evaluate(
        path, episodes=32, seats=[("pursuer", "cascade"), ("evader", "cascade")]
    )

    caught = summary["caught_e"]
    assert caught > 0.9, f"the cascade caught the cascade in only {caught:.2f} of episodes"
    assert out.name == "evader_cascade_pursuer_cascade"


def test_eval_flies_the_recorded_env_or_another_preset(tmp_path):
    recorded = chase.PRESETS["default"].replace(arena=(30.0, 30.0, 16.0))
    run = runner.new_run(config.TrainConfig(task="chase", preset="default"), recorded, tmp_path)
    assert evaluate.eval_env(run)[1].arena == (30.0, 30.0, 16.0)
    assert evaluate.eval_env(run, "small")[1].arena == chase.PRESETS["small"].arena

    params = avoidance.PRESETS["d6"]
    run = runner.new_run(config.TrainConfig(task="avoidance", preset="d6"), params, tmp_path)
    assert evaluate.eval_env(run)[1].length == params.length_max, "the eval values were not applied"


def test_eval_writes_what_the_viewer_needs(tmp_path):
    path, _ = runner.train(small(), root=tmp_path)
    out, _ = evaluate.evaluate(path, episodes=16)

    episodes = np.load(out / "episodes.npz")
    assert episodes["return"].shape == (16,), "per-episode arrays, one row per episode"

    traj = np.load(out / "trajectory.npz")
    assert traj["x"].shape == (evaluate.TRAJECTORIES, 700, 2, 3)
    assert traj["R"].shape == (evaluate.TRAJECTORIES, 700, 2, 3, 3)
    assert traj["goal"].shape == (evaluate.TRAJECTORIES, 700, 2, 3)

    header = json.loads((out / "header.json").read_text())
    assert header["roles"] == ["drone", "drone"] and header["arena"] == [32.0, 32.0, 16.0]

    spec = json.loads((out / "eval.json").read_text())
    assert spec["eval_seed"] == evaluate.EVAL_SEED and spec["episodes"] == 16


def test_a_dead_episode_stops_counting(tmp_path):
    path, _ = runner.train(small(), root=tmp_path)
    out, _ = evaluate.evaluate(path, episodes=64)

    episodes = np.load(out / "episodes.npz")
    died = episodes["death_rate"] > 0.0
    assert died.any(), "an untrained policy never crashed"
    assert np.all(episodes["ep_length"][died] < 700), "a crashed episode ran the full length"


@pytest.mark.parametrize("pursuers", (1, 3))
def test_every_episode_has_exactly_one_outcome(pursuers):
    params = chase.EnvParams(roles=("pursuer",) * pursuers + ("evader",))
    per = fixed_match({"pursuer": "frpn", "evader": "apf"}, params)
    total = sum(np.asarray(per[k]) for k in OUTCOMES)
    assert np.allclose(total, 1.0), f"outcomes sum to {sorted(set(total.round(3)))}"


def test_frpn_catches_the_path_evader():
    caught = float(np.mean(fixed_match({"pursuer": "frpn", "evader": "paths"})["caught_e"]))
    assert caught > 0.6, f"FRPN caught the path evader in only {caught:.2f} of episodes"
