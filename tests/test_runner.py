"""Check saved run data, config hashes, and the loss after restoring a checkpoint."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

from swarm import envs
from swarm.learn import config, ppo, runner


def small(**overrides):
    return config.TrainConfig(
        task="a_to_b", preset="hover", num_envs=8, num_steps=8, total_timesteps=8 * 8 * 4,
        num_minibatches=2, checkpoint_every=2, log_every=100, **overrides
    )


def test_run_writes_itself(tmp_path):
    cfg = small()
    path, _ = runner.train(cfg, root=tmp_path)

    spec = json.loads((path / "config.json").read_text())
    assert spec["git"]["commit"], "a run with no commit cannot be reproduced"
    assert spec["train"]["seed"] == cfg.seed
    assert spec["env"]["max_steps"] == 500, "the env config is not on the record"

    rows = list(csv.DictReader((path / "metrics.csv").open()))
    assert len(rows) == cfg.num_updates
    assert int(rows[-1]["steps"]) == cfg.total_timesteps

    saved = sorted(p.name for p in (path / "checkpoints").glob("*.pkl"))
    assert saved == ["0000002.pkl", "0000004.pkl"]
    assert (path / "latest.pkl").exists() and (path / "stdout.log").exists()


def test_the_hash_follows_the_config(tmp_path):
    _, env_params = envs.make("a_to_b", "hover")

    def digest(cfg, where):
        return runner.new_run(cfg, env_params, tmp_path / where).name.split("_")[-1]

    assert digest(small(), "a") == digest(small(), "b"), "same config, two names"
    assert digest(small(), "c") != digest(small(lr=1e-3), "d"), "two configs, one name"


def test_checkpoint_carries_on_exactly(tmp_path):
    cfg = small()
    path, carry = runner.train(cfg, root=tmp_path)

    env, env_params = envs.make(cfg.task, cfg.preset)
    init, update = ppo.make(cfg, env, env_params)
    restored = runner.restore(init(jax.random.PRNGKey(cfg.seed)), runner.load(path))

    _, live_metrics = update(*carry)
    _, restored_metrics = update(*restored)
    assert float(live_metrics["policy_loss"]) == float(restored_metrics["policy_loss"])
