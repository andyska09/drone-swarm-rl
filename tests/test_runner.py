"""Check saved run data, config hashes, and that a run resumes exactly."""

import csv
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from swarm import envs
from swarm.envs import avoidance
from swarm.learn import config, pool, runner


def small(**overrides):
    return config.TrainConfig(
        task="circle_swap", preset="pair", num_envs=8, num_steps=8, total_timesteps=8 * 8 * 4,
        num_minibatches=2, checkpoint_every=2, log_every=100, **overrides
    )


def test_run_writes_itself(tmp_path):
    cfg = small()
    path, _ = runner.train(cfg, root=tmp_path)

    spec = json.loads((path / "config.json").read_text())
    assert spec["git"]["commit"], "a run with no commit cannot be reproduced"
    assert spec["train"]["seed"] == cfg.seed
    assert spec["env"]["max_steps"] == 700, "the env config is not on the record"

    rows = list(csv.DictReader((path / "metrics.csv").open()))
    assert len(rows) == cfg.num_updates
    assert int(rows[-1]["steps"]) == cfg.total_timesteps

    saved = sorted(p.name for p in (path / "checkpoints").glob("*.pkl"))
    assert saved == ["0000002.pkl", "0000004.pkl"]
    assert (path / "latest.pkl").exists() and (path / "stdout.log").exists()


def test_the_hash_follows_the_config(tmp_path):
    _, env_params = envs.make("circle_swap", "pair")

    def digest(cfg, where):
        return runner.new_run(cfg, env_params, tmp_path / where).name.split("_")[-1]

    assert digest(small(), "a") == digest(small(), "b"), "same config, two names"
    assert digest(small(), "c") != digest(small(lr=1e-3), "d"), "two configs, one name"


def rows(file):
    with file.open(newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("kind", ("plain", "pool", "curriculum"))
def test_a_resumed_run_trains_on_exactly(tmp_path, monkeypatch, kind):
    """The run stops in its last checkpoint, after its last rows. The resume runs the last
    update again from the checkpoint before, and must write the same rows."""

    pooled = kind == "pool"
    cfg = dataclasses.replace(small(), checkpoint_every=1)
    if pooled:
        # Generations 0 and 1 of two updates, each update one per side.
        cfg = config.TrainConfig(
            task="chase", preset="default", num_envs=64, num_steps=8, num_minibatches=2,
            total_timesteps=8 * 64 * 8, generations=1, checkpoint_every=1, log_every=100,
        )
        monkeypatch.setattr(pool, "HEDGE_EPISODES", 4)
    if kind == "curriculum":
        # The drones go blind right after update 3, the checkpoint the resume starts from.
        cfg = dataclasses.replace(cfg, task="avoidance", preset="d_inf")
        harder = lambda params, row: params.replace(sight=0.0) if row["update"] == 3 else None
        monkeypatch.setattr(avoidance, "curriculum", harder)

    class Stop(Exception):
        pass

    save = runner.save

    def stop_at_update_4(path, state):
        if state.update == 4:
            raise Stop
        save(path, state)

    monkeypatch.setattr(runner, "save", stop_at_update_4)
    with pytest.raises(Stop):
        runner.train(cfg, root=tmp_path)
    monkeypatch.setattr(runner, "save", save)

    (path,) = tmp_path.iterdir()
    metrics = rows(path / "metrics.csv")
    assert int(metrics[-1]["steps"]) == cfg.total_timesteps, "steps did not count every learner"
    hedge = rows(path / "hedge.csv") if pooled else []
    runner.resume(path)

    same = lambda r: {k: v for k, v in r.items() if k != "seconds"}
    assert [same(r) for r in rows(path / "metrics.csv")] == [same(r) for r in metrics]
    if pooled:
        assert rows(path / "hedge.csv") == hedge, "the resumed phase weighed its pool again"
        phase = [r for r in hedge if (r["generation"], r["training"]) == ("1", "pursuer")]
        assert [r["member"] for r in phase] == ["paths", "evader_00"]
