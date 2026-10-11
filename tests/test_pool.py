"""Check the Hedge weights and that a resumed pool run keeps its pool."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from swarm.learn import config, pool, runner


def rows(file):
    with file.open(newline="") as f:
        return list(csv.DictReader(f))


def test_hedge_weights_follow_eq_12():
    c = np.array([0.8, 0.5, 0.0])
    weights = pool.hedge(1.0 - c)
    assert np.allclose(weights, 0.7 / 3 + 0.3 * c**2 / np.sum(c**2))
    assert np.isclose(weights.sum(), 1.0)

    # The learner beats every opponent, so no c is above 0: the second term is uniform.
    assert np.allclose(pool.hedge([1.0, 1.0, 1.0, 1.0]), 0.25)


def test_a_resume_mid_generation_keeps_its_pool(tmp_path, monkeypatch):
    # Generations 0 and 1 make four phases of two updates each.
    cfg = config.TrainConfig(
        task="chase", preset="default", num_envs=64, num_steps=8, num_minibatches=2,
        total_timesteps=8 * 64 * 8, generations=1, checkpoint_every=100, log_every=100,
    )
    monkeypatch.setattr(pool, "HEDGE_EPISODES", 4)

    class Crash(Exception):
        pass

    write = runner.append_row

    def crash_at_update_8(file, row):
        if row.get("update") == 8:
            raise Crash
        write(file, row)

    monkeypatch.setattr(runner, "append_row", crash_at_update_8)
    with pytest.raises(Crash):
        pool.train(cfg, root=tmp_path)
    monkeypatch.setattr(runner, "append_row", write)

    # The run stopped in the pursuer's phase of generation 1, after the evader of
    # generation 1 had joined its pool.
    (path,) = tmp_path.iterdir()
    assert (path / "pool" / "evader_01.pkl").exists()
    hedge, metrics = rows(path / "hedge.csv"), rows(path / "metrics.csv")

    pool.resume(path)

    phase = [r for r in rows(path / "hedge.csv") if (r["generation"], r["training"]) == ("1", "pursuer")]
    assert [r["member"] for r in phase] == ["paths", "evader_00"], "the pool took in its own generation"
    assert rows(path / "hedge.csv") == hedge, "the restarted phase weighted its pool differently"

    again = rows(path / "metrics.csv")
    assert [int(r["update"]) for r in again] == list(range(1, 9))
    same = lambda r: {k: v for k, v in r.items() if k != "seconds"}
    assert same(again[6]) == same(metrics[6]), "the restarted phase trained differently"
