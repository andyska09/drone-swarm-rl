"""Check the Hedge weights."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from swarm.learn import pool


def test_hedge_weights():
    c = np.array([0.8, 0.5, 0.0])
    weights = pool.hedge(1.0 - c)
    assert np.allclose(weights, 0.7 / 3 + 0.3 * c**2 / np.sum(c**2))
    assert np.isclose(weights.sum(), 1.0)

    # The learner beats every opponent, so no c is above 0: the second term is uniform.
    assert np.allclose(pool.hedge([1.0, 1.0, 1.0, 1.0]), 0.25)
