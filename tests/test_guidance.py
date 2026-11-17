import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from swarm.sim import dynamics, guidance


def test_drone_limit_is_finite_at_full_climb_in_a_batch():
    climb = jnp.tile(jnp.array([1.0, 0.0, 400.0]), (4, 1))
    limit = jax.jit(jax.vmap(guidance.drone_limit, in_axes=(0, None)))
    assert bool(jnp.all(jnp.isfinite(limit(climb, dynamics.default_params()))))
