import importlib

import flax.struct
import jax.numpy as jnp


@flax.struct.dataclass
class Obs:
    actor: dict
    critic: jnp.ndarray
    own: dict = None
    pairs: dict = None
    world: jnp.ndarray = None


def role_slices(roles):
    return {name: jnp.array([i for i, r in enumerate(roles) if r == name]) for name in dict.fromkeys(roles)}


def make(task, preset):
    env = importlib.import_module(f"swarm.envs.{task}")
    return env, env.PRESETS[preset]
