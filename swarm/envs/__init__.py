"""Shared observation layout and task loading."""

import importlib

import flax.struct
import jax.numpy as jnp


@flax.struct.dataclass
class Obs:
    """Per-drone observations; batching adds leading dimensions to these shapes."""

    own: jnp.ndarray  # (N, f)
    neighbors: jnp.ndarray  # (N, K, f), last column flags same side
    neighbor_mask: jnp.ndarray  # (N, K)
    target: jnp.ndarray  # (N, f)
    target_mask: jnp.ndarray  # (N,)


DRONE_AXIS = {
    "own": -2,
    "neighbors": -3,
    "neighbor_mask": -2,
    "target": -2,
    "target_mask": -1,
}


def flat(obs):
    """Flatten each drone's observations, zero masked features, and include the masks."""

    lead = obs.own.shape[:-1]
    return jnp.concatenate(
        [
            obs.own,
            (obs.neighbors * obs.neighbor_mask[..., None]).reshape(lead + (-1,)),
            obs.neighbor_mask,
            obs.target * obs.target_mask[..., None],
            obs.target_mask[..., None],
        ],
        axis=-1,
    )


def take(obs, drones):
    """Select drone indices while preserving leading batch dimensions."""

    return obs.replace(
        **{f: jnp.take(getattr(obs, f), drones, axis=a) for f, a in DRONE_AXIS.items()}
    )


def make(task, preset):
    env = importlib.import_module(f"swarm.envs.{task}")
    return env, env.PRESETS[preset]
