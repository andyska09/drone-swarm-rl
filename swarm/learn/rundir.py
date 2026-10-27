"""Read a run directory back: its config, its env and its saved files. And the commit
that a run or an eval records.

evaluate and pool read runs here, because runner imports them: `learn/` imports one way,
rundir, ppo, vecenv, seats, evaluate, pool, runner."""

import dataclasses
import importlib
import json
import pickle
import subprocess
from pathlib import Path

import jax.numpy as jnp

from swarm.learn import config


def git_state():
    def out(*args):
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()

    return {
        "commit": out("git", "rev-parse", "HEAD"),
        "dirty": bool(out("git", "status", "--porcelain")),
    }


def load(path, name="latest.pkl"):
    with (Path(path) / name).open("rb") as f:
        return pickle.load(f)


def load_config(run):
    spec = json.loads((Path(run) / "config.json").read_text())
    cfg = config.TrainConfig(**spec["train"])
    cfg.hidden = tuple(cfg.hidden)
    return cfg


def load_env(run):
    """The task module and the env the run trained in, rebuilt from its config.json."""

    spec = json.loads((Path(run) / "config.json").read_text())
    env = importlib.import_module(f"swarm.envs.{spec['train']['task']}")
    return env, _from_json(env.EnvParams, spec["env"])


def _from_json(cls, record):
    """JSON turned the nested classes into dicts, and the tuples and arrays into lists."""

    kinds = {f.name: f.type for f in dataclasses.fields(cls)}
    values = {}
    for name, value in record.items():
        kind = kinds.get(name)
        if dataclasses.is_dataclass(kind):
            value = _from_json(kind, value)
        elif kind is tuple:
            value = tuple(value)
        elif kind is jnp.ndarray:
            value = jnp.asarray(value)
        values[name] = value
    return cls(**values)
