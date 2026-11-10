"""Read a run directory back: its config, its env and its saved files. Write its log and
its csv files. And the commit that a run or an eval records.

evaluate and pool read and write runs here, because runner imports them: `learn/` imports one way,
rundir, ppo, vecenv, seats, evaluate, pool, runner."""

import csv
import dataclasses
import importlib
import json
import pickle
import subprocess
from pathlib import Path

import jax.numpy as jnp

from swarm.learn import config

ROOT = Path(__file__).resolve().parents[2]


def git_state():
    def out(*args):
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()

    return {
        "commit": out("git", "rev-parse", "HEAD"),
        "dirty": bool(out("git", "status", "--porcelain")),
    }


def check_code(run):
    """Stop unless swarm/ on disk is the code of the commit that trained the run."""

    git = json.loads((Path(run) / "config.json").read_text())["git"]
    commit = git["commit"]
    if not commit or git["dirty"]:
        print(f"warning: {run} trained on code that no commit holds, so its code is not checked",
              flush=True)
        return
    diff = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", commit, "--", "swarm/"],
        capture_output=True, text=True, check=False,
    )
    if diff.returncode:
        raise SystemExit(f"this repo lacks {commit[:7]}, the commit that trained {run}: "
                         + diff.stderr.strip())
    if diff.stdout:
        here = ROOT / "scratch" / f"code_{commit[:7]}"
        raise SystemExit(
            f"swarm/ differs from {commit[:7]}, the commit that trained {run}:\n{diff.stdout}"
            f"To use the run's own code:\n"
            f"  git worktree add --detach {here} {commit[:7]}\n"
            f"  cd {here}\n"
            f"and run the same command there, with the run as {Path(run).resolve()}"
        )


def load(path, name="latest.pkl"):
    with (Path(path) / name).open("rb") as f:
        return pickle.load(f)


def say(path, line):
    print(line, flush=True)
    with (path / "stdout.log").open("a") as f:
        f.write(line + "\n")


def append_row(file, row):
    with file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


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
