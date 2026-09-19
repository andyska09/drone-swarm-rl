"""Run PPO updates and save configuration, metrics, and checkpoints."""

import csv
import dataclasses
import hashlib
import json
import pickle
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

from swarm import envs
from swarm.learn import ppo

PROGRESS = ("return", "ep_length", "death_rate")


def _git():
    def out(*args):
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()

    return {
        "commit": out("git", "rev-parse", "HEAD"),
        "dirty": bool(out("git", "status", "--porcelain")),
    }


def _jsonable(x):
    if isinstance(x, (np.ndarray, jax.Array)):
        return np.asarray(x).tolist()
    if dataclasses.is_dataclass(x):
        return {f.name: _jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    return x


def new_run(cfg, env_params, root="runs"):
    spec = {"train": _jsonable(cfg), "env": _jsonable(env_params)}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")

    path = Path(root) / f"{cfg.task}_{cfg.preset}_s{cfg.seed}_{stamp}_{digest}"
    (path / "checkpoints").mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({**spec, "git": _git()}, indent=2, sort_keys=True)
    )
    return path


def say(path, line):
    print(line, flush=True)
    with (path / "stdout.log").open("a") as f:
        f.write(line + "\n")


def append_metrics(path, row):
    with (path / "metrics.csv").open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def save(path, update, carry):
    train_state, vec, obs, key = carry
    weights = {
        "update": update,
        "params": jax.device_get({r: t.params for r, t in train_state.items()}),
    }
    with (path / "checkpoints" / f"{update:07d}.pkl").open("wb") as f:
        pickle.dump(weights, f)

    # Numbered files store policy snapshots; latest.pkl also stores training state.
    with (path / "latest.pkl").open("wb") as f:
        pickle.dump(
            {
                **weights,
                "opt_state": jax.device_get(
                    {r: t.opt_state for r, t in train_state.items()}
                ),
                "step": {r: int(t.step) for r, t in train_state.items()},
                "vec": jax.device_get(vec),
                "obs": jax.device_get(obs),
                "key": jax.device_get(key),
            },
            f,
        )


def load(path, name="latest.pkl"):
    with (Path(path) / name).open("rb") as f:
        return pickle.load(f)


def restore(carry, saved):
    train_state = {
        role: t.replace(
            params=saved["params"][role],
            opt_state=saved["opt_state"][role],
            step=saved["step"][role],
        )
        for role, t in carry[0].items()
    }
    return train_state, saved["vec"], saved["obs"], saved["key"]


def _wandb(cfg, path):
    import wandb  # Optional dependency, loaded only when logging is enabled.

    return wandb.init(
        project="drone-swarm",
        name=path.name,
        config=dataclasses.asdict(cfg),
        dir=str(path),
    )


def train(cfg, root="runs", use_wandb=False):
    env, env_params = envs.make(cfg.task, cfg.preset)
    path = new_run(cfg, env_params, root)
    init, update = ppo.make(cfg, env, env_params)
    tracker = _wandb(cfg, path) if use_wandb else None

    carry = init(jax.random.PRNGKey(cfg.seed))
    say(path, f"{path}  {cfg.num_updates} updates of {cfg.num_envs} envs x {cfg.num_steps} steps")

    start = time.time()
    for i in range(cfg.num_updates):
        carry, metrics = update(*carry)
        elapsed = time.time() - start
        steps = (i + 1) * cfg.num_envs * cfg.num_steps
        row = {
            "update": i + 1,
            "steps": steps,
            "seconds": round(elapsed, 1),
            **{k: float(v) for k, v in metrics.items()},
        }
        append_metrics(path, row)
        if tracker is not None:
            tracker.log(row, step=steps)
        if i == 0 or (i + 1) % cfg.log_every == 0:
            say(
                path,
                f"{i + 1:6d} {steps:>12,} {steps / elapsed:>8.0f}/s  "
                + "  ".join(f"{k} {row[k]:8.3f}" for k in PROGRESS),
            )
        if (i + 1) % cfg.checkpoint_every == 0 or i + 1 == cfg.num_updates:
            save(path, i + 1, carry)

    if tracker is not None:
        tracker.finish()
    return path, carry
