"""Run PPO updates and save configuration, metrics, and checkpoints.

One loop runs every run, as a list of phases. A plain run has one phase, in which every
role learns. A pool run has one phase per generation, with one learner for each side
(pool.py). latest.pkl holds all that the loop needs, so every run resumes from it."""

import csv
import dataclasses
import functools
import hashlib
import json
import pickle
import time
from pathlib import Path

import jax
import numpy as np

from swarm import envs
from swarm.learn import pool, ppo, rundir, vecenv


def progress_columns(row, names):
    """A per-role metric arrives as `<role>_<name>`, so every role of it prints."""

    return [k for p in names for k in row if k == p or k.endswith("_" + p)]


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
        json.dumps({**spec, "git": rundir.git_state()}, indent=2, sort_keys=True)
    )
    return path


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


def _cut(file, keep):
    """Keep only the rows of a csv file that `keep` accepts."""

    if not file.exists():
        return
    with file.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if keep(row)]
        fields = reader.fieldnames
    with file.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def snapshot(path, update, ts):
    weights = {"update": update, "params": jax.device_get({r: t.params for r, t in ts.items()})}
    with (path / "checkpoints" / f"{update:07d}.pkl").open("wb") as f:
        pickle.dump(weights, f)
    return weights


def save(path, update, state):
    """A numbered snapshot of the weights, and latest.pkl with all that a resume needs."""

    ts = state["ts"]
    weights = snapshot(path, update, ts)
    with (path / "latest.pkl").open("wb") as f:
        pickle.dump(
            {
                **weights,
                "opt_state": jax.device_get({r: t.opt_state for r, t in ts.items()}),
                "step": {r: int(t.step) for r, t in ts.items()},
                "key": jax.device_get(state["key"]),
                "learners": jax.device_get(state["learners"]),
                # Not as numpy: the update compiles the env params in, and XLA folds a
                # numpy constant, which rounds differently.
                "env_params": state["env_params"],
            },
            f,
        )


def start_wandb(cfg, path):
    import wandb  # Optional dependency, loaded only when logging is enabled.

    return wandb.init(
        project="drone-swarm",
        name=path.name,
        config=dataclasses.asdict(cfg),
        dir=str(path),
    )


def _schedule(cfg):
    """The learners' sides and the updates of a phase. In a pool run the two learners
    share each update, so `--steps` counts the env steps of both."""

    sides = pool.SIDES if cfg.generations else (None,)
    per_phase = cfg.num_updates // len(sides) // (cfg.generations + 1)
    if per_phase == 0:
        raise ValueError(f"{cfg.num_updates} updates cannot fill {cfg.generations + 1} phases")
    return sides, per_phase


def _updates(cfg, env, env_params, sides):
    """Each learner's compiled update, and how its seat starts an episode."""

    if not cfg.generations:
        return {None: ppo.make(cfg, env, env_params)[1]}, {}
    seats = {side: pool.seat(cfg, env, env_params, side) for side in sides}
    updates = {side: ppo.make(cfg, env, env_params, seats[side])[1] for side in sides}
    return updates, {side: seats[side][2] for side in sides}


def train(cfg, root="runs", use_wandb=False):
    env, env_params = envs.make(cfg.task, cfg.preset)
    if cfg.generations:
        pool.check(cfg, env_params)
    path = new_run(cfg, env_params, root)
    ts, vec, obs, key = ppo.make(cfg, env, env_params)[0](jax.random.PRNGKey(cfg.seed))
    # A pool run makes its learners as each phase starts.
    learners = [] if cfg.generations else [
        {"side": None, "args": (), "vec": vec, "obs": obs, "key": key}
    ]
    state = {"update": 0, "ts": ts, "env_params": env_params, "key": key, "learners": learners}
    return _run(cfg, env, path, state, use_wandb)


def resume(path, use_wandb=False):
    """Run a run on from its last checkpoint."""

    path = Path(path)
    cfg = rundir.load_config(path)
    env, _ = rundir.load_env(path)
    saved = rundir.load(path)
    ts = ppo.make(cfg, env, saved["env_params"])[0](jax.random.PRNGKey(cfg.seed))[0]
    ts = {
        role: t.replace(
            params=saved["params"][role],
            opt_state=saved["opt_state"][role],
            step=saved["step"][role],
        )
        for role, t in ts.items()
    }

    # What ran after the checkpoint goes, so the files read as one run. A phase that has
    # not started yet weighs its pool again.
    done = saved["update"]
    _, per_phase = _schedule(cfg)
    started = -(-done // per_phase)
    _cut(path / "metrics.csv", lambda row: int(row["update"]) <= done)
    _cut(path / "hedge.csv", lambda row: int(row["generation"]) < started)
    say(path, f"resume at update {done}")

    state = {k: saved[k] for k in ("update", "env_params", "key", "learners")}
    return _run(cfg, env, path, state | {"ts": ts}, use_wandb)


def _phase(cfg, env, env_params, path, generation, ts, key, learners, starts, win_rate):
    """Start a generation of a pool run. Each side gets fresh episodes against its rival's
    pool, and keeps its reward scale. -> the learners, ts and key."""

    scale = {learner["side"]: learner["vec"].rew_norm for learner in learners}
    fresh = []
    for side in pool.SIDES:
        opponents, rows = pool.opponents(path, generation, side, ts, win_rate)
        for row in rows:
            append_row(path / "hedge.csv", row)
        say(
            path,
            f"generation {generation}: the {side} learns against "
            + ", ".join(
                f"{r['member']} (won {r['win_rate']:.2f}, weight {r['weight']:.2f})" for r in rows
            ),
        )

        key, k_reset, k_learn = jax.random.split(key, 3)
        start = functools.partial(starts[side], pool=opponents)
        obs, vec = vecenv.reset(k_reset, env, env_params, cfg, start)
        if side in scale:
            vec = vec.replace(rew_norm=scale[side])
        # The std shrinks within a phase, and the learner must explore again against a new pool.
        if cfg.reset_std:
            ts = {**ts, side: pool.with_std(ts[side], cfg.init_std)}
        learns = {side: True, pool.RIVAL[side]: False}
        fresh.append(
            {"side": side, "args": (learns, opponents), "vec": vec, "obs": obs, "key": k_learn}
        )
    return fresh, ts, key


def _run(cfg, env, path, state, use_wandb):
    sides, per_phase = _schedule(cfg)
    total = per_phase * (cfg.generations + 1)
    env_params, ts, key, learners = (state[k] for k in ("env_params", "ts", "key", "learners"))
    updates, starts = _updates(cfg, env, env_params, sides)
    win_rate = pool.scorer(cfg, env, env_params) if cfg.generations else None
    slices = ppo.role_slices(env_params.roles)
    batch = len(sides) * cfg.num_envs * cfg.num_steps

    tracker = start_wandb(cfg, path) if use_wandb else None
    say(
        path,
        f"{path}  {total} updates of {len(sides)} x {cfg.num_envs} envs x {cfg.num_steps} "
        f"steps in {cfg.generations + 1} phases",
    )
    first, clock = state["update"], time.time()

    for u in range(first, total):
        generation = u // per_phase
        if cfg.generations and u % per_phase == 0:
            learners, ts, key = _phase(
                cfg, env, env_params, path, generation, ts, key, learners, starts, win_rate
            )

        for learner in learners:
            side = learner["side"]
            carry, metrics = updates[side](
                ts, learner["vec"], learner["obs"], learner["key"], *learner["args"]
            )
            ts, learner["vec"], learner["obs"], learner["key"] = carry

            seconds = round(time.time() - clock, 1)
            row = {"update": u + 1, "steps": (u + 1) * batch, "seconds": seconds}
            frozen = set()
            if side is not None:
                row |= {"generation": generation, "training": side}
                # The frozen side's columns of these stay empty.
                frozen = {ppo.column(pool.RIVAL[side], s, slices) for s in pool.LEARNER_STATS}
            row |= {k: "" if k in frozen else float(v) for k, v in metrics.items()}
            append_row(path / "metrics.csv", row)
            if tracker is not None:
                logged = {f"{side}/{k}" if side else k: v for k, v in row.items() if v != ""}
                tracker.log(logged, step=row["steps"])
            if u == first or (u + 1) % cfg.log_every == 0:
                speed = (u + 1 - first) * batch / (time.time() - clock)
                say(
                    path,
                    f"{u + 1:6d} {row['steps']:>12,} {speed:>8.0f}/s  "
                    + (f"{side}  " if side else "")
                    + "  ".join(f"{k} {row[k]:8.3f}" for k in progress_columns(row, env.PROGRESS)),
                )

        # Before the save, so a resume starts with the params of the next update.
        if hasattr(env, "curriculum"):
            harder = env.curriculum(env_params, row)
            if harder is not None:
                env_params = harder
                updates, starts = _updates(cfg, env, env_params, sides)
        if cfg.generations and (u + 1) % per_phase == 0:
            for side in sides:
                pool.save_member(path, side, generation, ts[side].params, u + 1)
        if (u + 1) % cfg.checkpoint_every == 0 or (u + 1) % per_phase == 0:
            state = {"ts": ts, "env_params": env_params, "key": key, "learners": learners}
            save(path, u + 1, state)

    if tracker is not None:
        tracker.finish()
    return path, {"ts": ts, "env_params": env_params, "key": key, "learners": learners}
