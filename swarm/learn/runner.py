"""Run PPO updates and save configuration, metrics, and checkpoints.

One loop runs every run, as a list of phases. A plain run has one phase, in which every
role learns. A run of a preset with `piloted` has one phase, in which the pilot flies one
role and the other roles learn. A pool run has one phase per generation, with one learner for
each side (pool.py). latest.pkl holds all that the loop needs, so every run resumes from
it."""

import csv
import dataclasses
import functools
import hashlib
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm import envs
from swarm.learn import pool, ppo, rundir, seats, vecenv

# A frozen role's columns of these stay empty in metrics.csv.
LEARNER_STATS = (
    "policy_loss", "value_loss", "entropy", "clipfrac", "approx_kl", "explained_var"
)


@dataclasses.dataclass
class State:
    """All that the loop carries from update to update. latest.pkl saves it."""

    update: int
    ts: dict
    env_params: object
    key: jax.Array
    learners: list


class OnePhase:
    """The phases of a plain run: one, in which every role learns. Piloted and
    pool.Phases have the same members."""

    sides = (None,)
    count = 1

    def rival_seat(self, env_params, side):
        """No pool flies in a plain run."""

        del env_params, side

    def begin(self, path, generation, env_params, ts, key, learners):
        del path, generation, env_params
        return learners, ts, key

    def end(self, path, generation, ts, update):
        del path, generation, ts, update


class Piloted:
    """The phase of a preset with `piloted`: its pilot flies one role as the only member of
    a pool of one block, and the other roles learn."""

    sides = (None,)
    count = 1

    def __init__(self, cfg, env, piloted):
        self.cfg, self.env = cfg, env
        self.role, self.pilot = piloted

    def rival_seat(self, env_params, side):
        del side
        return seats.pool_seat(self.cfg, self.env, env_params, self.role, self.pilot)

    def begin(self, path, generation, env_params, ts, key, learners):
        del path, generation, learners
        # The block's weights never fly, because its member, 0, is the pilot.
        weights = jax.tree.map(lambda x: jnp.zeros_like(x)[None], ts[self.role].params["actor"])
        one_block = (jnp.zeros(1, jnp.int32), weights)
        key, k_reset, k_learn = jax.random.split(key, 3)
        start = functools.partial(self.rival_seat(env_params, None)[2], pool=one_block)
        obs, vec = vecenv.reset(k_reset, self.env, env_params, self.cfg, start)
        learning = {role: role != self.role for role in ts}
        return [ppo.Learner(vec, obs, k_learn, learning, one_block)], ts, key

    def end(self, path, generation, ts, update):
        del path, generation, ts, update


def progress_columns(row, names):
    """A per-role metric arrives as `<name>_<r>`, so every role of it prints."""

    return [k for p in names for k in row if k == p or k.startswith(p + "_")]


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


def save(path, state):
    """A numbered snapshot of the weights, and latest.pkl with all that a resume needs."""

    ts = state.ts
    weights = snapshot(path, state.update, ts)
    with (path / "latest.pkl").open("wb") as f:
        pickle.dump(
            {
                **weights,
                "opt_state": jax.device_get({r: t.opt_state for r, t in ts.items()}),
                "step": {r: int(t.step) for r, t in ts.items()},
                "key": jax.device_get(state.key),
                "learners": jax.device_get(state.learners),
                # Not as numpy: the update compiles the env params in, and XLA folds a
                # numpy constant, which rounds differently.
                "env_params": state.env_params,
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


def _phases(cfg, env, env_params):
    if env_params.piloted:
        if cfg.generations:
            role, pilot = env_params.piloted
            raise ValueError(f"{cfg.preset} has no pools: {pilot} flies its {role}")
        return Piloted(cfg, env, env_params.piloted)
    return pool.Phases(cfg, env, env_params) if cfg.generations else OnePhase()


def _per_phase(cfg, phases):
    """The updates of one phase. Every learner runs once in each update, so `--steps`
    counts the env steps of all of them."""

    per_phase = cfg.num_updates // len(phases.sides) // phases.count
    if per_phase == 0:
        raise ValueError(f"{cfg.num_updates} updates cannot fill {phases.count} phases")
    return per_phase


def _updates(cfg, env, env_params, phases):
    """Each learner's compiled update. A pool side's update flies its rival's pool."""

    return {
        side: ppo.make(cfg, env, env_params, phases.rival_seat(env_params, side))[1]
        for side in phases.sides
    }


def train(cfg, root="runs", use_wandb=False):
    env, env_params = envs.make(cfg.task, cfg.preset)
    phases = _phases(cfg, env, env_params)
    path = new_run(cfg, env_params, root)
    ts, learner = ppo.make(cfg, env, env_params)[0](jax.random.PRNGKey(cfg.seed))
    # A pool run's phases replace this learner with one for each side.
    state = State(0, ts, env_params, learner.key, [learner])
    return _run(cfg, env, path, state, phases, use_wandb)


def resume(path, use_wandb=False):
    """Run a run on from its last checkpoint."""

    path = Path(path)
    cfg = rundir.load_config(path)
    env, _ = rundir.load_env(path)
    saved = rundir.load(path)
    phases = _phases(cfg, env, saved["env_params"])
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
    started = -(-done // _per_phase(cfg, phases))
    _cut(path / "metrics.csv", lambda row: int(row["update"]) <= done)
    _cut(path / "hedge.csv", lambda row: int(row["generation"]) < started)
    rundir.say(path, f"resume at update {done}")

    state = State(done, ts, saved["env_params"], saved["key"], saved["learners"])
    return _run(cfg, env, path, state, phases, use_wandb)


def _run(cfg, env, path, state, phases, use_wandb):
    per_phase = _per_phase(cfg, phases)
    total = per_phase * phases.count
    updates = _updates(cfg, env, state.env_params, phases)
    slices = envs.role_slices(state.env_params.roles)
    batch = len(phases.sides) * cfg.num_envs * cfg.num_steps

    tracker = start_wandb(cfg, path) if use_wandb else None
    rundir.say(
        path,
        f"{path}  {total} updates of {len(phases.sides)} x {cfg.num_envs} envs x {cfg.num_steps} "
        f"steps in {phases.count} phases",
    )
    first, clock = state.update, time.time()

    for u in range(first, total):
        generation = u // per_phase
        if u % per_phase == 0:
            state.learners, state.ts, state.key = phases.begin(
                path, generation, state.env_params, state.ts, state.key, state.learners
            )

        for i, learner in enumerate(state.learners):
            side = learner.side
            (state.ts, state.learners[i]), metrics = updates[side](state.ts, learner)

            seconds = round(time.time() - clock, 1)
            row = {"update": u + 1, "steps": (u + 1) * batch, "seconds": seconds}
            if side is not None:
                row |= {"generation": generation, "training": side}
            frozen = {
                ppo.column(role, s, slices)
                for role, trains in learner.train.items()
                if not trains
                for s in LEARNER_STATS
            }
            row |= {k: "" if k in frozen else float(v) for k, v in metrics.items()}
            rundir.append_row(path / "metrics.csv", row)
            if tracker is not None:
                logged = {f"{side}/{k}" if side else k: v for k, v in row.items() if v != ""}
                tracker.log(logged, step=row["steps"])
            if u == first or (u + 1) % cfg.log_every == 0:
                speed = (u + 1 - first) * batch / (time.time() - clock)
                rundir.say(
                    path,
                    f"{u + 1:6d} {row['steps']:>12,} {speed:>8.0f}/s  "
                    + (f"{side}  " if side else "")
                    + "  ".join(f"{k} {row[k]:8.3f}" for k in progress_columns(row, env.PROGRESS)),
                )

        # Before the save, so a resume starts with the params of the next update.
        if hasattr(env, "curriculum"):
            harder = env.curriculum(state.env_params, row)
            if harder is not None:
                state.env_params = harder
                updates = _updates(cfg, env, state.env_params, phases)
        state.update = u + 1
        if state.update % per_phase == 0:
            phases.end(path, generation, state.ts, state.update)
        if state.update % cfg.checkpoint_every == 0 or state.update % per_phase == 0:
            save(path, state)

    if tracker is not None:
        tracker.finish()
    return path, state
