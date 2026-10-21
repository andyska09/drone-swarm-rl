"""Pool training for chase, AMSPBH: each side learns in turn against a growing pool of
frozen opponents, weighted by a Hedge sampler. The design:
research/notes/task_chase_training.md."""

import csv
import math
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm import envs
from swarm.learn import evaluate, ppo, runner, seats, vecenv

GAMMA_H, ETA_H = 0.7, 2.0
HEDGE_EPISODES = 600
BLOCKS = 64
# The first member of each pool, and the side that learns first in a generation.
SEEDS = {"pursuer": "frpn", "evader": "paths"}
ORDER = ("evader", "pursuer")
RIVAL = {"pursuer": "evader", "evader": "pursuer"}
# A side wins on these outcomes, and `fail_both` is half a win to each side.
WINS = {
    "pursuer": ("caught_evader", "crash_evader", "OOB_evader"),
    "evader": ("timeout", "crash_pursuer", "OOB_pursuer"),
}
# The frozen side's columns of these stay empty in metrics.csv.
LEARNER_STATS = (
    "policy_loss", "value_loss", "entropy", "clipfrac", "approx_kl", "explained_var"
)


def hedge(win_rates):
    """An opponent's weight grows with how often it beats the learner."""

    c = 1.0 - np.asarray(win_rates, float)
    power = c**ETA_H
    share = power / power.sum() if power.sum() > 0 else np.full(len(c), 1.0 / len(c))
    return GAMMA_H / len(c) + (1.0 - GAMMA_H) * share


def blocks(weights):
    """Each block's member. A member of weight p gets p × BLOCKS blocks, rounded by
    largest remainder."""

    quota = np.asarray(weights) * BLOCKS
    count = np.floor(quota).astype(int)
    count[np.argsort(count - quota, kind="stable")[: BLOCKS - count.sum()]] += 1
    return np.repeat(np.arange(len(count)), count)


def phases(cfg):
    """(generation, learner) in the order they run."""

    return [(g, side) for g in range(cfg.generations + 1) for side in ORDER]


def members(side, generation):
    """The pool of `side` at the end of generation − 1: its seed, then its policies."""

    return [SEEDS[side]] + [f"{side}_{g:02d}" for g in range(generation)]


def with_std(t, std):
    """The TrainState with its actor's std set to `std` for every action."""

    actor = t.params["actor"]["params"]
    log_std = jnp.full_like(actor["log_std"], math.log(std))
    return t.replace(params={**t.params, "actor": {"params": {**actor, "log_std": log_std}}})


def match(cfg, env, env_params, sources, role, episodes):
    """`role`'s win rate over the eval seeds, both sides flying their mean. `sources`
    gives each role a name from SOURCES, or None for a network. The returned function
    takes the networks' weights, {role: params}."""

    slices = ppo.role_slices(env_params.roles)
    keys = jax.random.split(jax.random.PRNGKey(evaluate.EVAL_SEED), episodes)
    fixed = {r: seats.scripted(env, env_params, s, slices[r]) for r, s in sources.items() if s}
    nets = {r: slices[r] for r, s in sources.items() if not s}

    def win_rate(weights):
        act, start = seats.combine(seats.net_policy(cfg, env, weights, nets), fixed, slices)

        def one(key):
            _, _, _, reward, info, live = evaluate.rollout(key, env, env_params, act, start)
            end = evaluate.summarize(reward, info, live, slices)
            return sum(end[k] for k in WINS[role]) + 0.5 * end["fail_both"]

        return jax.vmap(one)(keys).mean()

    return jax.jit(win_rate)


def load_member(path, side, name):
    return runner.load(Path(path) / "pool", f"{name}.pkl")["params"][side]


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


def train(cfg, root="runs", use_wandb=False):
    env, env_params = envs.make(cfg.task, cfg.preset)
    if set(env_params.roles) != set(ORDER):
        raise ValueError(f"pool training needs pursuers and an evader, not {env_params.roles}")
    path = runner.new_run(cfg, env_params, root)
    (path / "pool").mkdir()
    ts, vec, _, key = ppo.make(cfg, env, env_params)[0](jax.random.PRNGKey(cfg.seed))
    return _run(cfg, env, env_params, path, ts, vec.rew_norm, key, 0, use_wandb)


def resume(path, use_wandb=False):
    """Run a pool run on from the start of its unfinished phase."""

    path = Path(path)
    cfg = evaluate.load_config(path)
    env, env_params = evaluate.load_env(path)
    saved = runner.load(path)
    init, _ = ppo.make(cfg, env, env_params)
    ts = runner.restore(init(jax.random.PRNGKey(cfg.seed)), saved)[0]

    order = phases(cfg)
    first = saved["update"] // (cfg.num_updates // len(order))
    if first == len(order):
        raise ValueError(f"{path} has run all {len(order)} phases")
    # What the unfinished phase wrote goes, so the files read as one run.
    _cut(path / "metrics.csv", lambda row: int(row["update"]) <= saved["update"])
    _cut(
        path / "hedge.csv",
        lambda row: order.index((int(row["generation"]), row["training"])) < first,
    )
    runner.say(path, f"resume at update {saved['update']}, phase {first}")
    return _run(
        cfg, env, env_params, path, ts, saved["vec"].rew_norm, saved["key"], first, use_wandb
    )


def _run(cfg, env, env_params, path, ts, rew_norm, key, first, use_wandb):
    order = phases(cfg)
    per_phase = cfg.num_updates // len(order)
    if per_phase == 0:
        raise ValueError(f"{cfg.num_updates} updates cannot fill {len(order)} phases")
    if cfg.num_envs % BLOCKS:
        raise ValueError(f"num_envs must be a multiple of {BLOCKS}, not {cfg.num_envs}")
    batch = cfg.num_envs * cfg.num_steps
    slices = ppo.role_slices(env_params.roles)

    # Each side learns against its rival's pool, whose member 0 is the rival's seed.
    pool_seats = {
        side: seats.pool_seat(cfg, env, env_params, RIVAL[side], SEEDS[RIVAL[side]])
        for side in ORDER
    }
    updates = {side: ppo.make(cfg, env, env_params, pool_seats[side])[1] for side in ORDER}
    matches = {}

    def win_rate(side, member, weights):
        sources = {side: None, RIVAL[side]: member if member in seats.SOURCES else None}
        named = (side, tuple(sorted(sources.items())))
        if named not in matches:
            matches[named] = match(cfg, env, env_params, sources, side, HEDGE_EPISODES)
        return float(matches[named](weights))

    tracker = runner.start_wandb(cfg, path) if use_wandb else None
    runner.say(
        path,
        f"{path}  {len(order)} phases of {per_phase} updates of "
        f"{cfg.num_envs} envs x {cfg.num_steps} steps",
    )
    clock = time.time()

    for i in range(first, len(order)):
        generation, side = order[i]
        rival = RIVAL[side]
        names = members(rival, generation)
        nets = {name: load_member(path, rival, name) for name in names[1:]}

        rates = [win_rate(side, names[0], {side: ts[side].params})] + [
            win_rate(side, name, {side: ts[side].params, rival: nets[name]})
            for name in names[1:]
        ]
        weights = hedge(rates)
        for name, w, p in zip(names, rates, weights):
            runner.append_row(
                path / "hedge.csv",
                {"generation": generation, "training": side, "member": name,
                 "win_rate": w, "weight": p},
            )
        runner.say(
            path,
            f"generation {generation}: the {side} learns against "
            + ", ".join(
                f"{n} (won {w:.2f}, weight {p:.2f})" for n, w, p in zip(names, rates, weights)
            ),
        )

        # The seed's blocks carry placeholder weights that never fly.
        member = blocks(weights)
        actors = [jax.tree.map(jnp.zeros_like, ts[rival].params["actor"])] + [
            nets[name]["actor"] for name in names[1:]
        ]
        stacked = jax.tree.map(lambda *a: jnp.stack(a), *[actors[m] for m in member])
        pool = (jnp.asarray(member), stacked)

        # Fresh episodes for the new pairing. Each side keeps its own reward scale.
        key, k = jax.random.split(key)
        obs, vec = vecenv.reset(
            k, env, env_params, cfg, lambda state: pool_seats[side][2](state, pool)
        )
        # The std shrinks within a phase, and the learner must explore again against a new pool.
        if cfg.reset_std:
            ts = {**ts, side: with_std(ts[side], cfg.init_std)}
        carry = (ts, vec.replace(rew_norm=rew_norm), obs, key)
        train = {side: True, rival: False}
        frozen = {ppo.column(rival, name, slices) for name in LEARNER_STATS}

        for u in range(i * per_phase, (i + 1) * per_phase):
            carry, metrics = updates[side](*carry, train, pool)
            row = {
                "update": u + 1,
                "steps": (u + 1) * batch,
                "seconds": round(time.time() - clock, 1),
                "generation": generation,
                "training": side,
                **{k: "" if k in frozen else float(v) for k, v in metrics.items()},
            }
            runner.append_row(path / "metrics.csv", row)
            if tracker is not None:
                tracker.log({k: v for k, v in row.items() if v != ""}, step=row["steps"])
            done = u + 1 - first * per_phase
            if done == 1 or (u + 1) % cfg.log_every == 0:
                speed = done * batch / (time.time() - clock)
                runner.say(
                    path,
                    f"{u + 1:6d} {row['steps']:>12,} {speed:>8.0f}/s  "
                    + "  ".join(f"{k} {row[k]:8.3f}" for k in runner.progress_columns(row)),
                )
            if (u + 1) % cfg.checkpoint_every == 0:
                runner.snapshot(path, u + 1, carry[0])

        ts, vec, _, key = carry
        rew_norm = vec.rew_norm
        with (path / "pool" / f"{side}_{generation:02d}.pkl").open("wb") as f:
            pickle.dump({"update": u + 1, "params": jax.device_get({side: ts[side].params})}, f)
        # latest.pkl only ever holds a phase boundary, so a resume restarts a phase.
        runner.save(path, u + 1, carry)

    if tracker is not None:
        tracker.finish()
    return path, carry
