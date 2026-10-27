"""Pools for chase, AMSPBH: in each generation both sides learn at once, each against
the other side's pool of frozen opponents, weighted by a Hedge sampler. The design:
research/notes/task_chase_training.md."""

import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm.learn import evaluate, ppo, rundir, seats

GAMMA_H, ETA_H = 0.7, 2.0
HEDGE_EPISODES = 600
BLOCKS = 64
# The first member of each pool.
SEEDS = {"pursuer": "frpn", "evader": "paths"}
SIDES = ("evader", "pursuer")
RIVAL = {"pursuer": "evader", "evader": "pursuer"}
# A side wins on these outcomes, and `fail_both` is half a win to each side.
WINS = {
    "pursuer": ("caught_evader", "crash_evader", "OOB_evader"),
    "evader": ("time_limit", "crash_pursuer", "OOB_pursuer", "collPP"),
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
    gives each role a name from the task's PILOTS, or None for a network. The returned
    function takes the networks' weights, {role: params}."""

    slices = ppo.role_slices(env_params.roles)
    keys = jax.random.split(jax.random.PRNGKey(evaluate.EVAL_SEED), episodes)
    fixed = {r: env.PILOTS[s](env_params, slices[r]) for r, s in sources.items() if s}
    nets = {r: slices[r] for r, s in sources.items() if not s}

    def win_rate(weights):
        act, start = seats.combine(seats.net_policy(cfg, env, weights, nets), fixed, slices)

        def one(key):
            _, _, _, reward, info, live = evaluate.rollout(key, env, env_params, act, start)
            end = evaluate.summarize(reward, info, live, slices)
            return sum(end[k] for k in WINS[role]) + 0.5 * end["fail_both"]

        return jax.vmap(one)(keys).mean()

    return jax.jit(win_rate)


def check(cfg, env_params):
    """A pool run needs pursuers and an evader, and blocks of one size."""

    if set(env_params.roles) != set(SIDES):
        raise ValueError(f"pool training needs pursuers and an evader, not {env_params.roles}")
    if cfg.num_envs % BLOCKS:
        raise ValueError(f"num_envs must be a multiple of {BLOCKS}, not {cfg.num_envs}")


def seat(cfg, env, env_params, side):
    """The seat of `side`'s rival: its pool, whose member 0 is the rival's seed."""

    return seats.pool_seat(cfg, env, env_params, RIVAL[side], SEEDS[RIVAL[side]])


def scorer(cfg, env, env_params):
    """win_rate(side, member, weights): `side`'s win rate against one member of its
    rival's pool. Each pairing compiles once."""

    matches = {}

    def win_rate(side, member, weights):
        sources = {side: None, RIVAL[side]: member if member in env.PILOTS else None}
        named = (side, tuple(sorted(sources.items())))
        if named not in matches:
            matches[named] = match(cfg, env, env_params, sources, side, HEDGE_EPISODES)
        return float(matches[named](weights))

    return win_rate


def opponents(path, generation, side, ts, win_rate):
    """`side`'s rival pool in `generation`, as it was at the end of generation − 1, each
    member weighted by Hedge. -> the pool, (each block's member, each block's actor
    weights), and its rows of hedge.csv."""

    rival = RIVAL[side]
    names = members(rival, generation)
    nets = {name: load_member(path, rival, name) for name in names[1:]}
    rates = [win_rate(side, names[0], {side: ts[side].params})] + [
        win_rate(side, name, {side: ts[side].params, rival: nets[name]}) for name in names[1:]
    ]
    weights = hedge(rates)

    # The seed's blocks carry placeholder weights that never fly.
    member = blocks(weights)
    actors = [jax.tree.map(jnp.zeros_like, ts[rival].params["actor"])] + [
        nets[name]["actor"] for name in names[1:]
    ]
    stacked = jax.tree.map(lambda *a: jnp.stack(a), *[actors[m] for m in member])
    rows = [
        {"generation": generation, "training": side, "member": n, "win_rate": w, "weight": p}
        for n, w, p in zip(names, rates, weights)
    ]
    return (jnp.asarray(member), stacked), rows


def load_member(path, side, name):
    return rundir.load(Path(path) / "pool", f"{name}.pkl")["params"][side]


def save_member(path, side, generation, params, update):
    folder = Path(path) / "pool"
    folder.mkdir(exist_ok=True)
    with (folder / f"{side}_{generation:02d}.pkl").open("wb") as f:
        pickle.dump({"update": update, "params": jax.device_get({side: params})}, f)
