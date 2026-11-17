import functools
import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm import envs
from swarm.learn import evaluate, ppo, rundir, seats, vecenv

GAMMA_H, ETA_H = 0.7, 2.0
HEDGE_EPISODES = 600
BLOCKS = 64
SEEDS = {"pursuer": "frpn", "evader": "paths"}
SIDES = ("evader", "pursuer")
RIVAL = {"pursuer": "evader", "evader": "pursuer"}
WINS = {
    "pursuer": ("caught_e", "crash_e", "OOB_e"),
    "evader": ("time_limit", "crash_p", "OOB_p", "coll_pp"),
}


def hedge(win_rates):
    c = 1.0 - np.asarray(win_rates, float)
    power = c**ETA_H
    share = power / power.sum() if power.sum() > 0 else np.full(len(c), 1.0 / len(c))
    return GAMMA_H / len(c) + (1.0 - GAMMA_H) * share


def blocks(weights):
    quota = np.asarray(weights) * BLOCKS
    count = np.floor(quota).astype(int)
    count[np.argsort(count - quota, kind="stable")[: BLOCKS - count.sum()]] += 1
    return np.repeat(np.arange(len(count)), count)


def members(side, generation):
    return [SEEDS[side]] + [f"{side}_{g:02d}" for g in range(generation)]


def with_std(t, std):
    actor = t.params["actor"]
    log_std = jnp.full_like(actor["params"]["log_std"], math.log(std))
    weights = {**actor["params"], "log_std": log_std}
    return t.replace(params={**t.params, "actor": {**actor, "params": weights}})


def match(cfg, env, env_params, pilots, role, episodes):
    slices = envs.role_slices(env_params.roles)
    keys = jax.random.split(jax.random.PRNGKey(evaluate.EVAL_SEED), episodes)
    fixed = {r: env.PILOTS[s](env_params, slices[r]) for r, s in pilots.items() if s}
    nets = {r: slices[r] for r, s in pilots.items() if not s}

    def win_rate(weights):
        act, start = seats.combine(seats.net_policy(cfg, env, weights, nets), fixed, slices)

        def one(key):
            _, _, _, reward, info, live = evaluate.rollout(key, env, env_params, act, start)
            end = evaluate.summarize(reward, info, live, slices)
            return sum(end[k] for k in WINS[role]) + 0.5 * end["fail_pe"]

        return jax.vmap(one)(keys).mean()

    return jax.jit(win_rate)


def check(cfg, env_params):
    if set(env_params.roles) != set(SIDES):
        raise ValueError(f"pool training needs pursuers and an evader, not {env_params.roles}")
    if cfg.num_envs % BLOCKS:
        raise ValueError(f"num_envs must be a multiple of {BLOCKS}, not {cfg.num_envs}")


def matches(cfg, env, env_params, episodes):
    compiled = {}

    def win_rate(role, pilots, weights):
        named = (role, tuple(sorted(pilots.items())))
        if named not in compiled:
            compiled[named] = match(cfg, env, env_params, pilots, role, episodes)
        return float(compiled[named](weights))

    return win_rate


def scorer(cfg, env, env_params):
    win_rate = matches(cfg, env, env_params, HEDGE_EPISODES)

    def score(side, member, weights):
        pilots = {side: None, RIVAL[side]: member if member in env.PILOTS else None}
        return win_rate(side, pilots, weights)

    return score


def opponents(path, generation, side, ts, win_rate):
    rival = RIVAL[side]
    names = members(rival, generation)
    nets = {name: load_member(path, rival, name) for name in names[1:]}
    rates = [win_rate(side, names[0], {side: ts[side].params})] + [
        win_rate(side, name, {side: ts[side].params, rival: nets[name]}) for name in names[1:]
    ]
    weights = hedge(rates)

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


class Phases:
    sides = SIDES

    def __init__(self, cfg, env, env_params):
        check(cfg, env_params)
        self.cfg, self.env = cfg, env
        self.count = cfg.generations + 1
        self.win_rate = scorer(cfg, env, env_params)

    def rival_seat(self, env_params, side):
        rival = RIVAL[side]
        return seats.pool_seat(self.cfg, self.env, env_params, rival, SEEDS[rival])

    def begin(self, path, generation, env_params, ts, key, learners):
        scale = {learner.side: learner.vec.rew_norm for learner in learners}
        fresh = []
        for side in SIDES:
            pool, rows = opponents(path, generation, side, ts, self.win_rate)
            for row in rows:
                rundir.append_row(path / "hedge.csv", row)
            rundir.say(
                path,
                f"generation {generation}: the {side} learns against "
                + ", ".join(
                    f"{r['member']} (won {r['win_rate']:.2f}, weight {r['weight']:.2f})"
                    for r in rows
                ),
            )

            key, k_reset, k_learn = jax.random.split(key, 3)
            start = functools.partial(self.rival_seat(env_params, side)[2], pool=pool)
            obs, vec = vecenv.reset(k_reset, self.env, env_params, self.cfg, start)
            if side in scale:
                vec = vec.replace(rew_norm=scale[side])
            if self.cfg.reset_std:
                ts = {**ts, side: with_std(ts[side], self.cfg.init_std)}
            train = {side: True, RIVAL[side]: False}
            fresh.append(ppo.Learner(vec, obs, k_learn, train, pool, side))
        return fresh, ts, key

    def end(self, path, generation, ts, update):
        for side in SIDES:
            save_member(path, side, generation, ts[side].params, update)
