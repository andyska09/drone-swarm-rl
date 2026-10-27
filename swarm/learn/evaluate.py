"""Measure a policy: fixed eval seeds, mean actions, no auto-reset, per-episode only."""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm.learn import ppo, rundir
from swarm.learn.seats import combine, net_policy

# Far from any training seed, so eval never flies an episode training saw.
EVAL_SEED = 1_000_000
TRAJECTORIES = 8


def checkpoint_name(text):
    if text.isdigit():
        return f"checkpoints/{int(text):07d}.pkl"
    return text if text.endswith(".pkl") else f"{text}.pkl"


def _saved(run, source, checkpoint):
    path = Path(run) if source == "checkpoint" else Path(source)
    if path.suffix == ".pkl":
        return rundir.load(path.parent, path.name)
    return rundir.load(path, checkpoint_name(checkpoint))


def make_policy(cfg, env, env_params, run, checkpoint, seats):
    """Who flies each role: this run's weights, another run's, or one of the task's
    PILOTS."""

    slices = ppo.role_slices(env_params.roles)
    for role in seats:
        if role not in slices:
            raise ValueError(f"{role!r} has no seat; roles are {sorted(slices)}")

    nets = {r: seats.get(r, "checkpoint") for r in slices if seats.get(r) not in env.PILOTS}
    saved = {r: _saved(run, s, checkpoint) for r, s in nets.items()}
    net = net_policy(
        cfg,
        env,
        {r: s["params"][r] for r, s in saved.items()},
        {r: slices[r] for r in nets},
    )
    step = next((s["update"] for s in saved.values()), None)

    fixed = {r: env.PILOTS[s](env_params, slices[r]) for r, s in seats.items() if s in env.PILOTS}
    return *combine(net, fixed, slices), step


def _hold(done, new, old):
    return jax.tree.map(lambda a, b: jnp.where(done, b, a), new, old)


def rollout(key, env, env_params, act, start):
    """One episode to its own end or to `max_steps`. Per step: drone, reference, action,
    reward, info, live."""

    _, state = env.reset(key, env_params)
    # A path seat puts its drone on its path, so the obs is built after.
    state, carry = start(state)
    obs = env.get_obs(state, env_params)

    def body(c, t):
        state, obs, carry, done = c
        action, carry = act(state, obs, carry)
        next_obs, next_state, reward, ended, info = env.step(
            key, state, action, env_params
        )
        # The step limit ends the episode at the last step, as in training.
        last = t == env_params.max_steps - 1
        info = {**info, "end": {**info["end"], "time_limit": last & ~ended}}
        # Nothing resets here, so hold the scene at the step the episode ended.
        state = _hold(done, next_state, state)
        obs = _hold(done, next_obs, obs)
        live = ~done
        return (state, obs, carry, done | ended), (
            state.drone,
            env.reference(state, env_params),
            action,
            reward * live,
            info,
            live,
        )

    _, out = jax.lax.scan(
        body, (state, obs, carry, jnp.bool_(False)), jnp.arange(env_params.max_steps)
    )
    return out


def summarize(reward, info, live, slices):
    """One number per metric of one episode, by the rules of `ppo.episode_metrics`."""

    steps = live.sum()
    info = {**info, "ep_return": jnp.cumsum(reward, axis=0), "ep_length": jnp.cumsum(live)}
    return ppo.episode_metrics(
        info,
        lambda x: jnp.take(x, steps - 1, axis=0),
        lambda x: jnp.sum(x * live) / steps,
        slices,
        lead=1,
    )


def eval_env(run, preset=None):
    """The env that eval flies: the run's recorded env, or another preset of its task,
    with the task's eval_params on top."""

    env, env_params = rundir.load_env(run)
    if preset is not None:
        env_params = env.PRESETS[preset]
    if hasattr(env, "eval_params"):
        env_params = env.eval_params(env_params)
    return env, env_params


def evaluate(run, episodes=1024, checkpoint="latest", seats=(), name=None, preset=None):
    run = Path(run)
    cfg = rundir.load_config(run)
    env, env_params = eval_env(run, preset)
    if preset is not None:
        cfg = dataclasses.replace(cfg, preset=preset)

    seats = dict(seats)
    act, start, step = make_policy(cfg, env, env_params, run, checkpoint, seats)

    keys = jax.random.split(jax.random.PRNGKey(EVAL_SEED), episodes)

    slices = ppo.role_slices(env_params.roles)

    def measure(key):
        _, _, _, reward, info, live = rollout(key, env, env_params, act, start)
        return summarize(reward, info, live, slices)

    per_episode = jax.jit(jax.vmap(measure))(keys)

    parts = [f"{r}_{Path(s).name}" for r, s in sorted(seats.items())]
    named = "_".join(parts + ([f"preset_{preset}"] if preset else []))
    out = run / "evals" / (name or named or checkpoint)
    out.mkdir(parents=True, exist_ok=True)

    summary = {k: float(ppo.mean_defined(v, 1.0)) for k, v in per_episode.items()}
    (out / "eval.json").write_text(
        json.dumps(
            {
                "task": cfg.task,
                "preset": cfg.preset,
                "seats": {r: seats.get(r, "checkpoint") for r in slices},
                "checkpoint": checkpoint if step is not None else None,
                "update": step,
                "episodes": episodes,
                "eval_seed": EVAL_SEED,
                "git": rundir.git_state(),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    np.savez(out / "episodes.npz", **{k: np.asarray(v) for k, v in per_episode.items()})
    save_replay(out, cfg, env, env_params, act, start, keys[:TRAJECTORIES])
    return out, summary


def save_replay(out, cfg, env, env_params, act, start, keys):
    """Fly the episodes of `keys` again and write what the viewer reads:
    trajectory.npz and header.json."""

    def trace(key):
        drone, goal, action, _, _, live = rollout(key, env, env_params, act, start)
        return drone, action, goal, live

    drone, action, goal, live = jax.jit(jax.vmap(trace))(keys)
    scenery = {}
    if hasattr(env, "scenery"):
        # The scenery stays fixed in an episode, so the reset state holds it.
        scenery = jax.vmap(
            lambda k: env.scenery(env.reset(k, env_params)[1], env_params)
        )(keys)
    np.savez(
        out / "trajectory.npz",
        goal=np.asarray(goal),
        live=np.asarray(live),
        action=np.asarray(action),
        **{k: np.asarray(v) for k, v in scenery.items()},
        **{
            f.name: np.asarray(getattr(drone, f.name))
            for f in drone.__dataclass_fields__.values()
        },
    )
    (out / "header.json").write_text(
        json.dumps(
            {
                "task": cfg.task,
                "preset": cfg.preset,
                "roles": list(env_params.roles),
                "center": list(env_params.center),
                "arena": list(env_params.arena),
                "hitbox": 0.5 * env_params.collision_dist,
                "policy_dt": env_params.policy_dt,
            },
            indent=2,
            sort_keys=True,
        )
    )
