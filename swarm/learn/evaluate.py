"""Measure a policy: fixed eval seeds, mean actions, no auto-reset, per-episode only."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm import envs
from swarm.learn import config, ppo, runner
from swarm.learn.seats import SOURCES, combine, net_policy, scripted

# Far from any training seed, so eval never flies an episode training saw.
EVAL_SEED = 1_000_000
TRAJECTORIES = 8


def load_config(run):
    spec = json.loads((Path(run) / "config.json").read_text())
    cfg = config.TrainConfig(**spec["train"])
    cfg.hidden = tuple(cfg.hidden)
    return cfg


def checkpoint_name(text):
    if text.isdigit():
        return f"checkpoints/{int(text):07d}.pkl"
    return text if text.endswith(".pkl") else f"{text}.pkl"


def _saved(run, source, checkpoint):
    path = Path(run) if source == "checkpoint" else Path(source)
    if path.suffix == ".pkl":
        return runner.load(path.parent, path.name)
    return runner.load(path, checkpoint_name(checkpoint))


def make_policy(cfg, env, env_params, run, checkpoint, seats):
    """Who flies each role: this run's weights, another run's, or one of SOURCES."""

    slices = ppo.role_slices(env_params.roles)
    for role, source in seats.items():
        if role not in slices:
            raise ValueError(f"{role!r} has no seat; roles are {sorted(slices)}")
        if SOURCES.get(source, role) not in (None, role):
            raise ValueError(f"{source!r} flies the {SOURCES[source]}, not the {role}")

    nets = {r: seats.get(r, "checkpoint") for r in slices if seats.get(r) not in SOURCES}
    saved = {r: _saved(run, s, checkpoint) for r, s in nets.items()}
    net = net_policy(
        cfg,
        env,
        {r: s["params"][r] for r, s in saved.items()},
        {r: slices[r] for r in nets},
    )
    step = next((s["update"] for s in saved.values()), None)

    fixed = {
        r: scripted(env, env_params, s, slices[r])
        for r, s in seats.items()
        if s in SOURCES
    }
    return *combine(net, fixed, slices), step


def _hold(done, new, old):
    return jax.tree.map(lambda a, b: jnp.where(done, b, a), new, old)


def rollout(key, env, env_params, act, start):
    """One episode to its own end. Per step: drone, reference, action, reward, info, live."""

    _, state = env.reset(key, env_params)
    # A path seat puts its drone on its path, so the obs is built after.
    state, carry = start(state)
    obs = env.get_obs(state, env_params)

    def body(c, _):
        state, obs, carry, done = c
        action, carry = act(state, obs, carry)
        next_obs, next_state, reward, ended, info = env.step(
            key, state, action, env_params
        )
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
        body, (state, obs, carry, jnp.bool_(False)), None, length=env_params.max_steps
    )
    return out


def _mean_defined(values):
    """Average the episodes that have a value. NaN means none of them did."""

    x = np.asarray(values, float)
    good = np.isfinite(x)
    return float(x[good].mean()) if good.any() else float("nan")


def summarize(reward, info, live, slices):
    """One number per metric, reduced the way the task asked for — see the env
    contract in `swarm/envs/__init__.py`."""

    steps = live.sum()
    at_end = lambda x: jnp.take(x, steps - 1)
    over_steps = lambda x: jnp.sum(x * live) / steps

    def split(name, x, per_drone):
        return ppo.by_role(name, x.astype(jnp.float32), slices, per_drone)

    out = {"ep_length": steps.astype(jnp.float32)}
    out |= split("return", reward.sum(0), True)
    out |= {k: at_end(v) for k, v in split("death_rate", 1.0 - info["alive"], True).items()}
    for name, x in info["end"].items():
        out |= {k: at_end(v) for k, v in split(name, x, x.ndim == 2).items()}
    for name, x in info["step"].items():
        out |= {k: over_steps(v) for k, v in split(name, x, x.ndim == 2).items()}
    return out


def evaluate(run, episodes=1024, checkpoint="latest", seats=(), name=None):
    run = Path(run)
    cfg = load_config(run)
    env, env_params = envs.make(cfg.task, cfg.preset)

    seats = dict(seats)
    act, start, step = make_policy(cfg, env, env_params, run, checkpoint, seats)

    keys = jax.random.split(jax.random.PRNGKey(EVAL_SEED), episodes)

    slices = ppo.role_slices(env_params.roles)

    def measure(key):
        _, _, _, reward, info, live = rollout(key, env, env_params, act, start)
        return summarize(reward, info, live, slices)

    def trace(key):
        drone, goal, action, _, _, live = rollout(key, env, env_params, act, start)
        return drone, action, goal, live

    per_episode = jax.jit(jax.vmap(measure))(keys)
    drone, action, goal, live = jax.jit(jax.vmap(trace))(keys[:TRAJECTORIES])

    named = "_".join(f"{r}_{Path(s).name}" for r, s in sorted(seats.items()))
    out = run / "evals" / (name or named or checkpoint)
    out.mkdir(parents=True, exist_ok=True)

    summary = {k: _mean_defined(v) for k, v in per_episode.items()}
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
                "git": runner.git_state(),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    np.savez(out / "episodes.npz", **{k: np.asarray(v) for k, v in per_episode.items()})
    np.savez(
        out / "trajectory.npz",
        goal=np.asarray(goal),
        live=np.asarray(live),
        action=np.asarray(action),
        **{
            f.name: np.asarray(getattr(drone, f.name))
            for f in drone.__dataclass_fields__.values()
        },
    )
    # `reference` is a real goal for a drone the cascade flies, and for a drone whose
    # obs carries a target. For anyone else it is only the baseline's input.
    obs0, _ = env.reset(jax.random.PRNGKey(EVAL_SEED), env_params)
    goal = [
        bool(obs0.target.shape[-1]) or seats.get(role) == "cascade"
        for role in env_params.roles
    ]

    # The net is chase-only.
    net = {
        f: getattr(env_params, f)
        for f in ("net_side", "net_offset")
        if hasattr(env_params, f)
    }
    (out / "header.json").write_text(
        json.dumps(
            {
                "task": cfg.task,
                "preset": cfg.preset,
                "roles": list(env_params.roles),
                "center": list(env_params.center),
                "arena": list(env_params.arena),
                "net": net or None,
                "goal": goal,
                "hitbox": 0.5 * env_params.collision_dist,
                "policy_dt": env_params.policy_dt,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return out, summary
