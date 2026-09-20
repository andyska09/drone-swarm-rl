"""Measure a policy: fixed eval seeds, mean actions, no auto-reset, per-episode only."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm import envs
from swarm.envs import core
from swarm.learn import config, ppo, runner
from swarm.sim import control

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


def net_policy(cfg, env, env_params, weights):
    slices = ppo.role_slices(env_params.roles, env_params.scripted)
    apply = ppo.make_apply(ppo.make_net(cfg, env), slices)

    def act(state, obs, carry):
        del state
        mean, _, _ = apply(weights, obs)
        return mean, carry

    return act, None


def cascade_policy(env, env_params):
    gains = control.cascade_gains(env_params.model)
    pids = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (env_params.n_drones,) + x.shape),
        control.cascade_init(),
    )

    def act(state, obs, pids):
        del obs

        def one(drone, goal, pid):
            (throttle, rate_ref), pid = control.cascade_outer(
                drone, goal, 0.0, pid, gains, env_params.model, env_params.policy_dt
            )
            return core.command_to_action(throttle, rate_ref, env_params), pid

        return jax.vmap(one)(state.drone, env.reference(state, env_params), pids)

    return act, pids


def _hold(done, new, old):
    return jax.tree.map(lambda a, b: jnp.where(done, b, a), new, old)


def rollout(key, env, env_params, act, carry):
    """One episode to its own end. Per step: drone, reference, action, reward, info, live."""

    obs, state = env.reset(key, env_params)

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


def summarize(reward, info, live):
    steps = live.sum()
    out = {"return": reward.sum(0).mean(), "length": steps.astype(jnp.float32)}
    for name, value in info.items():
        x = value.astype(jnp.float32).reshape(value.shape[0], -1).mean(-1)
        out[f"{name}_mean"] = jnp.sum(x * live) / steps
        out[f"{name}_final"] = jnp.take(x, steps - 1)
    return out


def evaluate(run, episodes=1024, checkpoint="latest", preset=None, policy="checkpoint",
             name=None):
    run = Path(run)
    cfg = load_config(run)
    env, env_params = envs.make(cfg.task, preset or cfg.preset)

    if policy == "cascade":
        act, carry = cascade_policy(env, env_params)
        step = None
    else:
        source = run if policy == "checkpoint" else Path(policy)
        saved = runner.load(source, checkpoint_name(checkpoint))
        act, carry = net_policy(cfg, env, env_params, saved["params"])
        step = saved["update"]

    keys = jax.random.split(jax.random.PRNGKey(EVAL_SEED), episodes)

    def measure(key):
        _, _, _, reward, info, live = rollout(key, env, env_params, act, carry)
        return summarize(reward, info, live)

    def trace(key):
        drone, goal, action, _, _, live = rollout(key, env, env_params, act, carry)
        return drone, action, goal, live

    per_episode = jax.jit(jax.vmap(measure))(keys)
    drone, action, goal, live = jax.jit(jax.vmap(trace))(keys[:TRAJECTORIES])

    out = run / "evals" / (name or (policy if policy != "checkpoint" else checkpoint))
    out.mkdir(parents=True, exist_ok=True)

    summary = {k: float(np.mean(v)) for k, v in per_episode.items()}
    (out / "eval.json").write_text(
        json.dumps(
            {
                "task": cfg.task,
                "preset": preset or cfg.preset,
                "policy": policy,
                "checkpoint": checkpoint if policy != "cascade" else None,
                "update": step,
                "episodes": episodes,
                "eval_seed": EVAL_SEED,
                "git": runner._git(),
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
        **{f.name: np.asarray(getattr(drone, f.name)) for f in drone.__dataclass_fields__.values()},
    )
    # `reference` is a real goal for a drone the cascade flies, and for a drone whose
    # obs carries a target. For anyone else it is only the baseline's input.
    obs0, _ = env.reset(jax.random.PRNGKey(EVAL_SEED), env_params)
    goal = [
        bool(obs0.target.shape[-1]) or role in env_params.scripted
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
                "preset": preset or cfg.preset,
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
