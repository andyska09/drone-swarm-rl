"""Measure a policy: fixed eval seeds, mean actions, no auto-reset, per-episode only."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm import envs
from swarm.envs import core
from swarm.learn import config, ppo, runner
from swarm.sim import control, guidance

# Far from any training seed, so eval never flies an episode training saw.
EVAL_SEED = 1_000_000
TRAJECTORIES = 8

# The role each hand-written source flies. The cascade flies any.
SOURCES = {
    "cascade": None,
    "frpn": "pursuer",
    "apf": "evader",
    "hover": "evader",
    "circle": "evader",
    "figure8": "evader",
}


def load_config(run):
    spec = json.loads((Path(run) / "config.json").read_text())
    cfg = config.TrainConfig(**spec["train"])
    cfg.hidden = tuple(cfg.hidden)
    return cfg


def checkpoint_name(text):
    if text.isdigit():
        return f"checkpoints/{int(text):07d}.pkl"
    return text if text.endswith(".pkl") else f"{text}.pkl"


def net_policy(cfg, env, weights, slices):
    apply = ppo.make_actor(ppo.make_net(cfg, env)[0], slices)

    def act(state, obs, carry):
        del state
        mean, _ = apply(weights, obs)
        return ppo.squash(mean), carry

    return act


def _per_row(tree, n):
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), tree)


def cascade_policy(env, env_params, rows):
    gains = control.cascade_gains(env_params.model)

    def act(state, obs, pids):
        del obs

        def one(drone, goal, pid):
            (throttle, rate_ref), pid = control.cascade_outer(
                drone, goal, 0.0, pid, gains, env_params.model, env_params.policy_dt
            )
            return core.command_to_action(throttle, rate_ref, env_params), pid

        drone = jax.tree.map(lambda a: a[rows], state.drone)
        return jax.vmap(one)(drone, env.reference(state, env_params)[rows], pids)

    return act, lambda state: (state, _per_row(control.cascade_init(), len(rows)))


def guided_policy(env, env_params, source, rows):
    """FRPN, APF or a path. The law gives a world acceleration, and MRS's
    acceleration and attitude levels turn it into the CTBR action a network sends."""

    model, dt = env_params.model, env_params.policy_dt
    gains = control.attitude_gains(max_rate_yaw=env_params.max_rate_yaw)
    pursuers = jnp.array([i for i, r in enumerate(env_params.roles) if r == "pursuer"])
    evader = env_params.roles.index("evader")
    lo = jnp.asarray(env_params.center) - core.half(env_params)
    hi = jnp.asarray(env_params.center) + core.half(env_params)
    n = len(rows)

    def fly(drone, a, pid):
        heading = jnp.arctan2(drone.R[1, 0], drone.R[0, 0])
        orientation, throttle = control.acceleration_controller(
            drone, guidance.drone_limit(a, model), heading, model
        )
        # MRS takes the square root of the thrust unguarded. A negative thrust, from a
        # reversal over 1 g, gives a zero throttle, as in Němeček's copy of MRS.
        throttle = jnp.where(jnp.isnan(throttle), 0.0, throttle)
        rate, pid = control.attitude_controller(drone, orientation, pid, gains, dt)
        return core.command_to_action(throttle, rate, env_params), pid

    def start(state):
        pids = _per_row(control.pid_init(), n)
        if source in ("frpn", "apf"):
            return state, (pids, None)
        draw = lambda k: guidance.draw_path(
            k, source, lo + env_params.margin, hi - env_params.margin
        )
        paths = jax.vmap(draw)(jax.random.split(state.key, n))
        on = jax.vmap(lambda p: guidance.path_at(source, p, 0.0))(paths)
        drone = state.drone.replace(
            x=state.drone.x.at[rows].set(on[:, 0]), v=state.drone.v.at[rows].set(on[:, 1])
        )
        return state.replace(drone=drone), (pids, paths)

    def act(state, obs, carry):
        del obs
        pids, paths = carry
        d = state.drone
        x, v = d.x[rows], d.v[rows]
        if source == "frpn":
            # `reference` is where the pursuer flies its net centre onto the evader.
            aim = env.reference(state, env_params)[rows]
            a = jax.vmap(guidance.frpn)(aim - x, d.v[evader] - v)
        elif source == "apf":
            a = jax.vmap(guidance.apf, in_axes=(0, 0, None, None, None, None))(
                x, v, d.x[pursuers], d.v[pursuers], lo, hi
            )
        else:
            target = jax.vmap(lambda p: guidance.path_at(source, p, state.time * dt))(paths)
            a = jax.vmap(guidance.follow)(x, v, target)
        if source in ("frpn", "apf"):
            a = jax.vmap(guidance.nemecek_limit)(a, v)

        action, pids = jax.vmap(fly)(jax.tree.map(lambda q: q[rows], d), a, pids)
        return action, (pids, paths)

    return act, start


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
        r: cascade_policy(env, env_params, slices[r])
        if s == "cascade"
        else guided_policy(env, env_params, s, slices[r])
        for r, s in seats.items()
        if s in SOURCES
    }
    if not fixed:
        return net, lambda state: (state, None), step

    def act(state, obs, carry):
        action, _ = net(state, obs, None)
        out = {}
        for role, (fly, _) in fixed.items():
            flown, out[role] = fly(state, obs, carry[role])
            action = action.at[slices[role]].set(flown)
        return action, out

    def start(state):
        carry = {}
        for role, (_, seat_start) in fixed.items():
            state, carry[role] = seat_start(state)
        return state, carry

    return act, start, step


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
