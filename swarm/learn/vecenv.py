import flax.struct
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class Normalizer:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


@flax.struct.dataclass
class VecState:
    env: any
    key: jnp.ndarray
    rew_norm: Normalizer
    ret: jnp.ndarray
    ep_return: jnp.ndarray
    ep_length: jnp.ndarray
    limit: jnp.ndarray
    flying: jnp.ndarray
    seat: any = None


def normalizer(width):
    return Normalizer(jnp.zeros(width), jnp.ones(width), jnp.full(width, 1e-4, dtype=float))


def welford(n, batch):
    batch_mean, batch_var, batch_count = batch.mean(0), batch.var(0), batch.shape[0]
    delta = batch_mean - n.mean
    total = n.count + batch_count
    m2 = (
        n.var * n.count
        + batch_var * batch_count
        + delta**2 * n.count * batch_count / total
    )
    return Normalizer(n.mean + delta * batch_count / total, m2 / total, total)


def normalized(n, x):
    n = jax.lax.stop_gradient(n)
    return (x - n.mean) / jnp.sqrt(n.var + 1e-8)


def select(pick, a, b):
    return jax.tree.map(
        lambda x, y: jnp.where(pick.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), a, b
    )


def at_limit(info, ended, reached):
    time_limit = reached & ~ended
    return time_limit, {**info, "end": {**info["end"], "time_limit": time_limit}}


def reset(key, env, params, cfg, start=None):
    key, k, k_time = jax.random.split(key, 3)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(k, cfg.num_envs), params
    )

    limit = params.max_steps - jax.random.randint(k_time, (cfg.num_envs,), 0, params.max_steps)
    seat = None
    if start is not None:
        env_state, seat = start(env_state)
    obs = jax.vmap(env.get_obs, in_axes=(0, None))(env_state, params)

    vec = VecState(
        env=env_state,
        key=key,
        rew_norm=normalizer(params.n_drones),
        ret=jnp.zeros((cfg.num_envs, params.n_drones)),
        ep_return=jnp.zeros((cfg.num_envs, params.n_drones)),
        ep_length=jnp.zeros(cfg.num_envs, jnp.int32),
        limit=limit.astype(jnp.int32),
        flying=jnp.ones((cfg.num_envs, params.n_drones), bool),
        seat=seat,
    )
    return obs, vec


def step(vec, action, env, params, cfg, learns=1.0, start=None):
    key, k_step, k_reset = jax.random.split(vec.key, 3)
    n = cfg.num_envs

    obs, env_state, reward, ended, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(k_step, n), vec.env, action, params
    )
    ep_length = vec.ep_length + 1
    time_limit, info = at_limit(info, ended, ep_length >= vec.limit)
    flying = info["flying"]
    bootstrap = flying & (time_limit[:, None] | (ended[:, None] & info["bootstrap"]))
    done = (ended | time_limit).astype(reward.dtype)

    info["done"] = done
    info["mask"] = vec.flying.astype(reward.dtype)
    info["cont"] = flying * (1.0 - done[:, None])
    info["bootstrap"] = bootstrap.astype(reward.dtype)
    info["ep_return"] = vec.ep_return + reward
    info["ep_length"] = ep_length

    fresh_obs, fresh_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(k_reset, n), params
    )
    seat = vec.seat
    if start is not None:
        fresh_state, fresh_seat = start(fresh_state)
        fresh_obs = jax.vmap(env.get_obs, in_axes=(0, None))(fresh_state, params)
        seat = select(done, fresh_seat, seat)
    info["final_obs"] = obs
    info["final_state"] = env_state
    env_state = select(done, fresh_state, env_state)
    next_obs = select(done, fresh_obs, obs)

    rew_norm = vec.rew_norm
    ret = vec.ret * cfg.gamma + reward
    if cfg.normalize_reward:
        stepped = welford(rew_norm, ret)
        rew_norm = jax.tree.map(lambda a, b: jnp.where(learns > 0, a, b), stepped, rew_norm)
        reward = reward / jnp.sqrt(rew_norm.var + 1e-8)

    vec = VecState(
        env=env_state,
        key=key,
        rew_norm=rew_norm,
        ret=ret * (1.0 - done)[:, None],
        ep_return=info["ep_return"] * (1.0 - done)[:, None],
        ep_length=info["ep_length"] * (1 - done.astype(jnp.int32)),
        limit=jnp.where(done > 0, params.max_steps, vec.limit).astype(jnp.int32),
        flying=flying | (done[:, None] > 0),
        seat=seat,
    )
    return next_obs, vec, reward, done, info
