"""Batch scenes, reset finished episodes, and normalize rewards."""

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
    seat: any = None  # the carry of a seat that flies some rows, per scene


def _welford(n, batch):
    batch_mean, batch_var, batch_count = batch.mean(0), batch.var(0), batch.shape[0]
    delta = batch_mean - n.mean
    total = n.count + batch_count
    m2 = (
        n.var * n.count
        + batch_var * batch_count
        + delta**2 * n.count * batch_count / total
    )
    return Normalizer(n.mean + delta * batch_count / total, m2 / total, total)


def select(pick, a, b):
    """Select leaves from a for the scenes `pick` marks and from b for the rest."""

    return jax.tree.map(
        lambda x, y: jnp.where(pick.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), a, b
    )


def reset(key, env, params, cfg, start=None):
    """`start(states) -> (states, carry)` begins each episode of a seat."""

    key, k, k_time = jax.random.split(key, 3)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(k, cfg.num_envs), params
    )

    # Shorten the first episodes to spread timeouts across updates.
    env_state = env_state.replace(
        time=jax.random.randint(k_time, (cfg.num_envs,), 0, params.max_steps)
    )
    seat = None
    if start is not None:
        env_state, seat = start(env_state)
    obs = jax.vmap(env.get_obs, in_axes=(0, None))(env_state, params)

    vec = VecState(
        env=env_state,
        key=key,
        # One scale per drone column, so a pursuer and an evader are not pooled.
        rew_norm=Normalizer(
            jnp.zeros(params.n_drones),
            jnp.ones(params.n_drones),
            jnp.full(params.n_drones, 1e-4),
        ),
        ret=jnp.zeros((cfg.num_envs, params.n_drones)),
        ep_return=jnp.zeros((cfg.num_envs, params.n_drones)),
        ep_length=jnp.zeros(cfg.num_envs, jnp.int32),
        seat=seat,
    )
    return obs, vec


def step(vec, action, env, params, cfg, learns=1.0, start=None):
    """`learns` is 1 for each drone column whose reward scale may move. `start` is
    as in `reset`."""

    key, k_step, k_reset = jax.random.split(vec.key, 3)
    n = cfg.num_envs

    obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(k_step, n), vec.env, action, params
    )
    done = done.astype(reward.dtype)
    nobody = jnp.zeros_like(info["alive"])
    flying = (info["alive"] & ~info.get("arrived", nobody)).astype(reward.dtype)
    stopped = info["died_this_step"] | info.get("arrived_this_step", nobody)

    info = dict(info)
    info["done"] = done
    info["mask"] = flying + stopped.astype(reward.dtype)
    info["cont"] = flying * (1.0 - done[:, None])
    info["bootstrap"] = flying * info["truncated"][:, None].astype(reward.dtype)
    info["ep_return"] = vec.ep_return + reward
    info["ep_length"] = vec.ep_length + 1

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

    # The terminal step carries the largest return, so measure before clearing.
    rew_norm = vec.rew_norm
    ret = vec.ret * cfg.gamma + reward
    if cfg.normalize_reward:
        stepped = _welford(rew_norm, ret)
        rew_norm = jax.tree.map(lambda a, b: jnp.where(learns > 0, a, b), stepped, rew_norm)
        reward = reward / jnp.sqrt(rew_norm.var + 1e-8)

    vec = VecState(
        env=env_state,
        key=key,
        rew_norm=rew_norm,
        ret=ret * (1.0 - done)[:, None],
        ep_return=info["ep_return"] * (1.0 - done)[:, None],
        ep_length=info["ep_length"] * (1 - done.astype(jnp.int32)),
        seat=seat,
    )
    return next_obs, vec, reward, done, info
