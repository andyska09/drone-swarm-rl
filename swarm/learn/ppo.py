"""PPO with separate weights per role and one compiled rollout and training update."""

import math

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax import lax

from swarm import envs
from swarm.learn import vecenv

LOG_2PI = math.log(2.0 * math.pi)

# Exclude episode control and bookkeeping fields from reported env metrics.
CONTROL_KEYS = frozenset(
    {
        "alive",
        "died_this_step",
        "truncated",
        "done",
        "mask",
        "cont",
        "bootstrap",
        "final_obs",
        "ep_return",
        "ep_length",
    }
)


class ActorCritic(nn.Module):
    num_actions: int
    hidden: tuple = (256, 256)
    activation: str = "tanh"
    init_log_std: float = -0.5

    @nn.compact
    def __call__(self, obs):
        x = envs.flat(obs)
        act = nn.tanh if self.activation == "tanh" else nn.relu

        def trunk(x):
            for h in self.hidden:
                x = act(
                    nn.Dense(
                        h, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
                    )(x)
                )
            return x

        mean = nn.Dense(
            self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(trunk(x))
        log_std = self.param(
            "log_std", lambda _: jnp.full((self.num_actions,), self.init_log_std)
        )
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            trunk(x)
        )
        return mean, log_std, jnp.squeeze(value, -1)


def sample(key, mean, log_std):
    return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)


def log_prob(action, mean, log_std):
    z = (action - mean) / jnp.exp(log_std)
    return -0.5 * jnp.sum(z**2 + 2.0 * log_std + LOG_2PI, axis=-1)


def entropy(log_std):
    return jnp.sum(log_std + 0.5 * (LOG_2PI + 1.0), axis=-1)


def role_slices(roles, scripted=()):
    """One entry per learned role. A scripted role gets no weights and no loss."""

    return {
        name: jnp.array([i for i, r in enumerate(roles) if r == name])
        for name in dict.fromkeys(roles)
        if name not in scripted
    }


def _mean(x, mask):
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def normalize_by_role(adv, mask, slices):
    """Centre and scale each role on its own. A pursuer and an evader have
    opposite signs and different sizes, so one shared scale flattens both."""

    out = jnp.zeros_like(adv)
    for drones in slices.values():
        a, m = adv[..., drones], mask[..., drones]
        centred = a - _mean(a, m)
        out = out.at[..., drones].set(centred / (jnp.sqrt(_mean(centred**2, m)) + 1e-8))
    return out


def make_net(cfg, env):
    return ActorCritic(env.NUM_ACTIONS, cfg.hidden, cfg.activation, cfg.init_log_std)


def make_apply(net, slices):
    """Apply each role's weights and return outputs in drone order."""

    def apply(params, obs):
        lead = obs.own.shape[:-1]
        mean = jnp.zeros(lead + (net.num_actions,))
        log_std = jnp.zeros(lead + (net.num_actions,))
        value = jnp.zeros(lead)
        for role, drones in slices.items():
            m, ls, v = net.apply(params[role], envs.take(obs, drones))
            mean = mean.at[..., drones, :].set(m)
            log_std = log_std.at[..., drones, :].set(jnp.broadcast_to(ls, m.shape))
            value = value.at[..., drones].set(v)
        return mean, log_std, value

    return apply


@flax.struct.dataclass
class Transition:
    obs: envs.Obs
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    cont: jnp.ndarray
    mask: jnp.ndarray


def make(cfg, env, env_params):
    assert cfg.num_updates > 0, "total_timesteps < num_envs * num_steps"
    assert (cfg.num_envs * cfg.num_steps) % cfg.num_minibatches == 0
    batch_size = cfg.num_envs * cfg.num_steps

    net = make_net(cfg, env)
    slices = role_slices(env_params.roles, env_params.scripted)
    apply = make_apply(net, slices)

    trained = tuple(r for r in slices if not cfg.train_roles or r in cfg.train_roles)

    def init(key):
        key, k_env, k_net = jax.random.split(key, 3)
        obs, vec = vecenv.reset(k_env, env, env_params, cfg)

        one_scene = jax.tree.map(lambda x: x[:1], obs)
        keys = jax.random.split(k_net, len(slices))

        if cfg.anneal_lr:

            def lr(count):
                done = count // (cfg.update_epochs * cfg.num_minibatches)
                return cfg.lr * (1.0 - done / cfg.num_updates)

        else:
            lr = cfg.lr

        # One TrainState per role, so gradient clipping and Adam never mix two
        # sides of a game, and a role can be frozen without touching the other.
        ts = {
            role: TrainState.create(
                apply_fn=None,
                params=net.init(k, envs.take(one_scene, drones)),
                tx=optax.chain(
                    optax.clip_by_global_norm(cfg.max_grad_norm),
                    optax.adam(lr, eps=1e-5),
                ),
            )
            for (role, drones), k in zip(slices.items(), keys)
        }
        return ts, vec, obs, key

    def weights(ts):
        return {role: t.params for role, t in ts.items()}

    def update(ts, vec, obs, key):
        def rollout(carry, _):
            ts, vec, obs, key = carry
            key, k = jax.random.split(key)
            mean, log_std, value = apply(weights(ts), obs)
            action = sample(k, mean, log_std)
            next_obs, vec, reward, done, info = vecenv.step(
                vec, action, env, env_params, cfg
            )
            transition = Transition(
                obs,
                action,
                log_prob(action, mean, log_std),
                value,
                reward,
                info["cont"],
                info["mask"],
            )
            return (ts, vec, next_obs, key), (transition, info)

        (ts, vec, obs, key), (traj, info) = lax.scan(
            rollout, (ts, vec, obs, key), None, length=cfg.num_steps
        )

        # Bootstrap timeouts from the final observation before the scene resets.
        # A diverged drone makes v_final NaN, and NaN * 0 is NaN, so select instead.
        _, _, v_final = apply(weights(ts), info["final_obs"])
        traj = traj.replace(
            reward=traj.reward
            + cfg.gamma * jnp.where(info["bootstrap"] > 0, v_final, 0.0)
        )

        _, _, last_value = apply(weights(ts), obs)

        def gae_step(carry, tr):
            gae, next_value = carry
            delta = tr.reward + cfg.gamma * next_value * tr.cont - tr.value
            gae = delta + cfg.gamma * cfg.gae_lambda * tr.cont * gae
            return (gae, tr.value), gae

        _, adv = lax.scan(
            gae_step,
            (jnp.zeros_like(last_value), last_value),
            traj,
            reverse=True,
            unroll=16,
        )
        targets = adv + traj.value

        def epoch(carry, _):
            ts, key = carry
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, batch_size)
            flat = jax.tree.map(
                lambda x: x.reshape((batch_size,) + x.shape[2:])[perm],
                (traj, adv, targets),
            )
            # An empty neighbor dimension prevents inferring the batch size with -1.
            minibatches = jax.tree.map(
                lambda x: x.reshape(
                    (cfg.num_minibatches, cfg.minibatch_size) + x.shape[1:]
                ),
                flat,
            )

            def minibatch(ts, batch):
                tr, a, target = batch
                a_n = normalize_by_role(a, tr.mask, slices)

                def loss_fn(params):
                    mean, log_std, value = apply(params, tr.obs)
                    ratio = jnp.exp(log_prob(tr.action, mean, log_std) - tr.log_prob)

                    policy = -_mean(
                        jnp.minimum(
                            ratio * a_n,
                            jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a_n,
                        ),
                        tr.mask,
                    )
                    value_loss = 0.5 * _mean((value - target) ** 2, tr.mask)
                    ent = _mean(entropy(log_std), tr.mask)
                    return policy + cfg.vf_coef * value_loss - cfg.ent_coef * ent, (
                        policy,
                        value_loss,
                        ent,
                        ratio,
                        tr.mask,
                    )

                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(weights(ts))
                policy, value_loss, ent, ratio, mask = aux
                stepped = {
                    role: t.apply_gradients(grads=grads[role]) if role in trained else t
                    for role, t in ts.items()
                }
                return stepped, {
                    "policy_loss": policy,
                    "value_loss": value_loss,
                    "entropy": ent,
                    "clipfrac": _mean(
                        (jnp.abs(ratio - 1.0) > cfg.clip_eps) * 1.0, mask
                    ),
                    "approx_kl": _mean((ratio - 1.0) - jnp.log(ratio + 1e-8), mask),
                }

            ts, stats = lax.scan(minibatch, ts, minibatches)
            return (ts, key), stats

        (ts, key), stats = lax.scan(epoch, (ts, key), None, length=cfg.update_epochs)

        done = info["done"]
        episodes = done.sum()

        def per_episode(x):
            return jnp.where(episodes > 0, (x * done).sum() / episodes, jnp.nan)

        var_y = jnp.var(targets)
        metrics = {
            "episodes": episodes,
            "return": per_episode(info["ep_return"].mean(-1)),
            "ep_length": per_episode(info["ep_length"]),
            "death_rate": per_episode(1.0 - info["alive"].mean(-1)),
            "explained_var": jnp.where(
                var_y > 0, 1.0 - jnp.var(targets - traj.value) / var_y, jnp.nan
            ),
            **jax.tree.map(lambda x: x[-1].mean(), stats),
        }
        for key_name, value in info.items():
            if key_name not in CONTROL_KEYS:
                metrics[key_name] = per_episode(
                    value.mean(-1) if value.ndim == 3 else value
                )

        return (ts, vec, obs, key), metrics

    return init, jax.jit(update)
