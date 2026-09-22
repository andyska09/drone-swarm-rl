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
LOG_2 = math.log(2.0)

MIN_STD = 1e-3  # keeps log(std) finite

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


def _inv_softplus(x):
    return math.log(math.expm1(x))


def _trunk(x, hidden, activation):
    act = nn.tanh if activation == "tanh" else nn.relu
    for h in hidden:
        x = act(
            nn.Dense(h, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        )
    return x


class Actor(nn.Module):
    """What flies. Local observations only, so it can run on a drone."""

    num_actions: int
    hidden: tuple = (256, 256)
    activation: str = "tanh"
    init_std: float = 0.6

    @nn.compact
    def __call__(self, obs):
        x = _trunk(envs.flat(obs), self.hidden, self.activation)
        head = lambda bias: nn.Dense(
            self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(bias)
        )(x)
        mean = head(0.0)
        std = nn.softplus(head(_inv_softplus(self.init_std))) + MIN_STD
        return mean, std


class Critic(nn.Module):
    hidden: tuple = (256, 256)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs, opp_action):
        x = _trunk(envs.flat_critic(obs, opp_action), self.hidden, self.activation)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return jnp.squeeze(value, -1)


def sample(key, mean, std):
    return mean + std * jax.random.normal(key, mean.shape)


def squash(draw):
    return jnp.tanh(draw)


def _log_slope(draw):
    """log(1 - tanh(draw)^2), stable at large |draw|."""

    return 2.0 * (LOG_2 - draw - jax.nn.softplus(-2.0 * draw))


def log_prob(draw, mean, std):
    """Log density of tanh(draw) under the squashed Gaussian."""

    z = (draw - mean) / std
    gauss = -0.5 * jnp.sum(z**2 + 2.0 * jnp.log(std) + LOG_2PI, axis=-1)
    return gauss - jnp.sum(_log_slope(draw), axis=-1)


def entropy(draw, std):
    """Entropy of the squashed Gaussian, estimated from one draw."""

    gauss = jnp.sum(jnp.log(std) + 0.5 * (LOG_2PI + 1.0), axis=-1)
    return gauss + jnp.sum(_log_slope(draw), axis=-1)


def role_slices(roles, scripted=()):
    """One entry per learned role. A scripted role gets no weights and no loss."""

    return {
        name: jnp.array([i for i, r in enumerate(roles) if r == name])
        for name in dict.fromkeys(roles)
        if name not in scripted
    }


def _mean(x, mask):
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def hold(training, stepped, old):
    """The stepped state if this role is training, the old one if it is frozen."""

    return jax.tree.map(lambda a, b: jnp.where(training, a, b), stepped, old)


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
    return Actor(env.NUM_ACTIONS, cfg.hidden, cfg.activation, cfg.init_std), Critic(
        cfg.hidden, cfg.activation
    )


def opponent_slices(roles, slices):
    """Every drone whose role is not mine. Teammates are left out."""

    return {
        role: jnp.array([i for i, r in enumerate(roles) if r != role], dtype=jnp.int32)
        for role in slices
    }


def opp_actions(action, opponents):
    """The opponents' actions this step, flat."""

    a = jnp.take(action, opponents, axis=-2)
    return a.reshape(a.shape[:-2] + (-1,))


def make_actor(net, slices):
    """Apply each role's actor and return outputs in drone order."""

    def apply(params, obs):
        lead = obs.own.shape[:-1]
        mean = jnp.zeros(lead + (net.num_actions,))
        # A scripted role has no weights, so its rows keep std 1 and log(std) stays finite.
        std = jnp.ones(lead + (net.num_actions,))
        for role, drones in slices.items():
            m, s = net.apply(params[role]["actor"], envs.take(obs, drones))
            mean = mean.at[..., drones, :].set(m)
            std = std.at[..., drones, :].set(s)
        return mean, std

    return apply


def make_critic(net, slices, opponents):
    """Apply each role's critic. `action` is every drone's action this step."""

    def apply(params, obs, action):
        value = jnp.zeros(obs.own.shape[:-1])
        for role, drones in slices.items():
            v = net.apply(
                params[role]["critic"],
                envs.take(obs, drones),
                opp_actions(action, opponents[role]),
            )
            value = value.at[..., drones].set(v)
        return value

    return apply


@flax.struct.dataclass
class Transition:
    obs: envs.Obs
    draw: jnp.ndarray  # pre-squash sample; atanh of the action is unstable at ±1
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

    actor_net, critic_net = make_net(cfg, env)
    slices = role_slices(env_params.roles, env_params.scripted)
    opponents = opponent_slices(env_params.roles, slices)
    actor = make_actor(actor_net, slices)
    critic = make_critic(critic_net, slices, opponents)

    # Whose turn it is. A traced flag, so alternating needs no second compile.
    every_turn = {r: not cfg.train_roles or r in cfg.train_roles for r in slices}

    def init(key):
        key, k_env, k_net = jax.random.split(key, 3)
        obs, vec = vecenv.reset(k_env, env, env_params, cfg)

        one_scene = jax.tree.map(lambda x: x[:1], obs)
        one_action = jnp.zeros((1, env_params.n_drones, env.NUM_ACTIONS))
        keys = jax.random.split(k_net, len(slices))

        if cfg.anneal_lr:

            def lr(count):
                done = count // (cfg.update_epochs * cfg.num_minibatches)
                return cfg.lr * (1.0 - done / cfg.num_updates)

        else:
            lr = cfg.lr

        def params(role, drones, k):
            k_a, k_c = jax.random.split(k)
            scene = envs.take(one_scene, drones)
            return {
                "actor": actor_net.init(k_a, scene),
                "critic": critic_net.init(
                    k_c, scene, opp_actions(one_action, opponents[role])
                ),
            }

        ts = {
            role: TrainState.create(
                apply_fn=None,
                params=params(role, drones, k),
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

    def value_at(ts, obs, key):
        """No action exists at this state yet, so draw one."""

        mean, std = actor(weights(ts), obs)
        return critic(weights(ts), obs, squash(sample(key, mean, std)))

    def update(ts, vec, obs, key, train=every_turn):
        def rollout(carry, _):
            ts, vec, obs, key = carry
            key, k = jax.random.split(key)
            mean, std = actor(weights(ts), obs)
            draw = sample(k, mean, std)
            action = squash(draw)
            value = critic(weights(ts), obs, action)
            next_obs, vec, reward, done, info = vecenv.step(
                vec, action, env, env_params, cfg
            )
            transition = Transition(
                obs,
                draw,
                action,
                log_prob(draw, mean, std),
                value,
                reward,
                info["cont"],
                info["mask"],
            )
            return (ts, vec, next_obs, key), (transition, info)

        (ts, vec, obs, key), (traj, info) = lax.scan(
            rollout, (ts, vec, obs, key), None, length=cfg.num_steps
        )

        key, k_final, k_last = jax.random.split(key, 3)

        # Bootstrap timeouts from the final observation before the scene resets.
        # A diverged drone makes v_final NaN, and NaN * 0 is NaN, so select instead.
        v_final = value_at(ts, info["final_obs"], k_final)
        traj = traj.replace(
            reward=traj.reward
            + cfg.gamma * jnp.where(info["bootstrap"] > 0, v_final, 0.0)
        )

        last_value = value_at(ts, obs, k_last)

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
                    mean, std = actor(params, tr.obs)
                    value = critic(params, tr.obs, tr.action)
                    ratio = jnp.exp(log_prob(tr.draw, mean, std) - tr.log_prob)

                    policy = -_mean(
                        jnp.minimum(
                            ratio * a_n,
                            jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a_n,
                        ),
                        tr.mask,
                    )
                    value_loss = 0.5 * _mean((value - target) ** 2, tr.mask)
                    ent = _mean(entropy(tr.draw, std), tr.mask)
                    return policy + cfg.vf_coef * value_loss - cfg.ent_coef * ent, (
                        policy,
                        value_loss,
                        ent,
                        ratio,
                        tr.mask,
                    )

                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(weights(ts))
                policy, value_loss, ent, ratio, mask = aux
                # Selecting the whole TrainState freezes the Adam moments and the
                # step count too, not only the weights.
                stepped = {
                    role: hold(train[role], t.apply_gradients(grads=grads[role]), t)
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
