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

KL_TARGET = 0.016
LR_MIN, LR_MAX = 1e-5, 1e-2


def adapt_lr(lr, kl):
    """Divide the lr by 1.5 when the KL is over twice the target, and multiply it by
    1.5 when the KL is under half of it."""

    return jnp.select(
        [kl > 2.0 * KL_TARGET, (kl > 0.0) & (kl < 0.5 * KL_TARGET)],
        [jnp.maximum(lr / 1.5, LR_MIN), jnp.minimum(lr * 1.5, LR_MAX)],
        lr,
    )


def lr_of(t):
    return t.opt_state.hyperparams["learning_rate"]


def with_lr(t, lr):
    state = t.opt_state
    return t.replace(
        opt_state=state._replace(hyperparams={**state.hyperparams, "learning_rate": lr})
    )


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
    init_std: float = 1.0

    @nn.compact
    def __call__(self, x):
        x = _trunk(x, self.hidden, self.activation)
        mean = nn.Dense(
            self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)
        log_std = self.param(
            "log_std", constant(math.log(self.init_std)), (self.num_actions,)
        )
        return mean, jnp.broadcast_to(jnp.exp(log_std), mean.shape)


class Critic(nn.Module):
    hidden: tuple = (256, 256)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        x = _trunk(x, self.hidden, self.activation)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return jnp.squeeze(value, -1)


def sample(key, mean, std):
    return mean + std * jax.random.normal(key, mean.shape)


def bound(draw):
    return jnp.clip(draw, -1.0, 1.0)


def log_prob(draw, mean, std):
    z = (draw - mean) / std
    return -0.5 * jnp.sum(z**2 + 2.0 * jnp.log(std) + LOG_2PI, axis=-1)


def entropy(std):
    return jnp.sum(jnp.log(std) + 0.5 * (LOG_2PI + 1.0), axis=-1)


def role_slices(roles):
    """Each role's drone indices."""

    return {
        name: jnp.array([i for i, r in enumerate(roles) if r == name])
        for name in dict.fromkeys(roles)
    }


def column(role, name, slices):
    """Metric column per role: `pursuer_return`. A one-role task keeps `return`."""

    return f"{role}_{name}" if len(slices) > 1 else name


def by_role(name, x, slices, per_drone):
    """One column for a scene fact, one per role for anything with a drone axis."""

    if not per_drone:
        return {name: x}
    return {column(r, name, slices): x[..., d].mean(-1) for r, d in slices.items()}


def mean_defined(x, weight):
    """The mean of x where weight > 0. A NaN says an entry has no value, so it is
    dropped, not counted. NaN if no entry is left."""

    weight = weight * jnp.isfinite(x)
    n = weight.sum()
    total = jnp.sum(jnp.where(weight > 0, x, 0.0))
    return jnp.where(n > 0, total / jnp.maximum(n, 1.0), jnp.nan)


def episode_metrics(info, at_end, over_steps, slices, lead):
    """Every metric that training and eval report.

    A metric with `lead` axes, the steps, is a scene fact. One more axis is a drone
    axis: it splits into one column per role, `<role>_<name>`. `at_end` reads a metric
    where each episode ends, and `over_steps` averages it over the steps."""

    def split(name, x):
        return by_role(name, x.astype(float), slices, x.ndim > lead)

    out = {"ep_length": at_end(info["ep_length"])}
    for name, x in (("return", info["ep_return"]), ("death_rate", 1.0 - info["alive"])):
        out |= {k: at_end(v) for k, v in split(name, x).items()}
    for name, x in info["end"].items():
        out |= {k: at_end(v) for k, v in split(name, x).items()}
    for name, x in info["step"].items():
        out |= {k: over_steps(v) for k, v in split(name, x).items()}
    return out


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


def opponent_slices(roles):
    """Every drone whose role is not mine. Teammates are left out."""

    return {
        role: jnp.array([i for i, r in enumerate(roles) if r != role], dtype=jnp.int32)
        for role in dict.fromkeys(roles)
    }


def opp_actions(action, opponents):
    """The opponents' actions this step, flat."""

    a = jnp.take(action, opponents, axis=-2)
    return a.reshape(a.shape[:-2] + (-1,))


def critic_input(obs, opp_action, rows):
    """The scene and the opponents' actions, one copy for each of a role's `rows` drones."""

    x = jnp.concatenate([obs.critic, opp_action], axis=-1)
    return jnp.broadcast_to(x[..., None, :], x.shape[:-1] + (rows, x.shape[-1]))


def make_actor(net, slices):
    """Apply each role's actor and return outputs in drone order."""

    def apply(params, obs):
        n = sum(rows.shape[-2] for rows in obs.actor.values())
        lead = obs.critic.shape[:-1] + (n,)
        mean = jnp.zeros(lead + (net.num_actions,))
        std = jnp.ones(lead + (net.num_actions,))
        for role, drones in slices.items():
            m, s = net.apply(params[role]["actor"], obs.actor[role])
            mean = mean.at[..., drones, :].set(m)
            std = std.at[..., drones, :].set(s)
        return mean, std

    return apply


def make_critic(net, slices, opponents):
    """Apply each role's critic. `action` is every drone's action this step."""

    def apply(params, obs, action):
        value = jnp.zeros(action.shape[:-1])
        for role, drones in slices.items():
            v = net.apply(
                params[role]["critic"],
                critic_input(obs, opp_actions(action, opponents[role]), len(drones)),
            )
            value = value.at[..., drones].set(v)
        return value

    return apply


@flax.struct.dataclass
class Transition:
    obs: envs.Obs
    draw: jnp.ndarray  # before the clip, which the log-prob needs
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    cont: jnp.ndarray
    mask: jnp.ndarray


def make(cfg, env, env_params, pool_seat=None):
    """`pool_seat` is a `seats.pool_seat`. It flies its rows over what the actors draw,
    and `update` then takes the pool as its last argument."""

    assert cfg.num_updates > 0, "total_timesteps < num_envs * num_steps"
    assert (cfg.num_envs * cfg.num_steps) % cfg.num_minibatches == 0
    batch_size = cfg.num_envs * cfg.num_steps

    actor_net, critic_net = make_net(cfg, env)
    slices = role_slices(env_params.roles)
    opponents = opponent_slices(env_params.roles)
    actor = make_actor(actor_net, slices)
    critic = make_critic(critic_net, slices, opponents)
    pool_rows, pool_act, pool_start = pool_seat or (None, None, None)

    # Whose turn it is. A traced flag, so alternating needs no second compile.
    every_turn = {r: True for r in slices}

    def init(key):
        key, k_env, k_net = jax.random.split(key, 3)
        obs, vec = vecenv.reset(k_env, env, env_params, cfg)

        one_scene = jax.tree.map(lambda x: x[:1], obs)
        one_action = jnp.zeros((1, env_params.n_drones, env.NUM_ACTIONS))
        keys = jax.random.split(k_net, len(slices))

        def params(role, drones, k):
            k_a, k_c = jax.random.split(k)
            opp = opp_actions(one_action, opponents[role])
            return {
                "actor": actor_net.init(k_a, one_scene.actor[role]),
                "critic": critic_net.init(k_c, critic_input(one_scene, opp, len(drones))),
            }

        ts = {
            role: TrainState.create(
                apply_fn=None,
                params=params(role, drones, k),
                # The lr sits in the optimizer state, so each role carries its own.
                tx=optax.inject_hyperparams(
                    lambda learning_rate: optax.chain(
                        optax.clip_by_global_norm(cfg.max_grad_norm),
                        optax.adam(learning_rate, eps=1e-5),
                    )
                )(learning_rate=cfg.lr),
            )
            for (role, drones), k in zip(slices.items(), keys)
        }
        return ts, vec, obs, key

    def weights(ts):
        return {role: t.params for role, t in ts.items()}

    def update(ts, vec, obs, key, train=every_turn, pool=None):
        # 1 for each drone whose role takes a step: only these rows enter the loss.
        learns = jnp.stack([jnp.asarray(train[r]) for r in env_params.roles]).astype(float)
        start = None if pool_seat is None else lambda state: pool_start(state, pool)

        def act(ts, state, obs, seat, key):
            mean, std = actor(weights(ts), obs)
            draw = sample(key, mean, std)
            action = bound(draw)
            if pool_seat is not None:
                flown, seat = pool_act(state, obs, seat, pool)
                action = action.at[:, pool_rows].set(flown)
            return action, draw, mean, std, seat

        def value_at(ts, state, obs, seat, key):
            """No action exists at this state yet, so draw one."""

            return critic(weights(ts), obs, act(ts, state, obs, seat, key)[0])

        def rollout(carry, _):
            ts, vec, obs, key = carry
            key, k, k_final = jax.random.split(key, 3)
            action, draw, mean, std, seat = act(ts, vec.env, obs, vec.seat, k)
            value = critic(weights(ts), obs, action)
            next_obs, vec, reward, done, info = vecenv.step(
                vec.replace(seat=seat), action, env, env_params, cfg, learns, start
            )
            # Bootstrap truncations from the final observation before the scene resets.
            # A diverged drone makes v_final NaN, also for a survivor that sees it, and
            # NaN * 0 is NaN, so select instead.
            v_final = value_at(
                ts, info.pop("final_state"), info.pop("final_obs"), seat, k_final
            )
            bootstrap = (info["bootstrap"] > 0) & jnp.isfinite(v_final)
            transition = Transition(
                obs,
                draw,
                action,
                log_prob(draw, mean, std),
                value,
                reward + cfg.gamma * jnp.where(bootstrap, v_final, 0.0),
                info["cont"],
                info["mask"] * learns,
            )
            return (ts, vec, next_obs, key), (transition, info)

        (ts, vec, obs, key), (traj, info) = lax.scan(
            rollout, (ts, vec, obs, key), None, length=cfg.num_steps
        )

        key, k_last = jax.random.split(key)
        last_value = value_at(ts, vec.env, obs, vec.seat, k_last)

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

                    # Per drone, so the same numbers can be reported per role.
                    clipped = -jnp.minimum(
                        ratio * a_n,
                        jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a_n,
                    )
                    squared = 0.5 * (value - target) ** 2
                    ent = entropy(std)

                    loss = (
                        _mean(clipped, tr.mask)
                        + cfg.vf_coef * _mean(squared, tr.mask)
                        - cfg.ent_coef * _mean(ent, tr.mask)
                    )
                    return loss, {
                        "policy_loss": clipped,
                        "value_loss": squared,
                        "entropy": ent,
                        "clipfrac": (jnp.abs(ratio - 1.0) > cfg.clip_eps) * 1.0,
                        "approx_kl": (ratio - 1.0) - jnp.log(ratio + 1e-8),
                    }

                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(weights(ts))
                stats = {
                    column(role, name, slices): _mean(x[..., drones], tr.mask[..., drones])
                    for name, x in aux.items()
                    for role, drones in slices.items()
                }

                def step(role, t):
                    if cfg.adaptive_lr:
                        lr = adapt_lr(lr_of(t), stats[column(role, "approx_kl", slices)])
                        t = with_lr(t, lr)
                    return t.apply_gradients(grads=grads[role])

                # Selecting the whole TrainState freezes the Adam moments, the lr and
                # the step count too, not only the weights.
                return {
                    role: hold(train[role], step(role, t), t) for role, t in ts.items()
                }, stats

            ts, stats = lax.scan(minibatch, ts, minibatches)
            return (ts, key), stats

        (ts, key), stats = lax.scan(epoch, (ts, key), None, length=cfg.update_epochs)

        done = info["done"]
        metrics = {"episodes": done.sum()} | episode_metrics(
            info,
            lambda x: mean_defined(x.astype(done.dtype), done),
            lambda x: x.mean(),
            slices,
            lead=2,
        )

        # Per role: the two sides of a game have opposite signs, so one pooled
        # variance rewards a critic that has only learned which role it is.
        for role, drones in slices.items():
            target, value = targets[..., drones], traj.value[..., drones]
            var_y = jnp.var(target)
            metrics[column(role, "explained_var", slices)] = jnp.where(
                var_y > 0, 1.0 - jnp.var(target - value) / var_y, jnp.nan
            )
        metrics |= jax.tree.map(lambda x: x[-1].mean(), stats)
        metrics |= {column(role, "lr", slices): lr_of(t) for role, t in ts.items()}

        return (ts, vec, obs, key), metrics

    return init, jax.jit(update)
