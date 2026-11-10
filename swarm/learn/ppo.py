"""PPO with separate weights per role and one compiled rollout and training update."""

import math

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
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
    attention: bool = False

    @nn.compact
    def __call__(self, *parts):
        x = _join(parts, self.attention, self.activation)
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
    attention: bool = False

    @nn.compact
    def __call__(self, *parts):
        x = _join(parts, self.attention, self.activation)
        x = _trunk(x, self.hidden, self.activation)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return jnp.squeeze(value, -1)


def _join(parts, attention, activation):
    """A network's one input: its flat input, or with `attention` the parts of
    `attention_input`, the teammates through a Perceiver."""

    if not attention:
        (x,) = parts
        return x
    own, rivals, mates = parts
    return jnp.concatenate([own, rivals, Perceiver(activation)(mates)], -1)


class Perceiver(nn.Module):
    """A set of tokens, (..., k, w), in any order and of any k, to one vector of a
    fixed width: learned queries attend to the tokens. No token gives zeros."""

    activation: str = "tanh"
    queries: int = 4
    heads: int = 4
    head_dim: int = 32

    @nn.compact
    def __call__(self, tokens):
        width = self.heads * self.head_dim
        lead = tokens.shape[:-2]
        if tokens.shape[-2] == 0:
            return jnp.zeros(lead + (self.queries * width,))
        # The inputs come unscaled, in metres and m/s, so a bounded layer comes first.
        tokens = _trunk(tokens, (width,), self.activation)
        q = self.param("queries", nn.initializers.normal(1.0), (self.queries, width))
        out = nn.MultiHeadDotProductAttention(self.heads, qkv_features=width)(
            jnp.broadcast_to(q, lead + q.shape), tokens
        )
        return out.reshape(lead + (-1,))


def sample(key, mean, std):
    return mean + std * jax.random.normal(key, mean.shape)


def bound(draw):
    return jnp.clip(draw, -1.0, 1.0)


def log_prob(draw, mean, std):
    z = (draw - mean) / std
    return -0.5 * jnp.sum(z**2 + 2.0 * jnp.log(std) + LOG_2PI, axis=-1)


def entropy(std):
    return jnp.sum(jnp.log(std) + 0.5 * (LOG_2PI + 1.0), axis=-1)


def column(role, name, slices):
    """Metric column per role, by the role's first letter: `return_p`. A one-role task
    keeps `return`."""

    if len(slices) == 1:
        return name
    if len({r[0] for r in slices}) < len(slices):
        raise ValueError(f"two roles start with the same letter: {list(slices)}")
    return f"{name}_{role[0]}"


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


def episode_metrics(info, end, steps, slices, lead):
    """Every metric that training and eval report.

    A metric with `lead` axes, the steps, is a scene fact. One more axis is a drone
    axis: it splits into one column per role, `<name>_<r>`. An `end` metric is read at
    the steps that `end` marks, where an episode ends, and a NaN leaves that episode out.
    A `step` metric is averaged over the steps, each weighed by `steps`."""

    def split(name, x):
        return by_role(name, x.astype(float), slices, x.ndim > lead)

    def at_end(x):
        return mean_defined(x, end)

    def over_steps(x):
        return jnp.sum(x * steps) / jnp.sum(steps)

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
    if cfg.net not in ("mlp", "attention"):
        raise ValueError(f"net is 'mlp' or 'attention', not {cfg.net!r}")
    attention = cfg.net == "attention"
    return Actor(
        env.NUM_ACTIONS, cfg.hidden, cfg.activation, cfg.init_std, attention
    ), Critic(cfg.hidden, cfg.activation, attention)


def sides(drones, n):
    """Columns out of n drones for the drones of one role: the drones of the other roles,
    and each drone's teammates, (len(drones), len(drones) - 1)."""

    mine = np.asarray(drones)
    rivals = np.setdiff1d(np.arange(n), mine)
    mates = [[j for j in mine if j != i] for i in mine]
    return rivals, np.array(mates, dtype=int).reshape(len(mine), len(mine) - 1)


def repeat_rows(x, n):
    """(..., N, w) to (..., n, N, w): one copy for each of n drones."""

    return jnp.broadcast_to(x[..., None, :, :], x.shape[:-2] + (n,) + x.shape[-2:])


def attention_input(obs, role, drones, action=None):
    """What an attention network of `role` reads: the own block, the drones of the other
    roles in one block, and each drone's teammates as a set. With `action` it is the
    critic's input: each drone also carries its state in the world frame, and each drone
    of another role its action this step."""

    n = len(drones)
    rivals, mates = sides(drones, obs.world.shape[-2])
    own, pairs = obs.own[role], obs.pairs[role]
    if action is None:
        rival = pairs[..., rivals, :]
    else:
        own = jnp.concatenate([own, obs.world[..., drones, :]], -1)
        pairs = jnp.concatenate([pairs, repeat_rows(obs.world, n)], -1)
        rival = jnp.concatenate(
            [pairs[..., rivals, :], repeat_rows(action, n)[..., rivals, :]], -1
        )
    rows = np.arange(n)[:, None]
    return own, rival.reshape(rival.shape[:-2] + (-1,)), pairs[..., rows, mates, :]


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


def critic_input(scene, opp_action, own):
    """The scene and the opponents' actions, then each drone's own actor input."""

    x = jnp.concatenate([scene, opp_action], axis=-1)
    x = jnp.broadcast_to(x[..., None, :], own.shape[:-1] + x.shape[-1:])
    return jnp.concatenate([x, own], axis=-1)


def with_scale(variables, x):
    """A network's variables and a scale for its input x. The scale sits with the weights,
    so whatever saves or flies a network takes its scale too."""

    return {**variables, "scale": {"input": vecenv.normalizer(x.shape[-1])}}


def scaled(variables, x):
    return vecenv.normalized(variables["scale"]["input"], x)


def rescale(variables, x):
    """The network, with its scale moved by the inputs x."""

    moved = vecenv.welford(variables["scale"]["input"], x.reshape(-1, x.shape[-1]))
    return {**variables, "scale": {"input": moved}}


def make_actor(net, slices):
    """Apply each role's actor and return outputs in drone order."""

    def apply(params, obs):
        n = sum(x.shape[-2] for x in obs.actor.values())
        lead = obs.critic.shape[:-1] + (n,)
        mean = jnp.zeros(lead + (net.num_actions,))
        std = jnp.ones(lead + (net.num_actions,))
        for role, drones in slices.items():
            actor = params[role]["actor"]
            if net.attention:
                x = attention_input(obs, role, drones)
            else:
                x = (scaled(actor, obs.actor[role]),)
            m, s = net.apply(actor, *x)
            mean = mean.at[..., drones, :].set(m)
            std = std.at[..., drones, :].set(s)
        return mean, std

    return apply


def make_critic(net, slices, opponents):
    """Apply each role's critic. `action` is every drone's action this step."""

    def apply(params, obs, action):
        value = jnp.zeros(action.shape[:-1])
        for role, drones in slices.items():
            critic = params[role]["critic"]
            if net.attention:
                x = attention_input(obs, role, drones, action)
            else:
                opp = opp_actions(action, opponents[role])
                own = scaled(params[role]["actor"], obs.actor[role])
                x = (critic_input(scaled(critic, obs.critic), opp, own),)
            v = net.apply(critic, *x)
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


@flax.struct.dataclass
class Learner:
    """One learner's part of an update, besides the weights: its envs, the roles that
    take a step, and the pool that flies its rivals, or None. `side` is the role it
    trains, or None when every role learns."""

    vec: vecenv.VecState
    obs: envs.Obs
    key: jnp.ndarray
    # {role: bool}, traced, so freezing a role needs no second compile.
    train: dict
    pool: tuple = None
    side: str = flax.struct.field(pytree_node=False, default=None)


def make(cfg, env, env_params, pool_seat=None):
    """`pool_seat` is a `seats.pool_seat`. It flies its rows over what the actors draw,
    with the pool that the learner carries."""

    assert cfg.num_updates > 0, "total_timesteps < num_envs * num_steps"
    assert (cfg.num_envs * cfg.num_steps) % cfg.num_minibatches == 0
    if cfg.net == "attention" and cfg.normalize_obs:
        raise ValueError("the attention nets read unscaled inputs: set normalize_obs=false")
    batch_size = cfg.num_envs * cfg.num_steps

    actor_net, critic_net = make_net(cfg, env)
    slices = envs.role_slices(env_params.roles)
    opponents = opponent_slices(env_params.roles)
    actor = make_actor(actor_net, slices)
    critic = make_critic(critic_net, slices, opponents)
    pool_rows, pool_act, pool_start = pool_seat or (None, None, None)

    def init(key):
        key, k_env, k_net = jax.random.split(key, 3)
        obs, vec = vecenv.reset(k_env, env, env_params, cfg)

        one_scene = jax.tree.map(lambda x: x[:1], obs)
        one_action = jnp.zeros((1, env_params.n_drones, env.NUM_ACTIONS))
        keys = jax.random.split(k_net, len(slices))

        def params(role, k):
            k_a, k_c = jax.random.split(k)
            if actor_net.attention:
                actor_in = attention_input(one_scene, role, slices[role])
                critic_in = attention_input(one_scene, role, slices[role], one_action)
                return {
                    "actor": actor_net.init(k_a, *actor_in),
                    "critic": critic_net.init(k_c, *critic_in),
                }
            opp = opp_actions(one_action, opponents[role])
            own, scene = one_scene.actor[role], one_scene.critic
            critic_in = critic_input(scene, opp, own)
            return {
                "actor": with_scale(actor_net.init(k_a, own), own),
                "critic": with_scale(critic_net.init(k_c, critic_in), scene),
            }

        ts = {
            role: TrainState.create(
                apply_fn=None,
                params=params(role, k),
                # The lr sits in the optimizer state, so each role carries its own.
                tx=optax.inject_hyperparams(
                    lambda learning_rate: optax.chain(
                        optax.clip_by_global_norm(cfg.max_grad_norm),
                        optax.adam(learning_rate, eps=1e-5),
                    )
                )(learning_rate=cfg.lr),
            )
            for role, k in zip(slices, keys)
        }
        return ts, Learner(vec, obs, key, {r: True for r in slices})

    def weights(ts):
        return {role: t.params for role, t in ts.items()}

    def update(ts, learner):
        vec, obs, key = learner.vec, learner.obs, learner.key
        train, pool = learner.train, learner.pool
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

        def rescaled(role, t):
            nets = {
                "actor": rescale(t.params["actor"], traj.obs.actor[role]),
                "critic": rescale(t.params["critic"], traj.obs.critic),
            }
            return hold(train[role], t.replace(params=nets), t)

        # After the epochs, so the rollout and the loss read the same inputs.
        if cfg.normalize_obs:
            ts = {role: rescaled(role, t) for role, t in ts.items()}

        done = info["done"]
        metrics = {"episodes": done.sum()} | episode_metrics(
            info, done, jnp.ones_like(done), slices, lead=2
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

        return (ts, learner.replace(vec=vec, obs=obs, key=key)), metrics

    return init, jax.jit(update)
