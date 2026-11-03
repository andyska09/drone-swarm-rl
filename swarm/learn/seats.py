"""Who flies a seat: a network, or one of the task's PILOTS. Eval and training both use it.

A seat flies some rows of one scene. `act(state, obs, carry)` gives their actions and
the next carry, and `start(state)` gives the state and the first carry of an episode.
"""

import jax
import jax.numpy as jnp

from swarm import envs
from swarm.learn import ppo, vecenv


def net_policy(cfg, env, weights, slices):
    apply = ppo.make_actor(ppo.make_net(cfg, env)[0], slices)

    def act(state, obs, carry):
        del state
        mean, _ = apply(weights, obs)
        return ppo.bound(mean), carry

    return act


def combine(net, fixed, slices):
    """One act and start for the scene. The net flies its roles, and each fixed seat,
    {role: (act, start)}, flies the rows of its own role."""

    if not fixed:
        return net, lambda state: (state, None)

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

    return act, start


def pool_seat(cfg, env, env_params, role, seed):
    """A pool of opponents flies `role` in a batch of scenes, one member per block of
    scenes. Member 0 is `seed`, one of the task's PILOTS. Any other member is a frozen
    network that flies its mean.

    `pool` is (member, weights): each block's member and each block's actor weights.
    Returns the rows it flies, and its act and start over the whole batch."""

    rows = envs.role_slices(env_params.roles)[role]
    seed_act, seed_start = env.PILOTS[seed](env_params, rows)
    actor = ppo.make_actor(ppo.make_net(cfg, env)[0], {role: rows})

    def is_seed(member, n):
        return jnp.repeat(member == 0, n // member.shape[0])

    def act(state, obs, carry, pool):
        member, weights = pool
        n, blocks = obs.critic.shape[0], member.shape[0]
        flown, carry = jax.vmap(seed_act)(state, obs, carry)
        split = jax.tree.map(lambda x: x.reshape((blocks, n // blocks) + x.shape[1:]), obs)
        mean, _ = jax.vmap(lambda w, o: actor({role: {"actor": w}}, o))(weights, split)
        net = ppo.bound(mean.reshape((n,) + mean.shape[2:]))[:, rows]
        return jnp.where(is_seed(member, n)[:, None, None], flown, net), carry

    def start(state, pool):
        member, _ = pool
        started, carry = jax.vmap(seed_start)(state)
        scenes = jax.tree.leaves(state)[0].shape[0]
        return vecenv.select(is_seed(member, scenes), started, state), carry

    return rows, act, start
