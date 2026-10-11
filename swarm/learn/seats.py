"""Who flies a seat: a network, the cascade, or a hand-written law. Eval and training
both use it.

A seat flies some rows of one scene. `act(state, obs, carry)` gives their actions and
the next carry, and `start(state)` gives the state and the first carry of an episode.
"""

import jax
import jax.numpy as jnp

from swarm.envs import core
from swarm.learn import ppo, vecenv
from swarm.sim import control, guidance

# The role each hand-written source flies. The cascade flies any.
SOURCES = {
    "cascade": None,
    "frpn": "pursuer",
    "apf": "evader",
    "hover": "evader",
    "circle": "evader",
    "figure8": "evader",
    "paths": "evader",
}
# `paths` draws one of these for each episode, all three equally likely.
PATHS = ("hover", "circle", "figure8")


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
    shapes = PATHS if source == "paths" else (source,)
    n = len(rows)

    def fly(drone, a, pid):
        heading = jnp.arctan2(drone.R[1, 0], drone.R[0, 0])
        orientation, throttle = control.acceleration_controller(
            drone, guidance.drone_limit(a, model), heading, model
        )
        # A negative thrust gives a NaN throttle, so use 0.
        throttle = jnp.where(jnp.isnan(throttle), 0.0, throttle)
        rate, pid = control.attitude_controller(drone, orientation, pid, gains, dt)
        return core.command_to_action(throttle, rate, env_params), pid

    def at(path, t):
        return jnp.stack([guidance.path_at(s, path, t) for s in shapes])[path["shape"]]

    def start(state):
        pids = _per_row(control.pid_init(), n)
        if source in ("frpn", "apf"):
            return state, (pids, None)

        def draw(k):
            drawn = [
                guidance.draw_path(k, s, lo + env_params.margin, hi - env_params.margin)
                for s in shapes
            ]
            shape = jax.random.randint(jax.random.fold_in(k, 1), (), 0, len(shapes))
            path = drawn[0]
            for i, other in enumerate(drawn[1:], 1):
                path = jax.tree.map(lambda a, b: jnp.where(shape == i, b, a), path, other)
            # Training starts an episode at any `time`, so a path counts from its own start.
            return path | {"shape": shape, "t0": state.time}

        paths = jax.vmap(draw)(jax.random.split(state.key, n))
        on = jax.vmap(lambda p: at(p, 0.0))(paths)
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
            target = jax.vmap(lambda p: at(p, (state.time - p["t0"]) * dt))(paths)
            a = jax.vmap(guidance.follow)(x, v, target)
        if source in ("frpn", "apf"):
            a = jax.vmap(guidance.nemecek_limit)(a, v)

        action, pids = jax.vmap(fly)(jax.tree.map(lambda q: q[rows], d), a, pids)
        return action, (pids, paths)

    return act, start


def scripted(env, env_params, source, rows):
    """The act and start of one of SOURCES."""

    if source == "cascade":
        return cascade_policy(env, env_params, rows)
    return guided_policy(env, env_params, source, rows)


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
    scenes. Member 0 is `seed`, one of SOURCES. Any other member is a frozen network
    that flies its mean.

    `pool` is (member, weights): each block's member and each block's actor weights.
    Returns the rows it flies, and its act and start over the whole batch."""

    rows = ppo.role_slices(env_params.roles)[role]
    seed_act, seed_start = scripted(env, env_params, seed, rows)
    actor = ppo.make_actor(ppo.make_net(cfg, env)[0], {role: rows})

    def is_seed(member, n):
        return jnp.repeat(member == 0, n // member.shape[0])

    def act(state, obs, carry, pool):
        member, weights = pool
        n, blocks = obs.own.shape[0], member.shape[0]
        flown, carry = jax.vmap(seed_act)(state, obs, carry)
        # An empty target block prevents inferring the block size with -1.
        split = jax.tree.map(lambda x: x.reshape((blocks, n // blocks) + x.shape[1:]), obs)
        mean, _ = jax.vmap(lambda w, o: actor({role: {"actor": w}}, o))(weights, split)
        net = ppo.squash(mean.reshape((n,) + mean.shape[2:]))[:, rows]
        return jnp.where(is_seed(member, n)[:, None, None], flown, net), carry

    def start(state, pool):
        member, _ = pool
        started, carry = jax.vmap(seed_start)(state)
        return vecenv.select(is_seed(member, state.time.shape[0]), started, state), carry

    return rows, act, start
