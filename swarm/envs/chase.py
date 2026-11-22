import flax.struct
import jax
import jax.numpy as jnp

from swarm.envs import Obs, core, role_slices
from swarm.sim import control, dynamics, guidance

NUM_ACTIONS = 4
PROGRESS = ("ep_length", "death_rate", "caught_e", "coll_pp")
PATHS = ("hover", "circle", "figure8")


@flax.struct.dataclass
class RewardConfig:
    progress: float = 0.25
    step: float = 4.0
    cmd: float = 0.02
    catch: float = 30.0
    coll_pe: float = 10.0
    coll_pp: float = 10.0
    fail: float = 30.0
    outside: float = 0.0


@flax.struct.dataclass
class EnvParams(core.Common):
    reward: RewardConfig = flax.struct.field(default_factory=RewardConfig)

    roles: tuple = flax.struct.field(pytree_node=False, default=("pursuer", "evader"))
    max_steps: int = flax.struct.field(pytree_node=False, default=1000)
    walls: bool = flax.struct.field(pytree_node=False, default=True)

    sense_pursuer: tuple = flax.struct.field(pytree_node=False, default=())
    sense_evader: tuple = flax.struct.field(pytree_node=False, default=())

    net_side: float = 1.0
    net_offset: float = 0.4
    body_catch: bool = flax.struct.field(pytree_node=False, default=False)
    net_kills: bool = flax.struct.field(pytree_node=False, default=True)

    spacing: float = 2.0

    waypoint_range: float = 3.0
    waypoint_reach: float = 0.2


@flax.struct.dataclass
class EnvState:
    drone: dynamics.State
    rate_pid: control.PIDState
    key: jnp.ndarray
    alive: jnp.ndarray
    time: jnp.ndarray


def _sides(params):
    return role_slices(params.roles)["pursuer"], params.roles.index("evader")


def _drop(params):
    return params.net_offset + 0.5 * params.net_side


def net_centre(drone, params):
    pursuers, _ = _sides(params)
    return drone.x[pursuers] - _drop(params) * drone.R[pursuers, :, 2]


def catch_gap(drone, params):
    pursuers, evader = _sides(params)
    point = drone.x[pursuers] if params.body_catch else net_centre(drone, params)
    return core.norm(drone.x[evader] - point)


def net_frame(drone, params):
    pursuers, _ = _sides(params)
    p = jnp.einsum("pji,pkj->pki", drone.R[pursuers], drone.x - drone.x[pursuers][:, None])
    return p.at[..., 2].add(_drop(params))


def in_net(drone, params):
    p = net_frame(drone, params)
    return jnp.all(jnp.abs(p[..., 1:]) <= 0.5 * params.net_side, axis=-1) & (
        jnp.abs(p[..., 0]) <= core.body_radius(params)
    )


def is_caught(drone, params):
    if params.body_catch:
        return coll_pe(drone, params)
    _, evader = _sides(params)
    return in_net(drone, params)[:, evader]


def net_collision(drone, params):
    pursuers, evader = _sides(params)
    hit = in_net(drone, params).at[:, evader].set(False)
    netter = jnp.zeros(params.n_drones, bool).at[pursuers].set(jnp.any(hit, axis=-1))
    return netter | jnp.any(hit, axis=0)


def get_obs(state, params):
    d = state.drone
    own = core.own_obs(d, core.wall_dist(d, params))
    sense = {"pursuer": params.sense_pursuer, "evader": params.sense_evader}
    slices = role_slices(params.roles)
    actor = {
        role: jnp.concatenate([own, core.others(d, params, True, sense[role])], -1)[rows]
        for role, rows in slices.items()
    }
    return Obs(
        actor=actor,
        critic=core.scene_obs(d),
        own={role: own[rows] for role, rows in slices.items()},
        pairs={role: core.pairs(d, params, sense[role])[rows] for role, rows in slices.items()},
        world=core.world(d),
    )


def crashed(drone, params):
    floor = params.center[2] - 0.5 * params.arena[2]
    return (drone.x[:, 2] < floor) | ~jnp.all(jnp.isfinite(drone.x), axis=-1)


def failed(drone, params):
    return core.out_of_arena(drone, params) if params.walls else crashed(drone, params)


def coll_pp(drone, params):
    _, evader = _sides(params)
    hit = core.hits(drone, params).at[evader].set(False).at[:, evader].set(False)
    hit = jnp.any(hit, axis=-1)
    if params.body_catch or not params.net_kills:
        return hit
    return hit | net_collision(drone, params)


def coll_pe(drone, params):
    pursuers, evader = _sides(params)
    return core.hits(drone, params)[pursuers, evader]


def reference(state, params):
    del state
    return jnp.full((params.n_drones, 3), jnp.nan)


def scenery(state, params):
    del state
    pursuers, _ = _sides(params)
    net = jnp.array([params.net_side, params.net_offset]) * (not params.body_catch)
    return {"net": jnp.zeros((params.n_drones, 2)).at[pursuers].set(net)}


def compute_reward(before, d, rate_ref, alive, fail, collide, caught, params):
    cfg = params.reward
    pursuers, evader = _sides(params)
    closed = catch_gap(before, params) - catch_gap(d, params)
    progress = jnp.zeros(params.n_drones).at[pursuers].set(cfg.progress * closed)

    hit = coll_pe(d, params)
    both = jnp.zeros(params.n_drones).at[pursuers].set(hit).at[evader].set(jnp.any(hit))

    clock = jnp.zeros(params.n_drones).at[evader].set(cfg.step) - cfg.coll_pe * both
    clock = clock.at[evader].add(-cfg.outside * core.out_of_arena(d, params)[evader])
    catch = jnp.full(params.n_drones, cfg.catch).at[evader].set(-cfg.catch)

    return (
        jnp.where(
            alive,
            params.policy_dt * (clock - cfg.cmd * core.norm(rate_ref)) + progress,
            0.0,
        )
        + catch * caught
        - cfg.coll_pp * collide
        - cfg.fail * fail
    )


def _waypoint(key, x, params):
    k_dir, k_len = jax.random.split(key)
    direction = jax.random.normal(k_dir, (3,))
    reach = jax.random.uniform(k_len, (), maxval=params.waypoint_range)
    centre = jnp.asarray(params.center)
    limit = core.half(params) - params.margin
    offset = x - centre + reach * direction / core.norm(direction)
    return centre + jnp.clip(offset, -limit, limit)


def _starts(key, x, free, params):
    centre = jnp.asarray(params.center)
    limit = core.half(params) - params.margin
    draws = centre + jax.random.uniform(key, (16,) + x.shape, minval=-limit, maxval=limit)
    return core.first_apart(jnp.where(free[:, None], draws, x), params.spacing)


def reset(key, params):
    n = params.n_drones
    k_pos, k_spawn, k_state = jax.random.split(key, 3)
    x = _starts(k_pos, jnp.zeros((n, 3)), jnp.ones(n, bool), params)
    drone = core.spawn(k_spawn, x, params)

    state = EnvState(
        drone=drone,
        rate_pid=control.pid_init((n, 3)),
        key=k_state,
        alive=jnp.ones(n, bool),
        time=jnp.int32(0),
    )
    return get_obs(state, params), state


def step(key, state, action, params):
    del key
    pursuers, evader = _sides(params)
    drone, rate_pid, rate_ref = core.fly(state.drone, state.rate_pid, action, params)

    caught = jnp.any(is_caught(drone, params)) & jnp.all(state.alive)
    fail = state.alive & failed(drone, params) & ~caught
    collide = state.alive & coll_pp(drone, params) & ~caught
    died = fail | collide
    alive = state.alive & ~died

    new_state = EnvState(
        drone=drone,
        rate_pid=rate_pid,
        key=state.key,
        alive=alive,
        time=state.time + 1,
    )

    reward = compute_reward(state.drone, drone, rate_ref, state.alive, fail, collide, caught, params)
    done = caught | ~jnp.all(alive)

    crash = fail & crashed(drone, params)
    oob = fail & ~crash
    crash_p, oob_p = jnp.any(crash[pursuers]), jnp.any(oob[pursuers])
    fail_p, fail_e = crash_p | oob_p | jnp.any(collide), fail[evader]

    on_net = net_frame(drone, params)[:, evader, 1:]
    info = {
        "alive": alive,
        "flying": alive,
        "bootstrap": jnp.full(params.n_drones, ~caught),
        "end": {
            "caught_e": caught,
            "fail_pe": fail_p & fail_e,
            "crash_p": crash_p & ~fail_e,
            "OOB_p": oob_p & ~crash_p & ~fail_e,
            "coll_pp": fail_p & ~crash_p & ~oob_p & ~fail_e,
            "crash_e": crash[evader] & ~fail_p,
            "OOB_e": oob[evader] & ~fail_p,
            "time_to_catch": jnp.where(caught, new_state.time * params.policy_dt, jnp.nan),
            "catch_offset": jnp.where(caught & (not params.body_catch), jnp.min(core.norm(on_net)), jnp.nan),
        },
        "step": {
            "distance": jnp.min(catch_gap(drone, params)),
            "coll_pe": jnp.any(coll_pe(drone, params)),
        },
    }
    return get_obs(new_state, params), new_state, reward, done, info


def _aim(drone, params):
    pursuers, evader = _sides(params)
    aim = drone.x[evader] + _drop(params) * drone.R[pursuers, :, 2]
    return jnp.broadcast_to(drone.x[evader], drone.x.shape).at[pursuers].set(aim)


def cascade(params, rows):
    _, evader = _sides(params)
    is_evader = (jnp.asarray(rows) == evader)[:, None]

    def start(state):
        k_wp, key = jax.random.split(state.key)
        waypoint = _waypoint(k_wp, state.drone.x[evader], params)
        return state, (core.per_row(control.cascade_init(), len(rows)), waypoint, key)

    def act(state, obs, carry):
        del obs
        pids, waypoint, key = carry
        d = state.drone
        key, k_wp = jax.random.split(key)
        reached = core.norm(waypoint - d.x[evader]) < params.waypoint_reach
        waypoint = jnp.where(reached, _waypoint(k_wp, d.x[evader], params), waypoint)
        points = jnp.where(is_evader, waypoint, _aim(d, params)[rows])
        own = jax.tree.map(lambda q: q[rows], d)
        action, pids = core.cascade(own, points, pids, params)
        return action, (pids, waypoint, key)

    return act, start


def frpn(params, rows):
    _, evader = _sides(params)

    def act(state, obs, pids):
        del obs
        d = state.drone
        own = jax.tree.map(lambda q: q[rows], d)
        a = jax.vmap(guidance.frpn)(_aim(d, params)[rows] - own.x, d.v[evader] - own.v)
        a = jax.vmap(guidance.speed_limit)(a, own.v)
        return core.accelerate(own, a, pids, params)

    return act, lambda state: (state, core.per_row(control.pid_init(), len(rows)))


def apf(params, rows, factor=1.0, limited=True):
    pursuers, _ = _sides(params)
    lo = jnp.asarray(params.center) - core.half(params)
    hi = jnp.asarray(params.center) + core.half(params)

    def act(state, obs, pids):
        del obs
        d = state.drone
        own = jax.tree.map(lambda q: q[rows], d)
        a = jax.vmap(guidance.apf, in_axes=(0, 0, None, None, None, None, None))(
            own.x, own.v, d.x[pursuers], d.v[pursuers], lo, hi, factor
        )
        if limited:
            a = jax.vmap(guidance.speed_limit)(a, own.v)
        return core.accelerate(own, a, pids, params)

    return act, lambda state: (state, core.per_row(control.pid_init(), len(rows)))


def paths(params, rows):
    dt, n = params.policy_dt, len(rows)
    lo = jnp.asarray(params.center) - core.half(params) + params.margin
    hi = jnp.asarray(params.center) + core.half(params) - params.margin

    def at(path, t):
        return jnp.stack([guidance.path_at(s, path, t) for s in PATHS])[path["shape"]]

    def start(state):
        def draw(k):
            drawn = [guidance.draw_path(k, s, lo, hi) for s in PATHS]
            shape = jax.random.randint(jax.random.fold_in(k, 1), (), 0, len(PATHS))
            path = drawn[0]
            for i, other in enumerate(drawn[1:], 1):
                path = jax.tree.map(lambda a, b: jnp.where(shape == i, b, a), path, other)
            return path | {"shape": shape}

        k_paths, k_place = jax.random.split(state.key)
        routes = jax.vmap(draw)(jax.random.split(k_paths, n))
        on = jax.vmap(lambda p: at(p, 0.0))(routes)
        others = jnp.ones(params.n_drones, bool).at[rows].set(False)
        x = _starts(k_place, state.drone.x.at[rows].set(on[:, 0]), others, params)
        drone = state.drone.replace(x=x, v=state.drone.v.at[rows].set(on[:, 1]))
        return state.replace(drone=drone), (core.per_row(control.pid_init(), n), routes)

    def act(state, obs, carry):
        del obs
        pids, routes = carry
        own = jax.tree.map(lambda q: q[rows], state.drone)
        target = jax.vmap(lambda p: at(p, state.time * dt))(routes)
        a = jax.vmap(guidance.follow)(own.x, own.v, target)
        action, pids = core.accelerate(own, a, pids, params)
        return action, (pids, routes)

    return act, start


PILOTS = {"cascade": cascade, "frpn": frpn, "apf": apf, "paths": paths}

SENSES = ((), ("attitude",), ("rpm",), ("attitude", "rpm"))

NEMECEK = EnvParams(
    arena=(20.0, 20.0, 10.0),
    center=(0.0, 0.0, 5.0),
    walls=False,
    reward=RewardConfig(outside=57.0),
)

PRESETS = (
    {
        "default": EnvParams(),
        "open": EnvParams(walls=False, reward=RewardConfig(outside=57.0)),
        "small": EnvParams(arena=(24.0, 24.0, 16.0)),
        "nemecek": NEMECEK,
        "3v1": EnvParams(roles=("pursuer",) * 3 + ("evader",)),
    }
    | {
        "_".join((f"paths_{p}v1",) + sense): EnvParams(
            roles=("pursuer",) * p + ("evader",),
            sense_pursuer=sense,
            piloted=("evader", "paths"),
        )
        for p in (1, 3)
        for sense in SENSES
    }
    | {
        "paths_3v1_body": EnvParams(
            roles=("pursuer",) * 3 + ("evader",),
            piloted=("evader", "paths"),
            body_catch=True,
            reward=RewardConfig(coll_pe=0.0),
        ),
        "paths_3v1_netpass": EnvParams(
            roles=("pursuer",) * 3 + ("evader",),
            piloted=("evader", "paths"),
            net_kills=False,
        ),
    }
    | {f"apf_{p}v1": EnvParams(roles=("pursuer",) * p + ("evader",), piloted=("evader", "apf")) for p in (1, 3)}
    | {"_".join(("default", "p", *p)): EnvParams(sense_pursuer=p) for p in SENSES[1:]}
    | {
        "_".join(("nemecek", "p", *p, "e", *e)): NEMECEK.replace(sense_pursuer=p, sense_evader=e)
        for p in SENSES[1:]
        for e in SENSES[1:]
    }
)
