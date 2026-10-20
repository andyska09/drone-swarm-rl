"""Two drones fly opposite ways along a bent square tube and must pass each other.

The design: research/notes/task_tunnel.md.
"""

import flax.struct
import jax
import jax.numpy as jnp

from swarm.envs import Obs, core
from swarm.sim import control, dynamics

NUM_ACTIONS = 4


@flax.struct.dataclass
class RewardConfig:
    """progress is paid per metre along the drone's own line, line per metre off that
    line per second, and crash once."""

    progress: float = 1.0
    line: float = 1.0
    crash: float = 10.0


@flax.struct.dataclass
class EnvParams(core.Common):
    reward: RewardConfig = flax.struct.field(default_factory=RewardConfig)

    roles: tuple = flax.struct.field(pytree_node=False, default=("drone", "drone"))
    max_steps: int = flax.struct.field(pytree_node=False, default=1500)

    # The tube is square, so `width` is also its height. `corner` is the angle between
    # the walls at an inner corner, measured through the solid.
    width: float = 2.4
    corner: float = 111.0
    middle: float = 2.4  # the inner wall of the middle piece
    leg: float = 10.0  # along the centre line
    cap: float = 1.0  # the tube is closed this far past each end of a line

    line_offset: float = 0.5  # from the centre line, sideways and up or down
    start_offset: float = 0.5
    start_anywhere: bool = True  # False starts each drone at the start of its line
    start_gap: float = 2.0  # the closest a start gets to the end of its line
    start_apart: float = 2.0  # the closest two drones start

    look_ahead: float = 3.0
    sight: float = float("inf")


@flax.struct.dataclass
class EnvState:
    drone: dynamics.State
    rate_pid: control.PIDState
    heading: jnp.ndarray  # the tunnel's yaw in the world
    offset: jnp.ndarray  # (n, 2): each line, moved sideways and up from the centre line
    progress: jnp.ndarray  # (n,): metres along the own line
    alive: jnp.ndarray
    time: jnp.ndarray
    closest: jnp.ndarray
    unseen: jnp.ndarray  # (n,): steps without sight of the other drone
    unseen_distance: jnp.ndarray  # (n,): distance from the line, summed over those steps


def yaw(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def to_tunnel(x, heading, params):
    """World to the tunnel frame: origin at the middle of the tube, x along the middle
    piece, y toward its outer wall, z up."""

    return (x - jnp.asarray(params.center)) @ yaw(heading)


def to_world(p, heading, params):
    return jnp.asarray(params.center) + p @ yaw(heading).T


def polyline(offset, leg, params):
    """(4, 2): the centre line moved `offset` toward the outer wall, left end first, seen
    from above. Its corners sit on the mitre lines, so it keeps its distance to the walls
    through the bends."""

    t = jnp.deg2rad(180.0 - params.corner)
    h = 0.5 * params.width
    up = jnp.array([0.0, 1.0])
    out_l = jnp.array([-jnp.sin(t), jnp.cos(t)])
    out_r = jnp.array([jnp.sin(t), jnp.cos(t)])
    away_l = jnp.array([-jnp.cos(t), -jnp.sin(t)])
    away_r = jnp.array([jnp.cos(t), -jnp.sin(t)])
    mitre_l = (out_l + up) / (1.0 + jnp.cos(t))
    mitre_r = (out_r + up) / (1.0 + jnp.cos(t))
    bend_l = jnp.array([-0.5 * params.middle, -h]) + h * mitre_l
    bend_r = jnp.array([0.5 * params.middle, -h]) + h * mitre_r
    return jnp.stack(
        [
            bend_l + leg * away_l + offset * out_l,
            bend_l + offset * mitre_l,
            bend_r + offset * mitre_r,
            bend_r + leg * away_r + offset * out_r,
        ]
    )


def walls(params):
    """(8, 2): the outline of the tube seen from above, a closed polygon."""

    h = 0.5 * params.width
    reach = params.leg + params.cap
    return jnp.concatenate([polyline(-h, reach, params), polyline(h, reach, params)[::-1]])


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _edges(poly):
    return poly, jnp.roll(poly, -1, axis=0)


def inside(p, poly):
    """p, (..., 2), lies inside the polygon."""

    a, b = _edges(poly)
    p = p[..., None, :]
    spans = (a[:, 1] > p[..., 1]) != (b[:, 1] > p[..., 1])
    dy = jnp.where(spans, b[:, 1] - a[:, 1], 1.0)
    x = a[:, 0] + (p[..., 1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / dy
    return jnp.sum(spans & (p[..., 0] < x), axis=-1) % 2 == 1


def wall_gap(p, poly):
    """Distance from p, (..., 2), to the nearest wall."""

    a, b = _edges(poly)
    ab = b - a
    t = jnp.clip(jnp.sum((p[..., None, :] - a) * ab, -1) / jnp.sum(ab**2, -1), 0.0, 1.0)
    return jnp.min(core.norm(p[..., None, :] - a - t[..., None] * ab), axis=-1)


def ray_length(p, d, poly):
    """How far the ray from p along d, both (..., 2), runs before it meets a wall."""

    a, b = _edges(poly)
    ab, ap = b - a, a - p[..., None, :]
    d = d[..., None, :]
    den = _cross(d, ab)
    safe = jnp.where(den == 0.0, 1.0, den)
    t, s = _cross(ap, ab) / safe, _cross(ap, d) / safe
    # A ray through a corner can miss both of its walls by rounding, so the ends get
    # a little slack.
    hit = (den != 0.0) & (t >= 0.0) & (s >= -1e-4) & (s <= 1.0 + 1e-4)
    return jnp.min(jnp.where(hit, t, jnp.inf), axis=-1)


def blocked(p, q, poly):
    """The segment from p to q, both (..., 2), crosses a wall."""

    a, b = _edges(poly)
    p, q = p[..., None, :], q[..., None, :]
    side = lambda o, x, y: jnp.sign(_cross(x - o, y - o))
    return jnp.any(
        (side(p, q, a) != side(p, q, b)) & (side(a, b, p) != side(a, b, q)), axis=-1
    )


def sees(p, x, params):
    """(n, n): drone i sees drone j. No wall stands between their centres, and j is
    closer than `sight`. The floor and the ceiling are flat, so only the walls block.
    `pair_dist` puts infinity on the diagonal, so no drone sees itself."""

    clear = ~blocked(p[:, None, :2], p[None, :, :2], walls(params))
    return clear & (core.pair_dist(x) < params.sight)


def lines(offset, params):
    """(n, 4, 2): each drone's line, in the direction it flies. Even drones start in the
    left leg, odd drones in the right one."""

    line = jax.vmap(lambda o: polyline(o, params.leg, params))(offset[:, 0])
    left = (jnp.arange(offset.shape[0]) % 2 == 0)[:, None, None]
    return jnp.where(left, line, line[:, ::-1])


def _knots(line):
    """Arc length at each corner of a line."""

    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(core.norm(line[1:] - line[:-1]))])


def project(p, line):
    """Arc length of the point on `line`, (4, 2), nearest to p, (2,)."""

    a, b = line[:-1], line[1:]
    ab = b - a
    t = jnp.clip(jnp.sum((p - a) * ab, -1) / jnp.sum(ab**2, -1), 0.0, 1.0)
    k = jnp.argmin(core.norm(p - a - t[:, None] * ab))
    return _knots(line)[k] + t[k] * core.norm(ab[k])


def point_at(s, line):
    knots = _knots(line)
    return jnp.stack([jnp.interp(s, knots, line[:, 0]), jnp.interp(s, knots, line[:, 1])])


def follow(p, offset, params):
    """Each drone's progress along its own line, that line's length, and the line."""

    line = lines(offset, params)
    return jax.vmap(project)(p[:, :2], line), jax.vmap(_knots)(line)[:, -1], line


def on_line(s, line, offset):
    """(n, 3): the point at arc length s on each line, in the tunnel frame."""

    return jnp.concatenate([jax.vmap(point_at)(s, line), offset[:, 1:]], axis=-1)


def wall_distances(drone, heading, p, params):
    """Distance to the tunnel walls ahead, behind, left and right, level with the floor."""

    nose, left = drone.R[:, :2, 0], drone.R[:, :2, 1]
    level = jnp.where(
        core.norm(nose)[:, None] > 1e-3, nose, jnp.stack([left[:, 1], -left[:, 0]], -1)
    )
    d = level / core.norm(level)[:, None]
    side = jnp.stack([-d[:, 1], d[:, 0]], -1)
    rays = jnp.stack([d, -d, side, -side], axis=1) @ yaw(heading)[:2, :2]
    return ray_length(p[:, None, :2], rays, walls(params))


def get_obs(state, params):
    d = state.drone
    p = to_tunnel(d.x, state.heading, params)
    s, _, line = follow(p, state.offset, params)
    seen = sees(p, d.x, params)

    def body(q):
        return core.to_body(d.R, (q - p) @ yaw(state.heading).T)

    others, mask = core.others(d, params)
    # The same order core.others sorts its rows in.
    _, idx = jax.lax.top_k(-core.pair_dist(d.x), params.n_visible)

    return Obs(
        own=jnp.concatenate(
            [
                core.to_body(d.R, d.v),
                d.R.reshape(-1, 9),
                d.omega,
                p[:, 2:3] + 0.5 * params.width,
                wall_distances(d, state.heading, p, params),
            ],
            axis=-1,
        ),
        others=others,
        others_mask=mask * jnp.take_along_axis(seen, idx, axis=1),
        target=jnp.concatenate(
            [
                body(on_line(s, line, state.offset)),
                body(on_line(s + params.look_ahead, line, state.offset)),
            ],
            axis=-1,
        ),
        # The critic gets the whole game in the tunnel frame.
        scene=jnp.concatenate(
            [
                p,
                d.v @ yaw(state.heading),
                (yaw(state.heading).T @ d.R).reshape(-1, 9),
                d.omega,
                s[:, None],
                state.offset,
                jnp.any(seen, axis=1, keepdims=True).astype(p.dtype),
            ],
            axis=-1,
        ).reshape(-1),
    )


def reference(state, params):
    """Where each drone is trying to fly: `look_ahead` along its own line. The cascade
    baseline flies to it."""

    p = to_tunnel(state.drone.x, state.heading, params)
    s, _, line = follow(p, state.offset, params)
    return to_world(
        on_line(s + params.look_ahead, line, state.offset), state.heading, params
    )


def scenery(state, params):
    """For the viewer: the corners of the tube floor in the world, (8, 3), and the
    height of the walls."""

    outline = walls(params)
    floor = jnp.concatenate(
        [outline, jnp.full_like(outline[:, :1], -0.5 * params.width)], axis=-1
    )
    return {
        "walls": to_world(floor, state.heading, params),
        "wall_height": jnp.asarray(params.width),
    }


def compute_reward(progress, distance, alive, died, params):
    cfg = params.reward
    earned = cfg.progress * progress - params.policy_dt * cfg.line * distance
    return jnp.where(alive, earned, 0.0) - cfg.crash * died


def starts(key, params):
    """(n, 3): a start for each drone on the centre line moved up to `start_offset`
    sideways and up or down. With `start_anywhere` it lies anywhere short of the last
    `start_gap` metres of the line, else at the start of the line."""

    n = params.n_drones
    k_side, k_reach = jax.random.split(key)
    side = jax.random.uniform(
        k_side, (n, 2), minval=-params.start_offset, maxval=params.start_offset
    )
    line = lines(side, params)
    room = jax.vmap(_knots)(line)[:, -1] - params.start_gap
    anywhere = jax.random.uniform(k_reach, (n,)) * room
    return on_line(jnp.where(params.start_anywhere, anywhere, 0.0), line, side)


def reset(key, params):
    n = params.n_drones
    k_heading, k_offset, k_start, k_spawn = jax.random.split(key, 4)
    heading = jax.random.uniform(k_heading, (), maxval=2.0 * jnp.pi)
    offset = jax.random.uniform(
        k_offset, (n, 2), minval=-params.line_offset, maxval=params.line_offset
    )

    # Keep the first of 16 draws in which no two drones start closer than start_apart.
    tries = jax.vmap(lambda k: starts(k, params))(jax.random.split(k_start, 16))
    apart = jax.vmap(lambda p: jnp.min(core.pair_dist(p)))(tries) >= params.start_apart
    p = tries[jnp.argmax(apart)]
    drone = core.spawn(k_spawn, to_world(p, heading, params), params)

    state = EnvState(
        drone=drone,
        rate_pid=control.pid_init((n, 3)),
        heading=heading,
        offset=offset,
        progress=follow(p, offset, params)[0],
        alive=jnp.ones(n, bool),
        time=jnp.int32(0),
        closest=jnp.min(core.pair_dist(drone.x)),
        unseen=jnp.zeros(n, jnp.int32),
        unseen_distance=jnp.zeros(n),
    )
    return get_obs(state, params), state


def step(key, state, action, params):
    del key
    drone, rate_pid, _ = core.fly(state.drone, state.rate_pid, action, params)
    p = to_tunnel(drone.x, state.heading, params)

    poly = walls(params)
    r = core.body_radius(params)
    wall = (
        ~inside(p[:, :2], poly)
        | (wall_gap(p[:, :2], poly) < r)
        | (0.5 * params.width - jnp.abs(p[:, 2]) < r)
        | ~jnp.all(jnp.isfinite(p), axis=-1)
    )
    contact = jnp.any(core.hits(drone, params), axis=-1)
    died = state.alive & (wall | contact)
    alive = state.alive & ~died

    s, length, line = follow(p, state.offset, params)
    distance = core.norm(p - on_line(s, line, state.offset))
    unseen = ~jnp.any(sees(p, drone.x, params), axis=1)

    new_state = EnvState(
        drone=drone,
        rate_pid=rate_pid,
        heading=state.heading,
        offset=state.offset,
        progress=s,
        alive=alive,
        time=state.time + 1,
        closest=jnp.minimum(state.closest, jnp.min(core.pair_dist(drone.x))),
        unseen=state.unseen + unseen,
        unseen_distance=state.unseen_distance + unseen * distance,
    )

    reward = compute_reward(s - state.progress, distance, state.alive, died, params)
    finished = jnp.any(alive & (s >= length - 1e-3))
    crashed = ~jnp.all(alive)
    limit = new_state.time >= params.max_steps
    # Only the time limit bootstraps. A finish or a crash ends the episode for both.
    truncated = limit & ~finished & ~crashed
    done = limit | finished | crashed

    info = {
        "alive": alive,
        "died_this_step": died,
        "truncated": truncated,
        "end": {
            "success": finished & ~crashed,
            "contact": jnp.any(died & contact),
            "wall": jnp.any(died & wall),
            "timeout": truncated,
            "closest": new_state.closest,
            "line_distance_unseen": jnp.where(
                new_state.unseen > 0,
                new_state.unseen_distance / jnp.maximum(new_state.unseen, 1),
                jnp.nan,
            ),
        },
        "step": {
            "line_distance": distance,
            "speed": (s - state.progress) / params.policy_dt,
        },
    }
    return get_obs(new_state, params), new_state, reward, done, info


def eval_params(params):
    """Eval flies the whole tunnel: each drone starts at the start of its line."""

    return params.replace(start_anywhere=False)


PRESETS = {
    "d_inf": EnvParams(),
    "d6": EnvParams(sight=6.0),
    "d5": EnvParams(sight=5.0),
    "d4": EnvParams(sight=4.0),
    "d3": EnvParams(sight=3.0),
}
