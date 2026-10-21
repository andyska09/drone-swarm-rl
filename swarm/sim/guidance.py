"""Hand-written opponents for eval: FRPN, APF and flown paths. Each one gives a drone
a world acceleration. Where the values come from:
research/notes/task_chase_baselines.md."""

import jax
import jax.numpy as jnp

# A path draws its radius (m) and speed (m/s) from these ranges. Then each axis
# multiplies the radius by its own factor from SCALE. A hover has radius zero.
SIZE = {"hover": (0.0, 0.0), "circle": (4.8, 8.0), "figure8": (3.6, 7.2)}
SCALE = (0.85, 1.0)
SPEED = (2.83, 5.66)
# How far each shape reaches from its centre along x and y, at radius 1.
REACH = {"hover": (1.0, 1.0), "circle": (1.0, 1.0), "figure8": (1.0, 0.5)}
# Gains of `follow`. K_V = 2·sqrt(K_P) brings the drone back to the path without overshoot.
K_P, K_V = 4.0, 4.0


def speed_limit(a, v, top_acc=20.0, top_speed=12.0):
    """Cap the acceleration at top_acc. At top_speed or faster, remove the part that
    points along the velocity, so the drone does not speed up."""

    a = a * jnp.minimum(1.0, top_acc / (jnp.linalg.norm(a) + 1e-9))
    u = v / (jnp.linalg.norm(v) + 1e-9)
    along = a @ u
    return jnp.where((jnp.linalg.norm(v) >= top_speed) & (along > 0.0), a - along * u, a)


def drone_limit(a, model):
    """Cut the acceleration down to what the motors can give. The vertical part gets
    thrust first and the sideways part gets the rest, so the drone never has to flip."""

    thrust = lambda rpm: model.n_motors * model.kf * rpm**2 / model.mass
    up = jnp.clip(a[2] + model.g, thrust(model.min_rpm), thrust(model.max_rpm))
    # At full climb this is 0, and float32 can round it below 0, which makes sqrt NaN.
    room = jnp.sqrt(jnp.maximum(thrust(model.max_rpm) ** 2 - up**2, 0.0))
    side = a[:2] * jnp.minimum(1.0, room / (jnp.linalg.norm(a[:2]) + 1e-9))
    return jnp.append(side, up - model.g)


def frpn(dp, dv, gain=6.677, weight=0.337):
    """Acceleration toward where the target will be in t_go seconds, mixed with a pull
    straight at it (`weight`). `dp` and `dv` are the target minus the pursuer."""

    t_go = jnp.linalg.norm(dp) / (jnp.linalg.norm(dv) + 1e-6)
    return gain * ((1.0 - weight) * (dp + dv * t_go) / (t_go**2 + 1e-6) + weight * dp)


def apf(x, v, x_p, v_p, lo, hi):
    """The evader's acceleration: away from every pursuer, and away from every face of
    the box [lo, hi] that is closer than r_w."""

    # k is the push, c the extra push per m/s of closing speed, e how fast the push
    # fades with distance. _p is for pursuers, _w for walls.
    k_p, k_w, c_p, c_w, r_w, e_p, e_w = 2.135, 57.89, 1.370, 29.71, 9.997, 1.032, 0.5652

    r = x - x_p
    dist = jnp.linalg.norm(r, axis=-1, keepdims=True)
    away = r / (dist + 1e-6)
    closing = jnp.maximum(jnp.sum((v_p - v) * away, axis=-1, keepdims=True), 0.0)
    push = jnp.sum((k_p + c_p * closing) / (dist**e_p + 1e-6) * away, axis=0)

    # Row 0 holds the low faces, which push toward +axis. Row 1 holds the high faces.
    gap = jnp.abs(jnp.stack([x - lo, hi - x]))
    inward = jnp.array([[1.0], [-1.0]])
    toward = jnp.maximum(-inward * v, 0.0)
    base = jnp.maximum(1.0 / (gap + 1e-6) - 1.0 / r_w, 0.0)
    wall = jnp.where(gap < r_w, (k_w + c_w * toward) * base**e_w, 0.0)
    return push + jnp.sum(inward * wall, axis=0)


def _unit(shape, theta):
    """The shape at radius 1 and angle theta. Rows are x and y. Columns are the
    position and its first and second derivative in theta."""

    s, c = jnp.sin(theta), jnp.cos(theta)
    if shape == "figure8":
        s2, c2 = jnp.sin(2.0 * theta), jnp.cos(2.0 * theta)
        return jnp.array([[s, c, -s], [0.5 * s2, c2, -2.0 * s2]])
    return jnp.array([[c, -s, -c], [s, c, -s]])


def draw_path(key, shape, lo, hi):
    """A flat path at a random place where all of it fits in the box [lo, hi], with a
    random size, speed, direction and start point. Returns its centre `c`, radii `r`,
    turn rate `w` in rad/s (the sign is the direction) and start angle `theta`."""

    k_r, k_s, k_v, k_t, k_c, k_d = jax.random.split(key, 6)
    radius = jax.random.uniform(k_r, (), minval=SIZE[shape][0], maxval=SIZE[shape][1])
    size = radius * jax.random.uniform(k_s, (2,), minval=SCALE[0], maxval=SCALE[1])
    speed = jax.random.uniform(k_v, (), minval=SPEED[0], maxval=SPEED[1])
    theta = jax.random.uniform(k_t, (), maxval=2.0 * jnp.pi)

    reach = jnp.append(size * jnp.array(REACH[shape]), 0.0)
    c = jax.random.uniform(k_c, (3,), minval=lo + reach, maxval=hi - reach)
    w = jnp.where(size.mean() > 0.0, speed / jnp.maximum(size.mean(), 1e-9), 0.0)
    w = jnp.where(jax.random.bernoulli(k_d), w, -w)
    return {"c": c, "r": size, "w": w, "theta": theta}


def path_at(shape, path, t):
    """The path's point at time t. The rows are its position, velocity and acceleration."""

    w = path["w"]
    k = path["r"][:, None] * _unit(shape, path["theta"] + w * t) * jnp.array([1.0, w, w**2])
    return jnp.zeros((3, 3)).at[:, :2].set(k.T).at[0].add(path["c"])


def follow(x, v, target):
    """The acceleration that keeps the drone on the path: the path's own acceleration,
    plus a pull toward the path's point and its velocity."""

    p, dp, ddp = target
    return ddp + K_P * (p - x) + K_V * (dp - v)
