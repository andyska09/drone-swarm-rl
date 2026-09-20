"""The pieces every task shares: params, the action, the plant loop, the obs blocks."""

import dataclasses

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np

from swarm.sim import control, dynamics

OTHER_FEATURES = 7  # body-frame offset and velocity of one drone, plus a role flag


@dataclasses.dataclass(frozen=True)
class Common:
    """Every task's EnvParams inherits this and adds only the fields it owns."""

    model: dynamics.Params = flax.struct.field(default_factory=dynamics.default_params)

    roles: tuple = flax.struct.field(pytree_node=False, default=("drone",))
    # A scripted role is flown inside step() and never enters the PPO loss.
    scripted: tuple = flax.struct.field(pytree_node=False, default=())
    n_neighbors: int = flax.struct.field(pytree_node=False, default=6)

    sim_dt: float = 0.005
    steps_per_action: int = flax.struct.field(pytree_node=False, default=2)
    max_steps: int = flax.struct.field(pytree_node=False, default=500)

    max_rate_rp: float = 4.0
    max_rate_yaw: float = 2.0

    center: tuple = flax.struct.field(pytree_node=False, default=(0.0, 0.0, 8.0))
    arena: tuple = flax.struct.field(pytree_node=False, default=(32.0, 32.0, 16.0))
    margin: float = 2.0

    start_vel_range: float = 0.5
    start_tilt: float = 0.2
    random_yaw: bool = flax.struct.field(pytree_node=False, default=True)

    # MRS kills on arm+prop+arm+prop; close_dist is where a soft cost can start.
    collision_dist: float = 0.8
    close_dist: float = 1.5

    @property
    def n_drones(self):
        return len(self.roles)

    @property
    def n_visible(self):
        """Seats in the `others` block, capped at n_neighbors."""

        return min(self.n_neighbors, len(self.roles) - 1)

    @property
    def policy_dt(self):
        return self.sim_dt * self.steps_per_action


def norm(v):
    return jnp.sqrt(jnp.sum(v**2, axis=-1) + 1e-12)


def to_body(R, v):
    return jnp.einsum("nji,nj->ni", R, v)


def half(params):
    return 0.5 * jnp.asarray(params.arena)


def rate_scale(params):
    return jnp.array([params.max_rate_rp, params.max_rate_rp, params.max_rate_yaw])


def action_to_command(action, params):
    """CTBR action in [-1,1] to a throttle in [0,1] and body rates in rad/s."""

    return 0.5 * (action[..., 0] + 1.0), action[..., 1:] * rate_scale(params)


def command_to_action(throttle, rate_ref, params):
    action = jnp.concatenate(
        [jnp.atleast_1d(2.0 * throttle - 1.0), rate_ref / rate_scale(params)]
    )
    return jnp.clip(action, -1.0, 1.0)


def fly(drone, pid, action, params):
    """One policy step of the rate loop and the plant. A reward can charge for
    the commanded rates, so they come back too."""

    throttle, rate_ref = action_to_command(jnp.clip(action, -1.0, 1.0), params)
    gains = control.cascade_gains(params.model)
    rate_gains, allocation, dt = gains[3], gains[4], params.sim_dt

    def substep(carry, _):
        drone, pid = carry

        def one(d, r, t, p):
            group, p = control.rate_controller(
                d, r, t, p, rate_gains, dt, d_on_measurement=True
            )
            return (
                dynamics.step(d, control.mixer(group, allocation), params.model, dt),
                p,
            )

        return jax.vmap(one)(drone, rate_ref, throttle, pid), None

    (drone, pid), _ = jax.lax.scan(
        substep, (drone, pid), None, length=params.steps_per_action
    )
    return drone, pid, rate_ref


def wall_dist(drone, params):
    """Distance to the arena wall along four axis. Parallel to the world floor."""

    nose, left = drone.R[:, :2, 0], drone.R[:, :2, 1]
    level = jnp.where(
        norm(nose)[:, None] > 1e-3, nose, jnp.stack([left[:, 1], -left[:, 0]], -1)
    )
    d = level / norm(level)[:, None]
    side = jnp.stack([-d[:, 1], d[:, 0]], -1)
    u = jnp.stack([d, -d, side, -side], axis=1)
    r = drone.x[:, None, :2] - jnp.asarray(params.center)[:2]
    return jnp.min((half(params)[:2] - r * jnp.sign(u)) / jnp.abs(u), axis=-1)


def own_obs(drone, params):
    # Height is a scalar, so it carries no frame and keeps the body-frame symmetry.
    return jnp.concatenate(
        [
            to_body(drone.R, drone.v),
            drone.R.reshape(drone.x.shape[0], 9),
            drone.omega,
            drone.x[:, 2:3],
            wall_dist(drone, params),
        ],
        axis=-1,
    )


def target_obs(drone, x, v=0.0):
    return jnp.concatenate(
        [to_body(drone.R, x - drone.x), to_body(drone.R, v - drone.v)], axis=-1
    )


def pair_dist(x):
    """Drone-to-drone distances, diagonal at infinity so nobody is their own neighbour."""

    d = norm(x[None, :, :] - x[:, None, :])
    return jnp.where(jnp.eye(x.shape[0], dtype=bool), jnp.inf, d)


def _same_role(roles):
    r = np.array(roles)
    return ((r[:, None] == r[None, :]) & ~np.eye(len(roles), dtype=bool)).astype(float)


def others(drone, params):
    """The K nearest drones, body frame, and a 0/1 mask for the empty seats.

    Every role goes in one block. The last feature says whether that drone shares
    my role, so a teammate and an opponent are told apart by a number, not by
    sitting in different blocks. Rows are sorted by distance.
    """

    near, idx = jax.lax.top_k(-pair_dist(drone.x), params.n_visible)
    take = lambda a: jnp.take_along_axis(a, idx[..., None], axis=1)
    rel = lambda a: jnp.einsum("nji,nkj->nki", drone.R, take(a))
    feat = jnp.concatenate(
        [
            rel(drone.x[None, :, :] - drone.x[:, None, :]),
            rel(drone.v[None, :, :] - drone.v[:, None, :]),
            jnp.take_along_axis(_same_role(params.roles), idx, axis=1)[..., None],
        ],
        axis=-1,
    )
    return feat, jnp.isfinite(near).astype(feat.dtype)


def body_radius(params):
    """The ball that wraps one drone: centre to the tip of a propeller."""

    return params.model.arm_length + params.model.prop_radius


def hits(drone, params):
    return pair_dist(drone.x) < params.collision_dist


def out_of_arena(drone, params):
    off = jnp.abs(drone.x - jnp.asarray(params.center))
    return jnp.any(off > half(params), axis=-1) | ~jnp.all(
        jnp.isfinite(drone.x), axis=-1
    )


def is_dead(drone, params):
    """The default rule. A task with different terminations writes its own."""

    return (
        out_of_arena(drone, params)
        | (drone.x[:, 2] < 0.0)
        | (drone.R[:, 2, 2] < 0.0)
        | jnp.any(hits(drone, params), axis=-1)
    )


def _attitude(key, yaw, max_tilt):
    k_axis, k_angle = jax.random.split(key)
    axis = jax.random.normal(k_axis, (3,))
    K = dynamics.skew(axis / norm(axis))
    a = jax.random.uniform(k_angle, (), minval=0.0, maxval=max_tilt)
    tilt = jnp.eye(3) + jnp.sin(a) * K + (1.0 - jnp.cos(a)) * (K @ K)

    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ tilt


def spawn(key, x, params):
    k_vel, k_att, k_yaw = jax.random.split(key, 3)
    n = x.shape[0]
    hover_rpm = jnp.sqrt(
        params.model.mass * params.model.g / (params.model.n_motors * params.model.kf)
    )
    yaw = jax.random.uniform(k_yaw, (n,), maxval=2.0 * jnp.pi)
    return dynamics.State(
        x=x,
        v=jax.random.uniform(
            k_vel, (n, 3), minval=-params.start_vel_range, maxval=params.start_vel_range
        ),
        R=jax.vmap(_attitude, in_axes=(0, 0, None))(
            jax.random.split(k_att, n),
            yaw if params.random_yaw else jnp.zeros(n),
            params.start_tilt,
        ),
        omega=jnp.zeros((n, 3)),
        rpm=jnp.full((n, 4), hover_rpm),
    )
