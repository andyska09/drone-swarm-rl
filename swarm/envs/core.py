"""The pieces every task shares: params, the action, the plant loop, the obs blocks."""

import dataclasses

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np

from swarm.envs import role_slices
from swarm.sim import control, dynamics, guidance


@dataclasses.dataclass(frozen=True)
class Common:
    """Every task's EnvParams inherits this and adds only the fields it owns."""

    model: dynamics.Params = flax.struct.field(default_factory=dynamics.default_params)

    roles: tuple = flax.struct.field(pytree_node=False, default=("drone",))
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

    # Two drones collide when their propeller tips touch: arm + prop + arm + prop.
    collision_dist: float = 0.8

    @property
    def n_drones(self):
        return len(self.roles)

    @property
    def n_visible(self):
        """Slots in the `others` block, capped at n_neighbors."""

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


def per_row(tree, n):
    """The tree repeated for n drones."""

    return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), tree)


def cascade(drones, points, pids, params):
    """The action of the full cascade that flies each drone to its point."""

    gains = control.cascade_gains(params.model)

    def one(drone, point, pid):
        (throttle, rate_ref), pid = control.cascade_outer(
            drone, point, 0.0, pid, gains, params.model, params.policy_dt
        )
        return command_to_action(throttle, rate_ref, params), pid

    return jax.vmap(one)(drones, points, pids)


def cascade_pilot(params, rows, goal):
    """A pilot: the cascade flies each drone of `rows` to its point, `goal(state)`."""

    def act(state, obs, pids):
        del obs
        return cascade(jax.tree.map(lambda a: a[rows], state.drone), goal(state), pids, params)

    return act, lambda state: (state, per_row(control.cascade_init(), len(rows)))


def accelerate(drones, a, pids, params):
    """The action that gives each drone its world acceleration `a`, through the
    cascade's acceleration and attitude levels."""

    model, dt = params.model, params.policy_dt
    gains = control.attitude_gains(max_rate_yaw=params.max_rate_yaw)

    def one(drone, a, pid):
        heading = jnp.arctan2(drone.R[1, 0], drone.R[0, 0])
        orientation, throttle = control.acceleration_controller(
            drone, guidance.drone_limit(a, model), heading, model
        )
        # A negative thrust gives a NaN throttle, so use 0.
        throttle = jnp.where(jnp.isnan(throttle), 0.0, throttle)
        rate, pid = control.attitude_controller(drone, orientation, pid, gains, dt)
        return command_to_action(throttle, rate, params), pid

    return jax.vmap(one)(drones, a, pids)


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


def box_dist(offset, R, half):
    nose, left = R[:, :2, 0], R[:, :2, 1]
    level = jnp.where(
        norm(nose)[:, None] > 1e-3, nose, jnp.stack([left[:, 1], -left[:, 0]], -1)
    )
    d = level / norm(level)[:, None]
    side = jnp.stack([-d[:, 1], d[:, 0]], -1)
    u = jnp.stack([d, -d, side, -side], axis=1)
    return jnp.min((half[:2] - offset[:, None, :2] * jnp.sign(u)) / jnp.abs(u), axis=-1)


def wall_dist(drone, params):
    """Distance to the arena wall along four axis. Parallel to the world floor."""

    return box_dist(drone.x - jnp.asarray(params.center), drone.R, half(params))


def own_obs(drone, walls):
    # Height is a scalar, so it carries no frame and keeps the body-frame symmetry.
    return jnp.concatenate(
        [
            to_body(drone.R, drone.v),
            drone.R.reshape(drone.x.shape[0], 9),
            drone.omega,
            drone.x[:, 2:3],
            walls,
        ],
        axis=-1,
    )


def scene_obs(drone):
    """The exact state (one vector) of every drone in world frame."""

    n = drone.x.shape[0]
    return jnp.concatenate(
        [drone.x, drone.v, drone.R.reshape(n, 9), drone.omega], axis=-1
    ).reshape(-1)


def target_obs(drone, x, v=0.0):
    return jnp.concatenate(
        [to_body(drone.R, x - drone.x), to_body(drone.R, v - drone.v)], axis=-1
    )


def pair_dist(x):
    """Drone-to-drone distances, diagonal at infinity so nobody is their own neighbour."""

    d = norm(x[None, :, :] - x[:, None, :])
    return jnp.where(jnp.eye(x.shape[0], dtype=bool), jnp.inf, d)


def same_role(roles):
    """(N, N, 1): 1 where drone j shares the role of drone i."""

    r = np.array(roles)
    same = (r[:, None] == r[None, :]) & ~np.eye(len(roles), dtype=bool)
    return same.astype(float)[..., None]


def others(drone, params, seen):
    """(N, 8K): the K nearest drones that each drone sees, nearest first.

    Each slot holds the offset and the velocity in the body frame, then 1 if that drone
    shares the role. Then come K 0/1 flags for the filled slots. An empty slot is all
    zeros. `seen` is (N, N), or True when every drone sees every other.
    """

    near, idx = jax.lax.top_k(-jnp.where(seen, pair_dist(drone.x), jnp.inf), params.n_visible)
    take = lambda a: jnp.take_along_axis(a, idx[..., None], axis=1)
    rel = lambda a: jnp.einsum("nji,nkj->nki", drone.R, take(a))
    slots = jnp.concatenate(
        [
            rel(drone.x[None, :, :] - drone.x[:, None, :]),
            rel(drone.v[None, :, :] - drone.v[:, None, :]),
            take(same_role(params.roles)),
        ],
        axis=-1,
    )
    filled = jnp.isfinite(near).astype(slots.dtype)
    flat = (slots * filled[..., None]).reshape(slots.shape[0], -1)
    return jnp.concatenate([flat, filled], axis=-1)


def by_role(x, roles):
    """Split x, one row per drone, into one matrix per role, in drone order."""

    return {role: x[drones] for role, drones in role_slices(roles).items()}


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


def yaw(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _attitude(key, heading, max_tilt):
    k_axis, k_angle = jax.random.split(key)
    axis = jax.random.normal(k_axis, (3,))
    K = dynamics.skew(axis / norm(axis))
    a = jax.random.uniform(k_angle, (), minval=0.0, maxval=max_tilt)
    tilt = jnp.eye(3) + jnp.sin(a) * K + (1.0 - jnp.cos(a)) * (K @ K)
    return yaw(heading) @ tilt


def spawn(key, x, params):
    k_vel, k_att, k_yaw = jax.random.split(key, 3)
    n = x.shape[0]
    hover_rpm = jnp.sqrt(
        params.model.mass * params.model.g / (params.model.n_motors * params.model.kf)
    )
    heading = jax.random.uniform(k_yaw, (n,), maxval=2.0 * jnp.pi)
    return dynamics.State(
        x=x,
        v=jax.random.uniform(
            k_vel, (n, 3), minval=-params.start_vel_range, maxval=params.start_vel_range
        ),
        R=jax.vmap(_attitude, in_axes=(0, 0, None))(
            jax.random.split(k_att, n),
            heading if params.random_yaw else jnp.zeros(n),
            params.start_tilt,
        ),
        omega=jnp.zeros((n, 3)),
        rpm=jnp.full((n, 4), hover_rpm),
    )
