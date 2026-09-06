"""The MRS controller cascade: position -> velocity -> acceleration -> attitude ->
rate -> mixer. Reimplements controllers/*.hpp. Only the heading (not heading-rate)
branch is here; that is the one the position command flows through."""

import flax.struct
import jax.numpy as jnp


@flax.struct.dataclass
class PIDState:
    integral: jnp.ndarray
    last_error: jnp.ndarray


@flax.struct.dataclass
class Gains:
    kp: jnp.ndarray
    kd: jnp.ndarray
    ki: jnp.ndarray
    saturation: jnp.ndarray
    antiwindup: jnp.ndarray


def pid_init(n=3):
    return PIDState(integral=jnp.zeros(n), last_error=jnp.zeros(n))


def pid_update(pid, error, gains, dt):
    difference = (error - pid.last_error) / dt
    total = gains.kp * error + gains.kd * difference + gains.ki * pid.integral

    clamped = jnp.clip(total, -gains.saturation, gains.saturation)
    out = jnp.where(gains.saturation > 0, clamped, total)

    winding = (gains.antiwindup > 0) & (jnp.abs(out) < gains.antiwindup)
    integral = jnp.where(winding, pid.integral + error * dt, pid.integral)

    return out, PIDState(integral=integral, last_error=error)


def _gains(kp, kd, ki, saturation, antiwindup):
    return Gains(kp=jnp.full(3, kp), kd=jnp.full(3, kd), ki=jnp.full(3, ki),
                 saturation=jnp.asarray(saturation) * jnp.ones(3),
                 antiwindup=jnp.full(3, antiwindup))


def position_gains(kp=2.0, kd=0.15, ki=0.2, max_velocity=6.0):
    return _gains(kp, kd, ki, max_velocity, 1.0)


def position_controller(state, position_ref, pid, gains, dt):
    return pid_update(pid, position_ref - state.x, gains, dt)


def velocity_gains(kp=2.0, kd=0.05, ki=0.01, max_acceleration=4.0):
    return _gains(kp, kd, ki, max_acceleration, 1.0)


def velocity_controller(state, velocity_ref, pid, gains, dt):
    return pid_update(pid, velocity_ref - state.v, gains, dt)


def acceleration_controller(state, acceleration_ref, heading, params):
    """Desired force to (orientation, throttle). No PID — pure geometry.

    The throttle is derived from the force projected on the *current* body z, and
    MRS takes its square root unguarded: an inverted drone gives NaN."""

    fd = (acceleration_ref + jnp.array([0.0, 0.0, params.g])) * params.mass
    body_z = fd / jnp.linalg.norm(fd)

    # Oblique projection of the desired heading onto the plane normal to body z.
    complement = jnp.eye(3) - jnp.outer(body_z, body_z)
    square = complement[:2, :2]
    projector = complement[:, :2] @ (jnp.linalg.inv(square.T @ square) @ square.T) @ jnp.eye(3)[:2]

    body_x = projector @ jnp.array([jnp.cos(heading), jnp.sin(heading), 0.0])
    body_x = body_x / jnp.linalg.norm(body_x)
    body_y = jnp.cross(body_z, body_x)
    body_y = body_y / jnp.linalg.norm(body_y)

    thrust = fd @ state.R[:, 2]
    throttle = (jnp.sqrt(thrust / (params.kf * params.n_motors)) - params.min_rpm) / (
        params.max_rpm - params.min_rpm
    )

    return jnp.column_stack([body_x, body_y, body_z]), throttle


def attitude_gains(kp=6.0, kd=0.05, ki=0.01, max_rate_roll_pitch=10.0, max_rate_yaw=1.0):
    return _gains(kp, kd, ki, [max_rate_roll_pitch, max_rate_roll_pitch, max_rate_yaw], 0.1)


def attitude_controller(state, orientation_ref, pid, gains, dt):
    e = 0.5 * (orientation_ref.T @ state.R - state.R.T @ orientation_ref)
    return pid_update(pid, jnp.array([e[1, 2], e[2, 0], e[0, 1]]), gains, dt)


def rate_gains(params, kp=4.0, kd=0.04, ki=0.0):
    # saturation < 0 disables the clamp, so the mixer is what bounds the output.
    j = jnp.diag(params.J)
    return Gains(kp=kp * j, kd=kd * j, ki=ki * j,
                 saturation=jnp.full(3, -1.0), antiwindup=jnp.full(3, 1.0))


def rate_controller(state, rate_ref, throttle, pid, gains, dt):
    torque, pid = pid_update(pid, rate_ref - state.omega, gains, dt)
    return jnp.append(torque, throttle), pid


def mixer_allocation(params):
    a = params.allocation_matrix
    inv = a.T @ jnp.linalg.inv(a @ a.T)

    roll_pitch = inv[:, :2] / jnp.linalg.norm(inv[:, :2], axis=1, keepdims=True)
    yaw = jnp.sign(inv[:, 2]) * (jnp.abs(inv[:, 2]) > 1e-2)

    return jnp.column_stack([roll_pitch, yaw, jnp.ones(params.n_motors)])


def mixer(control_group, allocation_inv):
    motors = allocation_inv @ control_group
    motors = motors + jnp.maximum(-motors.min(), 0.0)

    throttle, mean, top = control_group[3], motors.mean(), motors.max()

    # The rescale rebuilds from the control group, so it drops the shift above.
    scale = throttle / jnp.where(mean == 0.0, 1.0, mean)
    rescaled = allocation_inv @ (control_group * jnp.array([scale, scale, scale, 1.0]))
    divided = motors / jnp.where(top > 0.0, top, 1.0)

    return jnp.where(top > 1.0, jnp.where(throttle > 1e-2, rescaled, divided), motors)


def cascade_gains(params):
    return (position_gains(), velocity_gains(), attitude_gains(),
            rate_gains(params), mixer_allocation(params))


def cascade_init():
    return tuple(pid_init() for _ in range(4))


def cascade_step(state, position_ref, heading, pids, gains, params, dt):
    """Position reference to four motor throttles, through all six rungs."""

    pos, vel, att, rate, allocation_inv = gains
    pos_pid, vel_pid, att_pid, rate_pid = pids

    velocity_ref, pos_pid = position_controller(state, position_ref, pos_pid, pos, dt)
    accel_ref, vel_pid = velocity_controller(state, velocity_ref, vel_pid, vel, dt)
    orientation, throttle = acceleration_controller(state, accel_ref, heading, params)
    rate_ref, att_pid = attitude_controller(state, orientation, att_pid, att, dt)
    group, rate_pid = rate_controller(state, rate_ref, throttle, rate_pid, rate, dt)

    return mixer(group, allocation_inv), (pos_pid, vel_pid, att_pid, rate_pid)
