"""Bottom two rungs of the MRS cascade: the PID, the rate controller, the mixer.
Reimplements controllers/{pid,rate_controller,mixer}.hpp."""

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
    saturation: float
    antiwindup: float


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


def rate_gains(params, kp=4.0, kd=0.04, ki=0.0):
    # saturation < 0 disables the clamp, so the mixer is what bounds the output.
    j = jnp.diag(params.J)
    return Gains(kp=kp * j, kd=kd * j, ki=ki * j, saturation=-1.0, antiwindup=1.0)


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
