"""Quadrotor plant. Reimplements ctu-mrs/mrs_multirotor_simulator's
multirotor_model.hpp in JAX. Deviations are listed in research/notes/choices.md."""

import flax.struct
import jax
import jax.numpy as jnp
import jax.scipy.linalg


@flax.struct.dataclass
class Params:
    mass: float
    g: float
    kf: float
    km: float
    prop_radius: float
    arm_length: float
    body_height: float
    motor_time_constant: float
    min_rpm: float
    max_rpm: float
    air_resistance_coeff: float
    J: jnp.ndarray
    allocation_matrix: jnp.ndarray
    external_force: jnp.ndarray
    external_moment: jnp.ndarray
    n_motors: int = flax.struct.field(pytree_node=False, default=4)


@flax.struct.dataclass
class State:
    x: jnp.ndarray  # world position (3,)
    v: jnp.ndarray  # world velocity (3,)
    R: jnp.ndarray  # body -> world rotation (3, 3)
    omega: jnp.ndarray  # body angular rate (3,)
    rpm: jnp.ndarray  # motor speeds (4,)


def rest_state(rpm=0.0):
    """At the origin, level, motionless, all motors at `rpm`."""

    return State(
        x=jnp.zeros(3),
        v=jnp.zeros(3),
        R=jnp.eye(3),
        omega=jnp.zeros(3),
        rpm=jnp.full(4, rpm),
    )


def default_params(**overrides):
    """x500 quadrotor, matching MultirotorModel::ModelParams."""

    mass = 2.0
    kf = 0.00000027087
    km = 0.07
    prop_radius = 0.15
    arm_length = 0.25
    body_height = 0.1

    J = jnp.diag(
        jnp.array(
            [
                mass * (3.0 * arm_length**2 + body_height**2) / 12.0,
                mass * (3.0 * arm_length**2 + body_height**2) / 12.0,
                mass * arm_length**2 / 2.0,
            ]
        )
    )

    # rows: roll, pitch, yaw, thrust. Maps squared motor RPM to torque and thrust.
    # 0.707 is MRS's literal
    allocation_matrix = jnp.array(
        [
            [-0.707, 0.707, 0.707, -0.707],
            [-0.707, 0.707, -0.707, 0.707],
            [-1.0, -1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    ) * jnp.array(
        [
            [arm_length * kf],
            [arm_length * kf],
            [km * 3.0 * prop_radius * kf],
            [kf],
        ]
    )

    params = Params(
        mass=mass,
        g=9.81,
        kf=kf,
        km=km,
        prop_radius=prop_radius,
        arm_length=arm_length,
        body_height=body_height,
        motor_time_constant=0.03,
        min_rpm=1170.0,
        max_rpm=7800.0,
        air_resistance_coeff=0.30,
        J=J,
        allocation_matrix=allocation_matrix,
        external_force=jnp.zeros(3),
        external_moment=jnp.zeros(3),
    )

    return params.replace(**overrides) if overrides else params


def orthonormalize(R):
    """See choices.md, fixed missing transpose in MRS ref"""

    L = jnp.linalg.cholesky(R.T @ R)
    return jax.scipy.linalg.solve_triangular(L, R.T, lower=True).T


def skew(w):
    return jnp.array(
        [
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ]
    )


def derivative(state, params):
    """Time derivative of the 18 rigid-body states. `rpm` is deliberately not integrated."""

    R = orthonormalize(state.R)

    wrench = params.allocation_matrix @ state.rpm**2
    torque, thrust = wrench[:3], wrench[3]

    # +eps only so the gradient stays finite at v = 0; it shifts |v| by <1e-14.
    speed = jnp.sqrt(jnp.sum(state.v**2) + 1e-12)
    drag = params.air_resistance_coeff * jnp.pi * params.arm_length**2 * speed * state.v

    gravity = jnp.array([0.0, 0.0, -params.g])
    v_dot = gravity + (thrust * R[:, 2] + params.external_force - drag) / params.mass

    omega_dot = jnp.linalg.solve(
        params.J,
        torque
        - jnp.cross(state.omega, params.J @ state.omega)
        + params.external_moment,
    )

    return State(
        x=state.v,
        v=v_dot,
        R=R @ skew(state.omega),
        omega=omega_dot,
        rpm=jnp.zeros_like(state.rpm),
    )


def advance(state, slope, h):
    return jax.tree.map(lambda a, b: a + h * b, state, slope)


def step(state, throttle, params, dt):
    """One RK4 step, then re-orthonormalize, then the motor lag. That order.

    `rpm` survives the integration untouched because `derivative` returns zero for
    it, so the motors hold one speed for the whole step."""

    k1 = derivative(state, params)
    k2 = derivative(advance(state, k1, dt / 2), params)
    k3 = derivative(advance(state, k2, dt / 2), params)
    k4 = derivative(advance(state, k3, dt), params)
    slope = jax.tree.map(lambda a, b, c, e: (a + 2 * b + 2 * c + e) / 6, k1, k2, k3, k4)

    state = advance(state, slope, dt)
    state = state.replace(R=orthonormalize(state.R))

    cmd = params.min_rpm + (params.max_rpm - params.min_rpm) * jnp.clip(throttle, 0.0, 1.0)
    lag = jnp.exp(-dt / params.motor_time_constant)
    return state.replace(rpm=lag * state.rpm + (1.0 - lag) * cmd)
