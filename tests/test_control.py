"""M2 gate. The closed rate loop must reproduce the C++ reference, and the mixer
must agree with MRS's normalized allocation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarm import control as c
from swarm import dynamics as d

GOLDEN = ROOT / "tests" / "golden"
DT = 0.01
TOL = 1e-9


def state_cols(n_cmd):
    o = 1 + n_cmd
    return {"x": slice(o, o + 3), "v": slice(o + 3, o + 6), "R": slice(o + 6, o + 15),
            "omega": slice(o + 15, o + 18), "rpm": slice(o + 18, o + 22)}


def assert_matches(traj, rows, cols):
    for field, sl in cols.items():
        want = rows[1:, sl]
        got = np.asarray(getattr(traj, field)).reshape(len(want), -1)
        np.testing.assert_allclose(got, want, atol=TOL, rtol=0, err_msg=field)


def rate_rollout(state, throttle, rate_ref, params, steps):
    gains, inv = c.rate_gains(params), c.mixer_allocation(params)

    def body(carry, _):
        st, pid = carry
        group, pid = c.rate_controller(st, rate_ref, throttle, pid, gains, DT)
        st = d.step(st, c.mixer(group, inv), params, DT)
        return (st, pid), st

    return jax.lax.scan(body, (state, c.pid_init()), None, length=steps)[1]


def attitude_rollout(state, orientation_ref, throttle, params, steps):
    att, rate = c.attitude_gains(), c.rate_gains(params)
    inv = c.mixer_allocation(params)

    def body(carry, _):
        st, att_pid, rate_pid = carry
        rate_ref, att_pid = c.attitude_controller(st, orientation_ref, att_pid, att, DT)
        group, rate_pid = c.rate_controller(st, rate_ref, throttle, rate_pid, rate, DT)
        st = d.step(st, c.mixer(group, inv), params, DT)
        return (st, att_pid, rate_pid), st

    init = (state, c.pid_init(), c.pid_init())
    return jax.lax.scan(body, init, None, length=steps)[1]


def velocity_rollout(state, velocity_ref, heading, params, steps):
    vel, att, rate = c.velocity_gains(), c.attitude_gains(), c.rate_gains(params)
    inv = c.mixer_allocation(params)

    def body(carry, _):
        st, vel_pid, att_pid, rate_pid = carry
        accel_ref, vel_pid = c.velocity_controller(st, velocity_ref, vel_pid, vel, DT)
        orientation, throttle = c.acceleration_controller(st, accel_ref, heading, params)
        rate_ref, att_pid = c.attitude_controller(st, orientation, att_pid, att, DT)
        group, rate_pid = c.rate_controller(st, rate_ref, throttle, rate_pid, rate, DT)
        st = d.step(st, c.mixer(group, inv), params, DT)
        return (st, vel_pid, att_pid, rate_pid), st

    init = (state, c.pid_init(), c.pid_init(), c.pid_init())
    return jax.lax.scan(body, init, None, length=steps)[1]


def position_rollout(state, position_ref, heading, params, steps):
    gains = c.cascade_gains(params)

    def body(carry, _):
        st, pids = carry
        throttles, pids = c.cascade_step(st, position_ref, heading, pids, gains, params, DT)
        st = d.step(st, throttles, params, DT)
        return (st, pids), st

    return jax.lax.scan(body, (state, c.cascade_init()), None, length=steps)[1]


def test_mixer_allocation_matches_reference():
    p = d.default_params()
    ref = dict(line.split() for line in (GOLDEN / "mixer_allocation.txt").read_text().splitlines())
    got = c.mixer_allocation(p)

    for i in range(4):
        for j in range(4):
            assert float(got[i, j]) == pytest.approx(float(ref[f"M{i}{j}"]), abs=1e-15), f"M{i}{j}"


def test_matches_rate_step_golden():
    rows = np.genfromtxt(GOLDEN / "rate_step.csv", delimiter=",", skip_header=1)
    throttle, rate_ref = rows[0, 1], jnp.asarray(rows[0, 2:5])

    traj = rate_rollout(d.rest_state(0.0), throttle, rate_ref, d.default_params(), len(rows) - 1)
    assert_matches(traj, rows, state_cols(4))


def test_matches_attitude_step_golden():
    rows = np.genfromtxt(GOLDEN / "attitude_step.csv", delimiter=",", skip_header=1)
    orientation_ref, throttle = jnp.asarray(rows[0, 1:10]).reshape(3, 3), rows[0, 10]

    traj = attitude_rollout(d.rest_state(0.0), orientation_ref, throttle,
                            d.default_params(), len(rows) - 1)
    assert_matches(traj, rows, state_cols(10))


def test_attitude_converges_to_the_reference():
    rows = np.genfromtxt(GOLDEN / "attitude_step.csv", delimiter=",", skip_header=1)
    orientation_ref, throttle = jnp.asarray(rows[0, 1:10]).reshape(3, 3), rows[0, 10]

    traj = attitude_rollout(d.rest_state(0.0), orientation_ref, throttle, d.default_params(), 300)

    assert float(jnp.abs(traj.R[-1] - orientation_ref).max()) < 1e-3
    assert float(jnp.abs(traj.omega[-1]).max()) < 1e-3


def test_level_throttle_maps_to_equal_motors():
    p = d.default_params()
    inv = c.mixer_allocation(p)
    motors = c.mixer(jnp.array([0.0, 0.0, 0.0, 0.4]), inv)

    np.testing.assert_allclose(np.asarray(motors), 0.4, atol=1e-15)


def test_mixer_preserves_throttle_as_the_mean():
    inv = c.mixer_allocation(d.default_params())
    for group in ([0.2, -0.1, 0.3, 0.5], [0.0, 0.0, 0.0, 0.1]):
        motors = c.mixer(jnp.asarray(group), inv)
        assert float(motors.mean()) == pytest.approx(group[3], abs=1e-15)


def test_mixer_shifts_negative_motors_up():
    inv = c.mixer_allocation(d.default_params())
    motors = c.mixer(jnp.array([0.9, 0.0, 0.0, 0.05]), inv)

    assert float(motors.min()) >= 0.0


def test_pid_kicks_on_the_first_step():
    # last_error starts at zero, so a step reference gives a derivative of error/dt.
    gains = c.Gains(kp=jnp.full(3, 2.0), kd=jnp.full(3, 0.5), ki=jnp.zeros(3),
                    saturation=-1.0, antiwindup=1.0)
    out, pid = c.pid_update(c.pid_init(), jnp.ones(3), gains, DT)

    assert float(out[0]) == pytest.approx(2.0 + 0.5 / DT)
    np.testing.assert_allclose(np.asarray(pid.last_error), 1.0)
    np.testing.assert_allclose(np.asarray(pid.integral), 0.0)

    out, _ = c.pid_update(pid, jnp.ones(3), gains, DT)
    assert float(out[0]) == pytest.approx(2.0)


def test_rate_controller_output_is_not_clamped():
    p = d.default_params()
    gains = c.rate_gains(p)
    ref, pid = jnp.full(3, 500.0), c.pid_init()
    group, _ = c.rate_controller(d.rest_state(0.0), ref, 0.5, pid, gains, DT)

    assert float(jnp.abs(group[:3]).max()) > 1.0
    assert float(group[3]) == 0.5


def test_rate_step_converges_without_steady_state_error():
    p = d.default_params()
    traj = rate_rollout(d.rest_state(0.0), 0.46536755610843011, jnp.array([0.0, 1.0, 0.0]), p, 300)
    omega = np.asarray(traj.omega)

    assert omega[-1, 1] == pytest.approx(1.0, abs=1e-6)
    assert abs(omega[-1, 0]) < 1e-12 and abs(omega[-1, 2]) < 1e-12
    assert omega[:, 1].max() < 1.25, "overshoot above 25%"
    assert np.all(np.abs(omega[150:, 1] - 1.0) < 1e-6), "still moving after 1.5 s"


def test_matches_velocity_step_golden():
    rows = np.genfromtxt(GOLDEN / "velocity_step.csv", delimiter=",", skip_header=1)
    velocity_ref, heading = jnp.asarray(rows[0, 1:4]), rows[0, 4]

    traj = velocity_rollout(d.rest_state(0.0), velocity_ref, heading,
                            d.default_params(), len(rows) - 1)
    assert_matches(traj, rows, state_cols(4))


def test_matches_position_step_golden():
    rows = np.genfromtxt(GOLDEN / "position_step.csv", delimiter=",", skip_header=1)
    position_ref, heading = jnp.asarray(rows[0, 1:4]), rows[0, 4]

    traj = position_rollout(d.rest_state(0.0), position_ref, heading,
                            d.default_params(), len(rows) - 1)
    assert_matches(traj, rows, state_cols(4))


def test_flies_a_to_b_and_settles():
    target = jnp.array([3.0, -2.0, 5.0])
    traj = position_rollout(d.rest_state(0.0), target, 0.5, d.default_params(), 1500)

    assert float(jnp.linalg.norm(traj.x[-1] - target)) < 0.02
    assert float(jnp.linalg.norm(traj.v[-1])) < 0.01
    assert float(jnp.abs(traj.x[-200:] - target).max()) < 0.02, "still drifting"


def test_hovers_from_a_perturbed_start():
    p = d.default_params()
    start = d.rest_state(0.0).replace(
        x=jnp.array([0.8, -0.5, 0.4]),
        v=jnp.array([-1.0, 0.6, 0.5]),
        omega=jnp.array([0.3, -0.2, 0.1]),
    )
    traj = position_rollout(start, jnp.zeros(3), 0.0, p, 1500)

    assert float(jnp.linalg.norm(traj.x[-1])) < 0.02
    assert float(jnp.linalg.norm(traj.v[-1])) < 0.01
