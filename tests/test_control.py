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

COLS = {
    "x": slice(5, 8),
    "v": slice(8, 11),
    "R": slice(11, 20),
    "omega": slice(20, 23),
    "rpm": slice(23, 27),
}


def rate_rollout(state, throttle, rate_ref, params, steps):
    gains, inv = c.rate_gains(params), c.mixer_allocation(params)

    def body(carry, _):
        st, pid = carry
        group, pid = c.rate_controller(st, rate_ref, throttle, pid, gains, DT)
        st = d.step(st, c.mixer(group, inv), params, DT)
        return (st, pid), st

    return jax.lax.scan(body, (state, c.pid_init()), None, length=steps)[1]


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

    for field, cols in COLS.items():
        want = rows[1:, cols]
        got = np.asarray(getattr(traj, field)).reshape(len(want), -1)
        np.testing.assert_allclose(got, want, atol=TOL, rtol=0, err_msg=field)


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
