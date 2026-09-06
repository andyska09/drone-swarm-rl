"""M1 gate. The plant must reproduce the C++ reference trajectories, and the
analytic cases must hold. Regenerate the references with tools/mrs_golden/build.sh."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarm import dynamics as d

GOLDEN = ROOT / "tests" / "golden"
SCENARIOS = ("free_fall", "hover", "tumble", "spin_down", "random")
DT = 0.01
TOL = 1e-9  # measured worst case is 1.1e-13 over 500 steps

CMD = slice(1, 5)
COLS = {
    "x": slice(5, 8),
    "v": slice(8, 11),
    "R": slice(11, 20),
    "omega": slice(20, 23),
    "rpm": slice(23, 27),
}


def load(name):
    return np.genfromtxt(GOLDEN / f"{name}.csv", delimiter=",", skip_header=1)


def state_at(rows, i):
    f = {k: jnp.asarray(rows[i, s]) for k, s in COLS.items()}
    return d.State(**{**f, "R": f["R"].reshape(3, 3)})


def rollout(state, cmds, params, dt=DT):
    def body(s, u):
        s = d.step(s, u, params, dt)
        return s, s

    return jax.lax.scan(body, state, cmds)[1]


def hover_rpm(params):
    return jnp.sqrt(params.mass * params.g / (params.n_motors * params.kf))


# | ---------------------------- goldens ---------------------------- |


def test_params_match_reference():
    p = d.default_params()
    ref = dict(line.split() for line in (GOLDEN / "params.txt").read_text().splitlines())

    assert p.n_motors == int(ref["n_motors"])
    for name in ("mass", "g", "kf", "km", "prop_radius", "arm_length", "body_height",
                 "motor_time_constant", "min_rpm", "max_rpm", "air_resistance_coeff"):
        assert float(getattr(p, name)) == float(ref[name]), name

    for i in range(3):
        assert float(p.J[i, i]) == float(ref[f"J{i}{i}"])
    assert np.array_equal(p.J, np.diag(np.diag(p.J)))

    for i in range(4):
        for j in range(4):
            assert float(p.allocation_matrix[i, j]) == float(ref[f"A{i}{j}"]), f"A{i}{j}"


@pytest.mark.parametrize("name", SCENARIOS)
def test_matches_golden_trajectory(name):
    rows = load(name)
    traj = rollout(state_at(rows, 0), jnp.asarray(rows[:-1, CMD]), d.default_params())

    for field, cols in COLS.items():
        want = rows[1:, cols]
        got = np.asarray(getattr(traj, field)).reshape(len(want), -1)
        np.testing.assert_allclose(got, want, atol=TOL, rtol=0, err_msg=f"{name}.{field}")


# | --------------------------- analytic ---------------------------- |


def test_free_fall_matches_half_g_t_squared():
    # Both drag and the min_rpm floor have to go, or the drone does not fall freely.
    p = d.default_params(air_resistance_coeff=0.0, min_rpm=0.0)
    traj = rollout(d.rest_state(0.0), jnp.zeros((100, 4)), p)

    assert float(traj.x[-1, 2]) == pytest.approx(-0.5 * 9.81, abs=1e-12)
    assert float(traj.v[-1, 2]) == pytest.approx(-9.81, abs=1e-12)


def test_hover_rpm_holds_position():
    p = d.default_params()
    u = (hover_rpm(p) - p.min_rpm) / (p.max_rpm - p.min_rpm)
    traj = rollout(d.rest_state(hover_rpm(p)), jnp.full((100, 4), u), p)

    assert float(jnp.abs(traj.x).max()) < 1e-14
    assert float(jnp.abs(traj.v).max()) < 1e-13


def test_equal_rpm_makes_no_torque():
    p = d.default_params()
    for rpm in (p.min_rpm, 3000.0, p.max_rpm):
        torque = p.allocation_matrix @ jnp.full(4, rpm) ** 2
        assert float(jnp.abs(torque[:3]).max()) < 1e-15
        assert float(torque[3]) > 0.0


def test_split_throttle_is_pure_pitch():
    # Motors 1 and 3 up, 0 and 2 down: pitch only, and only while R is still level.
    p = d.default_params()
    rpm = hover_rpm(p) + jnp.array([-500.0, 500.0, -500.0, 500.0])
    dot = d.derivative(d.rest_state().replace(rpm=rpm), p)

    assert float(dot.omega[1]) > 1.0
    assert abs(float(dot.omega[0])) < 1e-12
    assert abs(float(dot.omega[2])) < 1e-12


def test_motor_lag_reaches_63_percent_after_one_time_constant():
    p = d.default_params()
    steps = int(round(p.motor_time_constant / DT))
    traj = rollout(d.rest_state(0.0), jnp.ones((steps, 4)), p)

    gap = p.max_rpm - 0.0
    assert float(traj.rpm[-1, 0]) == pytest.approx((1 - np.exp(-1)) * gap, rel=1e-12)


def test_rotation_stays_orthonormal_over_1e5_steps():
    p = d.default_params()
    cmds = jnp.tile(jnp.array([0.55, 0.40, 0.52, 0.44]), (100_000, 1))
    traj = rollout(d.rest_state(0.0), cmds, p)

    R = traj.R[-1]
    assert float(jnp.abs(R.T @ R - jnp.eye(3)).max()) < 1e-6
    assert float(jnp.abs(traj.omega[-1]).max()) > 1.0, "the drone never started spinning"


# | ------------------------- infrastructure ------------------------- |


def test_step_preserves_dtypes_and_shapes():
    p = d.default_params()
    before = d.rest_state(3000.0)
    after = d.step(before, jnp.full(4, 0.5), p, DT)

    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
        assert a.dtype == b.dtype and a.shape == b.shape


def test_vmap_matches_python_loop():
    p = d.default_params()
    rng = np.random.default_rng(0)
    n = 8

    states = [
        d.rest_state(0.0).replace(
            x=jnp.asarray(rng.normal(size=3)),
            v=jnp.asarray(rng.normal(size=3)),
            R=jnp.asarray(np.linalg.qr(rng.normal(size=(3, 3)))[0]),
            omega=jnp.asarray(rng.normal(size=3)),
            rpm=jnp.asarray(rng.uniform(1170, 7800, size=4)),
        )
        for _ in range(n)
    ]
    cmds = jnp.asarray(rng.uniform(size=(n, 4)))

    stacked = jax.tree.map(lambda *a: jnp.stack(a), *states)
    batched = jax.vmap(d.step, in_axes=(0, 0, None, None))(stacked, cmds, p, DT)
    looped = jax.tree.map(
        lambda *a: jnp.stack(a), *[d.step(s, u, p, DT) for s, u in zip(states, cmds)]
    )

    for a, b in zip(jax.tree.leaves(batched), jax.tree.leaves(looped)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12, rtol=0)


def test_float32_tracks_float64():
    """Runs a rollout in a child process, where x64 is off and everything is float32."""

    out = subprocess.run(
        [sys.executable, __file__], cwd=ROOT, capture_output=True, text=True, check=True
    )
    got = np.array([float(t) for t in out.stdout.split()])

    rows = load("spin_down")
    traj = rollout(state_at(rows, 0), jnp.asarray(rows[:-1, CMD]), d.default_params())
    want = np.concatenate([traj.x[-1], traj.v[-1], traj.omega[-1]])

    np.testing.assert_allclose(got, want, atol=1e-3, rtol=0)


def _float32_final_state():
    assert jnp.zeros(1).dtype == jnp.float32, "child process must not have x64 enabled"

    p = d.default_params()
    rows = load("spin_down").astype(np.float32)
    traj = rollout(state_at(rows, 0), jnp.asarray(rows[:-1, CMD]), p)
    return np.concatenate([traj.x[-1], traj.v[-1], traj.omega[-1]])


if __name__ == "__main__":
    print(" ".join(repr(float(a)) for a in _float32_final_state()))
