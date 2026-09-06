# Golden trajectory generator

Runs the MRS C++ model and dumps reference trajectories to `tests/golden/`.
The JAX simulator is tested by replaying the same inputs and matching the rows.

```bash
./build.sh          # patch, compile, regenerate tests/golden/
```

Needs Eigen and Boost (`brew install eigen boost`; override with `EIGEN_INC` /
`BOOST_INC`) and the MRS clone in `research/code_sources/`. That directory is
gitignored, so on a fresh clone the CSVs are the only copy of the reference —
you only need to rebuild if a scenario changes.

`transpose.diff` fixes the rotation re-orthonormalization: MRS computes
`R·L⁻¹` where `L` is the Cholesky factor of `RᵀR`, which does not give an
orthonormal matrix and drifts ~3e-3 off SO(3). The correct form is `R·L⁻ᵀ`.
See [choices.md](../../research/notes/choices.md).

## CSV format

One row per step, `%.17g`. Row `k` holds the state after `k` steps and the
throttle `u` applied to reach row `k+1`; the last row repeats the previous `u`,
which is unused.

```
step, u0..u3, x0..x2, v0..v2, R00..R22 (row-major), w0..w2, rpm0..rpm3
```

`params.txt` is the model parameters the trajectories were generated with. The
test asserts the Python defaults match it, so the two cannot drift apart.

## Scenarios

| file | steps | what it covers |
|------|-------|----------------|
| `free_fall` | 100 | gravity, quadratic drag, the `min_rpm` floor |
| `hover` | 100 | the equilibrium |
| `tumble` | 500 | all three torque axes, orthonormalization under rotation |
| `spin_down` | 200 | the gyroscopic term `ω × Jω`, isolated from the allocation matrix |
| `random` | 200 | everything nonzero, command changing every step |
| `rate_step` | 300 | the closed rate loop: PID, mixer, plant |
| `attitude_step` | 300 | attitude rung on top of the rate loop |
| `velocity_step` | 500 | velocity and acceleration rungs |
| `position_step` | 1500 | the whole cascade, flying to (3, -2, 5) |

All use `dt = 0.01`, ground and takeoff patch off, x500 defaults.

The closed-loop scenarios run through `UavSystem`, so their command columns hold
the reference rather than motor throttles: `throttle, rate_x, rate_y, rate_z` for
`rate_step`, the nine entries of `Rd` plus `throttle` for `attitude_step`, and
`vx, vy, vz, heading` / `px, py, pz, heading` for the other two.
`mixer_allocation.txt` is MRS's normalized inverse allocation.