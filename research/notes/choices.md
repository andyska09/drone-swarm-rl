# Design choices

Decisions taken for the JAX simulator, and every place it differs from the MRS
C++ reference (`research/code_sources/mrs_multirotor_simulator`). One line of
reasoning each. Update this file when a decision changes.

## The one deviation from the C++

MRS re-orthonormalizes the rotation matrix with a Cholesky of `RᵀR`:

```cpp
llt(R.transpose() * R);  P = llt.matrixL();  R = R * P.inverse();
```

`matrixL()` returns `L` with `RᵀR = L Lᵀ`, so `R L⁻¹` has Gram
`L⁻ᵀ L Lᵀ L⁻¹`, which is not the identity. The correct projection is `R L⁻ᵀ`.

Measured on a 100 Hz tumble, `‖RᵀR − I‖`:

| steps | MRS as written | with the transpose |
|-------|----------------|--------------------|
| 100   | 1.6e-12        | 3.6e-16            |
| 1000  | 1.4e-02        | 3.2e-16            |
| 100000| 3.3e-03        | 5.5e-16            |

MRS drifts ~3e-3 off SO(3) and stays there. **We use `R L⁻ᵀ`.** Everything
else is a faithful reimplementation.

Consequence: our reference is patched MRS, not stock MRS. The patch is a
two-line diff kept with the golden-trajectory harness.

## Model

- State is `(x, v, R, ω, rpm)`. `R` is a 3×3 matrix, not a quaternion. It is
  what MRS integrates, so it is what we can cross-check against. Revisit at M5
  only with a profile in hand.
- RK4 over the 18 rigid-body states. Not Euler: M1's value is the cross-check,
  and the integrator is not the bottleneck next to a policy forward pass.
- **RPM is held constant across the four RK4 stages**, then updated by the exact
  exponential lag `rpm ← e^(−dt/τ)·rpm + (1 − e^(−dt/τ))·rpm_cmd`. MRS does this
  (its derivative reads the member, not the integrated state) and it is correct:
  a zero-order hold on the actuator, which is what an ESC does between updates.
  The exponential is also more accurate than RK4-ing a first-order lag.
- Orthonormalization runs inside the derivative (every stage) *and* after the
  step, as in MRS.
- Physics kept: quadratic drag `0.30 · π · arm² · |v|² · v̂`, world-frame and
  isotropic. External force and moment as `Params` fields, zeroed — that is where
  wind and downwash go later.
- Physics dropped: ground contact off, takeoff patch off. The takeoff patch pins
  `z` to the spawn height until commanded thrust exceeds 0.9× hover, which
  silently breaks most tests.
- X500 parameters copied from `ModelParams`. Hover RPM is 4255.39.

## Interfaces

- `step(state, throttle_cmd, params, dt)` with `throttle_cmd ∈ [0,1]⁴`, clamped
  and mapped to `[min_rpm, max_rpm]` inside. Matches MRS, and the mixer's output
  wires straight in with no adapter.
- `min_rpm = 1170` is kept. Zero throttle is therefore 1170 RPM, not 0 — the
  motors never fully stop through the command interface. Tests that want true
  free fall use a params variant with `min_rpm = 0`.
- The CTBR action is `(throttle ∈ [0,1], ω_x, ω_y, ω_z)`. Normalized throttle,
  not Newtons, matching MRS. Revisit only if we train across airframes.
- dt = 0.01 (100 Hz). Physics, rate loop and policy all at one rate.

## Numerics

- MRS is `double` throughout. Cross-check tests run under
  `jax_enable_x64`, so a tolerance failure is a logic bug and not roundoff.
- Training runs in float32. `dynamics.py` is dtype-agnostic; one test asserts f32
  and f64 agree to ~1e-4 over 1 s.

## Batching

Everything is written for a single drone and lifted with `jax.vmap` at the env
boundary. No hand-rolled batch dimensions.

This costs a discipline: no Python `if` on a value that came out of an array, no
`.item()`, no shapes that depend on values. MRS's `if (v.norm() != 0)
v.normalize()` becomes a `jnp.where` with a safe norm
(`sqrt(sum(v*v) + 1e-12)`, because `norm` has a NaN gradient at zero). Same for
the mixer's saturation branches.

A `test_vmap_matches_loop` gate makes this real rather than aspirational.

## Controllers

- PID carries `(integral, last_error)` explicitly as `PIDState`. No hidden
  object state, because the whole loop has to be jittable.
- Three MRS behaviours replicated as-is, not fixed:
  - **The rate controller's output is never clamped.** It is constructed with
    `saturation = -1`, and the clamp is guarded by `if (saturation > 0)`. Despite
    the docs saying roll/pitch/yaw are normalized to `[-1,1]`, they are not — the
    mixer's desaturation is what actually bounds them. Load-bearing.
  - **Derivative kick on the first step.** `last_error` starts at 0, so a step
    reference gives `d = 1/dt`. With `kd·J_xx` the first output is double the
    steady-state one. Bounded, and it stops mattering at M3 when a policy rather
    than a step generates `ω_ref`.
  - **The integral is dead** — `ki = 0.0` by default.
- Mixer desaturation replicated exactly, branchless. Its rescale divides by
  `mean(motors)/throttle`, which can be near zero, so the divide is guarded —
  JAX evaluates both sides of a `where`.
- Rate gains `kp = 4.0`, `kd = 0.04`, `ki = 0.0`, each scaled by the matching
  diagonal element of `J`.
- The upper rungs are plain 3-axis PIDs on the error, with MRS's gains and
  saturations: position `2.0 / 0.15 / 0.2`, clamped to 6 m/s; velocity
  `2.0 / 0.05 / 0.01`, clamped to 4 m/s^2; attitude `6.0 / 0.05 / 0.01`, clamped to
  10 rad/s in roll and pitch and 1 rad/s in yaw. Unlike the rate loop these do
  clamp, and their integrals are live.
- The acceleration rung has no PID. It turns a desired force into an orientation
  by oblique projection, and into a throttle by projecting that force on the
  *current* body z. MRS takes the square root of that unguarded, so an inverted
  drone yields NaN. Replicated as-is.
- Only the heading branch of the cascade is implemented, not the heading-rate
  branch. A position command flows through the heading branch; the `TiltHdgRate`
  variant and its `getYawRateIntrinsic` machinery are unused.

## Testing

- `pytest`. `jax_enable_x64` set in `conftest.py`; goldens parametrized over the
  scenario list.
- **Golden trajectories are the primary gate.** The patched C++ is run once per
  scenario, the full state dumped at every step to CSV, and the CSVs committed.
  The test replays the same inputs in JAX and asserts agreement to 1e-9.
  `research/code_sources/` is gitignored, so the committed CSVs and the harness
  are the only record — a fresh clone has no C++.
- Scenarios: free fall; hover; tumble (throttles `0.55, 0.40, 0.52, 0.44`, 500
  steps); spin-down from nonzero `ω` at equal throttle; one seeded random state.
  A sixth covers the rate loop once it exists (`ω_ref = (0, 1, 0)`), which turns
  "converges without oscillating" into a number.
- Hover alone is not a gate. It is an equilibrium, so a transposed allocation
  matrix, a flipped yaw sign, swapped inertia axes, or a missing `ω × Jω` all
  pass it. The torque path is only exercised while tumbling.
- Analytic tests kept alongside, as a sanity layer: free fall against
  `min_rpm = 0, drag = 0` params (`z = −4.905` after 1 s — with stock params it
  is −4.378, drag and the RPM floor each account for part of the gap);
  orthonormality over 1e5 steps; zero net torque at equal RPM; 63.2% of the motor
  step after one time constant; and instantaneous `ω̇` pure about +y for a
  two-up/two-down throttle split, asserted on `derivative()` at t=0 rather than
  on a rollout — once `R` tilts, the drone accelerates sideways and the test
  stops being true.

## Revisit later

- Quaternions instead of `R`, at M5, if a profile says the matrix costs. Note the
  cost is not just the rewrite: a quaternion model cannot be cross-checked
  against the C++ at all.
- Euler instead of RK4, at M5, if RK4 turns out to be more than ~5% of step time.
- The unclamped rate-controller output, at M3.
- Attitude rate normalization, in future move it to newtons so it drone body doesnt matter. 
