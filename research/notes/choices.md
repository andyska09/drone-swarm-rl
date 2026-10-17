# Design choices

Decisions for the simulator and for what every task shares: the env contract, the
learner, eval and replay. Also every place we differ from the MRS C++ reference
(`research/code_sources/mrs_multirotor_simulator`). One line of reasoning each. A
task's own settings live in its `task_<name>.md`, and pool training lives in
[AMSPBH.md](AMSPBH.md). Update this file when a decision changes.

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
  what MRS integrates, so it is what we can cross-check against. Revisit only
  with a profile in hand.
- RK4 over the 18 rigid-body states. Not Euler: the plant's value is the
  cross-check against the C++, and the integrator is not the bottleneck next to a
  policy forward pass.
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
- The reference trajectory tests use dt = 0.01 (100 Hz), matching MRS's test
  harness. The environment runs physics and the rate controller at 200 Hz, and
  the policy at 100 Hz. See the environment section below.

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

A `test_vmap_matches_python_loop` gate makes this real rather than aspirational.

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
    steady-state one. This is manageable with smooth commands from the cascade
    controller. A policy that samples actions from a Gaussian can change the
    requested rate `ω_ref` abruptly at every step, causing repeated spikes.
    To avoid these command-induced spikes, `pid_update` accepts an optional
    `measurement`: the derivative term then uses the negative rate of change of
    the measured angular velocity instead of the rate error. The environment
    enables this with `d_on_measurement=True`. The default keeps MRS's behavior
    for the reference trajectory tests.
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
- Plant scenarios: free fall; hover; tumble (throttles `0.55, 0.40, 0.52, 0.44`,
  500 steps); spin-down from nonzero `ω` at equal throttle; one seeded random
  state. Controller scenarios: the mixer allocation, and a step for each rung of
  the cascade: rate (`ω_ref = (0, 1, 0)`), attitude, velocity and position. The
  rate step turns "converges without oscillating" into a number.
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

## Environment

- **One file per task.** Each task lives in `swarm/envs/<task>.py` with `reset`,
  `step`, `get_obs`, `reference` and a `PRESETS` dict. `swarm/envs/__init__.py`
  holds the contract every task file meets, the shared `Obs`, and
  `make(task, preset)`, which loads a task by name, so adding a task needs no
  registry. `core.py` holds what the tasks share. The trainer handles automatic
  episode resets, so we do not need gymnax's auto-reset base class. Single-drone
  and swarm tasks use the same interface, with a drone dimension on observations,
  actions and rewards.
- **Update rates.** Physics, the rate controller and the mixer run at 200 Hz
  (`sim_dt = 0.005`). The policy runs at 100 Hz, so each action is held for two
  simulation steps (`steps_per_action = 2`). We chose 200 Hz to better resolve
  the motor's 0.03 s response time: six simulation steps per time constant,
  compared with three at 100 Hz. The rate controller is slower, with an
  approximate time constant of 0.25 s at `kp = 4.0`, so either rate is sufficient
  for it. The extra physics step costs roughly 1% by operation count: about
  1,000 floating-point operations for RK4 versus 70,000 for a policy evaluation.
- **Observation fields.** Every task fills the same
  `Obs(own, others, others_mask, target, scene)`. `own` holds the drone's velocity
  `Rᵀv`, all nine entries of its rotation matrix `R`, its angular velocity `ω`, its
  height, and four distances to the arena walls along rays level with the floor.
  `others` holds the K nearest drones, `(K, 7)`: body-frame offset, body-frame
  velocity, and a flag that says whether that drone shares my role. `others_mask`
  marks the filled seats, and `flat` zeroes the empty ones, so the vector width
  never changes. `target` holds the position error `Rᵀ(goal − x)` and the velocity
  error `Rᵀ(v_goal − v)`, and a task with no goal gives it width 0. `scene` is the
  critic's input: every drone's exact state. Keeping the other drones in their own
  field lets an attention network read them directly.
- **Rotation representation.** Observations include all nine entries of `R`.
  This choice follows SimpleFlight's reported result of roughly 64% worse
  performance with quaternions.
- **Coordinate frame.** Position error, velocity and velocity error use the
  drone's body axes, matching the axes of its actions. Multiplying by `Rᵀ` does
  this conversion explicitly, so the policy does not have to learn it. Angular
  velocity `ω` already uses body axes. `R` still describes the drone's orientation
  relative to the world.
- **Motor speed is omitted from observations.** The rate controller reads the
  measured `ω` at every simulation step and compensates for motor lag. We rely
  on that feedback instead of giving `rpm` to the policy. `quad-swarm-rl` and
  SimpleFlight also omit it.
- **Actions.** Each action contains four values in `[−1,1]`: one throttle and
  three requested angular velocities. Throttle maps to `[0,1]`; roll and pitch
  rates map to ±4 rad/s, and yaw to ±2 rad/s. A new policy outputs values near
  zero, which gives a throttle near 0.5, close to the hover value of 0.4654.
  `quad-swarm-rl` allows ±31.4 rad/s for more acrobatic flight.
- **One centre.** `center` is the middle of the arena box, and each task spawns
  its drones and draws its goals around it.
- **Drone failure.** `core.is_dead` is the default rule: a drone fails if it
  leaves the arena box, goes below z = 0, flips over (`R[2,2] < 0`), touches
  another drone, or has a NaN or infinite position. Flipping over also causes
  problems for `acceleration_controller`, which can return NaN when inverted. A
  task with other rules writes its own `is_dead`. After failure, `alive` stays
  false, and the drone receives no further rewards or penalties.
- **State after failure.** Every task ends the episode when any drone fails, so
  no failed drone has to stay frozen while others fly on. `evaluate.py` freezes
  the whole scene at episode end to keep later scan steps from changing metrics
  or replay.
- **Episode endings.** The time limit, `max_steps` in each task, is a
  truncation: training stops the rollout but still uses the estimated future
  value `V(s_T)` for surviving drones. A failed drone has zero future value.
  `done` tells the trainer to reset the scene; `died_this_step` identifies drones
  that just failed; `truncated` says that every drone still alive bootstraps. That
  is the time limit, and in `chase` also a failure without a catch: only a catch is
  terminal for the survivor.
- **The cascade uses the policy interface.** Every action follows the same path:
  action → rate controller → mixer → physics. There is no separate `action_mode`.
  `control.cascade_outer` runs the cascade's position, velocity, acceleration and
  attitude stages, producing throttle and body-rate commands (CTBR).
  `command_to_action` converts these to the four normalized action values that a
  network would supply. Both controllers use the same rewards, starting states
  and episode length, so their results can be compared directly. In this setup,
  the outer controllers run at 100 Hz instead of 200 Hz, and their rate commands
  are limited to ±4 rad/s for roll/pitch and ±2 rad/s for yaw.
- **Controller state.** `EnvState` stores only the rate controller's `PIDState`.
  The cascade policy stores the PID states for its position, velocity and
  attitude controllers.
- **Staggered clocks.** The trainer gives each scene a random clock value on the
  first reset, and then calls `get_obs` again. Scenes that start together also end
  together, and then most updates finish no episode and report nothing. Later
  resets are not staggered, so no episode after the first is cut short. This is
  the one place where the trainer uses a field of the env state by name (`time`).

## Learner

- **The bound is a clip, and the std is one learned number per action.** The action
  is `clip(mean + std·noise, −1, 1)`, the std is the same in every state, and
  `init_std = 1.0`. The log-prob and the entropy are the plain Gaussian's, in closed
  form. The rollout keeps the draw before the clip, because the log-prob needs it.
  AMSPBH does not state its std. This form is the default of rsl_rl, rl_games and
  skrl, the PPO libraries of Isaac Lab, and of Němeček's SB3. It replaced Gavin's
  per-state std with a `tanh` squash after the pool run
  `chase_default_s0_20261011-103639_797201`: the pursuer's roll/pitch std fell from
  0.43/0.33 to 0.06/0.16, and from generation 6 no pursuer beat a new evader. A new
  std head on `pursuer_13` lifted its win rate against `evader_07` from 0.20 to 0.95
  in 50 updates. A per-state std can shrink just where precision pays, near the
  evader, and that is where the pursuer must find its next move.
- **`ent_coef` is 0.001, AMSPBH's value, and Gavin's 0.01 does not transfer.** Fourteen `chase` runs,
  and the two with `ent_coef = 0.01` are the only two with a broken std: 112 to
  60784 under the clip, 371 to 942 under the squash. The other twelve all end
  between 0.02 and 0.51, and the two runs that solve `chase default` (`caught`
  0.996 and 1.000) end at 0.08 to 0.13. PPO shrinks the std by itself once
  precision pays — that is `E[A(u²−1)] < 0` — and the bonus fights exactly that.
  At 0.001 the pull on the mean in the next point is ten times weaker, so it
  crosses the policy's own pull near `s = 150`, not `s = 15`.
- **Why the bonus breaks each design, which is not the same reason.** Write `H` for
  the entropy, `m` for the mean, `s` for the std, `u = (z−m)/s`. Under the clip,
  `∂H/∂log s = +1`, a constant that never decays, so noise pays for ever and
  `∂H/∂m = 0` leaves the mean free to grow with it — the old runs kept `m/s ≈ 1`
  and still flew. Under the squash, `∂H/∂log s ≈ 1 − 1.6s` flips sign above
  `s ≈ 0.63`, which is the fix, but a second term appears: `∂H/∂m ≈ −2·sign(m)`,
  a constant pull of the mean to zero, worth `0.02` at `ent_coef = 0.01` against
  the policy's own `≈ ρ/s`. They cross near `s = 15`. Past that the entropy term
  owns the mean, `m/s → 0`, and because the rate loop averages the command the
  drone flies `E[tanh z] ≈ 0.8·m/s ≈ 0`. That run ended at `distance` 11.40 m,
  where an untrained policy starts, against 0.5 to 4.0 m for every trained one.
- **`gamma` defaults to 0.998**, a 5 s horizon at `policy_dt = 0.01 s`. Chase sets
  0.99 ([task_chase.md](task_chase.md), Training). Why 0.998:
  [archive/task_a_to_b.md](archive/task_a_to_b.md).
- The lr, the adaptive lr and its rule come from AMSPBH: [AMSPBH.md](AMSPBH.md), PPO.

## Evaluation and replay

- **One number per metric, and the task picks which.** `info` carries two dicts:
  `end` is read at the step the episode ended, `step` is averaged over the steps.
  Reporting both flavours of every key, as evaluation used to, gives two columns
  where only one is meaningful and no name saying which — `rho_mean` was a
  per-episode score divided by episode length. A NaN in `end` means the episode
  has no value for that metric, and it is dropped from the average instead of
  counted as a zero. Task-specific pass thresholds belong in tests and notes.
- **Every metric with a drone axis is reported per role**, as `pursuer_return`;
  a one-role task keeps the bare name. Averaging over the drone axis cancels the
  two sides of a game: in `chase` the clock and the catch have opposite signs, so
  a pooled `return` measured only the crash bill, and a pooled `death_rate`
  capped at 0.5 in a 1v1. The loss statistics split the same way, so a frozen
  role cannot move the number its trainer reads.
- **Seeds and actions.** Evaluation uses a separate fixed seed,
  `EVAL_SEED = 1_000_000`, and mean policy actions.
- **No auto-reset.** Evaluation runs each episode once and freezes its final
  state. Later scan steps add no reward. It reports each episode's `ep_length` in
  policy steps; training reports the average length of episodes that finished.
- **Results stay with the run.** Evaluation writes to
  `runs/<run>/evals/<name>/`, keeping results with the run they measure.
- **Policy selection.** One flag names one seat: `--seat ROLE SOURCE`, where the
  source is this run's checkpoint, another run's, a `.pkl`, or a hand-written source
  from `seats.SOURCES` (`cascade`, `frpn`, `apf`, `paths`). A role nobody names
  flies this run's own checkpoint. That is what lets a duel measure one side
  against a fixed opponent.
- **Replay data.** The viewer reads task, preset, arena, roles and `policy_dt`
  from `header.json`, and state from `trajectory.npz`. It does not import the
  environment.
- **Trajectory format.** `np.savez` writes uncompressed ZIP files. The viewer
  reads the NPZ and NPY headers directly in JavaScript without an unzip library.
  Using `savez_compressed` would require adding decompression.
- **Opening replays.** `run/replay.py` serves the repo and lists evaluations at
  `/evals.json` for the viewer's dropdown. Opening the viewer through `file://`
  requires drag-and-drop or the file picker because the page cannot fetch local
  files.
- **Recorded replay.** We save trajectories after the compiled XLA rollout and
  replay them with a timeline and episode picker. This avoids the realtime pacing
  and frame-dropping used by `quad-swarm-rl` to render during rollouts.
- **Visual cues.** The shadow and velocity arrow follow
  `quadrotor_multi_visualization.py`. The shadow helps show altitude by spreading
  and fading as the drone rises. The arrow projects the current velocity 0.4 s
  ahead.

## Revisit later

- Quaternions instead of `R`, if a profile says the matrix costs. Note the cost
  is not just the rewrite: a quaternion model cannot be cross-checked against the
  C++ at all.
- Euler instead of RK4, if RK4 turns out to be more than ~5% of step time.
- Attitude rate normalization, in future move it to newtons so it drone body doesnt matter.
