# Design choices

Decisions for the simulator and for what every task shares: the env contract, the
learner, eval and replay. Also every place we differ from the MRS C++ reference
(`research/code_sources/mrs_multirotor_simulator`). One line of reasoning each. The
shared model itself is in [tasks/defaults.md](tasks/defaults.md), and a task's own model,
pool training included, is in its `tasks/task_<name>.md`. The runs behind a decision are
in `experiment_logs/`. Update this file when a decision changes.

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

The values are in [tasks/defaults.md](tasks/defaults.md). This section keeps the reasons.

- **One file per task.** Each task lives in `swarm/envs/<task>.py` with `reset`,
  `step`, `get_obs`, `reference` and a `PRESETS` dict. `swarm/envs/__init__.py`
  holds the contract every task file meets, the shared `Obs`, and
  `make(task, preset)`, which loads a task by name, so adding a task needs no
  registry. The contract between `envs/` and `learn/` stays narrow: an obs the learner
  never looks inside, three flags per drone, the pilot names and the metric shapes. So a
  new task changes nothing in `learn/`. `core.py` holds what the tasks share. The trainer handles automatic
  episode resets, so we do not need gymnax's auto-reset base class. Single-drone
  and swarm tasks use the same interface, with a drone dimension on observations,
  actions and rewards.
- **Update rates.** We chose 200 Hz physics to better resolve
  the motor's 0.03 s response time: six simulation steps per time constant,
  compared with three at 100 Hz. The rate controller is slower, with an
  approximate time constant of 0.25 s at `kp = 4.0`, so either rate is sufficient
  for it. The extra physics step costs roughly 1% by operation count: about
  1,000 floating-point operations for RK4 versus 70,000 for a policy evaluation.
- **Observation fields.** Each task builds `Obs(actor={role: matrix}, critic=vector)`:
  one matrix per role, one row per drone, and the task picks the columns. The learner
  never looks inside a row, so a task can change its obs with no change in `learn/`.
  `core.own_obs` takes the wall distances from the task: the arena box
  (`core.wall_dist`), or the task's own box, as the avoidance tube (`core.box_dist`).
  `core.others` sorts the drones that each drone sees and zeroes the empty seats, so the
  row width never changes.
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
- **Actions.** A new policy outputs values near zero, which gives a throttle near
  0.5, close to the hover value of 0.4654. `quad-swarm-rl` allows ±31.4 rad/s for
  more acrobatic flight, where we allow ±4 rad/s in roll and pitch.
- **One centre.** `center` is the middle of the arena box, and each task spawns
  its drones and draws its goals around it.
- **Drone failure.** The default rule kills a drone that flips over, because
  `acceleration_controller` can return NaN when the drone is inverted. A task with
  other rules writes its own `is_dead`. Contact at 0.80 m between centres is the MRS
  rule (`multirotor_simulator.cpp:342`, `crit_dist = arm+prop+arm+prop`), and MRS
  ships `collisions: {enabled: true, crash: true}`, so contact kills there too. The
  plant has no contact force, so the distance test is the whole contact model.
- **State after failure.** Every task ends the episode when any drone fails, so
  no failed drone has to stay frozen while others fly on. `evaluate.py` freezes
  the whole scene at episode end to keep later scan steps from changing metrics
  or replay.
- **Arrival.** In `avoidance`, a drone that reaches its goal leaves the episode, and
  the other drone flies on. The arrived drone stays frozen where it arrived, and the
  other drone can neither see it nor touch it. An arrived drone has `flying` false and
  `alive` true. So an arrival ends that drone's training steps with zero future value,
  as a failure does, but `death_rate` does not count it.
- **Episode endings.** `step` gives three flags per drone, `alive`, `flying` and
  `bootstrap`, and the scene flag `done`. The trainer computes the rest: the loss uses
  a drone's step if the drone was flying before it, and the return goes on while the
  drone flies and the scene goes on. The trainer owns the step limit, `max_steps`: at
  the limit every flying drone bootstraps from `V(s_T)`, unless the task ended the
  episode at the same step. When the task ends it, `bootstrap` says which flying drones
  bootstrap. A failed drone has zero future value. In `chase` only a catch is terminal
  for the survivors.
- **The cascade uses the policy interface.** Every action follows the same path:
  action → rate controller → mixer → physics. There is no separate `action_mode`. The
  cascade is a pilot, `core.cascade_pilot`. `control.cascade_outer` runs the cascade's
  position, velocity, acceleration and attitude stages, producing throttle and
  body-rate commands (CTBR). `command_to_action` converts these to the four normalized
  action values that a network would supply. The pilots that give a world acceleration
  (FRPN, APF, paths) go through `core.accelerate`, the acceleration and attitude stages. Both controllers use the same rewards, starting states
  and episode length, so their results can be compared directly. In this setup,
  the outer controllers run at 100 Hz instead of 200 Hz, and their rate commands
  are limited to ±4 rad/s for roll/pitch and ±2 rad/s for yaw.
- **Controller state.** `EnvState` stores only the rate controller's `PIDState`.
  The cascade pilot stores the PID states for its position, velocity and
  attitude controllers.
- **Staggered clocks.** The trainer keeps a step count and a step limit for each
  scene. The first episode's limit is random in [1, `max_steps`], and every later
  limit is `max_steps`. Scenes that start together also end together, and then most
  updates finish no episode and report nothing. The task's own `time` starts at 0 in
  every episode, so the trainer reads no field of the env state, and a deadline of the
  task, as avoidance's 10 s `timeout`, reads the true time.
- **Curriculum.** A task can change its params during training: it gives
  `curriculum(params, metrics)`. The training loop calls it after every update with
  that update's metrics. When it returns new params, the runner builds the update again
  with `ppo.make`, because the params are constants inside the compiled update. One
  rebuild compiled in about 2 s on the Mac (CPU). The other way, passing the params
  into the update as data, makes every float param of every task an input instead of
  a constant, and changes `ppo.py`.
- **Resume.** Every run resumes from its last checkpoint, and the resumed run is
  bit-exact. `latest.pkl` holds the weights, the optimizer state, the key, each
  learner's envs and pool, and the env params of a curriculum. Two details make it
  bit-exact: the reward-scale count has a fixed dtype, and the env params are pickled as
  JAX arrays, because XLA folds a numpy constant and rounds it differently. The
  curriculum runs before the save, so a resume starts with the params of the next
  update.

## Learner

- **The bound is a clip, and the std is one learned number per action.** The action
  is `clip(mean + std·noise, −1, 1)`, the std is the same in every state, and
  `init_std = 1.0`. The log-prob and the entropy are the plain Gaussian's, in closed
  form. The rollout keeps the draw before the clip, because the log-prob needs it.
  AMSPBH does not state its std. This form is the default of rsl_rl, rl_games and
  skrl, the PPO libraries of Isaac Lab, and of Němeček's SB3. A per-state std can
  shrink just where precision pays, near the evader, and that is where the pursuer
  must find its next move. The runs behind it:
  [experiment_logs/chase_pools_log.md](experiment_logs/chase_pools_log.md).
- **`ent_coef` is 0.001, AMSPBH's value, and Gavin's 0.01 does not transfer.** PPO
  shrinks the std by itself once precision pays, and the bonus fights exactly that. At
  0.01 the std breaks. The math: [learning/05-RL.md](learning/05-RL.md). The runs:
  [experiment_logs/chase_pools_log.md](experiment_logs/chase_pools_log.md).
- **`gamma` defaults to 0.998**, a 5 s horizon at `policy_dt = 0.01 s`. A 2 s horizon
  (0.995) was too short for a 6.9 m flight that takes 3 to 4 s. Chase sets 0.99
  ([tasks/task_chase.md](tasks/task_chase.md), Training).
- **The lr starts at 2e-4 and then follows the KL**, AMSPBH's lr and target with
  rsl_rl's rule. After the forward pass of each minibatch, the learner's mean
  `approx_kl` sets the lr, and the step of that same minibatch uses it, as in rsl_rl.
  `approx_kl` is a sample estimate of KL(old ‖ new), the KL that rsl_rl computes in
  closed form. The target is 0.016. KL > 0.032: lr ÷ 1.5. 0 < KL < 0.008: lr × 1.5. The
  lr stays in [1e-5, 1e-2]. Each role has its own lr in its own TrainState, so a
  frozen role keeps its lr. `adaptive_lr=false` keeps the lr at `lr` for the whole run.

## Evaluation and replay

- **One number per metric, and the task picks which.** `info` carries two dicts:
  `end` is read at the step the episode ended, `step` is averaged over the steps.
  Reporting both flavours of every key gives two columns where only one is
  meaningful, and no name says which. A NaN in `end` means the episode
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
- **Policy selection.** One `--seat` flag names one seat: the role, then who flies it.
  That is another run, a `.pkl`, or one of the task's `PILOTS` (`cascade` in every
  task, and `frpn`, `apf` and `paths` in chase). Any pilot can fly any seat. A role
  nobody names flies this run's own checkpoint. That is what lets a duel measure one side
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
- Collective thrust in newtons instead of the normalized throttle in the CTBR action,
  so the airframe does not matter.
- One death rule for all tasks. `core.is_dead` kills a drone that flips over, and
  `chase` and `avoidance` let it fly on.
