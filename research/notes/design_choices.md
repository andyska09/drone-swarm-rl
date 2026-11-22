# Design choices

Decisions for the simulator and for what every task shares: the env contract, the
learner, eval and replay. The Chase section holds the reasons for the values of the two
chase specs. Also every place we differ from the MRS C++ reference
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

- `R` is a 3×3 matrix, not a quaternion. It is what MRS integrates, so it is
  what we can cross-check against.
- RK4 over the 18 rigid-body states. Not Euler: the plant's value is the
  cross-check against the C++, and the integrator is not the bottleneck next to a
  policy forward pass.
- **RPM is held constant across the four RK4 stages**, then updated by the exact
  exponential lag. MRS does this (its derivative reads the member, not the
  integrated state) and it is correct: a zero-order hold on the actuator, which is
  what an ESC does between updates. The exponential is also more accurate than
  RK4-ing a first-order lag.
- Orthonormalization runs inside the derivative (every stage) *and* after the
  step, as in MRS.
- Physics dropped: ground contact off, takeoff patch off. The takeoff patch pins
  `z` to the spawn height until commanded thrust exceeds 0.9× hover, which
  silently breaks most tests.

## Interfaces

- `step(state, throttle_cmd, params, dt)` with `throttle_cmd ∈ [0,1]⁴`, clamped
  and mapped to `[min_rpm, max_rpm]` inside. Matches MRS, and the mixer's output
  wires straight in with no adapter.
- `min_rpm = 1170` is kept. Zero throttle is therefore 1170 RPM, not 0 — the
  motors never fully stop through the command interface. Tests that want true
  free fall use a params variant with `min_rpm = 0`.
- The CTBR action is `(throttle ∈ [0,1], ω_x, ω_y, ω_z)`. Normalized throttle,
  not Newtons, matching MRS.
- The reference trajectory tests use dt = 0.01 (100 Hz), matching MRS's test
  harness.

## Numerics

- MRS is `double` throughout. Cross-check tests run under
  `jax_enable_x64`, so a tolerance failure is a logic bug and not roundoff.

## Batching

Everything is written for a single drone and lifted with `jax.vmap` at the env
boundary. No hand-rolled batch dimensions.

This costs a discipline: no Python `if` on a value that came out of an array, no
`.item()`, no shapes that depend on values.

## Controllers

- PID carries `(integral, last_error)` explicitly as `PIDState`. No hidden
  object state, because the whole loop has to be jittable.
- Three MRS behaviours replicated as-is, not fixed:
  - **The rate controller's output is never clamped.** MRS constructs it with
    `saturation = -1`, and guards the clamp by `if (saturation > 0)`. We give it an
    infinite limit. Despite
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
- The acceleration rung turns a desired force into a throttle by projecting that
  force on the *current* body z. MRS takes the square root of that unguarded, so an
  inverted drone yields NaN. We clamp the projection at 0: a propeller only pushes,
  so an inverted drone gets the lowest throttle.
- Only the heading branch of the cascade is implemented, not the heading-rate
  branch. A position command flows through the heading branch; the `TiltHdgRate`
  variant and its `getYawRateIntrinsic` machinery are unused.

## Testing

- **Golden trajectories are the primary gate.** The patched C++ is run once per
  scenario, the full state dumped at every step to CSV, and the CSVs committed.
  The test replays the same inputs in JAX and asserts agreement to 1e-9.
  `research/code_sources/` is gitignored, so the committed CSVs and the harness
  are the only record — a fresh clone has no C++.
- Hover alone is not a gate. It is an equilibrium, so a transposed allocation
  matrix, a flipped yaw sign, swapped inertia axes, or a missing `ω × Jω` all
  pass it. The torque path is only exercised while tumbling.
- Analytic tests kept alongside, as a sanity layer. The torque-direction test
  asserts the instantaneous `ω̇` on `derivative()` at t=0 rather than on a
  rollout — once `R` tilts, the drone accelerates sideways and the test stops
  being true.

## Environment

The values are in [tasks/defaults.md](tasks/defaults.md). This section keeps the reasons.

- **One file per task.** `make(task, preset)` loads a task by name, so adding a task
  needs no registry. The contract between `envs/` and `learn/` stays narrow
  ([tasks/env_contract.md](tasks/env_contract.md)): an obs the learner never looks
  inside, three flags per drone, the pilot names and the metric shapes. So a new task
  changes nothing in `learn/`. The trainer handles automatic episode resets, so we do
  not need gymnax's auto-reset base class. Single-drone and swarm tasks use the same
  interface, with a drone dimension on observations, actions and rewards.
- **A preset names the setup of a run, all but the hyperparameters**: the box, the roles,
  what each role senses, and with `piloted` a pilot that flies one role in training. So
  one task holds every setup of one game. Chase has two specs, one for one pursuer and
  one for three, so that each one stands beside the papers of its kind on the spec page.
- **Update rates.** We chose 200 Hz physics to better resolve
  the motor's 0.03 s response time: six simulation steps per time constant,
  compared with three at 100 Hz. The rate controller is slower, with an
  approximate time constant of 0.25 s at `kp = 4.0`, so either rate is sufficient
  for it. The extra physics step costs roughly 1% by operation count: about
  1,000 floating-point operations for RK4 versus 70,000 for a policy evaluation.
- **What a drone senses of the others.** In `chase`, each role has its own sensed
  blocks, so one side can sense a block that the other side does not. The motor
  speeds are scaled, because about 4255 rpm at hover saturates the first tanh layer.
  `avoidance` mirrors the frame of each drone on its own, and the attitude block is not
  mirrored. So `avoidance` cannot use it as it is.
- **Rotation representation.** Observations include all nine entries of `R`.
  This choice follows SimpleFlight's reported result of roughly 64% worse
  performance with quaternions.
- **Coordinate frame.** Position error, velocity and velocity error use the
  drone's body axes, matching the axes of its actions. Multiplying by `Rᵀ` does
  this conversion explicitly, so the policy does not have to learn it.
- **Motor speed is omitted from the own observation.** The rate controller reads the
  measured `ω` at every simulation step and compensates for motor lag. We rely
  on that feedback instead of giving `rpm` to the policy. `quad-swarm-rl` and
  SimpleFlight also omit it.
- **Actions.** A new policy outputs values near zero, which gives a throttle near
  0.5, close to the hover value of 0.4654.
- **Contact.** Contact at 0.80 m between centres is the MRS
  rule (`multirotor_simulator.cpp:342`, `crit_dist = arm+prop+arm+prop`), and MRS
  ships `collisions: {enabled: true, crash: true}`, so contact kills there too. The
  plant has no contact force, so the distance test is the whole contact model.
- **State after failure.** Every task ends the episode when any drone fails, so
  no failed drone has to stay frozen while others fly on. `evaluate.py` freezes
  the whole scene at episode end to keep later scan steps from changing metrics
  or replay.
- **Arrival.** An arrived drone has `flying` false and `alive` true. So an arrival
  ends that drone's training steps with zero future value, as a failure does, but
  `death_rate` does not count it.
- **The cascade uses the policy interface.** Every action follows the same path:
  action → rate controller → mixer → physics. There is no separate `action_mode`. So
  a pilot and a network use the same rewards, starting states and episode length, and
  their results can be compared directly. The cost: the outer controllers run at
  100 Hz instead of 200 Hz, and their rate commands pass the action limits.
- **Staggered clocks.** The first episode of each scene has a random step limit.
  Scenes that start together also end together, and then most
  updates finish no episode and report nothing. The task's own `time` starts at 0 in
  every episode, so the trainer reads no field of the env state, and a deadline of the
  task, as avoidance's 10 s `timeout`, reads the true time.
- **Curriculum.** When a task changes its params, the runner builds the update again,
  because the params are constants inside the compiled update. One
  rebuild compiled in about 2 s on the Mac (CPU). The other way, passing the params
  into the update as data, makes every float param of every task an input instead of
  a constant, and changes `ppo.py`.
- **Resume.** Every run resumes from its last checkpoint, and the resumed run is
  bit-exact. Two details make it
  bit-exact: the reward-scale count has a fixed dtype, and the env params are pickled as
  JAX arrays, because XLA folds a numpy constant and rounds it differently. The
  curriculum runs before the save, so a resume starts with the params of the next
  update.

## Learner

These are the defaults of `swarm/learn/config.py`. Chase against a pilot runs on other
values: see the last item.

- **The bound is a clip, and the std is one learned number per action.**
  AMSPBH does not state its std. This form is the default of rsl_rl, rl_games and
  skrl, the PPO libraries of Isaac Lab, and of Němeček's SB3. A per-state std can
  shrink just where precision pays, near the evader, and that is where the pursuer
  must find its next move. The runs behind it:
  [experiment_logs/chase_1v1/02_pools.md](experiment_logs/chase_1v1/02_pools.md).
- **The critic reads each drone's own actor input**, so each drone gets its own value.
  [MAPPO](https://arxiv.org/abs/2103.01955)
  calls this input the Agent-Specific Global State. With one input for all three pursuers
  of `paths_3v1`, the critic cannot show which pursuer flies into a hit, so the blame
  falls on all three. There the std grew to 12–29, and the catch stayed under 0.04. The
  runs: [09_critic_input](experiment_logs/chase_3v1_before/09_critic_input.md).
- **The input scale moves once per update, after the epochs.** We store
  raw obs and scale them again in the loss, so a move during the rollout makes the loss
  read other inputs than the rollout did. In
  the 1v1 pools the scale gave no clear gain
  ([06_pools_input_scale](experiment_logs/chase_1v1/06_pools_input_scale.md)). In
  `paths_3v1` it lowered the catch
  ([09_critic_input](experiment_logs/chase_3v1_before/09_critic_input.md)).
- **The attention net has a norm on its Perceiver output.** `net=attention` reads the
  teammates through a Perceiver, so the size of its input does not depend on their
  number. A LayerNorm with no scale and no bias sits on the output of each query. Without
  it, the output grows from 0.59 to about 2000 and saturates the first layer of the
  trunk, so the policy reads neither its own block nor the evader. The runs:
  [21_attention](experiment_logs/chase_3v1_before/21_attention.md) and
  [22_attention_norm](experiment_logs/chase_3v1_before/22_attention_norm.md).
- **`ent_coef` defaults to 0.001, AMSPBH's value, and Gavin's 0.01 does not transfer.** At
  0.01 the std breaks. The math: [learning/05-RL.md](learning/05-RL.md). The runs:
  [experiment_logs/chase_1v1/01_duel.md](experiment_logs/chase_1v1/01_duel.md). The best
  value depends on the lr. In `paths_3v1` with a constant lr, 0.003 catches more than
  0.001 at every lr ([02_lr_ent](experiment_logs/chase_3v1/02_lr_ent.md)). With the
  adaptive lr, 0.003 blew the std up
  ([13_ent_adaptive_lr](experiment_logs/chase_3v1_before/13_ent_adaptive_lr.md)).
- **`gamma` defaults to 0.998**, a 5 s horizon at `policy_dt = 0.01 s`. A 2 s horizon
  (0.995) was too short for a 6.9 m flight that takes 3 to 4 s. Chase sets its own value.
  The pool runs set 0.99, Gavin's value at the same 100 Hz
  ([tasks/task_chase_1v1.md](tasks/task_chase_1v1.md), Training). The runs against a pilot set
  0.995: see the last item.
- **The lr starts at 2e-4 and then follows the KL**, AMSPBH's lr and target with
  rsl_rl's rule. One difference from rsl_rl: `approx_kl` is a sample estimate of
  KL(old ‖ new), the KL that rsl_rl computes in closed form.
  In `paths_3v1` the rule stops the learning. The std falls, and with a small std a small
  change of the mean is a large KL, so the rule lowers the lr to its floor
  ([12_freeze](experiment_logs/chase_3v1_before/12_freeze.md)).
- **Chase against a pilot has its own base.** The `paths_*` and `apf_*` presets run with
  the flags of [04_new_base](experiment_logs/chase_3v1/04_new_base.md):
  `--set normalize_obs=false --set adaptive_lr=false --set lr=1.25e-4 --set ent_coef=0.003
  --set gamma=0.995 --set clip_eps=0.1 --set num_steps=128`. This base catches 0.975 in
  `paths_3v1`. The experiments `01` to `04` in
  [experiment_logs/chase_3v1/](experiment_logs/chase_3v1/) hold the runs behind each
  value. The defaults in `config.py` stay as they are, because a change there also
  changes the pool runs and the other tasks.

## Evaluation and replay

- **One number per metric, and the task picks which**: read at the end of the episode,
  or averaged over the steps.
  Reporting both flavours of every key gives two columns where only one is
  meaningful, and no name says which. Task-specific pass thresholds belong in tests and
  notes.
- **Every metric with a drone axis is reported per role**, as `return_p`;
  a one-role task keeps the bare name. Averaging over the drone axis cancels the
  two sides of a game: in `chase` the clock and the catch have opposite signs, so
  a pooled `return` measured only the crash bill, and a pooled `death_rate`
  capped at 0.5 in a 1v1. The loss statistics split the same way, so a frozen
  role cannot move the number its trainer reads.
- **Policy selection.** Eval names who flies each role. A role
  nobody names flies the pilot that the preset gives it in training, else this run's own
  checkpoint. That is what lets a duel measure one side against a fixed opponent.
- **Trajectory format.** `np.savez` writes uncompressed ZIP files. The viewer
  reads the NPZ and NPY headers directly in JavaScript without an unzip library.
  Using `savez_compressed` would require adding decompression.
- **Recorded replay.** We save trajectories after the compiled XLA rollout and
  replay them with a timeline and episode picker. This avoids the realtime pacing
  and frame-dropping used by `quad-swarm-rl` to render during rollouts.
- **Visual cues.** The shadow and velocity arrow follow
  `quadrotor_multi_visualization.py`. The shadow helps show altitude by spreading
  and fading as the drone rises.

## Chase

The reasons for the values of the two chase specs,
[tasks/task_chase_1v1.md](tasks/task_chase_1v1.md) and
[tasks/task_chase_3v1.md](tasks/task_chase_3v1.md). Gavin is Gavin & Bronz 2026 and
Gavin, Lacroix & Bronz 2026. Roncero is Sánchez Roncero et al. 2026, arXiv v3. Němeček
is his thesis and his code. Their setups are in [literature/specs/](literature/specs/),
and the runs in `experiment_logs/chase_1v1/`, `experiment_logs/chase_3v1_before/` and
`experiment_logs/chase_3v1/`.

### The task

- **The net has no thickness.** A drone is a ball of 0.40 m, the arm plus the propeller,
  and a ball touches a plane when its centre is within one radius of it. At 15 m/s the
  evader moves 0.15 m per policy step, so it cannot skip through the net.
- **A capture wins over a death in the same step**, because the net square reaches into
  the 0.80 m contact distance.
- **Only a catch is terminal for the survivors.** Neither side gains when the other
  fails, which is Gavin's rule. Roncero also bootstraps past an out-of-bounds (§III-B).
- **`coll_pe` is a cost, and the episode goes on.** The pursuer must reach the evader to
  catch it, and an end there teaches it to stand off (Gavin).
- **A flipped drone keeps flying.** Gavin ends an episode on an exit and on a contact
  between pursuers, and says nothing about the attitude.
- **The drones start 2 m apart.** A drone touches a net only within 1.54 m of the
  pursuer's centre, and a contact needs 0.80 m. So no catch and no contact can happen at
  step 0.
- **The pursuer earns progress in place of a clock and a distance cost.** With the clock,
  the rest of an episode was worth about −5 to a pursuer that lives on, so a death also
  ended its costs. The 0.25 per m is Roncero's weight, and the catch of 30 is our pick.
- **The other weights are Gavin's.** His per-step weights are divided by the policy step,
  so ours are per second and do not depend on the control rate.
- **A `coll_pp` costs 10, Gavin's value.** A cost of 30 gave the same catch, 0.212
  against 0.208, and a cost of 0 lowers it
  ([03_breadth](experiment_logs/chase_3v1/03_breadth.md)).
- **`open` and `nemecek` have Němeček's arena rules.** Only the ground ends an episode,
  and the evader pays `outside` while it is outside the box. The 57 is his ratio: 0.1 per
  step outside against 0.007 per step for survival, times our clock of 4. `nemecek` is the
  box of his main runs.
- **`small` exists** because the policies of a `default` pool run, put in a 24 × 24 × 16 m
  box, caught the newest evader in 0.39 of the episodes, against 0.17 in 32 × 32 m.
- **The evader is in slot 0 of the others.** With the old order, nearest first, the
  evader of a 3v1 run was in the last slot in 71 % of the steps, so a teammate sat in
  slot 0.

### Pool training

- **The method is AMSPBH of Roncero** (§III-C and §IV-A). Where the paper is silent, we
  follow Němeček's code.
- **Pools replaced the swap of the duel.** No duel run gave a good pursuer and a good
  evader together ([01_duel](experiment_logs/chase_1v1/01_duel.md)).
- **Both sides learn at once**, each against the other side's pool up to generation
  k − 1. So neither side needs the other's new weights, and a run has 14 phases in place
  of 28.
- **The std reset at each phase is ours.** The std shrinks in a phase, and the learner
  must explore again against the new pool. Without it the pursuer stopped learning after
  generation 7 ([02_pools](experiment_logs/chase_1v1/02_pools.md)).
- **FRPN and paths stay in the pools in every generation**, as in the paper (Fig. 6,
  §IV-C).
- **A frozen network flies its mean action**, as in Němeček's code.
- **64 blocks with $\gamma_h$ = 0.7** give each member at least 3 blocks, in a pool of up
  to 14 members.
- **A `coll_pp` is an evader win.** One outcome holds for any number of pursuers, in the
  order: caught, `fail_pe`, a pursuer failure, an evader failure, the time limit.

### The pilots

- **The lineup is Roncero's** (§IV-B): an FRPN pursuer, an APF evader and three path
  evaders. The cascade is the easy opponent, with MRS's defaults.
- **FRPN and APF have Němeček's code and values** (thesis, Tables 8.1 and 8.6). Roncero
  does not publish all of his, and Němeček flew the same X500.
- **FRPN and APF fly under a limit of 20 m/s² and 12 m/s.** Němeček's simulator cuts every
  drone's speed to 12 m/s after each step. We cannot do that, because the network flies
  the same physics. Without the limit FRPN reached about 19 m/s, and against a hover
  evader it left the arena in 0.39 of the episodes.
- **A pursuer pilot aims at the point one net drop above the evader**, so that its net
  lands on the evader, not its body.
- **The paths are Roncero's, 8 times larger and √8 times faster.** That keeps the turn
  acceleration $v^2/r$ of each shape.
- **The path picks where the evader starts.** Near a corner no full-size path fits around
  a random start, and a shrunk path keeps its lap time, so it is also slower.
- **A pilot keeps its heading**, and the cascade code stays as it is, because it must
  match the C++ goldens.

## Revisit later

- Quaternions instead of `R`, if a profile says the matrix costs. Note the cost
  is not just the rewrite: a quaternion model cannot be cross-checked against the
  C++ at all.
- Euler instead of RK4, if RK4 turns out to be more than ~5% of step time.
- Collective thrust in newtons instead of the normalized throttle in the CTBR action,
  so the airframe does not matter.
- One death rule for all tasks. `core.is_dead` kills a drone that flips over, and
  `chase` and `avoidance` let it fly on.
