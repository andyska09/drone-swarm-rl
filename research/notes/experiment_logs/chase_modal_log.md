# chase_modal — run log

Every run in which a drone senses more of the other drones than their offset and velocity:
what it was, what it scored, and why the setup is as it is. The presets are in
[task_chase.md](../tasks/task_chase.md). The runs with the base obs are in
[chase_pools_log.md](chase_pools_log.md).

## The question

Three pursuers that share one policy hit each other often. chase_modal asks if a pursuer
that senses more of each other drone hits its teammates less and catches more. The evader
flies a fixed path, so only the sensing changes between runs.

## Design decisions, 2026-10-28

- **The steps are base, +attitude, +rpm and +attitude+rpm.** Base is the chase obs: the
  offset and the velocity of every other drone, at any distance. The user picked the
  steps. The rotation rate of the other drones is not a step: the user meant the tilt.
- **The spec says attitude, not tilt.** `defaults.md` uses tilt for the part of $R$ without
  the heading. The block is the whole $R_i^\top R_j$. So it also gives the heading of
  drone $j$, the side its net faces, and the place of each of its motors.
- **Why the attitude can matter.** From the offset and the velocity, a policy without
  memory can only extend a straight line. The attitude gives the direction of drone $j$'s
  thrust, so the direction it accelerates in now. The motor speeds give the size of that
  thrust and the torques, 0.03 s after drone $j$ commanded them (the motor lag). So they
  are close to its action. But a thrust and a torque in the body of drone $j$ say little
  without its attitude.
- **rpm is a share of the top speed, $n / n_{\max}$.** A raw motor speed is about 4255 rpm
  at hover, and it saturates the first tanh layer. So rpm is the one input that is not in
  raw SI units. Hover is 0.55, and the range is 0.15 to 1.
- **The sensed blocks go to every other drone, the evader too.** In 1v1 the only other
  drone is the evader, so the blocks are the pursuer's advantage over it. So 1v1 shows if
  a pursuer uses the blocks at all. It is also the control for 3v1: a gain that 3v1 shows
  and 1v1 does not comes from the teammates.
- **The task fixes the evader to paths.** The user picked this over a train setting that
  names a pilot for any task. So `--task chase_modal` always trains only the pursuers. Eval
  still needs `--seat evader paths`, because eval does not read the task's pilot yet.
- **The box rules stay those of chase `default`.** The rules for leaving the box change the
  results a lot, but the user keeps the defaults for now.

## Before the first run

Generation 0 of the 3v1 pool run `chase_3v1_s0_20261027-211843_000913` is the run `3v1`:
three pursuers learn against paths only, for 50M steps. At its last update, the training
episodes ended so:

| `caught_e` | `crash_p` | `OOB_p` | `coll_pp` | `time_limit` |
|---|---|---|---|---|
| 0.03 | 0.59 | 0.23 | 0.11 | 0.05 |

- With its mean action, this pursuer then won 0.01 of 600 episodes against paths.
- In the later generations of that run, `coll_pp` grew to 0.64 of the pursuer's training
  episodes. Against paths, no pursuer of the run won more than 0.07.
- FRPN, which does not avoid its teammates, catches paths in 0.75 of the episodes in 1v1.
  In 3v1 it catches in 0.44, and two pursuers hit each other in 0.36 (256 episodes).
- In three older 1v1 pool runs in `default`, with older code, the generation-0 pursuer won
  0.59, 0.50 and 0.12 against paths (`797201`, `967c5e`, `9e4242`).

## Runs, 2026-10-28

24 runs on MetaCentrum, commit `23b65a8`, `--set gamma=0.99`, seeds 0–2 for each preset.
1v1 ran 2e8 steps (762 updates), 3v1 ran 5e8 steps (1907 updates). Eval: `--seat evader
paths`, 1024 episodes, mean actions, the last checkpoint. The eval is close to the
training numbers of the last 5 % of the updates.

| preset | `caught_e` | `coll_pp` | `time_to_catch` |
|---|---|---|---|
| `1v1` | 0.96 [0.95 0.94 0.98] | — | 3.7 s [3.9 3.8 3.3] |
| `1v1_attitude` | 0.99 [0.96 1.00 1.00] | — | 2.7 s [3.9 2.2 2.0] |
| `1v1_rpm` | 0.98 [1.00 0.97 0.97] | — | 3.0 s [2.6 3.1 3.2] |
| `1v1_attitude_rpm` | 0.97 [0.91 1.00 1.00] | — | 3.0 s [4.3 2.1 2.5] |
| `3v1` | 0.05 [0.03 0.05 0.07] | 0.63 [0.79 0.65 0.45] | — |
| `3v1_attitude` | 0.05 [0.05 0.03 0.07] | 0.61 [0.59 0.46 0.78] | — |
| `3v1_rpm` | 0.06 [0.07 0.07 0.04] | 0.57 [0.55 0.73 0.44] | — |
| `3v1_attitude_rpm` | 0.07 [0.07 0.08 0.07] | 0.66 [0.67 0.67 0.62] | — |

Each cell is the mean of the three seeds, then seeds 0, 1 and 2.

- **1v1 catches in every preset.** A sensed block may make the catch faster: 7 of the 9
  runs with one catch in 2.0–3.2 s, and the fastest base run needs 3.3 s. But a run that
  learned sooner also had more updates left to get faster, and 3 seeds are few.
- **3v1 fails in every preset, and the sensing changes nothing.** The seeds of one preset
  spread more (`coll_pp` 0.46–0.78) than the presets differ (0.57–0.66).
- **More steps did not help 3v1.** In training, `caught_e` is 0.03–0.06 at 25 % of the run
  and 0.05–0.08 at the end. `crash_p` + `OOB_p` fell from 0.96 to 0.02–0.15, and `coll_pp`
  rose in their place: 0.02–0.16 at 10 % of the run, 0.33–0.75 at the end.
- **The 3v1 spread does not fall.** The std of the actions starts at 1. In 1v1 it ends at
  0.17–0.23. In 3v1 it ends at 0.4–1.3, and in `3v1` s0 it grew to 3.2.
- **The pursuers hit each other mid-chase, away from the evader.** In the eval of `3v1`
  s0, the median `coll_pp` comes at 4.6 s, and 4 % come before 2 s. In 4 of its 8 saved
  episodes, two pursuers fly into each other at 4–9 m/s, 5–8 m from the evader.
- **The evader has no fixed slot in the obs.** `others` lists the drones nearest first. In
  that eval, the evader is in the last of the 3 slots in 71 % of the steps, and in slot 0
  or 1 in the rest. In 1v1 it is always the one slot.

## The merge into chase, 2026-10-28

- **chase_modal is now part of chase.** The user picked one task, in which a preset names
  the whole setup: the box, the number of pursuers, what each role senses, and if the
  paths pilot flies the evader in training. The 8 presets are now `paths_1v1`,
  `paths_1v1_attitude`, … `paths_3v1_attitude_rpm` of `chase`.
- **In the paths presets, only the pursuer senses the blocks.** Before, the evader sensed
  them too. The evader never flies, so its input does not reach the pursuer.
- The run folders `chase_modal_*` come from the old task. The code has no task of that
  name now, so `run/eval.py` cannot load them.
- **Eval flies the preset's pilot.** A `paths_*` run needs no `--seat evader paths` now.
  Before, a run evaluated without it flew the untrained evader network.

## The coll_pp cost, 2026-10-28

- **A `coll_pp` now costs 30, the same as a crash.** The user picked this. Every run above
  paid 10.
- **Why.** In 3v1, `crash_p` + `OOB_p` fell and `coll_pp` rose in their place. At γ = 0.99,
  the rest of the episode is worth about −5 to a pursuer that lives on: the −4/s clock
  and the distance cost, over a 1 s horizon. So a `coll_pp` at 10 cost only about 5 more
  than to live on, and a crash at 30 about 25 more.
- Only a preset with more than one pursuer has a `coll_pp`, so the 1v1 presets do not
  change. The pool preset `3v1` changes too.

## Sensing in the pools, 2026-10-28

- **The question.** Does a side win more when it senses more of the other drone? In the
  runs against the paths evader above, a pursuer that sensed a block caught a little
  faster.
- **The grid.** The user picked 3 × 3 presets: each side senses the attitude, the rpm or
  both, `nemecek_p_<pursuer>_e_<evader>`. No preset has a side that senses nothing, and
  the grid has no new base run. The user's word for the attitude is tilt.
- **The setup is that of `b9225b`:** `nemecek`, `generations=13`, `gamma=0.99`,
  `ent_coef=0`, the std reset and 1.4e9 steps. Two seeds for each preset, so 18 runs.
- **Each role senses its own blocks.** Before, the blocks went to every drone, so a run
  could not give a block to one side only.
- **First question, the user's pick: does a pursuer that senses more beat learned
  evaders more often?** Against the paths evader, the blocks changed little. So only the
  pursuer senses more, in `default_p_attitude`, `default_p_rpm` and
  `default_p_attitude_rpm`, two seeds each. The matrix of each run is read against the
  matrix of `9e4242` ([chase_pools_log.md](chase_pools_log.md)), which has the same
  setup with no blocks.

## The evader's slot, 2026-11-01

- **A pursuer now finds the evader in slot 0 of `others`.** The drones of another role
  come first, then those of the drone's own role, each nearest first. The user picked
  this.
- **Why.** In the eval of `3v1` s0, the evader was in the last slot in 71 % of the steps.
  So a teammate sat in slot 0 most of the time, and a policy that flies at the drone in
  slot 0 flies at a teammate.
- **Not a separate target block.** That block would change the width of the obs, so no
  old `chase` run would load. The role flag stays in each slot, so the width stays.
- In 1v1 there is only one slot, and `circle_swap` and `avoidance` have one role, so their
  obs does not change. The old 3v1 runs still load, but they now see the new order.

## The pursuer that senses more, 2026-11-01

6 runs on MetaCentrum, commit `edea2f4`, `--set gamma=0.99 --set generations=13 --set
ent_coef=0`, 1.4e9 steps, seeds 0 and 1. Only the pursuer senses more. The evader senses
nothing extra, as in `9e4242`.

The matrix, in the columns of
[chase_pools_log.md](chase_pools_log.md#pool-training-results). Each cell is the mean of
the two seeds, then seeds 0 and 1:

| preset | lower triangle | older evaders | diagonal | last pair |
|---|---|---|---|---|
| `9e4242`, no blocks | 0.61 | 0.66 | 0.27 | 0.20 |
| `default_p_attitude` | 0.73 [0.64 0.81] | 0.78 [0.70 0.87] | 0.37 [0.28 0.46] | 0.31 [0.18 0.45] |
| `default_p_rpm` | 0.77 [0.73 0.81] | 0.81 [0.77 0.85] | 0.52 [0.47 0.57] | 0.59 [0.58 0.61] |
| `default_p_attitude_rpm` | 0.88 [0.87 0.89] | 0.93 [0.92 0.93] | 0.60 [0.57 0.63] | 0.66 [0.44 0.88] |

The evals, 1024 episodes each, the last checkpoint. Each cell is the pursuer's win rate,
seeds 0 and 1:

| preset | last pursuer against last evader | FRPN against last evader | last pursuer against cascade |
|---|---|---|---|
| `9e4242`, no blocks | 0.20 | 0.04 | 0.95 |
| `default_p_attitude` | 0.18, 0.45 | 0.04, 0.12 | 0.99, 0.99 |
| `default_p_rpm` | 0.58, 0.61 | 0.19, 0.03 | 0.97, 0.97 |
| `default_p_attitude_rpm` | 0.44, 0.88 | 0.16, 0.08 | 0.99, 0.92 |

- **A pursuer that senses more wins more against learned evaders.** Every run is above
  `9e4242` in every matrix column, but `default_p_attitude` seed 0: its diagonal is equal
  (0.28 against 0.27), and its last pair is lower (0.18 against 0.20).
- **More blocks, higher numbers.** In all 4 matrix columns, attitude < rpm < attitude and
  rpm. With both blocks, the pursuer wins 0.93 against older evaders, against 0.66.
- **Against the paths evader, the blocks changed little** (Runs, 2026-10-28, above).
  Against learned evaders, they change a lot.
- **The last evaders are hard for FRPN in every run.** FRPN wins 0.03–0.19 against them,
  and 0.04 against the last evader of `9e4242`.
- **The last pair is noisy.** It is one cell, and its two seeds spread from 0.44 to 0.88
  in `default_p_attitude_rpm`. The lower triangle spreads at most 0.17 between seeds.
- **Limits.** `9e4242` is one seed, and it ran on the code of 2026-10-14, where the sides
  learned in turn. Each matrix plays a pursuer only against the evaders of its own run.

## Runs with both fixes, 2026-11-01

8 runs on MetaCentrum, commit `de90325`: a `coll_pp` costs 30 and the evader is in slot 0.
The four `paths_3v1` presets, seeds 0 and 1, `--set gamma=0.99`, `ent_coef` 0.001, 5e8
steps. Eval: 1024 episodes, mean actions, the last checkpoint. Each new cell is the mean
of the two seeds. The old cells are the table of Runs, 2026-10-28, with 3 seeds.

| preset | `caught_e` | old | `coll_pp` | old | `crash_p` |
|---|---|---|---|---|---|
| `paths_3v1` | 0.005 | 0.05 | 0.44 | 0.63 | 0.53 |
| `paths_3v1_attitude` | 0.024 | 0.05 | 0.64 | 0.61 | 0.30 |
| `paths_3v1_rpm` | 0.008 | 0.06 | 0.53 | 0.57 | 0.45 |
| `paths_3v1_attitude_rpm` | 0.024 | 0.07 | 0.46 | 0.66 | 0.51 |

In training, the last 5 % of the updates, against seeds 0 and 1 of the old runs:

| | old | new |
|---|---|---|
| `caught_e` | 0.05–0.08 | 0.03–0.05 |
| `coll_pp` | 0.33–0.70 | 0.20–0.33 |
| `crash_p` + `OOB_p` | 0.03–0.16 | 0.33–0.56 |
| the pursuer's entropy at the end | 1.7–10.3 | 18–22 |

- **The spread blew up in all 8 runs.** The entropy rose for the whole run, so the std
  ended at about 20–60 for each action. The actions are clipped to [−1, 1], so almost
  every action in training was a random full push.
- **The adaptive lr rose to its cap of 0.01** in the second half of each run. In the old
  runs it stayed near 0.001. With a huge std, one update changes the policy little, so
  the rule raises the lr.
- **In training, the cost of 30 only swapped one death for another.** `coll_pp` fell, and
  `crash_p` + `OOB_p` rose in its place. A pursuer still died in about 3 of 4 episodes.
- **The eval tells little.** It flies the mean action, which the drones almost never flew
  in training. `OOB_p` is 0.18–0.39 in training and 0.001–0.019 in the eval.
- **Why the spread grows.** With a std far above 1, most draws hit the clip, so a larger
  std changes the flight little. The reward then does not push the std down, and the
  entropy bonus pushes it up. In 1v1 the reward pulled the std down early: the entropy
  fell from 5.4 to −1.0.
- So these runs cannot show what the two fixes do. **Next, the user's pick:** the same 8
  runs with `--set ent_coef=0`.

## Runs with `ent_coef=0`, 2026-11-02

8 runs on MetaCentrum: the runs above with `--set ent_coef=0`. Commit `de90325`, the four
`paths_3v1` presets, seeds 0 and 1, `--set gamma=0.99`, 5e8 steps. Eval as above. Each
cell is the mean of the two seeds.

| preset | `caught_e` | `coll_pp` | `crash_p` |
|---|---|---|---|
| `paths_3v1` | 0.019 | 0.50 | 0.45 |
| `paths_3v1_attitude` | 0.019 | 0.42 | 0.55 |
| `paths_3v1_rpm` | 0.022 | 0.37 | 0.51 |
| `paths_3v1_attitude_rpm` | 0.004 | 0.33 | 0.65 |

The four batches side by side, each over seeds 0 and 1 of its four presets, so 8 runs.
"Training" is the mean of the last 5 % of the updates. "At ±1" is the share of the eval
steps in which the mean thrust, roll or pitch action of a pursuer sits at the clip.

| | 1v1, 2026-10-28 | 3v1, 2026-10-28 | 3v1, `ent_coef` 0.001 | 3v1, `ent_coef` 0 |
|---|---|---|---|---|
| the pursuer's std per action, at the end | 0.15–0.33 | 0.27–3.7 | 18–87 | 5.9–14 |
| mean action at ±1, eval | 1–25 % | 13–73 % | 87–99 % | 75–95 % |
| `caught_e`, training | 0.88–1.00 | 0.05–0.08 | 0.03–0.05 | 0.04–0.05 |
| `caught_e`, eval | 0.91–1.00 | 0.03–0.08 | 0.004–0.030 | 0.004–0.035 |
| the critic's explained variance, training | 0.87–0.96 | 0.21–0.56 | 0.31–0.53 | 0.38–0.53 |

- **Without the bonus, the spread still grew in all 8 runs.** It grew less than with
  0.001, but the std ended at 6–14 for each action. So the bonus is not the cause, and
  the guess in the section above is wrong.
- **The pursuers fly full on or full off.** The mean action sits at ±1 in most eval steps.
  So the eval, which flies the mean, tells little.
- **In training, `caught_e` rose to about 0.04–0.05 in the first 30–50 % of the run, then
  stayed flat.** At the end, `coll_pp` was 0.20–0.49, and `crash_p` + `OOB_p` 0.29–0.61.
- **The adaptive lr rose again**, to 0.002–0.01 at the end.
- **The critic fits worse as training goes on.** Its explained variance is 0.66–0.87 in
  the first 10 % of the run, and 0.38–0.53 in the last 5 %.
- **Part of the bad fit comes from the critic's design.** It gets one input for all three
  pursuers, so it gives them one value. With FRPN pursuers in `paths_3v1`, 512 episodes,
  26 % of the spread of the pursuers' returns lies between the pursuers of one scene. So a
  critic of this kind explains at most about 0.74. Gavin & Bronz also give all pursuers
  one value, and their pursuers learn.

### Why the spread grows: a test with FRPN

FRPN flew the pursuers with Gaussian noise of std σ on each action, then the clip, 512
episodes for each σ. "Return" is the mean episode return of one pursuer, as eval reports
it: no discount and no bootstrap.

| σ | 1v1 caught | 1v1 return | 3v1 caught | 3v1 `coll_pp` | 3v1 `OOB_p` | 3v1 return |
|---|---|---|---|---|---|---|
| 0 | 0.76 | −16 | 0.49 | 0.32 | 0.17 | −13 |
| 0.3 | 0.71 | −19 | 0.47 | 0.33 | 0.17 | −14 |
| 1 | 0.25 | −41 | 0.19 | 0.35 | 0.29 | −25 |
| 3 | 0.04 | −47 | 0.04 | 0.05 | 0.82 | −23 |
| 10 | 0.00 | −43 | 0.00 | 0.01 | 0.98 | −19 |

- **In 1v1, noise only hurts.** From σ = 1, less noise is much better, and more noise is
  worse. So the reward pulls the std down, as in the 1v1 runs.
- **In 3v1, σ = 1 is the worst point, and the return gets better in both directions.**
  With more noise, the pursuers stop hitting each other and leave the box one by one. A
  `coll_pp` kills two pursuers, and an exit kills one.
- **A new policy starts at σ = 1** (`init_std`). So in 3v1, PPO can get out of the worst
  point by adding noise. The runs did.
- With `ent_coef` 0.001, the std ended at 0.27–3.7 with a `coll_pp` cost of 10, and at
  18–87 with 30. The slot change came at the same time, so the two cannot be told apart.
- FRPN is not the learned policy, and PPO discounts and bootstraps. So the test is a
  strong sign, not a proof.

## The Beta policy, 2026-11-03

- **Every policy now draws its actions from a Beta, as in
  [Chou et al. 2017](../../papers/chou17a.md).** The user picked this. The network gives
  α and β for each action, each a softplus plus 1. The action is 2x − 1 with
  x ~ Beta(α, β), so it lies in [−1, 1] without a clip. The Gaussian with one learned std
  per action and the clip is gone, with no switch back.
- **Why.** Past ±1, the clip gave the same action for every draw, so the reward could not
  pull back the mean or the std. In the 3v1 runs above, the std grew to 6–87, and the mean
  sat at ±1 in most steps. The widest Beta is uniform on [−1, 1], with a std of 0.58,
  below the Gaussian's start of 1. So the random full pushes, with which noise helped 3v1
  in the FRPN test, cannot happen. The log-prob is also that of the action the drone
  flies, which removes the bias near the bounds that Chou shows.
- **What else changed.**
  - A new actor starts at α = β ≈ 1.69, a std of about 0.48 for each action. The entropy
    of one action is at most log 2 ≈ 0.69, that of the uniform Beta.
  - Eval and a frozen opponent fly the Beta's mean.
  - The pool phases no longer reset the std, because the Beta has no std to reset.
  - Old runs no longer load for eval, the matrix or a resume. They need the commit before
    this change. Their saved evals still open in the viewer.
- **Smoke check.** 59 updates of `paths_3v1` on the Mac, 256 envs: the entropy stayed at
  2.1–2.5 of its maximum 2.77, the KL stayed near its target, and nothing turned NaN. The
  eval ran.
- **Next:** the 8 runs of 2026-11-02 again, with the Beta.

## Open

| question | state |
|---|---|
| why 3v1 does not learn | the spread grows in every 3v1 run with both fixes, also with `ent_coef=0`. The FRPN test points to noise that keeps the pursuers apart. Next: the Beta policy, whose noise has a ceiling (2026-11-03). Other options: start from a trained 1v1 pursuer, or a fixed lr of 5e-4 as in Gavin's 1v1 paper. FRPN catches 0.44–0.49 in 3v1, so the task can be done |
| a sensed block in 1v1 | a hint of faster catches. More seeds, or the time to catch at a fixed `caught_e`, can tell it from faster learning |
| the sensing steps in 3v1 | cannot be read until 3v1 learns |
| does a pursuer that senses more beat learned evaders more often | yes, in the 6 runs of `default_p_*`. The base `9e4242` is one seed on older code |
| what a sensed block gives a side | the 18 runs of the grid, not run yet |
