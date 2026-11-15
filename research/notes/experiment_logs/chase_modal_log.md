# chase_modal — run log

Every run in which a drone senses more of the other drones than their offset and velocity:
what it was, what it scored, and why the setup is as it is. The presets are in
[task_chase.md](../tasks/task_chase.md). The runs with the base obs are in
[chase_pools_log.md](chase_pools_log.md).

## Facts

- The number to beat in 3v1: `caught_e` 0.605 and 0.837 at the last checkpoint, and 0.682 and 0.910 at the best one, in 4096 episodes (`730968`, `9eb432`; base `paths_3v1`, `gamma=0.99`, `normalize_obs=false`, lr 5e-4 constant, `ent_coef=0.003`, 5e8 steps, seeds 0 and 1). FRPN catches 0.524. 2026-11-11 ([§](#seed-0-730968))
- `ent_coef=0.003` with the constant lr of 5e-4 lifts the 3v1 catch from 0.498 (seed 0) to 0.605 (seed 0) and 0.837 (seed 1). With a net 2–5 m away the pursuers fly 1.81 and 2.63 m/s toward the evader, against 0.71. 2026-11-11 ([§](#seed-0-730968))
- With `ent_coef=0.003` and the constant lr, `coll_pp` is 0.91 and 0.83 of the episodes with no catch (seeds 0 and 1), a median 1.60 and 1.26 m from the evader. The pursuers die in 0.394 and 0.163 of the episodes. 2026-11-11 ([§](#seed-0-730968))
- With `ent_coef=0.003`, the eval catch also falls after its best checkpoint: 0.914 at update 850, 0.785 at 1200 and 0.850 at 1907 (seed 1), and 0.684 at 1050, 0.604 at 1500 and 0.628 at 1907 (seed 0). 2026-11-11 ([§](#runs-with-ent_coef0003-and-a-constant-lr-2026-11-11))
- Against the APF evader, which no policy met in training, `9eb432` catches 0.719, and 0.856 at update 850; `730968` catches 0.477, and 0.612 at update 1050; `be848f` catches 0.454 and our FRPN 0.279. 2026-11-11 ([§](#against-the-apf-evader-and-gavin-nv1))
- The sensed blocks with the constant lr, two seeds each: the catch is 0.440 and 0.366 (attitude), 0.549 and 0.548 (rpm), 0.642 and 0.580 (both), against 0.498 for the one base seed. 2026-11-11 ([§](#runs-with-the-sensed-blocks-and-a-constant-lr-2026-11-11))
- No 3v1 policy reads the rpm block: with it fixed, the catch changes by −0.006 to +0.016. With the evader's attitude fixed, the catch falls by 0.117–0.290. 2026-11-11 ([§](#runs-with-the-sensed-blocks-and-a-constant-lr-2026-11-11))
- The attention network (`--set net=attention`) with a norm on its Perceiver output catches 0.326 and 0.267 in 3v1 with the adaptive lr, against 0.213 and 0.252 for the MLP with the same settings (4096 episodes; `91b18c`, `2cdecf`). No attention run has the constant lr. 2026-11-11 ([§](#the-attention-runs-with-the-norm))
- Without the norm the attention network does not learn: the Perceiver output grows from 0.59 to about 2000, every unit of the first layer is saturated from update 50 on, and the policy reads neither its own block nor the evader. The catch is 0.008 and 0.009, and `OOB_p` 0.78 and 0.82 (`b087ac`, `da9e33`). 2026-11-11 ([§](#why-they-did-not-learn))
- The 3v1 attention weights fly one pursuer with no new training: in `paths_1v1` the catch is 0.322 and 0.143, and `OOB_p` 0.004 and 0.001. 2026-11-11 ([§](#one-pursuer-with-the-3v1-weights))
- The critic input for each drone lifted eval `caught_e` from 0.01–0.04 to 0.10–0.11 with input scaling and 0.18–0.24 without; the std no longer blows up (0.03–0.10). 2026-11-04 ([§](#runs-with-a-critic-input-for-each-drone-2026-11-04))
- 10 times the steps did not lift the catch (5e9 steps: 0.184 and 0.265). It goes flat early, so 5e8 steps are enough for a test. 2026-11-04 ([§](#long-runs-with-a-critic-input-for-each-drone-2026-11-04))
- `gamma=0.995` gave no gain: eval `caught_e` 0.157 and 0.176, against 0.183 and 0.242 at `gamma=0.99`. 2026-11-04 ([§](#runs-with-gamma0995-2026-11-04))
- The progress reward (0.25 per m, a catch of 30) did not lift the catch: 0.212 and 0.253 in 4096 episodes; over the three rewards, ANOVA p = 0.83. It catches sooner (median 4.2 and 5.2 s, against 5.5–6.5 s), and `coll_pp` is 0.32–0.54. 2026-11-10 ([§](#runs-with-the-progress-reward-2026-11-10))
- Every 3v1 policy learned with the adaptive lr holds its nets on a ring: 2.8–4.1, 4.3–6.4 and 5.7–8.6 m from the evader at 4–8 s. With a net 2–5 m away, it flies 2.1–3.8 m/s around the evader and 0.26–0.45 m/s toward it. FRPN closes at 4.8–6.6 m/s. 2026-11-10 ([§](#runs-with-the-progress-reward-2026-11-10))
- With `ent_coef=0.001`, the learned 3v1 catch does not depend on the evader's path (hover 0.16–0.26, figure-eight 0.18–0.28); FRPN catches 0.60 and 0.41. 2026-11-10 ([§](#runs-with-the-progress-reward-2026-11-10)). With 0.003, `9eb432` catches 0.874 and 0.784, and `730968` 0.685 and 0.530. 2026-11-11 ([§](#runs-with-ent_coef0003-and-a-constant-lr-2026-11-11))
- The learned `coll_pp` come a median 3.1–4.1 m from the evader, and the second-nearest pursuer is in 92–96 % of them. 2026-11-10 ([§](#runs-with-the-progress-reward-2026-11-10))
- With the constant lr, `coll_pp` is 0.55–0.80 of the failures, near the evader (nearest net a median 1.9–2.1 m away). Against his APF evader, Gavin's learned pursuers die in 0.069–0.140 of the episodes, his FRPN in 0.394; ours with `ent_coef=0.001` die in 0.433–0.478, our FRPN in 0.476. 2026-11-10 ([§](#against-gavin-nv1-2026-11-10))
- With the constant lr, the eval catch falls from the best checkpoint to the last (0.562 → 0.487, 0.597 → 0.499), while the KL rises from 0.006 to 0.022–0.023 and the clip fraction from 0.08 to 0.28–0.29. 2026-11-10 ([§](#the-failures-and-the-late-decline))
- A constant lr of 5e-4 (`--set adaptive_lr=false --set lr=5e-4`) lifts the 3v1 catch from 0.212 to 0.498 (4096 episodes, one seed); the best checkpoint catches 0.562. The std stays at about 0.05. 2026-11-10 ([§](#runs-with-a-constant-lr-and-netpass-2026-11-10))
- Training in `paths_3v1_netpass` lifts the catch to 0.388 with the adaptive lr (0.335 flown in `paths_3v1`), and to 0.509 with the constant lr (one seed each). 2026-11-10 ([§](#runs-with-a-constant-lr-and-netpass-2026-11-10))
- With the same policies, removing the net kill lifts the catch by 0.034 and 0.028 (McNemar p < 1e-25); 0.74–0.75 of the saved episodes run to `time_limit`. 2026-11-10 ([§](#probes-on-the-trained-policies-2026-11-10))
- The ring stays when the teammates are moved 15 m away in the input: nearest net 3.1–3.5 m at 4 s, catch 0.188–0.240. 2026-11-10 ([§](#probes-on-the-trained-policies-2026-11-10))
- From the ring, closing at > 4 m/s gives a higher own return than moving away in all six runs (by 0.35–1.34), with 14–25 times the own catch within 1 s and at most +0.018 own `coll_pp`. 2026-11-10 ([§](#probes-on-the-trained-policies-2026-11-10))
- A `coll_pp` cost of 10 did not lift the catch: eval `caught_e` 0.250 and 0.166, against 0.183 and 0.242 at 30. Eval `coll_pp` rose to 0.18 and 0.47. 2026-11-10 ([§](#runs-with-a-coll_pp-cost-of-10-2026-11-10))
- With the adaptive lr, no `ent_coef` holds the std (two values): at 0.001 it falls to 0.03–0.05; at 0.003 it blew up to 124–18010 and eval `caught_e` fell to 0.006–0.027. 2026-11-05 ([§](#runs-with-ent_coef0003-2026-11-05)). With the constant lr of 5e-4, 0.003 does not blow up in two seeds: the std ends at 0.11–0.16, and at 0.21 and 0.38 for the yaw rate. 2026-11-11 ([§](#runs-with-ent_coef0003-and-a-constant-lr-2026-11-11))
- The std falls as about c / update, and the lr follows it (lr / std 1.1–1.7e-3) down to its floor, so the policy stops moving. 2026-11-05 ([§](#the-std-and-the-lr-over-training))
- The frozen policy is not at a top: a β = 0.25 blend toward FRPN in all three lifts its score, −4.33 to −4.02 and −4.35 to −4.22. 2026-11-05 ([§](#flying-on-from-the-frozen-states))
- At γ = 0.99, all three on FRPN score below the learned policy at `coll_pp` cost 30 (−5.63 against −4.33), above it at 10 (−3.05 against −4.27). 2026-11-05 ([§](#flying-on-from-the-frozen-states))
- FRPN in 3v1 with action noise σ catches 0.49 at σ = 0, 0.47 at 0.3, and 0.19 at 1, where a new policy starts. 2026-11-02 ([§](#why-the-spread-grows-a-test-with-frpn))
- A pursuer that senses more wins more against learned evaders: lower triangle 0.73 (attitude), 0.77 (rpm), 0.88 (both), against 0.61 for `9e4242`, one seed on older code. 2026-11-01 ([§](#the-pursuer-that-senses-more-2026-11-01))

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

## Runs with the Beta, 2026-11-03

8 runs on MetaCentrum, commit `c90447f`: the four `paths_3v1` presets, seeds 0 and 1,
`--set gamma=0.99`, 5e8 steps. The user ran them with the default `ent_coef` 0.001, not 0,
so the Gaussian batch that matches them is that of 2026-11-01. Eval: `--seat evader paths`,
1024 episodes, mean actions, the last checkpoint. Each cell is the lowest and the highest
value of the 8 runs of a batch. "The last 5 %" is the mean of the last 5 % of the updates.

| | 1v1, 2026-10-28 | 3v1, `ent_coef` 0.001 | 3v1, `ent_coef` 0 | 3v1 Beta, `ent_coef` 0.001 |
|---|---|---|---|---|
| the pursuer's std per action, at the end | 0.15–0.33 | 18–87 | 5.9–14 | 0.12–0.28 |
| mean action at the edge, eval | 1–25 % | 87–99 % | 75–95 % | 0 % |
| the adaptive lr, the last 5 % | 0.0004–0.0006 | 0.01 | 0.002–0.01 | 0.0003–0.0006 |
| `caught_e`, training | 0.88–1.00 | 0.03–0.05 | 0.04–0.05 | 0.02–0.04 |
| `caught_e`, eval | 0.91–1.00 | 0.004–0.030 | 0.004–0.035 | 0.020–0.040 |
| the critic's explained variance, the last 5 % | 0.87–0.96 | 0.31–0.53 | 0.38–0.53 | 0.54–0.73 |

- The Beta's spread changes with the state. So its std is the mean over the steps of the 8
  saved eval episodes of each run, with the obs rebuilt from the saved states. The mean of
  the rebuilt Beta matches the flown action to 1e-5.
- "At the edge" is the share of the eval steps in which the mean thrust, roll or pitch
  action sits at the clip. The Beta's mean cannot reach ±1, so for the Beta it is
  |mean| ≥ 0.9, a looser test.

The eval outcomes of the three 3v1 batches:

| | Gaussian, `ent_coef` 0.001 | Gaussian, `ent_coef` 0 | Beta, `ent_coef` 0.001 |
|---|---|---|---|
| `caught_e` | 0.004–0.030 | 0.004–0.035 | 0.020–0.040 |
| `coll_pp` | 0.32–0.76 | 0.18–0.59 | 0.07–0.17 |
| `crash_p` | 0.22–0.65 | 0.33–0.80 | 0.01–0.07 |
| `OOB_p` | 0.00–0.02 | 0.00–0.03 | 0.09–0.18 |
| `time_limit` | 0.00–0.04 | 0.00–0.15 | 0.57–0.75 |

- **The Beta fixed the spread.** The std ended as in 1v1, the mean action never sat near
  ±1, the adaptive lr stayed low, and the critic fit better. In training, the pursuer's
  entropy fell from 1.2–1.7 in the first 10 % of the run to −0.8 to −1.7 at the end. Its
  maximum is 2.77.
- **The pursuers stay alive.** In the eval, 57–75 % of the episodes run to the step limit.
  Eval and training now agree, because the drones fly close to their mean.
- **The catch did not change.** In training, `caught_e` rose to about 0.03 in the first
  20–30 % of the run, then stayed flat. In the same time, `crash_p` + `OOB_p` fell from
  0.8–0.96 to about 0.3.
- **The pursuers hold back.** In the 64 saved eval episodes, the nearest net stays 8–11 m
  from the evader for all 10 s, and the other two pursuers stay 10–20 m away. The pursuers
  fly at about 3.7 m/s, close to the evader's 3.1–3.6 m/s. The 1v1 pursuer flies at
  5.5–8.9 m/s and closes from 14 m to 2.5 m in 2 s.
- **Flying in pays more over a whole episode.** The eval return of a pursuer is −47 to
  −50. FRPN flies straight in and gets −13 in 3v1, in the FRPN test above.
- **A guess, not tested.** With γ = 0.99 a reward loses half its weight every 0.7 s, so a
  catch 3 s ahead counts 5 % of its size. Holding back at 8 m costs 0.8 per second, next to
  the clock of 4 per second, which a pursuer pays either way. Flying in is risky: the three
  pursuers share one policy, so they fly in together, and a `coll_pp` costs 30 to each of
  the two. A test: the same runs with `--set gamma=0.995`.
- **The code went back to the Gaussian of `de90325`**, the user's pick, to look at another
  cause. The Beta stays in commit `c90447f`. These 8 runs load for eval, the matrix or a
  resume only on that commit. Their saved evals still open in the viewer.

## Runs with input scaling, 2026-11-04

4 runs on MetaCentrum, commit `636ab7a`: base `paths_3v1` only, seeds 0 and 1, tanh,
`--set gamma=0.99`, `ent_coef` 0.001, 5e8 steps. Two runs scale the inputs, and two run with
`--set normalize_obs=false` on the same code. What the scaling is and why:
[chase_pools_log.md](chase_pools_log.md), Input scaling. Eval: 1024 episodes, mean actions,
the last checkpoint. "Training" is the mean of the last 100 updates. The std is its final
value for each action.

| run | scaling | seed | `caught_e` eval | `coll_pp` eval | `crash_p` eval | `time_limit` eval | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|
| `a2abcb` | on | 0 | 0.024 | 0.88 | 0.03 | 0.06 | 0.028 | 2520–4544 | 0.42 |
| `3849a5` | on | 1 | 0.025 | 0.55 | 0.40 | 0.02 | 0.035 | 1290–1423 | 0.25 |
| `2d0b98` | off | 0 | 0.035 | 0.34 | 0.56 | 0.03 | 0.045 | 18–29 | 0.59 |
| `6fce56` | off | 1 | 0.009 | 0.44 | 0.39 | 0.16 | 0.029 | 12–27 | 0.55 |

The std of each run over training, the mean over the four actions:

| update | 100 | 250 | 400 | 600 | 900 | 1300 | 1900 |
|---|---|---|---|---|---|---|---|
| on, s0 | 1.17 | 1.63 | 3.55 | 10.3 | 69.8 | 622 | 3238 |
| on, s1 | 1.18 | 1.38 | 2.61 | 7.39 | 32.1 | 185 | 1289 |
| off, s0 | 1.08 | 1.28 | 1.91 | 3.79 | 8.90 | 13.4 | 24.4 |
| off, s1 | 1.15 | 1.38 | 1.64 | 3.20 | 7.42 | 9.39 | 17.5 |

- **Scaling did not fix 3v1.** `caught_e` stays near 0.03 with and without it. The 1v1
  runs of 2026-10-28 reach about 0.9 against the same evader.
- **The spread blew up in all 4 runs, and much more with scaling.** The adaptive lr reached
  its cap of 0.01 in all 4. We do not know why scaling makes the spread grow faster.
- **With scaling, the pursuers stay alive sooner.** Around update 140, `ep_length` is
  600–720 with scaling and 250–300 without it.
- **The critic fits worse with scaling.** Its explained variance is 0.25–0.42, against
  0.55–0.59.
- **The spread is a symptom.** The Beta runs of 2026-11-03 kept the std small, and they
  caught as rarely.
- **The eval tells little**, for the reason in the runs of 2026-11-01.
- **Next, the user's pick:** the same 4 runs with `--set activation=relu`, because Gavin
  uses ReLU, and its units do not saturate like tanh units. Base sensing only, until 3v1
  learns.

## A critic input for each drone, 2026-11-04

- **The critic now reads each drone's own actor input**, after the scene and the opponents'
  actions. In `paths_3v1` that is 120 numbers, up from 76. So each pursuer gets its own
  value. The critic weights stay shared, and the actor does not change. There is no switch,
  the user's pick.
- **Why.** With one input for all three pursuers, the critic cannot show which of them flies
  into a hit. So the blame falls on all three, and flying in looks bad for each of them.
- **It is a known input.** [MAPPO](https://arxiv.org/pdf/2103.01955) calls it the
  Agent-Specific Global State, and our old input is its EP. It tests this only in StarCraft,
  with one reward for the whole team. Gavin's pursuers learn with one input for all. So
  this is a test, not a sure fix.
- **The own block uses the actor's mean and std**, so there is nothing new to save.
- **Runs from before this change cannot resume.** Eval and the matrix still work on them,
  because they use only the actor.
- No run has used this input yet. The 3v1 runs go after the ReLU results.

## Runs with ReLU, 2026-11-04

2 runs on MetaCentrum, commit `636ab7a`, so with the old critic input: base `paths_3v1`,
seeds 0 and 1, `--set activation=relu --set normalize_obs=false --set gamma=0.99`, 5e8 steps.
The pair with scaling on did not run. Training is the mean of the last 100 updates. The
tanh column is the two runs of 2026-11-04 with scaling off.

| | ReLU, s0 | ReLU, s1 | tanh |
|---|---|---|---|
| `caught_e` | 0.005 | 0.004 | 0.029–0.045 |
| `OOB_p` | 0.84 | 0.89 | 0.26–0.36 |
| `coll_pp` | 0.007 | 0.008 | 0.12–0.24 |
| `time_limit` | 0 | 0 | 0.17–0.23 |
| `distance` | 9.9 | 9.9 | 7.4–7.9 |
| explained variance | 0.98 | 0.99 | 0.55–0.59 |
| final std: thrust, roll, pitch, yaw | 1.2, 0.007, 0.007, 9.2 | 0.16, 0.008, 0.006, 0.84 | 12–29 |

The flight numbers below come from 512 episodes per run with sampled actions, as in
training, from the last checkpoint.

- **The spread did not blow up.** The mean std fell from 1 to 0.05–0.15.
- **The pursuers do not chase.** They stay level, and the roll and pitch actions stay near
  0. The net stays about 10 m from the evader. The yaw action sits at the clip in 85–91 % of
  the steps.
- **One pursuer drifts out of the box.** In 98–99 % of the episodes exactly one pursuer dies.
  The median end is at 3.5–4 s, at about 4 m/s, level. In seed 0 it leaves through any face,
  and in seed 1 mostly through a wall. That ends the episode.
- **The spread is not the cause.** The std blew up with tanh. It stayed small with the Beta
  and with ReLU. In all three, the pursuers did not fly in.
- **The eval tells little here too.** In seed 0 the mean thrust action is −0.97, so the mean
  flight falls: `crash_p` is 0.999 at about 1.1 s.

## Runs with a critic input for each drone, 2026-11-04

4 runs on MetaCentrum, commit `2d15eb5`: base `paths_3v1`, seeds 0 and 1, tanh,
`--set gamma=0.99`, 5e8 steps, two with scaling and two with `--set normalize_obs=false`. They
differ from the 4 tanh runs of Runs with input scaling only by the critic input. Eval: 1024
episodes, mean actions, the last checkpoint, on the same commit. Training is the mean of the
last 100 updates. The std is its final value for each action.

| run | scaling | seed | `caught_e` eval | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | time to catch | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `a2abcb` | on | 0 | 0.099 | 0.019 | 0.007 | 0.009 | 0.87 | 5.4 s | 0.093 | 0.04–0.07 | 0.62 |
| `3849a5` | on | 1 | 0.112 | 0.091 | 0.003 | 0.017 | 0.78 | 5.9 s | 0.109 | 0.07–0.10 | 0.69 |
| `2d0b98` | off | 0 | 0.183 | 0.097 | 0.009 | 0.015 | 0.70 | 6.1 s | 0.173 | 0.04–0.05 | 0.60 |
| `6fce56` | off | 1 | 0.242 | 0.032 | 0.011 | 0.001 | 0.71 | 6.0 s | 0.233 | 0.03 | 0.66 |

- **These are the first 3v1 runs that learn to fly in.** `caught_e` rose from 0.01–0.04 in the
  eval to 0.10–0.24, and pursuer deaths fell. The critic input was the main problem.
- **The spread no longer blows up.** The std falls from 1 to 0.03–0.10, as in 1v1, and the
  adaptive lr stays low. So the blow-up came from the critic's one value for all three.
- **The eval and training now agree**, because the drones fly close to their mean.
- **The pursuers come close, then stay near.** In the 8 saved eval episodes of each run, the
  nearest pursuer goes from 13 m to about 3–5 m by 4–6 s, and most episodes reach the step
  limit. FRPN catches 0.44–0.49, so there is still room.
- **Scaling off catches about twice as often as scaling on**, in both seeds. We do not know
  why. These are two seeds of each.
- **The catch still rises at the end without scaling.** From update 1300 to 1900 it goes from
  0.16 to 0.18 and from 0.22 to 0.24. With scaling it is almost flat, 0.09–0.11.

## Long runs with a critic input for each drone, 2026-11-04

2 runs on MetaCentrum, commit `2d15eb5`: base `paths_3v1`, seeds 0 and 1, tanh,
`--set gamma=0.99 --set normalize_obs=false`, 5e9 steps. That is 10 times the steps of
`2d0b98` and `6fce56`, and it matches Gavin's steps per side. They took 1 h 58 min and 1 h 31
min. Eval: 1024 episodes, mean actions, the last checkpoint, on the same commit. Training is
the mean of the last 100 updates. The std is its final value for each action.

| run | steps | seed | `caught_e` eval | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | time to catch | `distance` | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `2d0b98` | 5e8 | 0 | 0.183 | 0.097 | 0.009 | 0.015 | 0.70 | 6.1 s | 5.2 m | 0.173 | 0.04–0.05 | 0.60 |
| `76ea05` | 5e9 | 0 | 0.184 | 0.028 | 0.002 | 0.001 | 0.79 | 5.9 s | 4.4 m | 0.188 | 0.005–0.008 | 0.63 |
| `6fce56` | 5e8 | 1 | 0.242 | 0.032 | 0.011 | 0.001 | 0.71 | 6.0 s | 4.5 m | 0.233 | 0.03 | 0.66 |
| `4e80e4` | 5e9 | 1 | 0.265 | 0.045 | 0.007 | 0.018 | 0.67 | 6.3 s | 4.3 m | 0.258 | 0.005–0.007 | 0.66 |

The 5e9 runs over training, as the mean of each 5e8 steps up to that mark:

| steps | 5e8 | 1e9 | 2e9 | 3e9 | 4e9 | 5e9 |
|---|---|---|---|---|---|---|
| `caught_e`, s0 | 0.124 | 0.153 | 0.173 | 0.180 | 0.183 | 0.186 |
| `caught_e`, s1 | 0.209 | 0.251 | 0.248 | 0.251 | 0.250 | 0.256 |
| mean std, s0 | 0.265 | 0.033 | 0.012 | 0.008 | 0.007 | 0.006 |
| mean std, s1 | 0.216 | 0.018 | 0.008 | 0.006 | 0.006 | 0.006 |

The catch by what the evader does, from a second flight of the same 1024 episodes. Its totals
match the eval. `hover` means a mean evader speed under 0.5 m/s, 333 episodes. The wall rows
use the evader's closest approach to a face of the box. FRPN flew the pursuers on the same
commit.

| catch rate | `2d0b98` | `76ea05` | `6fce56` | `4e80e4` | FRPN |
|---|---|---|---|---|---|
| all | 0.183 | 0.184 | 0.242 | 0.265 | 0.510 |
| `hover` | 0.18 | 0.10 | 0.25 | 0.23 | 0.59 |
| moving | 0.18 | 0.22 | 0.24 | 0.28 | 0.47 |
| moving, under 3 m from a face | 0.13 | 0.21 | 0.21 | 0.24 | 0.32 |
| moving, over 5 m from a face | 0.25 | 0.26 | 0.31 | 0.31 | 0.55 |

- **10 times the steps did not lift the catch.** The eval moved by 0.00 and 0.02, and the
  noise on one `caught_e` is about ±0.013.
- **The catch goes flat early.** Seed 1 stops at about 1e9 steps, and seed 0 almost stops at
  about 2.5e9. At the end it changes by less than 0.01 per 1e9 steps. At 5e8 steps it is almost
  final, so 5e8 steps are enough for a test.
- **The two seeds stay apart**, at 0.18 and 0.26. Each one stays with its own flight.
- **The std and the lr freeze.** The std keeps falling to 0.006. `lr_p` reaches its floor of
  1e-5 at about 2e9 steps in seed 1 and 3.5e9 in seed 0. With a small std, a small change of
  the mean is a large KL, so the adaptive lr keeps falling. The KL still sits at 0.023–0.027 at
  the floor. The timing fits the flat catch, but it does not prove the cause.
- **The pursuers do not finish, and evader speed is not the limit.** They catch a `hover` no
  more often than a moving evader, and FRPN catches 0.59 of the `hover` paths. In the 8 saved
  eval episodes of each 5e9 run, the nearest net mostly stays 1–5 m away after 4 s.
- **The walls and the hits are not the main cause.** The catch is only a little lower near a
  face, and `coll_pp` stays at 0.03–0.05.
- **In seed 0, fewer hits gave no more catches.** `coll_pp` fell from 0.097 to 0.028, and those
  episodes moved to `time_limit`, from 0.70 to 0.79.
- **FRPN catches 0.51 here.** The older 0.44 and 0.49 came from 256 and 512 episodes.
- **`ent_coef=0.01` came up against the low std, and we dropped it.** It is Gavin's value, and
  it does not transfer: [design_choices.md](../design_choices.md).
- **Next, the user's pick:** `--set gamma=0.995`, seeds 0 and 1, 5e8 steps. Only gamma differs
  from `2d0b98` and `6fce56`. The chase code did not change from `2d15eb5` to `11324cb`: only
  helpers moved.

## Runs with `gamma=0.995`, 2026-11-04

2 runs on MetaCentrum, commit `11324cb`: base `paths_3v1`, seeds 0 and 1, tanh,
`--set gamma=0.995 --set normalize_obs=false`, 5e8 steps. Only gamma differs from `2d0b98` and
`6fce56`. A step is 0.01 s, so the reward horizon $1/(1-\gamma)$ goes from 1 s to 2 s. Eval:
1024 episodes, mean actions, the last checkpoint, on the same commit. Training is the mean of
the last 100 updates. The std is its final value for each action.

| run | gamma | seed | `caught_e` eval | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | time to catch | `distance` | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `2d0b98` | 0.99 | 0 | 0.183 | 0.097 | 0.009 | 0.015 | 0.70 | 6.1 s | 5.2 m | 0.173 | 0.04–0.05 | 0.60 |
| `714794` | 0.995 | 0 | 0.157 | 0.053 | 0.006 | 0.020 | 0.76 | 5.9 s | 4.7 m | 0.172 | 0.02–0.05 | 0.57 |
| `6fce56` | 0.99 | 1 | 0.242 | 0.032 | 0.011 | 0.001 | 0.71 | 6.0 s | 4.5 m | 0.233 | 0.03 | 0.66 |
| `4180cc` | 0.995 | 1 | 0.176 | 0.080 | 0.027 | 0.027 | 0.69 | 5.7 s | 5.0 m | 0.174 | 0.03–0.04 | 0.57 |

The same 1024 episodes, flown again on `11324cb`. The totals match the eval. The groups are
those of Long runs. "Closest net" is the nearest any net comes to the evader in an episode.

| | `2d0b98` | `714794` | `6fce56` | `4180cc` |
|---|---|---|---|---|
| catch, `hover` | 0.18 | 0.14 | 0.25 | 0.20 |
| catch, moving | 0.18 | 0.16 | 0.24 | 0.16 |
| catch, moving, under 3 m from a face | 0.13 | 0.15 | 0.21 | 0.16 |
| catch, moving, over 5 m from a face | 0.25 | 0.25 | 0.31 | 0.21 |
| closest net under 2 m | 0.74 | 0.76 | 0.76 | 0.79 |
| closest net under 1 m | 0.37 | 0.35 | 0.45 | 0.39 |
| catch, of the episodes under 1 m | 0.49 | 0.45 | 0.54 | 0.46 |
| mean net distance after 4 s | 3.9 m | 3.4 m | 2.9 m | 3.4 m |

- **`gamma=0.995` did not lift the catch.** The eval is lower in both seeds, by 0.025 and 0.066,
  and the noise on one `caught_e` is about ±0.012. In training, seed 0 is equal and seed 1 is
  0.06 lower. The two `gamma=0.99` seeds also differ by 0.06, so with two seeds each we cannot
  say that it is worse. We can say that it gave no gain.
- **Both seeds end at the same training catch**, 0.17. Seed 1 stops rising at about 3e8 steps,
  where `6fce56` was still rising at 5e8.
- **The nets come close, and the last part fails.** A net comes under 2 m in 74–79 % of the
  episodes and under 1 m in 35–45 %. Only about half of the episodes under 1 m end in a catch.
  Gamma did not change this.
- **The critic did not get better.** The explained variance is 0.57 in both, below 0.60–0.66.
  The value loss cannot be compared, because the reward scale changes with gamma.
- **A possible reason, not tested.** With `gae_lambda` 0.95, the advantage weighs about 18 steps
  (0.18 s) at gamma 0.995 and 17 steps at gamma 0.99. So a longer horizon reaches the actor only
  through the critic.
- **The saved episodes can differ from the eval rows.** In `714794`, 3 of the 8 saved episodes
  end at another step than the same episode in `episodes.npz`. Eval flies them in a second run
  with the same keys. We think that rounding causes this, but we did not test it.

## Why the learning freezes, 2026-11-05

Checks on the runs above, with no training. The 5e9 runs flew on a copy of their own commit,
`2d15eb5`. The scripts are in `scratch/freeze/` (gitignored).

### The std and the lr over training

Each cell is the mean of the updates within 10 of the column. The std is
$\exp(\text{entropy\_p}/4 - 1.4189)$. `b4511f` is `1v1` seed 0 of 2026-10-28, which has 762
updates.

| update | 100 | 190 | 400 | 762 | 1000 | 3000 | 10000 | 19000 |
|---|---|---|---|---|---|---|---|---|
| std, `76ea05` | 1.01 | 0.92 | 0.42 | 0.14 | 0.10 | 0.029 | 0.008 | 0.006 |
| std, `4e80e4` | 0.97 | 0.81 | 0.31 | 0.11 | 0.070 | 0.016 | 0.006 | 0.006 |
| std, `b4511f` | 0.79 | 0.58 | 0.33 | 0.18 | — | — | — | — |
| `caught_e`, `76ea05` | 0.05 | 0.08 | 0.17 | 0.11 | 0.11 | 0.16 | 0.17 | 0.18 |
| `caught_e`, `4e80e4` | 0.05 | 0.13 | 0.20 | 0.25 | 0.25 | 0.25 | 0.26 | 0.26 |
| `caught_e`, `b4511f` | 0.14 | 0.20 | 0.79 | 0.96 | — | — | — | — |

- **The lr follows the std.** In the four 3v1 runs with the critic input for each drone, lr / std
  stays at 1.1–1.7e-3 from update 190 until the lr reaches its floor. In 1v1 it is 1.4–2.7e-3.
  Adam moves each weight by about the lr, whatever the size of the gradient. The KL measures the
  move of the mean in units of the std. So the KL rule holds the lr at a fixed share of the std.
- **The critic slows down with the actor.** Both are in one optimizer, with one lr.
- **The std falls as about c / update**, with c about 50–100, from update 1000 until the lr
  reaches its floor.
- **1v1 has a similar curve, but it learned in time.** Its catch rose from 0.16–0.23 to 0.87–0.94
  between updates 190 and 600, while its std fell from about 0.6 to 0.25.
- **In seed 1 the catch goes flat earlier than Long runs says.** `4e80e4` has 93 % of its final
  training catch at 2.5e8 steps. `76ea05` has 64 % there, and 89 % at update 3000.

### What the frozen pursuers do

The last checkpoint of each 5e9 run, 1024 eval episodes, mean actions.

- **They fly in a formation behind the evader.** After 6 s, the median net distances, nearest
  first, are 2.6–3.0, 4.2–4.6 and 5.7–6.3 m. From 5 s on, the nearest pursuer closes in at
  0.25 m/s or less. The two closest pursuers stay a median 4.1–4.8 m apart.
- **The nearest does not go in when the others are far.** From 4 to 9 s, with the second net
  7–10 m away, the nearest net stays a median 3.6 m (`4e80e4`) and 4.3 m (`76ea05`) away. With
  the second net 3–4 m away, it stays 2.6 m away.

### Flying on from the frozen states

From the same eval episodes, the pursuers flew on from 4 s in other ways. The flight before 4 s
is the same in every way, because the keys are the same and the flight is deterministic. Only
the episodes in which every drone flies and the nearest net is 2–6 m away count: 779 of 1024 in
`4e80e4` and 773 in `76ea05`.

- "Nearest" is the pursuer whose net is nearest at 4 s. The other two fly the learned policy.
- The blend is action = (1 − β) × learned + β × FRPN. β = 0 is the learned policy.
- The score is the discounted return from 4 s, the mean of the three pursuers, at γ = 0.99. At
  an end that is no catch, each pursuer that still flies gets
  $-\Delta t\,(4 + 0.1\,d)/(1 - \gamma)$, the value of holding at its last net distance $d$.
- Catch and `coll_pp` are the outcomes of the rest of the episode.

Each cell is `4e80e4`, then `76ea05`:

| β | nearest only: score | nearest only: catch | all three: score | all three: catch | all three: `coll_pp` |
|---|---|---|---|---|---|
| 0 (learned) | −4.33, −4.35 | 0.21, 0.13 | −4.33, −4.35 | 0.21, 0.13 | 0.04, 0.03 |
| 0.10 | −4.25, −4.30 | 0.24, 0.16 | −4.22, −4.29 | 0.27, 0.20 | 0.03, 0.04 |
| 0.25 | −4.13, −4.24 | 0.33, 0.19 | −4.02, −4.22 | 0.41, 0.29 | 0.10, 0.10 |
| 0.50 | −3.57, −3.96 | 0.62, 0.36 | −3.97, −4.25 | 0.56, 0.40 | 0.40, 0.41 |
| 0.75 | −2.70, −2.97 | 0.80, 0.70 | −5.16, −5.27 | 0.45, 0.41 | 0.53, 0.59 |
| 1 (FRPN) | −2.32, −2.15 | 0.84, 0.81 | −5.63, −6.36 | 0.44, 0.38 | 0.53, 0.60 |

The score of all three at β = 1, at other settings, from the same flights:

| setting | learned | all three on FRPN |
|---|---|---|
| γ = 0.99, `coll_pp` cost 30 | −4.33, −4.35 | −5.63, −6.36 |
| γ = 0.99, `coll_pp` cost 10 | −4.27, −4.33 | −3.05, −3.35 |
| γ = 0.995, `coll_pp` cost 30 | −8.22, −8.45 | −8.69, −9.92 |
| γ = 0.998, `coll_pp` cost 30 | −19.00, −20.23 | −13.16, −15.02 |

- **The frozen policy is not at a top.** A small step toward FRPN raises the score, for the
  nearest only and for all three. The gain is inside the noise at β = 0.05, about 2.4 standard
  errors at 0.10, and clear at 0.25.
- **The score of the nearest rises up to β = 1.** Its catch rises from 0.21 to 0.84 and from 0.13
  to 0.81.
- **All three have a limit.** Their score is highest at β = 0.25–0.5 and falls after it.
  `coll_pp` jumps from 0.10 to 0.40 between β = 0.25 and 0.50. At β = 1 the score is below the
  learned one. So past this limit, the pursuers need the rule "go in only if I am the nearest".
- **A `coll_pp` cost of 10, or γ = 0.998, removes the fall at β = 1.** γ = 0.995 does not.
- **Limits.** One start time, one direction, mean actions, and the estimated value for the
  pursuers that still fly.

### Why the std falls

`6fce56` was restored from its `latest.pkl`, at update 1907 and a std of 0.03, with its
optimizer state. It then ran 4 updates on a copy of the code that saves the gradient of
`log_std` at each minibatch. Nothing was written into the run.

| epoch | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| gradient of the loss on `log_std` | −0.00089 | +0.00050 | +0.00099 | +0.00140 |

- A positive gradient lowers the std. Each cell is the mean over the 4 actions and the 4 updates,
  and its standard error is at most 0.00028.
- **On new samples, the reward gives almost no pull on the std.** The entropy weight alone gives
  −0.001, and epoch 1 gives −0.00089.
- **The pull down comes in epochs 2–4**, which use the same samples again. Without the entropy
  weight, the mean over the 4 epochs is about +0.0015. This test cannot tell if the cause is the
  fit to the noise of the batch, or the clip.
- **The gradient shows no shared direction.** The odd and the even minibatches gave actor
  gradients with a cosine of −0.26 in epoch 1 and −0.65 in epochs 2–4. A negative cosine means
  that the steps undo each other. We guess that the clip pulls the policy back, but we did not
  test it.
- **Limits.** One run, 4 updates, one std.

### What this means

- **The policy stops moving while its score can still rise.** The std and the lr fall together,
  and the critic's lr falls with them. A model, not measured: the signal in the gradient scales
  with the std, and the step with the lr, so the progress per update scales with 1 / update². By
  this model, updates 1000–19000 add about as much as updates 500–1000. So more steps cannot help.
- **The `gamma=0.995` runs fit this.** Their std fell to 0.02–0.05, with lr / std at 1.1–1.6e-3.
- **A second limit sits higher up.** A shared step past β = 0.25–0.5 meets `coll_pp`.
- **This goes against a decision.** [design_choices.md](../design_choices.md) says that PPO
  shrinks the std once precision pays. At a std of 0.03, the reward gives no such pull.

## Runs with `ent_coef=0.003`, 2026-11-05

2 runs on MetaCentrum, commit `11324cb`: base `paths_3v1`, seeds 0 and 1, tanh,
`--set gamma=0.99 --set normalize_obs=false --set ent_coef=0.003`, 5e8 steps. Only `ent_coef`
differs from `2d0b98` and `6fce56`. Eval: 1024 episodes, mean actions, the last checkpoint, on
`d05c85d`, which differs from `11324cb` only in `random_goals.py`. Training is the mean of the
last 100 updates. The std is its final value for each action.

- **Why.** In Why the learning freezes, the pull down on the std is about 0.0015, and the entropy
  weight of 0.001 is smaller. A weight of 0.003 was to hold the std up, so that the policy keeps
  moving.

| run | `ent_coef` | seed | `caught_e` eval | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | time to catch | `distance` | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `2d0b98` | 0.001 | 0 | 0.183 | 0.097 | 0.009 | 0.015 | 0.70 | 6.1 s | 5.2 m | 0.173 | 0.04–0.05 | 0.60 |
| `2cda6a` | 0.003 | 0 | 0.006 | 0.282 | 0.479 | 0.000 | 0.23 | 2.1 s | 9.0 m | 0.040 | 328–18010 | 0.64 |
| `6fce56` | 0.001 | 1 | 0.242 | 0.032 | 0.011 | 0.001 | 0.71 | 6.0 s | 4.5 m | 0.233 | 0.03 | 0.66 |
| `503bac` | 0.003 | 1 | 0.027 | 0.228 | 0.281 | 0.005 | 0.46 | 4.1 s | 8.0 m | 0.047 | 124–679 | 0.62 |

The mean std and the training catch over training:

| update | 100 | 190 | 400 | 600 | 1000 | 1500 | 1890 |
|---|---|---|---|---|---|---|---|
| std, `2cda6a` | 1.35 | 1.78 | 3.2 | 5.9 | 31 | 213 | 1178 |
| std, `503bac` | 1.38 | 1.96 | 3.0 | 3.3 | 5.2 | 41 | 198 |
| `caught_e`, `2cda6a` | 0.05 | 0.06 | 0.09 | 0.14 | 0.05 | 0.05 | 0.03 |
| `caught_e`, `503bac` | 0.05 | 0.07 | 0.11 | 0.21 | 0.14 | 0.04 | 0.05 |

- **The std grew from the first updates, in both seeds, and blew up.** The adaptive lr rose to its
  cap of 0.01 at about update 1000 and 1300.
- **Up to update 600, the catch kept pace.** At a std of 3–6, the training catch was 0.14 and
  0.21. The runs with 0.001 had 0.12–0.16 (seed 0) and 0.18–0.22 (seed 1) at the same update.
  Then the catch fell to 0.04–0.05.
- **The guess missed a known fact.** Past ±1, the clip hides the noise, so the reward does not
  push a large std down (2026-11-01). In the FRPN test, more noise past σ = 1 even paid in 3v1.
  The entropy weight is a constant push up, and the pull down looked constant too. So the std
  either falls for ever (0.001) or grows for ever (0.003), and no `ent_coef` holds it. This rests
  on two values.
- **So this batch does not test the freeze.** The std never stayed at 0.1–0.5.
- **Next, proposed, not yet picked:** hold the std at 0.3, with a `stop_gradient` on `log_std` and
  `--set init_std=0.3`. In the FRPN test, a σ of 0.3 cost little (catch 0.49 to 0.47), and 1 cost
  a lot (0.19).

## Runs with a `coll_pp` cost of 10, 2026-11-10

2 runs on MetaCentrum, commit `c8d89d5`: base `paths_3v1`, seeds 0 and 1, tanh,
`--set gamma=0.99 --set normalize_obs=false`, 5e8 steps. A `coll_pp` costs 10, not 30 (commit
`9f50d40`). Only the cost differs from `2d0b98` and `6fce56`: the rest of the `swarm/` diff from
`2d15eb5` moves code into helpers. Eval: 1024 episodes, mean actions, the last checkpoint, on
`3642bd8`, which has the same `swarm/` as `c8d89d5`. Training is the mean of the last 100
updates. The std is its final value for each action.

- **Why.** In Flying on from the frozen states, all three pursuers on FRPN score −3.05 and −3.35
  at a cost of 10, above the learned −4.27 and −4.33. At 30 they score below the learned policy.

| run | cost | seed | `caught_e` eval | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | time to catch | `distance` | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `2d0b98` | 30 | 0 | 0.183 | 0.097 | 0.009 | 0.015 | 0.70 | 6.1 s | 5.2 m | 0.173 | 0.04–0.05 | 0.60 |
| `a573b5` | 10 | 0 | 0.250 | 0.176 | 0.017 | 0.034 | 0.52 | 6.1 s | 5.0 m | 0.238 | 0.02–0.03 | 0.68 |
| `6fce56` | 30 | 1 | 0.242 | 0.032 | 0.011 | 0.001 | 0.71 | 6.0 s | 4.5 m | 0.233 | 0.03 | 0.66 |
| `052a62` | 10 | 1 | 0.166 | 0.468 | 0.011 | 0.012 | 0.34 | 5.5 s | 5.6 m | 0.180 | 0.03–0.04 | 0.58 |

Training over the updates, the mean of each range:

| updates | 1–100 | 101–200 | 201–500 | 501–1000 | 1001–1907 |
|---|---|---|---|---|---|
| `caught_e`, `2d0b98` (30) | 0.04 | 0.07 | 0.10 | 0.13 | 0.17 |
| `caught_e`, `a573b5` (10) | 0.06 | 0.16 | 0.21 | 0.24 | 0.23 |
| `caught_e`, `6fce56` (30) | 0.04 | 0.09 | 0.16 | 0.18 | 0.22 |
| `caught_e`, `052a62` (10) | 0.05 | 0.14 | 0.17 | 0.18 | 0.18 |
| `coll_pp`, `2d0b98` (30) | 0.18 | 0.07 | 0.06 | 0.06 | 0.09 |
| `coll_pp`, `a573b5` (10) | 0.27 | 0.42 | 0.33 | 0.18 | 0.15 |
| `coll_pp`, `6fce56` (30) | 0.15 | 0.05 | 0.03 | 0.03 | 0.04 |
| `coll_pp`, `052a62` (10) | 0.28 | 0.37 | 0.49 | 0.47 | 0.49 |

- **The catch did not rise.** The mean eval `caught_e` is 0.208 at a cost of 10 and 0.212 at 30.
  Seed 0 rose by 0.067 and seed 1 fell by 0.076. The noise on one `caught_e` is about ±0.013.
- **The hits rose.** Eval `coll_pp` is 0.176 and 0.468, against 0.097 and 0.032.
- **The hits took the place of `time_limit`.** It fell from 0.70–0.71 to 0.34–0.52. `crash_p` and
  `OOB_p` stay at 0.001–0.034.
- **At 30, the pursuers first learn to stay apart. At 10 they do not.** In updates 101–200,
  `coll_pp` is 0.05–0.07 at 30 and 0.37–0.42 at 10.
- **At 10, the catch rises faster, then goes flat at the same level.** In updates 101–200 the
  training catch is 0.14–0.16 at 10 and 0.07–0.09 at 30. In updates 1001–1907 it is 0.18–0.23 at
  10 and 0.17–0.22 at 30.
- **The std falls as at 30.** It ends at 0.02–0.04, against 0.03–0.05.
- **Decision, with the user:** the `coll_pp` cost stays at 10, and the pursuer gets a new
  reward ([The progress reward](#the-progress-reward-2026-11-10), below).

## The progress reward, 2026-11-10

- **The pursuer earns progress in place of the clock and the distance cost.** It earns 0.25
  for each metre that its own net closes on the evader. The user picked the shape and the
  weight. Sánchez Roncero pays the same 0.25 per metre, and Němeček 0.5.
- **Why.** At γ = 0.99, the rest of the episode was worth about −5 to a pursuer that lives on
  ([The coll_pp cost](#the-coll_pp-cost-2026-10-28)), so a death also ended its costs. With
  progress, a pursuer that lives on can still earn. Geles et al. 2026 pay a racer the same
  way: 1 for each metre it closes on the next gate, and no clock.
- **Each pursuer's own net, not the team's.** The user picked it.
- **A catch now pays 30, not 10.** The user picked it. The evader pays the same 30.
- **The rest stays:** `coll_pp` 10, fail 30, `coll_pe` 10 per s, cmd 0.02 per rad/s. The
  evader keeps its clock.
- **The catch was already shared.** One catch pays every pursuer the same, so it is the
  captor and helper reward of de Souza et al. 2021 with captor = helper.

## Runs with the progress reward, 2026-11-10

2 runs on MetaCentrum, commit `18a4c40`: base `paths_3v1`, seeds 0 and 1, tanh,
`--set gamma=0.99 --set normalize_obs=false`, 5e8 steps. The reward is the one of
[The progress reward](#the-progress-reward-2026-11-10). The rest is as in `a573b5` and
`052a62`. Eval: 1024 episodes, mean actions, the last checkpoint, on `79e1f0e` with the eval
code check removed. On seed 0 it gives the same summary as `18a4c40`. Training is the mean of
the last 100 updates. The std is its final value for each action.

| run | reward | seed | `caught_e` eval | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | time to catch | `distance` | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `a573b5` | cost 10 | 0 | 0.250 | 0.176 | 0.017 | 0.034 | 0.52 | 6.1 s | 5.0 m | 0.238 | 0.02–0.03 | 0.68 |
| `c6de8e` | progress | 0 | 0.194 | 0.561 | 0.024 | 0.014 | 0.21 | 4.6 s | 5.3 m | 0.202 | 0.018–0.022 | 0.55 |
| `052a62` | cost 10 | 1 | 0.166 | 0.468 | 0.011 | 0.012 | 0.34 | 5.5 s | 5.6 m | 0.180 | 0.03–0.04 | 0.58 |
| `398c2a` | progress | 1 | 0.255 | 0.325 | 0.026 | 0.037 | 0.36 | 5.4 s | 4.8 m | 0.258 | 0.016–0.019 | 0.47 |

Training over the updates, the mean of each range:

| updates | 1–100 | 101–200 | 201–500 | 501–1000 | 1001–1907 |
|---|---|---|---|---|---|
| std, `c6de8e` | 0.92 | 0.75 | 0.36 | 0.086 | 0.030 |
| std, `398c2a` | 0.92 | 0.71 | 0.33 | 0.080 | 0.027 |
| `caught_e`, `c6de8e` | 0.09 | 0.16 | 0.20 | 0.21 | 0.21 |
| `caught_e`, `398c2a` | 0.09 | 0.18 | 0.22 | 0.25 | 0.26 |
| `coll_pp`, `c6de8e` | 0.30 | 0.40 | 0.39 | 0.51 | 0.53 |
| `coll_pp`, `398c2a` | 0.34 | 0.53 | 0.34 | 0.36 | 0.34 |

- The highest std is 1.00 and 1.01, at updates 2 and 3. At the end it is 0.020 and 0.018,
  and `lr_p` is 4.4e-5 and 3.9e-5.
- Over updates 1001–1907, a line fit to the training catch moves by −0.004 and 0.000 per
  1000 updates. In `a573b5`, `052a62`, `2d0b98` and `6fce56` it moves by +0.004 to +0.031.

### The behaviour, from 4096 episodes

A script flew 4096 eval episodes of each run, with the eval seeds and mean actions, and
saved every step. The cost-10 runs flew on `c8d89d5` and the cost-30 runs on `2d15eb5`, their
own commits. FRPN flew the pursuers in the same episodes. The script runs 256 episodes at a
time, and the eval 1024. That changes the outcome of 13 of the first 1024 episodes of
`c6de8e`, and its catch by 0.003.

The outcomes, with the 95 % Wilson interval of the catch:

| run | reward | `caught_e` | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | median time to catch |
|---|---|---|---|---|---|---|---|
| `c6de8e` | progress | 0.212 [0.200, 0.225] | 0.537 | 0.031 | 0.013 | 0.207 | 4.2 s |
| `398c2a` | progress | 0.253 [0.240, 0.267] | 0.324 | 0.030 | 0.034 | 0.358 | 5.2 s |
| `a573b5` | cost 10 | 0.245 [0.232, 0.258] | 0.155 | 0.013 | 0.031 | 0.556 | 6.5 s |
| `052a62` | cost 10 | 0.177 [0.166, 0.189] | 0.459 | 0.010 | 0.010 | 0.345 | 5.5 s |
| `2d0b98` | cost 30 | 0.175 [0.164, 0.187] | 0.093 | 0.009 | 0.010 | 0.712 | 6.4 s |
| `6fce56` | cost 30 | 0.243 [0.230, 0.256] | 0.044 | 0.013 | 0.005 | 0.695 | 6.0 s |
| FRPN | — | 0.524 [0.509, 0.539] | 0.322 | 0.012 | 0.142 | 0.000 | 1.5 s |

- The two seeds of one reward differ by 4.4–7.6 standard errors in `caught_e`, and by
  8.9–31.6 in `coll_pp`.
- A one-way ANOVA over the three rewards, with the two seeds as the repeats: `caught_e`
  F(2, 3) = 0.19, p = 0.83. `coll_pp` F = 2.90, p = 0.20. `time_limit` F = 7.97, p = 0.06.

The catch by the evader's path. Each path has 1344–1386 episodes. The evader's mean speed
is 0.02 m/s on the hover, 4.27 on the circle and 4.42 on the figure-eight.

| path | `c6de8e` | `398c2a` | `a573b5` | `052a62` | `2d0b98` | `6fce56` | FRPN |
|---|---|---|---|---|---|---|---|
| hover | 0.210 | 0.260 | 0.224 | 0.170 | 0.156 | 0.219 | 0.603 |
| circle | 0.211 | 0.225 | 0.228 | 0.184 | 0.174 | 0.225 | 0.559 |
| figure-eight | 0.216 | 0.275 | 0.282 | 0.177 | 0.195 | 0.284 | 0.413 |

The distance from the evader to each net, nearest / second / third, the median over the
episodes still running (m). Only 83 FRPN episodes run past 4 s, and 1 past 8 s.

| time | 1 s | 2 s | 4 s | 8 s |
|---|---|---|---|---|
| `c6de8e` | 6.5 / 10.1 / 14.1 | 3.4 / 5.1 / 7.1 | 3.3 / 4.9 / 6.7 | 3.4 / 4.9 / 6.6 |
| `398c2a` | 7.0 / 10.6 / 14.7 | 4.0 / 6.0 / 8.0 | 3.0 / 4.6 / 6.2 | 2.8 / 4.3 / 5.7 |
| the four older runs | 7.5–7.8 / 11.5–11.9 / 15.9–16.4 | 4.9–5.3 / 7.1–8.5 / 9.4–11.7 | 3.3–4.1 / 5.2–6.4 / 7.0–8.6 | 2.8–3.7 / 4.5–5.8 / 6.2–7.6 |
| FRPN | 4.7 / 9.3 / 14.6 | 2.2 / 4.3 / 7.0 | — | — |

The median closing speed of a net on the evader, by its distance (m/s):

| distance | 0–1 m | 1–2 m | 2–3 m | 3–5 m | 5–8 m | 8–12 m | 12–40 m |
|---|---|---|---|---|---|---|---|
| `c6de8e` | 0.88 | 0.34 | 0.28 | 0.42 | 0.85 | 2.77 | 5.96 |
| `398c2a` | 0.79 | 0.26 | 0.22 | 0.32 | 0.75 | 2.82 | 5.61 |
| the four older runs | 0.56–0.89 | 0.17–0.27 | 0.15–0.23 | 0.21–0.33 | 0.43–0.65 | 1.20–1.89 | 3.29–4.54 |
| FRPN | 4.15 | 4.79 | 5.82 | 6.63 | 7.70 | 8.19 | 6.21 |

The motion of a pursuer relative to the evader, while its net is 2–5 m away. Medians, from
every 10th step. The speed of the pursuers is over all their steps.

| run | toward the evader | around it | turn rate around it | around it, hover path | pursuer speed, median / top 10 % |
|---|---|---|---|---|---|
| `c6de8e` | 0.45 m/s | 3.72 m/s | 1.08 rad/s | 3.45 m/s | 5.43 / 9.98 m/s |
| `398c2a` | 0.34 | 2.97 | 0.86 | 2.59 | 4.32 / 8.05 |
| `a573b5` | 0.37 | 2.61 | 0.73 | 2.27 | 3.82 / 6.64 |
| `052a62` | 0.34 | 3.75 | 1.08 | 3.40 | 5.20 / 8.70 |
| `2d0b98` | 0.26 | 3.24 | 0.87 | 2.84 | 4.34 / 7.12 |
| `6fce56` | 0.29 | 2.07 | 0.62 | 1.67 | 3.42 / 6.20 |
| FRPN | 6.60 | 2.78 | 0.90 | 2.01 | 9.34 / 12.70 |

The `coll_pp`, at the step of the hit. Rank 1 is the pursuer whose net is nearest to the
evader. "Same way" is the share of pairs that turned around the evader in the same direction
0.5 s before the hit. "Base" is the same share for any two pursuers with their nets 2–5 m
away.

| run | hits | body on body | ranks 1+2 / 2+3 / 1+3 | nearest net to the evader, median | under 2 m | pair within 2 m before, median | pair speed, median | same way | base |
|---|---|---|---|---|---|---|---|---|---|
| `c6de8e` | 2200 | 0.47 | 0.57 / 0.35 / 0.08 | 3.60 m | 0.14 | 0.21 s | 5.22 m/s | 0.62 | 0.57 |
| `398c2a` | 1328 | 0.44 | 0.52 / 0.40 / 0.08 | 3.26 m | 0.17 | 0.27 s | 4.17 m/s | 0.65 | 0.57 |
| `a573b5` | 633 | 0.39 | 0.57 / 0.37 / 0.06 | 3.22 m | 0.20 | 0.29 s | 3.99 m/s | 0.68 | 0.53 |
| `052a62` | 1879 | 0.53 | 0.58 / 0.36 / 0.06 | 3.90 m | 0.10 | 0.17 s | 6.07 m/s | 0.60 | 0.53 |
| `2d0b98` | 382 | 0.30 | 0.50 / 0.44 / 0.06 | 4.05 m | 0.08 | 0.20 s | 4.57 m/s | 0.54 | 0.49 |
| `6fce56` | 179 | 0.30 | 0.64 / 0.32 / 0.04 | 3.09 m | 0.22 | 0.38 s | 3.40 m/s | 0.72 | 0.54 |
| FRPN | 1320 | 0.66 | 0.85 / 0.11 / 0.04 | 1.32 m | 0.74 | 0.12 s | 11.92 m/s | 0.62 | 0.61 |

Per second of the episodes still running, a `coll_pp` comes at 0.08–0.12 in `c6de8e` and at
0.02–0.07 in `398c2a`, in each bin from 1 s to 10 s.

The episodes with no catch. "Closest" is the median of each episode's closest net. The off-net
columns are shares of the episodes with a net within 1 m, at its closest step. The net square
needs the evader within 0.40 m across it, 0.5 m sideways and 0.5 m up or down. The height is
that of the nearest pursuer over the evader, at all steps with its net within 3 m, the catches
included. A level pursuer's net centre hangs 0.9 m under it. A body touch is centres within
0.8 m, before the end of the episode.

| run | closest | a net within 1 m | closing at the closest step | off across / sideways / up-down | height q10 / q50 / q90 | a body touch |
|---|---|---|---|---|---|---|
| `c6de8e` | 1.44 m | 0.25 | 0.03 m/s | 0.52 / 0.30 / 0.45 | −1.36 / 0.19 / 1.89 m | 0.187 |
| `398c2a` | 1.24 m | 0.33 | 0.15 m/s | 0.43 / 0.35 / 0.48 | −1.13 / 0.37 / 1.99 m | 0.247 |
| `a573b5` | 1.31 m | 0.30 | 0.09 m/s | 0.45 / 0.37 / 0.47 | −1.03 / 0.49 / 2.01 m | 0.160 |
| `052a62` | 1.47 m | 0.25 | −0.02 m/s | 0.54 / 0.23 / 0.47 | −1.17 / 0.33 / 2.01 m | 0.200 |
| `2d0b98` | 1.49 m | 0.23 | 0.00 m/s | 0.44 / 0.33 / 0.49 | −1.29 / 0.35 / 2.14 m | 0.149 |
| `6fce56` | 1.42 m | 0.26 | 0.05 m/s | 0.41 / 0.36 / 0.48 | −1.28 / 0.12 / 1.78 m | 0.232 |
| FRPN | 0.90 m | 0.62 | 1.21 m/s | 0.37 / 0.30 / 0.55 | −0.21 / 0.60 / 1.39 m | 0.592 |

The angle at the evader between the two nearest pursuers, at steps with both nets within
4 m: median 74° and 86° in the new runs, 82–101° in the four older runs and 78° for FRPN.
Under 30°: 0.09 and 0.06, 0.02–0.07, and 0.09.

- **The std did not blow up.** It fell from 1.0 to 0.016–0.022, and `lr_p` to 3.9–4.4e-5.
  Over updates 1001–1907 the training catch was flat.
- **The catch did not change with the reward.** In 4096 episodes it is 0.212 and 0.253,
  against 0.175–0.245 in the four older runs. The ANOVA gives p = 0.83. The two seeds of each
  reward differ by 4.4–7.6 standard errors.
- **The catch came sooner:** a median of 4.2 and 5.2 s, against 5.5–6.5 s.
- **`coll_pp` is 0.32–0.54,** against 0.16–0.46 at cost 10 and 0.04–0.09 at cost 30.
  `time_limit` fell to 0.21–0.36.
- **Far from the evader, the new pursuers close faster.** Beyond 12 m they close at 5.6–6.0
  m/s, against 3.3–4.5. At 2 s the nearest net is 3.4–4.0 m away, against 4.9–5.3 m.
- **At 1–5 m, all six policies close at 0.15–0.42 m/s.** FRPN closes at 4.8–6.6 m/s there.
- **All six policies hold their nets on a ring around the evader.** At 4 s and 8 s the nets
  are 2.8–4.1, 4.3–6.4 and 5.7–8.6 m away. With a net 2–5 m away, the pursuer moves at
  2.1–3.8 m/s around the evader and at 0.26–0.45 m/s toward it. On the hover path it moves
  at 1.7–3.5 m/s around it.
- **The learned catch does not depend on the path:** hover 0.16–0.26, circle 0.17–0.23,
  figure-eight 0.18–0.28. FRPN catches 0.60 of the hovers and 0.41 of the figure-eights.
- **The close passes do not close.** In 23–33 % of the episodes with no catch, a net came
  within 1 m. At its closest step it closed at −0.02 to 0.15 m/s, and the evader was off the
  net in each direction in 23–54 % of those episodes.
- **The `coll_pp` come on the ring.** The nearest net is a median 3.1–4.1 m from the evader.
  The second pursuer is in 92–96 % of the pairs. A pair is within 2 m for a median 0.17–0.38 s
  before the hit. In 47–70 % of the hits a net touches a pursuer. In every run, the share of
  pairs that turned the same way (0.54–0.72) is higher than the base (0.49–0.57).
- **The two nearest pursuers come from different sides,** at a median 74–101°.
- **Decision, with the user:** next, a body catch, as the preset `paths_3v1_body`
  ([The body catch](#the-body-catch-2026-11-10), below), one run. The user rejected a 3v1
  start from a trained 1v1 pursuer: its three copies would all fly in and hit each other,
  and the training would bring them back to the ring.

## The body catch, 2026-11-10

- **`paths_3v1_body` is `paths_3v1` with a body catch.** A pursuer that touches the evader,
  centres within 0.8 m, catches it. The pursuers carry no net, so a `coll_pp` is a touch of
  two bodies. A touch of the evader costs nothing ($\lambda_{\text{PE}}$ = 0), and the progress
  counts the distance from the evader to the pursuer's centre.
- **A new preset, not a change of the task.** The user picked it, so the net task stays.
- **Why.** The user expects the task to be easier without a net. In the runs above, the close
  passes missed the net square in each of its three directions, and in 47–70 % of the
  `coll_pp` a net touched a pursuer.
- **One run, seed 0.** The user submits it, to see if the catch rises clearly.

## Probes on the trained policies, 2026-11-10

Three probes on the last checkpoints, with no new training: 4096 episodes, the eval seeds,
mean actions, on `79e1f0e` with the presets `paths_3v1_body` and `paths_3v1_netpass` added.

### The net kill

`run/eval.py` flew each policy in `paths_3v1` and in `paths_3v1_netpass`, with the same keys.
So each episode is the same in both up to its first net touch of a pursuer. FRPN flew the
pursuers in the same episodes. "Saved" are the `coll_pp` episodes of `paths_3v1` that end
in another way in `paths_3v1_netpass`.

| policy | catch | `coll_pp` | `time_limit` | episodes to a catch / away from it | McNemar p | saved | saved, then catch / `time_limit` / `crash_p` / `OOB_p` | still `coll_pp` |
|---|---|---|---|---|---|---|---|---|
| `c6de8e` | 0.213 → 0.248 | 0.538 → 0.385 | 0.206 → 0.321 | 140 / 0 | 2.7e-32 | 627 | 0.223 / 0.753 / 0.019 / 0.005 | 1575 |
| `398c2a` | 0.252 → 0.280 | 0.326 → 0.213 | 0.357 → 0.441 | 113 / 0 | 2.2e-26 | 464 | 0.244 / 0.737 / 0.006 / 0.013 | 873 |
| FRPN | 0.524 → 0.562 | 0.322 → 0.276 | 0.000 → 0.001 | 154 / 0 | 2.3e-35 | 191 | 0.806 / 0.010 / 0.031 / 0.152 | 1129 |

### Ghost teammates

In each pursuer's actor input, the offsets of the two teammate slots moved to 15 m, along
the same direction. Their velocities stayed. "All" changes the input of all three pursuers,
"nearest" only of the pursuer nearest to the evader. The real teammates still flew.

| policy | input | catch | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | nets at 4 s | closing at 1–2 / 2–3 / 3–5 m | closest net, no catch |
|---|---|---|---|---|---|---|---|---|---|
| `c6de8e` | real | 0.212 | 0.537 | 0.031 | 0.013 | 0.207 | 3.3 / 4.9 / 6.7 m | 0.34 / 0.28 / 0.42 m/s | 1.44 m |
| `c6de8e` | all | 0.188 | 0.533 | 0.100 | 0.030 | 0.150 | 3.5 / 5.2 / 7.1 m | 0.38 / 0.40 / 0.56 m/s | 1.50 m |
| `c6de8e` | nearest | 0.197 | 0.552 | 0.045 | 0.017 | 0.190 | 3.5 / 5.2 / 7.0 m | 0.39 / 0.34 / 0.50 m/s | 1.49 m |
| `398c2a` | real | 0.253 | 0.324 | 0.030 | 0.034 | 0.358 | 3.0 / 4.6 / 6.2 m | 0.26 / 0.22 / 0.32 m/s | 1.24 m |
| `398c2a` | all | 0.197 | 0.311 | 0.129 | 0.082 | 0.282 | 3.4 / 4.9 / 6.7 m | 0.34 / 0.30 / 0.45 m/s | 1.47 m |
| `398c2a` | nearest | 0.240 | 0.342 | 0.057 | 0.047 | 0.314 | 3.1 / 4.7 / 6.3 m | 0.37 / 0.37 / 0.47 m/s | 1.29 m |

### The payoff of closing in

From the 4096-episode records of "Runs with the progress reward". A sample is a pursuer at
every 10th step up to 6 s, alive, with its net 2–5 m from the evader, in an episode that
goes on past that step. G is its own discounted return from that step, at γ = 0.99, with the
run's own reward. Its standard error is 0.01–0.04. "Within 1 s" means that the episode ends
within 100 steps: by a catch in this pursuer's net, or by a `coll_pp` with this pursuer in
the pair.

G, by the pursuer's closing speed on the evader (m/s):

| run | < 0 | 0–0.5 | 0.5–1 | 1–2 | 2–4 | > 4 |
|---|---|---|---|---|---|---|
| `c6de8e` | 0.04 | 0.31 | 0.36 | 0.47 | 0.67 | 1.34 |
| `398c2a` | 0.37 | 0.63 | 0.67 | 0.69 | 1.01 | 1.71 |
| `a573b5` | −4.19 | −4.13 | −4.11 | −4.07 | −3.95 | −3.65 |
| `052a62` | −4.30 | −4.31 | −4.24 | −4.18 | −4.20 | −3.95 |
| `2d0b98` | −4.47 | −4.42 | −4.40 | −4.34 | −4.25 | −3.94 |
| `6fce56` | −4.09 | −4.10 | −4.12 | −4.07 | −3.94 | −3.54 |

This pursuer catches within 1 s:

| run | < 0 | 0–0.5 | 0.5–1 | 1–2 | 2–4 | > 4 |
|---|---|---|---|---|---|---|
| `c6de8e` | 0.004 | 0.009 | 0.011 | 0.017 | 0.027 | 0.056 |
| `398c2a` | 0.003 | 0.006 | 0.008 | 0.015 | 0.031 | 0.064 |
| `a573b5` | 0.004 | 0.007 | 0.011 | 0.017 | 0.035 | 0.063 |
| `052a62` | 0.002 | 0.005 | 0.009 | 0.014 | 0.024 | 0.050 |
| `2d0b98` | 0.003 | 0.006 | 0.008 | 0.012 | 0.022 | 0.056 |
| `6fce56` | 0.003 | 0.006 | 0.008 | 0.014 | 0.029 | 0.072 |

This pursuer is in a `coll_pp` within 1 s:

| run | < 0 | 0–0.5 | 0.5–1 | 1–2 | 2–4 | > 4 |
|---|---|---|---|---|---|---|
| `c6de8e` | 0.067 | 0.069 | 0.072 | 0.076 | 0.081 | 0.082 |
| `398c2a` | 0.039 | 0.036 | 0.036 | 0.036 | 0.037 | 0.034 |
| `a573b5` | 0.016 | 0.015 | 0.015 | 0.017 | 0.018 | 0.019 |
| `052a62` | 0.050 | 0.055 | 0.057 | 0.056 | 0.062 | 0.068 |
| `2d0b98` | 0.006 | 0.007 | 0.005 | 0.006 | 0.006 | 0.007 |
| `6fce56` | 0.004 | 0.005 | 0.005 | 0.005 | 0.006 | 0.005 |

On the hover path alone, G goes from < 0 to > 4 m/s as 0.07 → 1.20 (`c6de8e`), 0.54 → 1.72
(`398c2a`), −4.22 → −3.85 (`a573b5`), −4.27 → −4.01 (`052a62`), −4.46 → −4.00 (`2d0b98`) and
−4.12 → −3.79 (`6fce56`).

- **Without the net kill, the same policies catch more,** by 0.034 and 0.028, and FRPN by
  0.038. Every changed episode moved toward a catch.
- **Most saved episodes of the learned policies run to the time limit:** 0.753 and 0.737 of
  them, against 0.010 for FRPN. 1575 and 873 `coll_pp` episodes still end in a `coll_pp`.
- **With ghost teammates, the ring stays.** The nearest net at 4 s is 3.1–3.5 m away, against
  3.0 and 3.3 m. At 1–5 m the nets close at 0.30–0.56 m/s, against 0.22–0.42. The catch is
  0.188–0.240, against 0.212 and 0.253. With "all", `crash_p` rises to 0.100 and 0.129.
- **From the ring, a pursuer that closes at > 4 m/s has a higher G than one that moves away,**
  in all six runs, by 0.35–1.34. On the hover path alone, by 0.26–1.18. In `052a62` and
  `6fce56`, G does not rise over the bands below 1 m/s.
- **Its own catch within 1 s is 14–25 times higher** at > 4 m/s than at < 0. Its own
  `coll_pp` within 1 s differs by −0.005 to +0.018.
- **Limits.** The payoff tables compare states inside the flights of each policy, with no
  change of the action. The ghost input shows teammates at 15 m together with the evader at
  about 3 m.

## Four runs, 2026-11-10

- **The user picked four seed-0 runs,** each with `--set gamma=0.99 --set normalize_obs=false`
  and 5e8 steps:

| | adaptive lr | lr 5e-4, constant |
|---|---|---|
| `paths_3v1` | `c6de8e` (done) | new |
| `paths_3v1_netpass` | new | new |

- **`paths_3v1_body`, seed 0, adaptive lr,** as in The body catch.
- **`paths_3v1_netpass`:** a net that touches a pursuer is no `coll_pp`. A touch of two bodies
  still is. The probes above measure the current policies only, so this tests what training
  under the rule does.
- **Why a constant lr.** Gavin et al. 2026 (1v1) train with a constant lr of 5e-4 (Table II).
  Our adaptive lr fell to 3.9–4.4e-5 in `c6de8e` and `398c2a`. Gavin's reward (Nv1, Table I)
  is our reward before the progress reward, term by term. His training differs from ours in
  the lr, a tanh squash of the action, an entropy weight of 0.01, and 15 × 1 epochs and
  minibatches, against our 4 × 32.

## Runs with a constant lr and netpass, 2026-11-10

3 runs on MetaCentrum, commit `fba6aec`, seed 0, tanh, `--set gamma=0.99 --set
normalize_obs=false`, 5e8 steps, the progress reward:

| run | preset | lr |
|---|---|---|
| `c6de8e` (before) | `paths_3v1` | adaptive |
| `be848f` | `paths_3v1` | 5e-4, constant (`--set adaptive_lr=false --set lr=5e-4`) |
| `4f3c3b` | `paths_3v1_netpass` | adaptive |
| `d11920` | `paths_3v1_netpass` | 5e-4, constant |

Eval on `fba6aec`: 1024 episodes, mean actions, the run's own preset, the last checkpoint
unless a table says otherwise. The behaviour tables come from 4096 episodes, as in "Runs with
the progress reward". The body run was not submitted. The two attention runs (`b087ac`,
`da9e33`) are in [The attention network](#the-attention-network-2026-11-11).

The training catch and the std, the mean of each 200 updates:

| updates | 1–200 | 201–400 | 401–600 | 601–800 | 801–1000 | 1001–1200 | 1201–1400 | 1401–1600 | 1601–1800 | 1801–1907 |
|---|---|---|---|---|---|---|---|---|---|---|
| catch, `c6de8e` | 0.13 | 0.19 | 0.21 | 0.22 | 0.21 | 0.21 | 0.21 | 0.20 | 0.21 | 0.20 |
| catch, `be848f` | 0.24 | 0.47 | 0.47 | 0.45 | 0.47 | 0.55 | 0.58 | 0.56 | 0.51 | 0.50 |
| catch, `4f3c3b` | 0.15 | 0.29 | 0.35 | 0.35 | 0.36 | 0.36 | 0.37 | 0.37 | 0.37 | 0.37 |
| catch, `d11920` | 0.30 | 0.57 | 0.61 | 0.62 | 0.62 | 0.60 | 0.57 | 0.57 | 0.56 | 0.53 |
| std, `c6de8e` | 0.835 | 0.434 | 0.174 | 0.089 | 0.055 | 0.041 | 0.033 | 0.027 | 0.023 | 0.020 |
| std, `be848f` | 0.669 | 0.256 | 0.136 | 0.094 | 0.078 | 0.066 | 0.059 | 0.054 | 0.051 | 0.050 |
| std, `4f3c3b` | 0.816 | 0.370 | 0.154 | 0.081 | 0.050 | 0.035 | 0.027 | 0.022 | 0.019 | 0.017 |
| std, `d11920` | 0.677 | 0.276 | 0.143 | 0.093 | 0.073 | 0.063 | 0.056 | 0.052 | 0.051 | 0.050 |

Over updates 1001–1907, the mean of:

| run | `lr_p` | `approx_kl_p` | `clipfrac_p` | explained variance |
|---|---|---|---|---|
| `c6de8e` | 7.0e-5 | 0.020 | 0.26 | 0.54 |
| `be848f` | 5e-4 | 0.022 | 0.28 | 0.77 |
| `4f3c3b` | 7.1e-5 | 0.021 | 0.26 | 0.59 |
| `d11920` | 5e-4 | 0.023 | 0.29 | 0.75 |

Eval at four checkpoints, 1024 episodes:

| checkpoint | 400 | 900 | 1300 | 1907 |
|---|---|---|---|---|
| `be848f` catch | 0.512 | 0.492 | 0.562 | 0.487 |
| `be848f` `coll_pp` | 0.283 | 0.397 | 0.292 | 0.280 |
| `d11920` catch | 0.592 | 0.597 | 0.573 | 0.499 |
| `d11920` `coll_pp` | 0.334 | 0.349 | 0.369 | 0.387 |

The outcomes, 4096 episodes, with the 95 % Wilson interval of the catch:

| run | catch | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | median time to catch |
|---|---|---|---|---|---|---|
| `c6de8e` | 0.212 [0.200, 0.225] | 0.537 | 0.031 | 0.013 | 0.207 | 4.2 s |
| `be848f` | 0.498 [0.482, 0.513] | 0.278 | 0.062 | 0.093 | 0.069 | 4.1 s |
| `4f3c3b` | 0.388 [0.373, 0.403] | 0.348 | 0.013 | 0.052 | 0.199 | 5.0 s |
| `d11920` | 0.509 [0.493, 0.524] | 0.392 | 0.022 | 0.064 | 0.013 | 3.2 s |
| FRPN | 0.524 [0.509, 0.539] | 0.322 | 0.012 | 0.142 | 0.000 | 1.5 s |

The netpass runs flown in `paths_3v1`, 1024 episodes: `4f3c3b` catches 0.335 with `coll_pp`
0.481, and `d11920` catches 0.396 with `coll_pp` 0.513.

The catch by the evader's path:

| path | `c6de8e` | `be848f` | `4f3c3b` | `d11920` | FRPN |
|---|---|---|---|---|---|
| hover | 0.210 | 0.511 | 0.379 | 0.528 | 0.603 |
| circle | 0.211 | 0.501 | 0.364 | 0.506 | 0.559 |
| figure-eight | 0.216 | 0.481 | 0.421 | 0.493 | 0.413 |

The ends per second, among the episodes still running, in each bin from 2 s to 10 s: catch
0.035–0.045 (`c6de8e`), 0.105–0.147 (`be848f`), 0.045–0.091 (`4f3c3b`), 0.136–0.208
(`d11920`) and 0.15–0.42 (FRPN).

The behaviour:

| | `c6de8e` | `be848f` | `4f3c3b` | `d11920` | FRPN |
|---|---|---|---|---|---|
| nets at 4 s, median (m) | 3.3 / 4.9 / 6.7 | 2.1 / 3.6 / 5.6 | 2.7 / 4.0 / 5.5 | 1.9 / 3.0 / 4.4 | 2.4 / 3.9 / 6.0 |
| nets at 8 s, median (m) | 3.4 / 4.9 / 6.6 | 1.9 / 3.5 / 5.7 | 2.5 / 3.8 / 5.1 | 1.8 / 2.7 / 4.0 | — |
| closing at 1–2 / 2–3 / 3–5 m (m/s) | 0.34 / 0.28 / 0.42 | 0.44 / 0.45 / 0.69 | 0.38 / 0.40 / 0.60 | 0.46 / 0.59 / 1.14 | 4.79 / 5.82 / 6.63 |
| closing at 5–8 / 8–12 / 12–40 m (m/s) | 0.85 / 2.77 / 5.96 | 1.40 / 4.20 / 5.66 | 1.26 / 4.00 / 5.82 | 3.67 / 7.69 / 7.04 | 7.70 / 8.19 / 6.21 |
| net 2–5 m away: toward / around the evader (m/s) | 0.45 / 3.72 | 0.71 / 2.81 | 0.54 / 3.47 | 0.94 / 3.53 | 6.60 / 2.78 |
| pursuer speed, median / top 10 % (m/s) | 5.43 / 9.98 | 5.26 / 10.70 | 5.32 / 8.96 | 6.52 / 12.69 | 9.34 / 12.70 |
| height over the evader, net within 3 m, q10 / q50 / q90 (m) | −1.36 / 0.19 / 1.89 | −0.67 / 0.60 / 1.69 | −0.81 / 0.77 / 2.25 | −0.66 / 0.57 / 1.66 | −0.21 / 0.60 / 1.39 |
| no catch: closest net, median; share within 1 m | 1.44 m; 0.25 | 1.12 m; 0.41 | 1.06 m; 0.44 | 1.08 m; 0.44 | 0.90 m; 0.62 |
| `coll_pp`: nearest net to the evader, median; share under 2 m | 3.60 m; 0.14 | 2.10 m; 0.46 | 2.70 m; 0.26 | 1.90 m; 0.55 | 1.32 m; 0.74 |
| `coll_pp`: ranks 1+2 / 2+3 / 1+3 | 0.57 / 0.35 / 0.08 | 0.68 / 0.26 / 0.06 | 0.52 / 0.36 / 0.12 | 0.63 / 0.28 / 0.10 | 0.85 / 0.11 / 0.04 |
| angle of the two nearest at the evader, median | 74° | 89° | 75° | 80° | 78° |

The payoff of closing in from the ring, as in "Probes on the trained policies":

| run | G, < 0 | 0–0.5 | 0.5–1 | 1–2 | 2–4 | > 4 m/s | own catch within 1 s, < 0 / > 4 | own `coll_pp` within 1 s, < 0 / > 4 |
|---|---|---|---|---|---|---|---|---|
| `be848f` | 2.21 | 2.92 | 3.01 | 3.02 | 3.10 | 3.05 | 0.011 / 0.089 | 0.045 / 0.067 |
| `4f3c3b` | 0.89 | 1.17 | 1.13 | 1.33 | 1.57 | 2.22 | 0.007 / 0.069 | 0.043 / 0.045 |
| `d11920` | 3.03 | 3.71 | 3.75 | 3.76 | 3.78 | 3.67 | 0.019 / 0.091 | 0.073 / 0.112 |

- **The constant lr more than doubles the catch on `paths_3v1`:** 0.498 against 0.212, in
  4096 episodes. The six adaptive-lr policies of "Runs with the progress reward" caught
  0.175–0.253.
- **With the constant lr, the std stays at about 0.05.** With the adaptive lr it falls to
  0.017–0.020. The training catch rose by update 400 (0.47 and 0.57), peaked at 0.58 and
  0.62, and ended at 0.50 and 0.53.
- **The best checkpoints catch more than the last.** `be848f` catches 0.562 at update 1300,
  and `d11920` 0.597 at update 900, against 0.487 and 0.499 at 1907.
- **netpass with the adaptive lr catches 0.388,** against 0.212 for `c6de8e`. Flown in
  `paths_3v1`, it catches 0.335.
- **With the constant lr, netpass changes the catch little:** 0.509 against 0.498. Flown in
  `paths_3v1`, `d11920` catches 0.396.
- **The ring moved in.** At 4 s the nearest net is 2.1 m (`be848f`) and 1.9 m (`d11920`) from
  the evader, against 3.3 m. At 3–5 m the nets close at 0.69 and 1.14 m/s, against 0.42.
- **The height over the evader is near FRPN's.** The median is 0.57–0.60 m, against 0.19 m
  for `c6de8e` and 0.60 m for FRPN.
- **The hits moved toward the evader.** At a `coll_pp`, the nearest net is a median 1.9–2.1 m
  away, against 3.6 m.
- **The learned catch still does not depend much on the path.** With the constant lr: hover
  0.51–0.53, figure-eight 0.48–0.49. FRPN: 0.60 and 0.41.
- **With the constant lr, G no longer rises above 0.5 m/s of closing:** 2.92–3.10 and
  3.67–3.78. In `4f3c3b` it still rises, from 0.89 to 2.22.
- **Limits.** One seed for each new run. The two seeds of one setup differed by 0.04–0.07 in
  the catch before.

### The failures and the late decline

The episodes with no catch, last checkpoint, 4096 episodes. A death is a `coll_pp`, an OOB or
a crash.

| run | no catch | `coll_pp` | `OOB_p` | `crash_p` | `time_limit` | deaths | `coll_pp` share of the deaths |
|---|---|---|---|---|---|---|---|
| `c6de8e` | 0.788 | 0.537 | 0.013 | 0.031 | 0.207 | 0.581 | 0.92 |
| `be848f` | 0.502 | 0.278 | 0.093 | 0.062 | 0.069 | 0.433 | 0.64 |
| `4f3c3b` | 0.612 | 0.348 | 0.052 | 0.013 | 0.199 | 0.413 | 0.84 |
| `d11920` | 0.491 | 0.392 | 0.064 | 0.022 | 0.013 | 0.478 | 0.82 |

The std, the KL and the clip fraction over training, the mean of each range:

| updates | 1–100 | 101–200 | 201–500 | 501–1000 | 1001–1907 |
|---|---|---|---|---|---|
| std, `be848f` | 0.840 | 0.499 | 0.221 | 0.093 | 0.057 |
| std, `d11920` | 0.838 | 0.516 | 0.238 | 0.091 | 0.055 |
| `approx_kl_p`, `be848f` | 0.0065 | 0.0088 | 0.0132 | 0.0187 | 0.0224 |
| `approx_kl_p`, `d11920` | 0.0064 | 0.0082 | 0.0120 | 0.0189 | 0.0232 |
| `approx_kl_p`, `c6de8e` | 0.0143 | 0.0174 | 0.0179 | 0.0197 | 0.0201 |
| `clipfrac_p`, `be848f` | 0.080 | 0.119 | 0.183 | 0.248 | 0.283 |
| `clipfrac_p`, `d11920` | 0.079 | 0.109 | 0.166 | 0.249 | 0.290 |
| `clipfrac_p`, `c6de8e` | 0.197 | 0.232 | 0.235 | 0.253 | 0.256 |

- **`coll_pp` is the largest failure in all four runs:** 0.55 and 0.80 of the episodes with no
  catch under the constant lr, and 0.64–0.92 of the deaths.
- **Under the constant lr, the hits come near the evader.** The nearest net is a median 1.9–2.1
  m away at a hit, and the two nearest pursuers are the pair in 0.63–0.68 of them.
- **The eval catch fell from the best checkpoint to the last:** 0.562 → 0.487 (`be848f`, update
  1300 → 1907) and 0.597 → 0.499 (`d11920`, update 900 → 1907). With the two evals taken as
  independent samples of 1024 episodes, that is z = 3.4 and 4.5.
- **Over the same run, the KL rose from 0.006 to 0.022–0.023, and the clip fraction from 0.08
  to 0.28–0.29,** while the std fell from 0.84 to 0.06. With the adaptive lr, the KL stayed at
  0.014–0.020.

## Against Gavin Nv1, 2026-11-10

His Table II, the pursuers against the APF evader, beside ours against `paths` (4096
episodes). His pursuer deaths are his pursuer crashes plus his double crashes. His `coll_pp`
ends the episode and costs 10, as ours, and only a body hit counts (§IV-C). His `noSP`
pursuers train with MAPPO against the fixed APF evader, with no pool (§V-B).

| pursuers | evader | catch | pursuer deaths | evader crash | evader free at the end |
|---|---|---|---|---|---|
| his FRPN | APF | 0.550 | 0.394 | 0.055 | 0.001 |
| his `noSP` | APF | 0.652 | 0.140 | 0.151 | 0.057 |
| his PFSP-CTBR | APF | 0.881 | 0.069 | 0.049 | 0.002 |
| our FRPN | `paths` | 0.524 | 0.476 | 0.000 | 0.000 |
| `be848f` (lr 5e-4) | `paths` | 0.498 | 0.433 | 0.000 | 0.069 |
| `d11920` (netpass, lr 5e-4) | `paths` | 0.509 | 0.478 | 0.000 | 0.013 |

His training beside ours. The Nv1 paper publishes no PPO settings, so the lr, the entropy
weight and the update come from his 1v1 paper (Table II).

| | Gavin | `be848f` |
|---|---|---|
| lr | 5e-4, constant | 5e-4, constant |
| action distribution | Gaussian, then a tanh squash | Gaussian, then a clip |
| entropy weight | 0.01 | 0.001 |
| update | 15 epochs × 1 batch, 128-step rollouts | 4 epochs × 32 minibatches, 64-step rollouts |
| activation | ReLU | tanh |
| steps | 5e9 for each side | 5e8 |
| the evader | learns, in a pool; it stays in a 12 × 12 × 6 m box in the centre | `paths`, fixed |
| reward | Table I | the progress reward; before it, his Table I term by term |

- **FRPN catches about as often in both setups:** 0.550 against his evader and 0.524 against
  ours.
- **His learned pursuers die less than his FRPN:** 0.140 (`noSP`) and 0.069 (PFSP-CTBR),
  against 0.394. Ours die about as often as our FRPN: 0.433 and 0.478, against 0.476.
- **His `noSP` catches 0.102 more than his FRPN, with no pool.** Ours catches 0.015–0.026 less
  than our FRPN.
- **His APF evader crashes in 0.049–0.151 of the episodes.** The `paths` evader never crashes.

## Runs with `ent_coef=0.003` and a constant lr, 2026-11-11

1 run on MetaCentrum, commit `fba6aec`: `9eb432`, `paths_3v1`, seed 1, tanh, `--set gamma=0.99
--set normalize_obs=false --set adaptive_lr=false --set lr=5e-4 --set ent_coef=0.003`, 5e8
steps, the progress reward. Its config differs from `be848f` only in `ent_coef` (0.003 against
0.001) and in the seed (1 against 0).

Eval on `7a9878f`, whose code differs from `fba6aec` only in the attention network: 1024
episodes, mean actions. The flights are 4096 episodes, of the last checkpoint and of update
850. The scripts are in `scratch/`: `train_bins.py`, `ckpt_sweep.py`, and `fly_record.py`
with `CKPT=850`.

Training, the mean of each 200 updates:

| updates | 1–200 | 201–400 | 401–600 | 601–800 | 801–1000 | 1001–1200 | 1201–1400 | 1401–1600 | 1601–1800 | 1801–1907 |
|---|---|---|---|---|---|---|---|---|---|---|
| catch, `be848f` | 0.24 | 0.47 | 0.47 | 0.45 | 0.47 | 0.55 | 0.58 | 0.56 | 0.51 | 0.50 |
| catch, `9eb432` | 0.36 | 0.69 | 0.85 | 0.90 | 0.90 | 0.83 | 0.80 | 0.82 | 0.83 | 0.83 |
| `coll_pp`, `be848f` | 0.36 | 0.30 | 0.41 | 0.42 | 0.38 | 0.28 | 0.28 | 0.33 | 0.35 | 0.30 |
| `coll_pp`, `9eb432` | 0.33 | 0.26 | 0.13 | 0.08 | 0.08 | 0.16 | 0.19 | 0.16 | 0.15 | 0.15 |
| std, `be848f` | 0.669 | 0.256 | 0.136 | 0.094 | 0.078 | 0.066 | 0.059 | 0.054 | 0.051 | 0.050 |
| std, `9eb432` | 0.862 | 0.549 | 0.363 | 0.263 | 0.210 | 0.189 | 0.182 | 0.176 | 0.166 | 0.160 |
| `return_p`, `be848f` | 5.27 | 13.76 | 13.18 | 12.61 | 13.19 | 16.19 | 17.35 | 16.27 | 14.55 | 13.74 |
| `return_p`, `9eb432` | 9.08 | 21.29 | 26.91 | 28.78 | 28.86 | 26.31 | 25.06 | 26.06 | 26.25 | 26.15 |
| time to catch, `9eb432` (s) | 4.79 | 3.76 | 3.08 | 2.75 | 2.45 | 2.19 | 2.08 | 2.14 | 2.17 | 2.12 |

Over updates 1001–1907, and the std of each action at the end (throttle, roll rate, pitch
rate, yaw rate):

| run | `approx_kl_p` | `clipfrac_p` | explained variance | std of each action |
|---|---|---|---|---|
| `be848f` | 0.022 | 0.28 | 0.77 | 0.055, 0.046, 0.044, 0.059 |
| `9eb432` | 0.014 | 0.20 | 0.81 | 0.134, 0.111, 0.112, 0.381 |

The first updates, beside the two runs with `ent_coef=0.003` and the adaptive lr (2026-11-05).
Those two also have the reward before the progress reward and a `coll_pp` cost of 30. The lr
is the median over updates 1–100, then over 101–200:

| update | 10 | 50 | 100 | 200 | 400 | lr, 1–100 | lr, 101–200 |
|---|---|---|---|---|---|---|---|
| std, `9eb432` | 0.996 | 0.969 | 0.868 | 0.687 | 0.437 | 5.0e-4 | 5.0e-4 |
| std, `503bac` | 1.017 | 1.127 | 1.391 | 2.014 | 3.043 | 3.0e-3 | 1.9e-3 |
| std, `2cda6a` | 1.016 | 1.078 | 1.357 | 1.841 | 3.201 | 3.0e-3 | 1.6e-3 |

Eval of `9eb432` at its checkpoints, 1024 episodes:

| update | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 850 | 900 | 1000 | 1100 | 1200 | 1300 | 1500 | 1700 | 1907 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catch | 0.554 | 0.766 | 0.778 | 0.832 | 0.875 | 0.907 | 0.910 | 0.914 | 0.880 | 0.869 | 0.844 | 0.785 | 0.810 | 0.786 | 0.818 | 0.850 |
| `coll_pp` | 0.392 | 0.161 | 0.192 | 0.149 | 0.108 | 0.070 | 0.070 | 0.070 | 0.109 | 0.115 | 0.145 | 0.206 | 0.181 | 0.192 | 0.165 | 0.125 |
| time to catch (s) | 3.99 | 3.90 | 3.16 | 2.92 | 2.84 | 2.79 | 2.65 | 2.56 | 2.39 | 2.26 | 2.19 | 2.12 | 2.07 | 2.07 | 2.05 | 2.16 |

The outcomes, 4096 episodes, with the 95 % Wilson interval of the catch. A death is a
`coll_pp`, an OOB or a crash:

| run | catch | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | deaths | `coll_pp` share of the deaths | median time to catch |
|---|---|---|---|---|---|---|---|---|
| `be848f` | 0.498 [0.482, 0.513] | 0.278 | 0.062 | 0.093 | 0.069 | 0.433 | 0.64 | 4.1 s |
| `9eb432`, last | 0.837 [0.825, 0.848] | 0.136 | 0.020 | 0.007 | 0.001 | 0.163 | 0.83 | 1.9 s |
| `9eb432`, update 850 | 0.910 [0.901, 0.919] | 0.074 | 0.006 | 0.010 | 0.000 | 0.090 | 0.82 | 2.4 s |
| FRPN | 0.524 [0.509, 0.539] | 0.322 | 0.012 | 0.142 | 0.000 | 0.476 | 0.68 | 1.5 s |

On the same 4096 episodes: `9eb432` catches 1707 that `be848f` misses and misses 318 that
`be848f` catches (McNemar z = 30.9). Update 850 catches 580 that the last checkpoint misses
and misses 279 that it catches (z = 10.3).

The catch by the evader's path, and by the start distance of the nearest net (the quarters
of all episodes, with edges at 6.8, 9.4 and 12.3 m):

| | `be848f` | `9eb432`, last | `9eb432`, update 850 | FRPN |
|---|---|---|---|---|
| hover | 0.511 | 0.874 | 0.931 | 0.603 |
| circle | 0.501 | 0.854 | 0.925 | 0.559 |
| figure-eight | 0.481 | 0.784 | 0.876 | 0.413 |
| nearest quarter | 0.57 | 0.85 | 0.93 | 0.60 |
| second quarter | 0.52 | 0.83 | 0.92 | 0.53 |
| third quarter | 0.46 | 0.84 | 0.90 | 0.50 |
| farthest quarter | 0.44 | 0.83 | 0.88 | 0.46 |

The ends per second, among the episodes still running, in the bins 1–2, 2–3 and 3–4 s:

| | `be848f` | `9eb432`, last | `9eb432`, update 850 | FRPN |
|---|---|---|---|---|
| catch | 0.059 / 0.105 / 0.138 | 0.443 / 0.553 / 0.534 | 0.278 / 0.551 / 0.662 | 0.359 / 0.333 / 0.419 |
| `coll_pp` | 0.041 / 0.073 / 0.070 | 0.065 / 0.106 / 0.101 | 0.024 / 0.037 / 0.051 | 0.213 / 0.219 / 0.304 |

The behaviour:

| | `be848f` | `9eb432`, last | `9eb432`, update 850 | FRPN |
|---|---|---|---|---|
| nets at 2 s, median (m) | 2.7 / 4.4 / 6.6 | 1.4 / 2.6 / 4.7 | 1.5 / 2.8 / 4.8 | 2.2 / 4.3 / 7.0 |
| nets at 4 s, median (m) | 2.1 / 3.6 / 5.6 | 1.3 / 2.7 / 4.5 | 1.1 / 1.9 / 3.4 | 2.4 / 3.9 / 6.0 |
| closing at 1–2 / 2–3 / 3–5 m (m/s) | 0.44 / 0.45 / 0.69 | 0.80 / 1.46 / 3.10 | 0.81 / 1.46 / 2.95 | 4.79 / 5.82 / 6.63 |
| closing at 5–8 / 8–12 / 12–40 m (m/s) | 1.40 / 4.20 / 5.66 | 6.05 / 7.62 / 6.69 | 5.75 / 7.39 / 6.66 | 7.70 / 8.19 / 6.21 |
| net 2–5 m away: toward / around the evader (m/s) | 0.71 / 2.81 | 2.63 / 2.70 | 2.47 / 2.00 | 6.60 / 2.78 |
| pursuer speed, median / top 10 % (m/s) | 5.26 / 10.70 | 6.84 / 13.38 | 5.85 / 12.06 | 9.34 / 12.70 |
| height over the evader, net within 3 m, q10 / q50 / q90 (m) | −0.67 / 0.60 / 1.69 | −0.54 / 0.64 / 1.51 | −0.37 / 0.63 / 1.54 | −0.21 / 0.60 / 1.39 |
| no catch: closest net, median; share within 1 m | 1.12 m; 0.41 | 0.91 m; 0.59 | 0.97 m; 0.52 | 0.90 m; 0.62 |
| `coll_pp`: nearest net to the evader, median; share under 2 m | 2.10 m; 0.46 | 1.26 m; 0.80 | 1.26 m; 0.80 | 1.32 m; 0.74 |
| `coll_pp`: ranks 1+2 / 2+3 / 1+3 | 0.68 / 0.26 / 0.06 | 0.75 / 0.18 / 0.07 | 0.77 / 0.17 / 0.06 | 0.85 / 0.11 / 0.04 |
| `coll_pp`: share with the two bodies under 0.8 m apart | 0.44 | 0.30 | 0.28 | 0.66 |
| `coll_pp`: the pair's distance 0.5 s before, median | 2.94 m | 3.83 m | 2.63 m | 6.01 m |
| `coll_pp`: speed of the pair at the hit, median (m/s) | 5.10 | 5.92 | 4.73 | 11.92 |
| angle of the two nearest at the evader, median | 89° | 92° | 93° | 78° |

The payoff of closing in with the own net 2–5 m from the evader, as in "Probes on the trained
policies":

| run | G, < 0 | 0–0.5 | 0.5–1 | 1–2 | 2–4 | > 4 m/s | own catch within 1 s, < 0 / > 4 | own `coll_pp` within 1 s, < 0 / > 4 |
|---|---|---|---|---|---|---|---|---|
| `be848f` | 2.21 | 2.92 | 3.01 | 3.02 | 3.10 | 3.05 | 0.011 / 0.089 | 0.045 / 0.067 |
| `9eb432`, last | 9.46 | 10.72 | 11.07 | 11.08 | 11.03 | 11.81 | 0.061 / 0.268 | 0.051 / 0.081 |
| `9eb432`, update 850 | 11.20 | 12.17 | 12.51 | 12.19 | 11.46 | 11.08 | 0.048 / 0.218 | 0.025 / 0.032 |

- **`ent_coef=0.003` with the constant lr lifts the catch from 0.498 to 0.837,** in 4096
  episodes at the last checkpoint, with one run each. Update 850 catches 0.910. FRPN catches
  0.524.
- **The std did not blow up.** It stayed under 1.01 and fell to 0.16: 0.11–0.13 for the
  throttle and the roll and pitch rates, and 0.38 for the yaw rate. With 0.001 it ends at
  0.04–0.06. In the two adaptive-lr runs with 0.003, the std was 1.36–1.39 at update 100, and
  the median of their lr over updates 1–100 was 3.0e-3, 6 times the constant lr.
- **The pursuers no longer hold on a ring.** With a net 2–5 m away they fly 2.63 m/s toward the
  evader, against 0.71. At 3–5 m they close at 3.10 m/s, against 0.69. FRPN closes at 6.60 and
  6.63 m/s.
- **It catches at FRPN's rate and hits less.** From 1 to 4 s it catches 0.44–0.55 of the
  running episodes per second, against 0.33–0.42 for FRPN and 0.06–0.14 for `be848f`. Its
  `coll_pp` rate is 0.07–0.11 per second, against 0.21–0.30 for FRPN.
- **The catch does not depend on the start distance:** 0.83–0.85 in each quarter. In `be848f`
  it falls from 0.57 to 0.44.
- **The figure-eight is the hardest path:** 0.784, against 0.874 for the hover.
- **`coll_pp` is 0.83 of the episodes with no catch, next to the evader.** At a hit the nearest
  net is a median 1.26 m from the evader, the two nearest pursuers are the pair in 0.75 of the
  hits, and the two bodies are under 0.8 m apart in 0.30 of them.
- **The catch falls after its best checkpoint here too.** The eval catch is 0.914 at update
  850, 0.785 at 1200 and 0.850 at 1907. In 4096 episodes it is 0.910 against 0.837.
- **The training return falls with it,** from 28.86 (updates 801–1000) to 25.06 (1201–1400).
  The time to catch keeps falling, from 2.56 s at update 850 to 2.07 s at 1300, while `coll_pp`
  rises from 0.070 to 0.181–0.206.
- **From update 850 to the last, the pursuers fly faster and hit more.** The median speed
  rises from 5.85 to 6.84 m/s, the speed around the evader from 2.00 to 2.70 m/s, and the
  speed of a pair at its hit from 4.73 to 5.92 m/s. The own `coll_pp` within 1 s rises from
  0.025–0.032 to 0.051–0.081.
- **At update 850, G is highest at a closing of 0.5–1 m/s** (12.51), and it is 11.08 above
  4 m/s. At the last checkpoint it is highest above 4 m/s (11.81).
- **The KL and the clip fraction are lower with 0.003:** 0.014 and 0.20 over updates
  1001–1907, against 0.022 and 0.28.
- **Limits.** `be848f` is seed 0 and `9eb432` is seed 1. Seed 0 with 0.003 is in the next
  part. No run has `ent_coef=0.003` with the adaptive lr and the progress reward.

### Seed 0 (`730968`)

`730968` is seed 0 of the same setup: its config differs from `9eb432` only in the seed. The
evals and the flights are as above. Its best checkpoint of the 18 that were checked is update
1050.

| | `be848f` (0.001, seed 0) | `730968` (0.003, seed 0) | `9eb432` (0.003, seed 1) |
|---|---|---|---|
| training catch, updates 801–1000 / 1001–1200 / 1801–1907 | 0.470 / 0.548 / 0.497 | 0.615 / 0.671 / 0.619 | 0.901 / 0.831 / 0.828 |
| training `coll_pp`, the same updates | 0.384 / 0.284 / 0.303 | 0.349 / 0.284 / 0.345 | 0.085 / 0.156 / 0.146 |
| std at the end, each action | 0.055, 0.046, 0.044, 0.059 | 0.157, 0.116, 0.120, 0.206 | 0.134, 0.111, 0.112, 0.381 |
| eval catch at the best checkpoint, 1024 episodes | 0.573 (1500) | 0.684 (1050) | 0.914 (850) |
| eval `coll_pp` at that checkpoint | 0.334 | 0.278 | 0.070 |
| eval catch at the last checkpoint, 1024 episodes | 0.487 | 0.628 | 0.850 |
| catch, 4096 episodes, last | 0.498 [0.482, 0.513] | 0.605 [0.590, 0.620] | 0.837 [0.825, 0.848] |
| `coll_pp` / `crash_p` / `OOB_p` / `time_limit`, last | 0.278 / 0.062 / 0.093 / 0.069 | 0.360 / 0.018 / 0.016 / 0.001 | 0.136 / 0.020 / 0.007 / 0.001 |
| catch, 4096 episodes, best checkpoint | — | 0.682 [0.668, 0.696] | 0.910 [0.901, 0.919] |
| `coll_pp`, 4096 episodes, best checkpoint | — | 0.275 | 0.074 |
| median time to catch, last | 4.1 s | 1.8 s | 1.9 s |
| hover / circle / figure-eight, last | 0.511 / 0.501 / 0.481 | 0.685 / 0.602 / 0.530 | 0.874 / 0.854 / 0.784 |
| net 2–5 m away: toward / around the evader (m/s), last | 0.71 / 2.81 | 1.81 / 3.08 | 2.63 / 2.70 |
| closing at 1–2 / 2–3 / 3–5 m (m/s), last | 0.44 / 0.45 / 0.69 | 0.80 / 1.13 / 2.06 | 0.80 / 1.46 / 3.10 |
| `coll_pp`: nearest net to the evader, median; share under 2 m | 2.10 m; 0.46 | 1.60 m; 0.66 | 1.26 m; 0.80 |
| `coll_pp`: speed of the pair at the hit, median (m/s) | 5.10 | 6.70 | 5.92 |
| against APF, 1024 episodes: catch, last / best checkpoint | 0.454 / — | 0.477 / 0.612 | 0.719 / 0.856 |

The eval catch of `730968` at its checkpoints, 1024 episodes:

| update | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 850 | 900 | 1000 | 1050 | 1100 | 1200 | 1300 | 1400 | 1500 | 1700 | 1907 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catch | 0.287 | 0.472 | 0.599 | 0.557 | 0.477 | 0.540 | 0.572 | 0.587 | 0.618 | 0.645 | 0.684 | 0.659 | 0.671 | 0.671 | 0.609 | 0.604 | 0.617 | 0.628 |
| `coll_pp` | 0.455 | 0.402 | 0.348 | 0.393 | 0.439 | 0.415 | 0.397 | 0.383 | 0.345 | 0.311 | 0.278 | 0.310 | 0.294 | 0.290 | 0.341 | 0.348 | 0.329 | 0.337 |

- **The two seeds with 0.003 differ by 0.232 in the catch:** 0.605 and 0.837 in 4096 episodes.
  At their best checkpoints they catch 0.682 and 0.910.
- **With the same seed 0, 0.003 catches 0.107 more than 0.001:** 0.605 against 0.498. At the
  best checkpoints it is 0.684 against 0.573 (1024 episodes).
- **Seed 0 hits far more than seed 1:** `coll_pp` is 0.360 against 0.136, and it is 0.91 of
  the episodes with no catch.
- **Seed 0 also goes in, but less:** with a net 2–5 m away it flies 1.81 m/s toward the
  evader, against 2.63 for seed 1 and 0.71 for `be848f`.
- **The std does not blow up in seed 0 either.** Its highest value is 1.02, and it ends at
  0.12–0.21.
- **The catch of seed 0 also falls after its best checkpoint:** 0.684 at update 1050, 0.604 at
  1500 and 0.628 at 1907.

### Against the APF evader and Gavin Nv1

Evals with `--seat evader apf`, 1024 episodes. No policy here trained against APF. Our FRPN
against APF is `--seat pursuer frpn --seat evader apf`. The rows against `paths` are the 4096
episodes above, and Gavin's rows are his Table II.

| pursuers | evader | catch | pursuer deaths | evader crash | evader free at the end |
|---|---|---|---|---|---|
| his FRPN | APF | 0.550 | 0.394 | 0.055 | 0.001 |
| his `noSP` | APF | 0.652 | 0.140 | 0.151 | 0.057 |
| his PFSP-CTBR | APF | 0.881 | 0.069 | 0.049 | 0.002 |
| our FRPN | `paths` | 0.524 | 0.476 | 0.000 | 0.000 |
| `be848f` | `paths` | 0.498 | 0.433 | 0.000 | 0.069 |
| `9eb432`, last | `paths` | 0.837 | 0.163 | 0.000 | 0.001 |
| `9eb432`, update 850 | `paths` | 0.910 | 0.090 | 0.000 | 0.000 |
| our FRPN | APF | 0.279 | 0.721 | 0.000 | 0.000 |
| `be848f` | APF | 0.454 | 0.393 | 0.000 | 0.153 |
| `9eb432`, last | APF | 0.719 | 0.276 | 0.002 | 0.003 |
| `9eb432`, update 850 | APF | 0.856 | 0.143 | 0.001 | 0.000 |

- **Against `paths`, `9eb432` dies in 0.163 of the episodes, and in 0.090 at update 850,**
  against 0.433 for `be848f` and 0.476 for our FRPN. Gavin's learned pursuers die in
  0.069–0.140.
- **Against APF, which it never met in training, `9eb432` catches 0.719, and 0.856 at update
  850.** `be848f` catches 0.454.
- **Our FRPN catches 0.279 against our APF, with `coll_pp` in 0.657 of the episodes.** His FRPN
  catches 0.550 against his APF.

## Runs with the sensed blocks and a constant lr, 2026-11-11

6 runs on MetaCentrum, commit `fba6aec`: `paths_3v1_attitude`, `paths_3v1_rpm` and
`paths_3v1_attitude_rpm`, seeds 0 and 1, tanh, `--set gamma=0.99 --set normalize_obs=false
--set adaptive_lr=false --set lr=5e-4`, 5e8 steps, `ent_coef` 0.001. The base is `be848f`
(`paths_3v1`, seed 0). The base has no seed 1 with these settings.

| run | the pursuer senses | seed |
|---|---|---|
| `be848f` | none | 0 |
| `b503f3` | attitude | 0 |
| `740128` | attitude | 1 |
| `873189` | rpm | 0 |
| `22b282` | rpm | 1 |
| `6e196e` | attitude, rpm | 0 |
| `1390aa` | attitude, rpm | 1 |

Eval on `7a9878f`: 1024 episodes, mean actions. The flights are 4096 episodes of the last
checkpoint.

The training catch, the mean of each 200 updates, and the std at the end:

| updates | 1–200 | 201–400 | 401–600 | 601–800 | 801–1000 | 1001–1200 | 1201–1400 | 1401–1600 | 1601–1800 | 1801–1907 | std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `be848f` | 0.24 | 0.47 | 0.47 | 0.45 | 0.47 | 0.55 | 0.58 | 0.56 | 0.51 | 0.50 | 0.051 |
| `b503f3` | 0.18 | 0.34 | 0.34 | 0.37 | 0.39 | 0.37 | 0.37 | 0.36 | 0.37 | 0.42 | 0.064 |
| `740128` | 0.22 | 0.36 | 0.31 | 0.36 | 0.39 | 0.42 | 0.42 | 0.42 | 0.37 | 0.37 | 0.061 |
| `873189` | 0.24 | 0.41 | 0.49 | 0.57 | 0.58 | 0.58 | 0.61 | 0.57 | 0.55 | 0.55 | 0.049 |
| `22b282` | 0.17 | 0.37 | 0.44 | 0.53 | 0.56 | 0.56 | 0.54 | 0.49 | 0.50 | 0.50 | 0.052 |
| `6e196e` | 0.27 | 0.44 | 0.55 | 0.56 | 0.60 | 0.64 | 0.62 | 0.63 | 0.64 | 0.64 | 0.049 |
| `1390aa` | 0.30 | 0.54 | 0.58 | 0.62 | 0.62 | 0.60 | 0.64 | 0.62 | 0.61 | 0.60 | 0.046 |

The eval catch at the checkpoints, 1024 episodes each. The mean is over the seven checkpoints
from 1000 to 1907:

| update | 200 | 400 | 600 | 800 | 900 | 1000 | 1100 | 1200 | 1300 | 1500 | 1700 | 1907 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `be848f` | 0.392 | 0.512 | 0.476 | 0.373 | 0.492 | 0.502 | 0.547 | 0.565 | 0.562 | 0.573 | 0.498 | 0.487 | 0.533 |
| `b503f3` | 0.238 | 0.353 | 0.386 | 0.385 | 0.371 | 0.389 | 0.361 | 0.399 | 0.383 | 0.367 | 0.385 | 0.461 | 0.392 |
| `740128` | 0.337 | 0.344 | 0.322 | 0.383 | 0.391 | 0.388 | 0.378 | 0.403 | 0.438 | 0.426 | 0.368 | 0.360 | 0.394 |
| `873189` | 0.368 | 0.463 | 0.546 | 0.600 | 0.581 | 0.572 | 0.604 | 0.606 | 0.636 | 0.530 | 0.541 | 0.522 | 0.573 |
| `22b282` | 0.298 | 0.377 | 0.446 | 0.577 | 0.597 | 0.563 | 0.543 | 0.530 | 0.549 | 0.467 | 0.472 | 0.542 | 0.524 |
| `6e196e` | 0.424 | 0.552 | 0.549 | 0.560 | 0.607 | 0.630 | 0.665 | 0.661 | 0.604 | 0.647 | 0.637 | 0.638 | 0.640 |
| `1390aa` | 0.500 | 0.522 | 0.630 | 0.624 | 0.604 | 0.604 | 0.591 | 0.588 | 0.626 | 0.589 | 0.593 | 0.550 | 0.592 |

The outcomes, 4096 episodes, with the 95 % Wilson interval of the catch:

| run | catch | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | deaths | `coll_pp` share of the deaths | median time to catch |
|---|---|---|---|---|---|---|---|---|
| `be848f` | 0.498 [0.482, 0.513] | 0.278 | 0.062 | 0.093 | 0.069 | 0.433 | 0.64 | 4.1 s |
| `b503f3` | 0.440 [0.425, 0.456] | 0.412 | 0.031 | 0.060 | 0.056 | 0.504 | 0.82 | 4.0 s |
| `740128` | 0.366 [0.351, 0.381] | 0.477 | 0.029 | 0.057 | 0.072 | 0.563 | 0.85 | 3.5 s |
| `873189` | 0.549 [0.533, 0.564] | 0.288 | 0.049 | 0.067 | 0.047 | 0.404 | 0.71 | 3.9 s |
| `22b282` | 0.548 [0.533, 0.564] | 0.314 | 0.018 | 0.034 | 0.085 | 0.366 | 0.86 | 4.0 s |
| `6e196e` | 0.642 [0.627, 0.656] | 0.290 | 0.025 | 0.021 | 0.023 | 0.335 | 0.87 | 3.4 s |
| `1390aa` | 0.580 [0.565, 0.595] | 0.334 | 0.031 | 0.053 | 0.002 | 0.418 | 0.80 | 2.2 s |

The catch by the evader's path, and the behaviour:

| | `be848f` | `b503f3` | `740128` | `873189` | `22b282` | `6e196e` | `1390aa` |
|---|---|---|---|---|---|---|---|
| hover | 0.511 | 0.435 | 0.368 | 0.544 | 0.529 | 0.688 | 0.607 |
| circle | 0.501 | 0.417 | 0.382 | 0.579 | 0.611 | 0.663 | 0.622 |
| figure-eight | 0.481 | 0.468 | 0.347 | 0.523 | 0.505 | 0.576 | 0.512 |
| nets at 4 s, median (m) | 2.1 / 3.6 / 5.6 | 2.0 / 3.2 / 4.6 | 2.6 / 4.0 / 5.6 | 2.1 / 3.8 / 6.1 | 2.3 / 3.9 / 6.1 | 1.5 / 2.6 / 4.0 | 1.6 / 2.7 / 4.4 |
| closing at 1–2 / 2–3 / 3–5 m (m/s) | 0.44 / 0.45 / 0.69 | 0.29 / 0.35 / 0.66 | 0.35 / 0.36 / 0.63 | 0.54 / 0.66 / 0.90 | 0.50 / 0.55 / 0.76 | 0.37 / 0.49 / 1.00 | 0.62 / 0.95 / 2.00 |
| closing at 5–8 / 8–12 m (m/s) | 1.40 / 4.20 | 2.00 / 5.66 | 1.85 / 5.95 | 1.75 / 4.78 | 1.49 / 3.96 | 3.87 / 7.37 | 4.58 / 6.98 |
| net 2–5 m away: toward / around the evader (m/s) | 0.71 / 2.81 | 0.54 / 2.90 | 0.64 / 3.25 | 0.88 / 2.81 | 0.75 / 2.70 | 0.88 / 2.57 | 1.64 / 3.27 |
| `coll_pp`: nearest net to the evader, median; share under 2 m | 2.10 m; 0.46 | 2.06 m; 0.48 | 2.36 m; 0.37 | 2.12 m; 0.46 | 2.23 m; 0.43 | 1.69 m; 0.63 | 1.55 m; 0.68 |
| `coll_pp`: ranks 1+2 / 2+3 / 1+3 | 0.68 / 0.26 / 0.06 | 0.62 / 0.29 / 0.09 | 0.60 / 0.33 / 0.08 | 0.68 / 0.27 / 0.05 | 0.66 / 0.27 / 0.07 | 0.59 / 0.31 / 0.09 | 0.69 / 0.22 / 0.09 |

A probe on the last checkpoints (`scratch/blind_sense.py`): the sensed columns of some drones
are set to a fixed value in the pursuer's input, and the drones fly as before. A fixed
attitude is the identity, which says that the other drone has the pursuer's own attitude. A
fixed rpm is 0.55, the hover value. 1024 episodes, the eval episodes. Each cell is the catch,
then `coll_pp`, then `crash_p`:

| run | nothing fixed | attitude of the evader | attitude of the teammates | rpm of the evader | rpm of the teammates |
|---|---|---|---|---|---|
| `b503f3` | 0.461 / 0.412 / 0.023 | 0.336 / 0.415 / 0.106 | 0.401 / 0.449 / 0.046 | — | — |
| `740128` | 0.360 / 0.473 / 0.030 | 0.243 / 0.450 / 0.099 | 0.301 / 0.474 / 0.064 | — | — |
| `873189` | 0.522 / 0.304 / 0.044 | — | — | 0.524 / 0.303 / 0.043 | 0.536 / 0.296 / 0.050 |
| `22b282` | 0.542 / 0.316 / 0.015 | — | — | 0.544 / 0.312 / 0.015 | 0.548 / 0.307 / 0.015 |
| `6e196e` | 0.638 / 0.293 / 0.020 | 0.406 / 0.437 / 0.096 | 0.553 / 0.393 / 0.025 | 0.632 / 0.300 / 0.020 | 0.649 / 0.294 / 0.021 |
| `1390aa` | 0.550 / 0.357 / 0.026 | 0.260 / 0.412 / 0.164 | 0.548 / 0.324 / 0.049 | 0.551 / 0.359 / 0.025 | 0.566 / 0.351 / 0.022 |

- **With the attitude alone, the pursuers catch less than the base seed:** 0.440 and 0.366,
  against 0.498. Over the checkpoints from 1000 on, the mean is 0.392 and 0.394, against 0.533.
  Their `coll_pp` is 0.412 and 0.477, against 0.278.
- **With the rpm alone, they catch about as the base seed:** 0.549 and 0.548. Over the
  checkpoints the mean is 0.573 and 0.524.
- **With both, they catch more:** 0.642 and 0.580, and 0.640 and 0.592 over the checkpoints.
  They catch sooner, at a median 3.4 and 2.2 s, against 4.1 s.
- **Two seeds of one preset differ by 0.075, 0.000 and 0.062 in the catch,** and the base has
  one seed. A preset differs from the base seed by −0.132 to −0.058 (attitude), +0.050 and
  +0.051 (rpm), and +0.082 to +0.144 (both).
- **No policy reads the rpm.** With the rpm fixed, the catch changes by −0.006 to +0.016 in
  the four runs that sense it. The 95 % interval of one such catch is about ±0.03.
- **The policies read the attitude of the evader.** With it fixed, the catch falls by 0.125,
  0.117, 0.232 and 0.290, and `crash_p` rises from 0.020–0.030 to 0.096–0.164.
- **They read the attitude of the teammates less:** the catch falls by 0.060, 0.059, 0.085 and
  0.002. In `6e196e`, `coll_pp` rises from 0.293 to 0.393.
- **In five of the six runs, the pursuers still hold back at 2–5 m:** they fly 0.54–0.88 m/s
  toward the evader and 2.57–3.25 m/s around it. `1390aa` flies 1.64 m/s toward it.
- **`coll_pp` is 0.71–0.87 of the deaths in all six runs.**
- **The eval catch falls after its best checkpoint in four of the six runs:** `740128` (0.438
  at update 1300, 0.360 at the end), `873189` (0.636 at 1300, 0.522), `22b282` (0.597 at 900,
  0.542) and `1390aa` (0.630 at 600, 0.550).
- **Limits.** One base seed. The probe shows that a policy reads a block, not that the block
  lifts the catch: the fixed attitude does not fit the pursuer's own attitude in its input.

## The attention network, 2026-11-11

A second network for the pursuer, `--set net=attention` (commit `79e1f0e`), after the encoder
of Geles et al. 2026. The size of its input does not depend on the number of teammates:

| part of the input | how the network reads it | size |
|---|---|---|
| own block | straight into the trunk | 20 |
| the evader: offset and velocity | straight into the trunk | 6 |
| each teammate: offset and velocity | a Perceiver: a layer of 128 with tanh on each teammate, then 4 learned queries that attend to the teammates with 4 heads of 32 | 4 × 128 = 512 |

- The trunk is 2 × 256 with tanh, as in the MLP, and it reads 538 values. The MLP reads the
  same 26 values, and the two teammates in two seats, the nearest first: 44 values.
- The critic has the same shape. Each part also holds the world state of its drone (18), and
  the evader's part holds its action (4).
- The network reads its inputs unscaled, so it needs `normalize_obs=false`. It has no LSTM.

6 runs on MetaCentrum: `paths_3v1`, tanh, `--set gamma=0.99 --set normalize_obs=false`, the
adaptive lr, `ent_coef` 0.001, 5e8 steps, the progress reward. The attention runs add `--set
net=attention`, and their config differs from the MLP runs only in `net`. The config of
`91b18c` and `2cdecf` also holds the two env fields that `fba6aec` added, at the values of
the old rules:

| run | network | seed | commit |
|---|---|---|---|
| `c6de8e`, `398c2a` (before) | MLP | 0, 1 | `18a4c40` |
| `b087ac`, `da9e33` | attention | 0, 1 | `79e1f0e` |
| `91b18c`, `2cdecf` | attention, with a norm on the Perceiver output | 0, 1 | `7a9878f` |

Eval: `run/eval.py --episodes 4096 --name paths_3v1_4096`, mean actions, the last checkpoint,
with the 95 % Wilson interval of the catch. `b087ac` and `da9e33` flew on `fba6aec`. The norm
has no weights, so their checkpoints also load on `7a9878f`, but the network is then another
one: fly them on code before `7a9878f`. The scripts are in `scratch/attention_analysis/`.

### The first attention runs

| run | network | seed | `caught_e` | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | `distance` |
|---|---|---|---|---|---|---|---|---|
| `c6de8e` | MLP | 0 | 0.213 [0.201, 0.226] | 0.538 | 0.031 | 0.013 | 0.206 | 5.25 m |
| `398c2a` | MLP | 1 | 0.252 [0.239, 0.266] | 0.326 | 0.030 | 0.034 | 0.357 | 4.76 m |
| `b087ac` | attention | 0 | 0.008 [0.005, 0.011] | 0.058 | 0.153 | 0.781 | 0.000 | 10.04 m |
| `da9e33` | attention | 1 | 0.009 [0.007, 0.013] | 0.088 | 0.074 | 0.822 | 0.007 | 10.21 m |

Training, the mean of each range of updates (`fixed_curves.py`):

| updates | 1–100 | 101–200 | 201–500 | 501–1000 | 1001–1500 | 1501–1907 |
|---|---|---|---|---|---|---|
| catch, `c6de8e` | 0.094 | 0.156 | 0.201 | 0.214 | 0.206 | 0.205 |
| catch, `398c2a` | 0.093 | 0.176 | 0.224 | 0.251 | 0.258 | 0.258 |
| catch, `b087ac` | 0.006 | 0.008 | 0.007 | 0.009 | 0.009 | 0.009 |
| catch, `da9e33` | 0.007 | 0.008 | 0.008 | 0.009 | 0.009 | 0.010 |
| `distance`, `c6de8e` (m) | 6.57 | 5.13 | 4.75 | 4.67 | 4.70 | 4.70 |
| `distance`, `398c2a` | 6.52 | 5.04 | 4.54 | 4.37 | 4.33 | 4.33 |
| `distance`, `b087ac` | 9.92 | 9.99 | 9.91 | 9.96 | 10.06 | 9.98 |
| `distance`, `da9e33` | 9.97 | 10.03 | 10.02 | 10.04 | 9.99 | 10.13 |
| `OOB_p`, `c6de8e` | 0.179 | 0.109 | 0.063 | 0.023 | 0.016 | 0.016 |
| `OOB_p`, `398c2a` | 0.182 | 0.054 | 0.054 | 0.031 | 0.031 | 0.031 |
| `OOB_p`, `b087ac` | 0.563 | 0.571 | 0.685 | 0.621 | 0.712 | 0.766 |
| `OOB_p`, `da9e33` | 0.534 | 0.646 | 0.642 | 0.693 | 0.698 | 0.708 |
| std, `c6de8e` | 0.920 | 0.751 | 0.358 | 0.086 | 0.035 | 0.023 |
| std, `398c2a` | 0.919 | 0.715 | 0.326 | 0.080 | 0.032 | 0.021 |
| std, `b087ac` | 0.909 | 0.744 | 0.558 | 0.344 | 0.201 | 0.120 |
| std, `da9e33` | 0.912 | 0.754 | 0.546 | 0.326 | 0.178 | 0.120 |
| `lr_p`, `c6de8e` | 3.3e-3 | 1.9e-3 | 6.9e-4 | 1.9e-4 | 8.4e-5 | 5.3e-5 |
| `lr_p`, `398c2a` | 3.3e-3 | 1.9e-3 | 7.0e-4 | 1.8e-4 | 7.0e-5 | 4.4e-5 |
| `lr_p`, `b087ac` | 5.1e-3 | 3.2e-3 | 2.5e-3 | 2.0e-3 | 1.7e-3 | 1.1e-3 |
| `lr_p`, `da9e33` | 4.4e-3 | 3.2e-3 | 2.7e-3 | 2.5e-3 | 1.7e-3 | 1.2e-3 |
| `policy_loss_p`, `c6de8e` | −0.0016 | −0.0033 | −0.0045 | −0.0045 | −0.0046 | −0.0045 |
| `policy_loss_p`, `398c2a` | −0.0019 | −0.0032 | −0.0038 | −0.0039 | −0.0039 | −0.0039 |
| `policy_loss_p`, `b087ac` | +0.0040 | +0.0043 | +0.0047 | +0.0049 | +0.0040 | +0.0046 |
| `policy_loss_p`, `da9e33` | +0.0038 | +0.0045 | +0.0050 | +0.0043 | +0.0050 | +0.0052 |

The flights of the pursuers, 512 eval episodes of each run (`fly.py`, `behav_*.py`). The
bearing row is the R² of the roll and pitch rate commands of the first 0.3 s against the
direction to the evader:

| | `b087ac` | `da9e33` | `c6de8e` | `398c2a` |
|---|---|---|---|---|
| yaw rate command, mean ± std | −0.66 ± 0.03 | −0.65 ± 0.14 | −0.41 ± 0.65 | +0.30 ± 0.57 |
| change of the action from step to step | 0.001–0.004 | 0.002–0.004 | 0.03–0.06 | 0.03–0.05 |
| the command follows the bearing, roll / pitch | 0.14 / 0.15 | 0.01 / 0.23 | 0.65 / 0.49 | 0.46 / 0.54 |
| episodes that end with a pursuer out of the arena | 398 | 410 | 6 | 17 |
| the exit: median time | 4.1 s | 5.0 s | — | — |
| the exit: median distance to the evader | 19.0 m | 19.9 m | — | — |
| the exit: the pursuer flies away from the evader | 92 % | 94 % | — | — |

- **The attention runs did not learn.** The catch is 0.008 and 0.009, against 0.213 and 0.252.
  `distance` stays at 9.9–10.1 m in every range of updates. In the MLP runs it falls to
  4.3–4.7 m.
- **A pursuer leaves the arena in 0.78 and 0.82 of the episodes.** It is 19–20 m from the
  evader at the exit, and in 92–94 % of the exits it flies away from it.
- **The pursuers give almost one action.** The yaw rate command is −0.66 ± 0.03 and −0.65 ±
  0.14, and the action changes by 0.001–0.004 from step to step, against 0.03–0.06. The
  bearing of the evader explains 0.01–0.23 of the roll and pitch commands, against 0.46–0.65.
- **`policy_loss_p` is above zero in every range of updates,** +0.0038 to +0.0052. In the MLP
  runs it is −0.0016 to −0.0046.
- **The std falls slower and the lr stays higher:** over updates 1501–1907 the std is 0.120,
  against 0.021–0.023, and `lr_p` is 1.1–1.2e-3, against 4.4–5.3e-5.

### Why they did not learn

The size of the Perceiver output over the updates of `b087ac`: the std over 1000 states that
`b087ac` flew, the mean over the 512 outputs (`audit_timeline.py`). In `da9e33` it is 2890
at update 1900.

| update | start | 50 | 500 | 1000 | 1900 |
|---|---|---|---|---|---|
| Perceiver output | 0.59 | 170 | 1356 | 2061 | 2022 |

The first layer of the trunk in `b087ac` at the end, by the part of the input (`audit_c.py`,
`check_fix.py`). The sum is the std of what the part adds to one unit before the tanh. The
shuffle is the RMS change of the mean action when the part is shuffled across the states:

| part of the input | its sum in a unit of the first layer | the action moves on a shuffle |
|---|---|---|
| own block | 2.8 | 0.000 |
| the evader | 2.7 | 0.000 |
| the teammates, through the Perceiver | 4.6e5 | 0.112 |

- **The Perceiver output grows from 0.59 to about 2000.** At update 50 it is 170.
- **From update 50 on, every unit of the first layer is saturated:** 100 % have |tanh| > 0.99.
  The teammates add 4.6e5 to a unit, the own block 2.8 and the evader 2.7.
- **The policy reads neither its own block nor the evader.** A shuffle of either moves the
  action by 0.000. In `c6de8e`, a shuffle of each part moves the action by 0.5–0.9.
- **The lr reaches its cap of 1e-2 in updates 1–2,** in all four runs.
- **A constant lr of 2e-4 holds the output** over 12 updates from the start (1024 envs,
  `audit_early.py`): it stays at 0.62–0.69. With the adaptive lr it goes from 0.62 to 2.4.
- **The checks that found no fault.** The own block and the evader's part are equal to the
  same columns of the MLP input (`audit_a.py`). On the first minibatch of an update, before a
  step, the PPO ratio is 1 for every row (`audit_ratio.py`).

### The norm

Commit `7a9878f` puts a LayerNorm with no scale and no bias on the output of each query. A
test on the Mac before the runs (`bed.py`): `paths_3v1`, seed 0, 80 updates of 1024 envs, the
flags of the runs. The numbers are the mean of updates 73–80:

| network | lr | `distance` | catch | `policy_loss_p` |
|---|---|---|---|---|
| MLP | adaptive | 5.84 m | 0.107 | −0.0044 |
| attention, no norm | adaptive | 9.94 m | 0.006 | +0.0048 |
| attention, no norm | 3e-4, constant | 5.82 m | 0.131 | −0.0055 |
| attention, a layer of 32 with tanh after the Perceiver | adaptive | 5.55 m | 0.110 | −0.0023 |
| attention, a LayerNorm with scale and bias | adaptive | 5.99 m | 0.073 | −0.0046 |
| attention, the LayerNorm of `7a9878f` | adaptive | 6.53 m | 0.071 | −0.0043 |

At update 80, on 1000 states that `b087ac` flew (`check_fix.py`):

| network | Perceiver output | shuffle: own block | shuffle: the evader | shuffle: the teammates |
|---|---|---|---|---|
| attention, no norm | 135.8 | 0.002 | 0.002 | 0.347 |
| attention, the LayerNorm of `7a9878f` | 0.63 | 0.277 | 0.393 | 0.401 |

- **The test shows the failure in 80 updates:** without the norm, `distance` stays at 9.94 m
  and the catch at 0.006, and the MLP reaches 5.84 m and 0.107.
- **Each of the three changes makes the network learn:** the constant lr, the layer of 32, and
  the LayerNorm. `distance` is 5.55–6.53 m and the catch 0.071–0.131.
- **With the norm the policy reads all three parts** at update 80: a shuffle moves the action
  by 0.277, 0.393 and 0.401. Without it, by 0.002, 0.002 and 0.347.
- **Limits.** One seed, 80 updates. The spread between the variants that learn (0.071–0.131)
  is one run each.

### The attention runs with the norm

| run | network | seed | `caught_e` | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | mean time to catch | `distance` |
|---|---|---|---|---|---|---|---|---|---|
| `c6de8e` | MLP | 0 | 0.213 [0.201, 0.226] | 0.538 | 0.031 | 0.013 | 0.206 | 4.5 s | 5.25 m |
| `398c2a` | MLP | 1 | 0.252 [0.239, 0.266] | 0.326 | 0.030 | 0.034 | 0.357 | 5.3 s | 4.76 m |
| `91b18c` | attention, norm | 0 | 0.326 [0.312, 0.341] | 0.448 | 0.035 | 0.026 | 0.164 | 4.5 s | 4.77 m |
| `2cdecf` | attention, norm | 1 | 0.267 [0.253, 0.280] | 0.307 | 0.021 | 0.012 | 0.394 | 4.9 s | 4.62 m |

Training, the mean of each range of updates, and of the last 100:

| updates | 1–100 | 101–200 | 201–500 | 501–1000 | 1001–1500 | 1501–1907 | last 100 |
|---|---|---|---|---|---|---|---|
| catch, `91b18c` | 0.091 | 0.157 | 0.245 | 0.319 | 0.338 | 0.337 | 0.338 |
| catch, `2cdecf` | 0.085 | 0.160 | 0.182 | 0.242 | 0.259 | 0.265 | 0.269 |
| `distance`, `91b18c` (m) | 6.51 | 4.92 | 4.38 | 4.16 | 4.10 | 4.11 | 4.12 |
| `distance`, `2cdecf` | 6.58 | 4.98 | 4.70 | 4.25 | 4.19 | 4.16 | 4.15 |
| std, `91b18c` | 0.914 | 0.685 | 0.303 | 0.070 | 0.027 | 0.017 | 0.015 |
| std, `2cdecf` | 0.905 | 0.653 | 0.305 | 0.076 | 0.031 | 0.020 | 0.017 |
| `lr_p`, `91b18c` | 3.0e-3 | 1.4e-3 | 5.5e-4 | 7.8e-5 | 2.6e-5 | 1.5e-5 | 1.4e-5 |
| `lr_p`, `2cdecf` | 2.7e-3 | 1.3e-3 | 5.3e-4 | 9.3e-5 | 3.5e-5 | 2.1e-5 | 1.8e-5 |
| `policy_loss_p`, `91b18c` | −0.0006 | −0.0021 | −0.0040 | −0.0059 | −0.0058 | −0.0056 | −0.0055 |
| `policy_loss_p`, `2cdecf` | −0.0011 | −0.0026 | −0.0034 | −0.0046 | −0.0048 | −0.0048 | −0.0048 |

Over the checkpoints, on 1000 states that `c6de8e` flew (`fixed_timeline.py`). The shuffle
columns are the own block / the evader / the teammates:

| update | 50 | 200 | 500 | 1000 | 1900 |
|---|---|---|---|---|---|
| attention output before the norm, `91b18c` | 15.3 | 30.1 | 35.1 | 35.4 | 35.3 |
| attention output before the norm, `2cdecf` | 15.0 | 27.6 | 32.4 | 32.7 | 32.8 |
| Perceiver output, `91b18c` | 0.58 | 0.47 | 0.45 | 0.45 | 0.45 |
| Perceiver output, `2cdecf` | 0.50 | 0.41 | 0.41 | 0.41 | 0.41 |
| first layer saturated, `91b18c` | 0.878 | 0.900 | 0.904 | 0.904 | 0.904 |
| first layer saturated, `2cdecf` | 0.867 | 0.886 | 0.893 | 0.893 | 0.893 |
| shuffle, `91b18c` | 0.42 / 0.27 / 0.29 | 0.49 / 0.44 / 0.35 | 0.55 / 0.57 / 0.45 | 0.57 / 0.61 / 0.46 | 0.57 / 0.61 / 0.46 |
| shuffle, `2cdecf` | 0.38 / 0.23 / 0.38 | 0.47 / 0.42 / 0.39 | 0.52 / 0.50 / 0.50 | 0.53 / 0.54 / 0.50 | 0.53 / 0.55 / 0.49 |

- **With the norm the attention network learns.** The catch is 0.326 and 0.267, `OOB_p` 0.026
  and 0.012, and `policy_loss_p` is below zero in every range.
- **It catches more than the MLP in both seeds,** by 0.113 (seed 0) and 0.015 (seed 1). Two
  seeds of one network differ by 0.039 (MLP) and 0.059 (attention).
- **The attention output before the norm still grows,** from 15 to 33–35. What the trunk gets
  stays at 0.41–0.58.
- **The policy reads all three parts** from update 50 to the end: a shuffle moves the action
  by 0.23–0.61. The first layer is 0.87–0.90 saturated. In `c6de8e`, on its own states, it is
  0.85.
- **The lr ends near its floor of 1e-5:** 1.4e-5 and 1.8e-5 over the last 100 updates, against
  4.3e-5 and 4.1e-5 in the MLP runs. The std ends at 0.015 and 0.017, against 0.020 and 0.018.
- **The training catch is flat from update 1000 on:** 0.338 and 0.337 (seed 0), 0.259 and
  0.265 (seed 1).
- **One update takes 1.25 s,** against 0.76 s for the MLP, so a run takes 43 min, against 27.
  `b087ac` and `da9e33` took 3.41 s for each update, in other jobs. The logs do not name the
  card.
- **Limits.** Two seeds. Every attention run has the adaptive lr, which catches 0.213–0.252
  with the MLP, against 0.605 and 0.837 with the constant lr and `ent_coef=0.003`. No
  attention run has those settings.

### One pursuer with the 3v1 weights

The last checkpoints of `91b18c` and `2cdecf`, flown in `paths_1v1` with no new training:
`run/eval.py --episodes 4096 --preset paths_1v1 --name paths_1v1_4096`, mean actions.

| run | seed | `caught_e` in 1v1 | `caught_e` in 3v1 | `crash_p` | `OOB_p` | `time_limit` | mean time to catch |
|---|---|---|---|---|---|---|---|
| `91b18c` | 0 | 0.322 [0.308, 0.337] | 0.326 | 0.033 | 0.004 | 0.640 | 5.8 s |
| `2cdecf` | 1 | 0.143 [0.132, 0.154] | 0.267 | 0.001 | 0.001 | 0.855 | 5.8 s |

- **The same weights fly one pursuer:** `OOB_p` is 0.004 and 0.001, and `crash_p` 0.033 and
  0.001.
- **Alone, seed 0 catches as often as three pursuers do:** 0.322, against 0.326. Seed 1
  catches 0.143, against 0.267.
- **Limits.** No run with more than three pursuers: there is no preset for it. No 1v1 run
  with these settings to compare with.

## The setups side by side, 2026-11-14

No new run. The table holds the eval at the last update (1907) of each `paths_3v1` run with
`gamma=0.99` and 5e8 steps: 1024 episodes, mean actions. The setups are sorted by their
average catch over the two seeds.

"Same code and reward" is yes for a run with the progress reward and the critic input for
each drone: the constant-lr runs (commit `fba6aec`), and `c6de8e` and `398c2a` (`18a4c40`).
It is no for the adaptive-lr sense runs of 2026-11-01 (`de90325`: before the critic input for
each drone, the reward before the progress reward) and for the adaptive-lr runs with
`ent_coef=0.003` of 2026-11-05 (`11324cb`: the reward before the progress reward, a `coll_pp`
cost of 30). Each sense run has `ent_coef=0.001`.

| the pursuer senses | lr | average catch | run | seed | catch | hits between pursuers | same code and reward |
|---|---|---|---|---|---|---|---|
| nothing, `ent_coef=0.003` | constant | 0.739 | `730968` | 0 | 0.628 | 33.7 % | yes |
| | | | `9eb432` | 1 | 0.850 | 12.5 % | yes |
| attitude and rpm | constant | 0.594 | `6e196e` | 0 | 0.638 | 29.3 % | yes |
| | | | `1390aa` | 1 | 0.550 | 35.7 % | yes |
| rpm | constant | 0.533 | `873189` | 0 | 0.522 | 30.4 % | yes |
| | | | `22b282` | 1 | 0.542 | 31.6 % | yes |
| nothing, `ent_coef=0.001` | constant | 0.487 (one seed) | `be848f` | 0 | 0.487 | 28.0 % | yes |
| | | | no run | 1 | — | — | — |
| attitude | constant | 0.411 | `b503f3` | 0 | 0.461 | 41.2 % | yes |
| | | | `740128` | 1 | 0.360 | 47.3 % | yes |
| nothing, `ent_coef=0.001` | adaptive | 0.225 | `c6de8e` | 0 | 0.194 | 56.1 % | yes |
| | | | `398c2a` | 1 | 0.255 | 32.5 % | yes |
| attitude | adaptive | 0.023 | `cabc36` | 0 | 0.030 | 53.0 % | no |
| | | | `05fb55` | 1 | 0.017 | 75.9 % | no |
| attitude and rpm | adaptive | 0.023 | `396719` | 0 | 0.017 | 32.4 % | no |
| | | | `7c45e0` | 1 | 0.030 | 59.0 % | no |
| nothing, `ent_coef=0.003` | adaptive | 0.017 | `2cda6a` | 0 | 0.006 | 28.2 % | no |
| | | | `503bac` | 1 | 0.027 | 22.8 % | no |
| rpm | adaptive | 0.008 | `150658` | 0 | 0.009 | 43.8 % | no |
| | | | `fade20` | 1 | 0.007 | 62.8 % | no |

- **Every constant-lr setup catches more than every adaptive-lr setup:** 0.411–0.739 against
  0.008–0.225.
- **One pair of setups differs only in the lr:** nothing sensed with `ent_coef=0.001` catches
  0.487 with the constant lr (one seed), against 0.225 with the adaptive lr.
- **Limits.** The rows with "no" differ from the constant-lr runs in more than the lr. The
  constant-lr base with `ent_coef=0.001` has one seed.

## Open

| question | state |
|---|---|
| why 3v1 does not learn | with the Gaussian, the spread grows in every 3v1 run with both fixes, also with `ent_coef=0`. The FRPN test points to noise that keeps the pursuers apart. The Beta (2026-11-03) kept the spread small and the pursuers alive, but they held back about 8 m from the evader and caught as rarely. The code is back on the Gaussian. Input scaling (2026-11-04) did not change the catch, and the spread grew more. ReLU (2026-11-04) stopped the blow-up, but the pursuers held still and caught less. The critic input for each drone (2026-11-04) stopped the blow-up and lifted the catch to 0.10–0.24, the best without scaling. The pursuers still stop 3–5 m short in most episodes. 10 times the steps (2026-11-04) left the catch at 0.18–0.26: it goes flat by 2.5e8–8e8 steps, the std falls to 0.006, and the lr reaches its floor. The pursuers catch a `hover` no more often than a moving evader. `gamma=0.995` (2026-11-04) gave no gain: the catch is 0.16–0.18. The freeze checks (2026-11-05): the frozen policy is not at a top, because small steps toward FRPN raise its score at γ = 0.99. The std falls as about c / update and the lr follows it, so the policy stops moving. `ent_coef=0.003` (2026-11-05) blew the std up. A `coll_pp` cost of 10 (2026-11-10) left the catch at 0.17–0.25, and the hits rose. The progress reward (2026-11-10) left the catch at 0.21–0.25 and raised the hits to 0.32–0.54. All six policies hold their nets 2.8–8.6 m from the evader and fly around it. The probes (2026-11-10): without the net kill the same policies catch 0.03 more, the ring stays with ghost teammates, and closing in from the ring gives a higher own return. A constant lr of 5e-4 (2026-11-10) lifted the catch to 0.498, and 0.562 at the best checkpoint, with one seed. netpass lifted it to 0.388 with the adaptive lr. With the constant lr, `coll_pp` is the largest failure (0.55–0.80 of the episodes with no catch), and the catch falls after its best checkpoint. `ent_coef=0.003` with the constant lr (2026-11-11) lifted the catch to 0.605 and 0.837 (seeds 0 and 1), and to 0.682 and 0.910 at the best checkpoints: the pursuers go in, `coll_pp` is 0.83–0.91 of the episodes with no catch, and the catch still falls after its best checkpoint. Not run yet: the body catch. Other option: hold the std at 0.3. FRPN catches 0.44–0.51 in 3v1, so the task can be done |
| a sensed block in 1v1 | a hint of faster catches. More seeds, or the time to catch at a fixed `caught_e`, can tell it from faster learning |
| the sensing steps in 3v1 | with the constant lr (2026-11-11), two seeds each: the attitude catches 0.37–0.44, the rpm 0.55, both 0.58–0.64, against 0.498 for one base seed. No policy reads the rpm. The base has one seed, and no sense run has `ent_coef=0.003` |
| does a pursuer that senses more beat learned evaders more often | yes, in the 6 runs of `default_p_*`. The base `9e4242` is one seed on older code |
| what a sensed block gives a side | the 18 runs of the grid, not run yet |
