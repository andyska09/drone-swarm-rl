# chase_modal — run log

Every run in which a drone senses more of the other drones than their offset and velocity:
what it was, what it scored, and why the setup is as it is. The presets are in
[task_chase.md](../tasks/task_chase.md). The runs with the base obs are in
[chase_pools_log.md](chase_pools_log.md).

## Facts

- The number to beat in 3v1: eval `caught_e` 0.183 and 0.242 (`2d0b98`, `6fce56`; base `paths_3v1`, `gamma=0.99`, `normalize_obs=false`, `coll_pp` cost 30, 5e8 steps). FRPN catches 0.51. 2026-11-04 ([§](#long-runs-with-a-critic-input-for-each-drone-2026-11-04))
- The critic input for each drone lifted eval `caught_e` from 0.01–0.04 to 0.10–0.11 with input scaling and 0.18–0.24 without; the std no longer blows up (0.03–0.10). 2026-11-04 ([§](#runs-with-a-critic-input-for-each-drone-2026-11-04))
- 10 times the steps did not lift the catch (5e9 steps: 0.184 and 0.265). It goes flat early, so 5e8 steps are enough for a test. 2026-11-04 ([§](#long-runs-with-a-critic-input-for-each-drone-2026-11-04))
- `gamma=0.995` gave no gain: eval `caught_e` 0.157 and 0.176, against 0.183 and 0.242 at `gamma=0.99`. 2026-11-04 ([§](#runs-with-gamma0995-2026-11-04))
- The progress reward (0.25 per m, a catch of 30) did not lift the catch: 0.212 and 0.253 in 4096 episodes; over the three rewards, ANOVA p = 0.83. It catches sooner (median 4.2 and 5.2 s, against 5.5–6.5 s), and `coll_pp` is 0.32–0.54. 2026-11-10 ([§](#runs-with-the-progress-reward-2026-11-10))
- Every learned 3v1 policy holds its nets on a ring: 2.8–4.1, 4.3–6.4 and 5.7–8.6 m from the evader at 4–8 s. With a net 2–5 m away, it flies 2.1–3.8 m/s around the evader and 0.26–0.45 m/s toward it. FRPN closes at 4.8–6.6 m/s. 2026-11-10 ([§](#runs-with-the-progress-reward-2026-11-10))
- The learned 3v1 catch does not depend on the evader's path (hover 0.16–0.26, figure-eight 0.18–0.28); FRPN catches 0.60 and 0.41. 2026-11-10 ([§](#runs-with-the-progress-reward-2026-11-10))
- The learned `coll_pp` come a median 3.1–4.1 m from the evader, and the second-nearest pursuer is in 92–96 % of them. 2026-11-10 ([§](#runs-with-the-progress-reward-2026-11-10))
- A `coll_pp` cost of 10 did not lift the catch: eval `caught_e` 0.250 and 0.166, against 0.183 and 0.242 at 30. Eval `coll_pp` rose to 0.18 and 0.47. 2026-11-10 ([§](#runs-with-a-coll_pp-cost-of-10-2026-11-10))
- No `ent_coef` holds the std (two values): at 0.001 it falls to 0.03–0.05; at 0.003 it blew up to 124–18010 and eval `caught_e` fell to 0.006–0.027. 2026-11-05 ([§](#runs-with-ent_coef0003-2026-11-05))
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

## Open

| question | state |
|---|---|
| why 3v1 does not learn | with the Gaussian, the spread grows in every 3v1 run with both fixes, also with `ent_coef=0`. The FRPN test points to noise that keeps the pursuers apart. The Beta (2026-11-03) kept the spread small and the pursuers alive, but they held back about 8 m from the evader and caught as rarely. The code is back on the Gaussian. Input scaling (2026-11-04) did not change the catch, and the spread grew more. ReLU (2026-11-04) stopped the blow-up, but the pursuers held still and caught less. The critic input for each drone (2026-11-04) stopped the blow-up and lifted the catch to 0.10–0.24, the best without scaling. The pursuers still stop 3–5 m short in most episodes. 10 times the steps (2026-11-04) left the catch at 0.18–0.26: it goes flat by 2.5e8–8e8 steps, the std falls to 0.006, and the lr reaches its floor. The pursuers catch a `hover` no more often than a moving evader. `gamma=0.995` (2026-11-04) gave no gain: the catch is 0.16–0.18. The freeze checks (2026-11-05): the frozen policy is not at a top, because small steps toward FRPN raise its score at γ = 0.99. The std falls as about c / update and the lr follows it, so the policy stops moving. `ent_coef=0.003` (2026-11-05) blew the std up. A `coll_pp` cost of 10 (2026-11-10) left the catch at 0.17–0.25, and the hits rose. The progress reward (2026-11-10) left the catch at 0.21–0.25 and raised the hits to 0.32–0.54. All six policies hold their nets 2.8–8.6 m from the evader and fly around it. Next: the body catch (2026-11-10). Other options: hold the std at 0.3, or a fixed lr of 5e-4 as in Gavin's 1v1 paper. FRPN catches 0.44–0.51 in 3v1, so the task can be done |
| a sensed block in 1v1 | a hint of faster catches. More seeds, or the time to catch at a fixed `caught_e`, can tell it from faster learning |
| the sensing steps in 3v1 | cannot be read until 3v1 learns |
| does a pursuer that senses more beat learned evaders more often | yes, in the 6 runs of `default_p_*`. The base `9e4242` is one seed on older code |
| what a sensed block gives a side | the 18 runs of the grid, not run yet |
