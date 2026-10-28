# chase_modal — run log

What each chase_modal run was and what it scored, and why the setup is as it is. The task
is now the paths presets of [task_chase.md](../tasks/task_chase.md).

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

## Open

| question | state |
|---|---|
| why 3v1 does not learn | candidates, none tested: every step of sensing gives the same `coll_pp`, so the reward (a `coll_pp` costs 10, a crash 30); the evader moves between the slots of the obs; the spread does not fall (`--set ent_coef=0` stopped its blow-up in avoidance). FRPN catches 0.44 in 3v1, so the task can be done |
| a sensed block in 1v1 | a hint of faster catches. More seeds, or the time to catch at a fixed `caught_e`, can tell it from faster learning |
| the sensing steps in 3v1 | cannot be read until 3v1 learns |
