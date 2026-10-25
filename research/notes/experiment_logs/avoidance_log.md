# Task avoidance — run log

The runs of the avoidance task, their results, and why the spec changed. The spec is in
[task_avoidance.md](../tasks/task_avoidance.md).

## Why the task exists, 2026-10-20

The task replaces the tunnel ([task_tunnel.md](../tasks/task_tunnel.md)). The tunnel runs hit
the walls, mostly at the first bend, before any meeting ([tunnel_log.md](tunnel_log.md)).
So the tunnel score mixed two skills: flying the tube and dodging. In open space there
are no walls, and only the sight range sets the warning time.

## Design decisions, 2026-10-20

| decision | why |
|---|---|
| a moving point at a set speed, not pay per metre | The tunnel paid per metre, and its policy flew as fast as it could. Pay for "on the line" alone lets a drone hover forever. A finish bonus alone makes the drone rush, because the discount makes an early bonus worth more. The moving point pays for accuracy only, and it sets the speed. |
| `target` holds the offset and the velocity difference to the moving point | SimpleFlight ([arXiv:2412.11764](https://arxiv.org/abs/2412.11764)) gives the path the same way, as its next 10 points. One point is enough for a straight line. |
| every episode starts at the start of the lines, in training too | A straight line is easy to fly. In the tunnel, random starts skipped the meeting in 40% of the episodes. |
| the meeting is picked first | As in the encounter models of Kochenderfer et al. 2010. A random meeting place and a random distance hide where the meeting will be. |
| miss 0–1 m, angle 90–180° | 80% of the encounters need a dodge. Angles below 180° break the head-on symmetry. |
| pay `exp(−d / 1 m)`, never negative | A crash never saves cost. |
| crash 30, no finish bonus | A drone that survives already earns the pay of every step. A finish bonus is a second setting for the same job. |
| the drones start at 3 m/s | From a hover, the moving point drives away. The drone then loses pay that has nothing to do with the dodge. |
| a flipped drone keeps flying | As in the tunnel and in `chase`. |
| the moving point has its own clock, `travelled` | In the first smoke run, episodes lasted 10 steps with 100% success. `vecenv.reset` sets `time` to a random value in the first episode of each env, and the moving point moved with `time`. |

## First runs, without the 2 m rule

`d_inf`, 1e9 steps, commit `fc963e2`. Eval: 1024 episodes, mean actions.

| run | seed | training success, last update | eval success | out | contact | distance to the moving point | success against the line follower |
|---|---|---|---|---|---|---|---|
| `f85270` | 0 | 91% | 1.7% | 97.6% | 0.8% | 3.3 m | 50.2% |
| `fe7ee8` | 1 | 95% | 66.6% | 31.5% | 2.0% | 4.5 m | 99.3% |
| `892728` | 2 | 89% | 54.3% | 0.2% | 45.5% | 3.7 m | 100% |

- **The drones did not follow their moving points.** In seed 1, the drone drifted about
  1 m in the first second. At the meeting, it was 3–7 m away. In the episodes where the
  lines pass closer than 0.8 m, the seed 1 drones came closer than 0.8 m in only 0–1.3%.
  Two line followers touch in 79% of them. So a success mostly meant "stayed in the
  arena for 4 s". The drift was already there at update 200.
- **The pay gave no pull back.** `exp(−d / 1 m)` is almost flat beyond 2 m. At 4 m, 1 m
  closer adds only 0.03 per second. A crash costs 30.
- **The spread of the actions blew up.** The entropy rose from 5.7 to 22–26, so the std
  grew from 1 to about 60. The adaptive lr stayed at its cap of 1e-2. The actions are
  clipped to [−1, 1], so in training almost every action was a random full push. At the
  end, the mean action was at the clip in 96–100% of the steps. That is why the eval
  with mean actions is much worse than training.
- Against the line follower, contact was 0%. But the policy did not dodge. Its drone was
  3–4 m off its line.

## The 2 m rule

- A drone that gets more than `max_point_distance` = 2 m from its moving point dies and
  pays the crash cost. The metric is `lost`.
- So drifting away is a failure. A success now means that the drone stayed near its line
  and passed the other drone.
- A dodge needs 0.8 m, so 2 m leaves room.
- We checked the rule. Two line followers are never lost, and they still touch in 78.8%
  of the episodes.

## Runs with the 2 m rule, 2026-10-20

`d_inf`, 1e9 steps, commit `a6907c4`, seeds 0–1. Eval: 1024 episodes, mean actions.
"Best" is the update with the highest training success.

| run | seed | checkpoint | success | contact | lost | distance to the moving point | against the line follower: success | contact |
|---|---|---|---|---|---|---|---|---|
| `70245b` | 0 | 1000, best | 98.9% | 1.1% | 0% | 0.66 m | 72.8% | 27.2% |
| `70245b` | 0 | last | 98.7% | 0.9% | 0.4% | 0.81 m | 74.6% | 25.4% |
| `0248af` | 1 | 1800, best | 85.6% | 5.1% | 9.3% | 1.08 m | 93.0% | 6.3% |
| `0248af` | 1 | last | 67.8% | 4.2% | 28.0% | 0.74 m | 58.3% | 26.0% |

- **The rule worked.** The drones now stay near their moving points: 0.7–1.1 m on
  average, against 3.3–4.5 m before. Training success rose to 97% in both seeds. Seed 0
  got there by update 600, seed 1 by update 1600.
- **The drones learned a habit, not a dodge.** Near the meeting, both drones swerve
  0.6–0.9 m to their own left. They do it in every kind of meeting, also when the lines
  miss by 0.8–1 m and no dodge is needed. In self-play this works, because both drones
  swerve the same way.
- **The line follower test catches the habit.** Seed 0 at update 1000, against the line
  follower:

  | the follower's line | contact |
  |---|---|
  | on the left | 56% |
  | on the right | 1.9% |
  | no dodge needed | 33% |

  The swerve moves the drone into a follower that would have missed it.
- Seed 1 also drops 1.3 m below its moving point near the meeting, and it stays low
  after it. So it passes under the follower, and its follower test at update 1800 is
  better: 6.3% contact. This is a second habit, not a dodge.
- **The spread blew up again.** The entropy rose from 5.6 to 21–26. Training success
  fell from 97% to 70% in seed 0 and to 84% in seed 1. At the end, the mean rate actions
  sat at the clip in 97–100% of the steps. This time the lr stayed at 1e-3 to 4e-3, under
  its cap. So the cap is not the cause.

Next: not decided. The options are a line follower in some training episodes, against
the habit, and `--set ent_coef=0`, against the spread.

## Redesign, 2026-10-21

The task is now a closed square tube. The spec is in
[task_avoidance.md](../tasks/task_avoidance.md). Everything above is about the old design.

Why:

- The task is a measuring instrument. The input is the sensor design, and the reading is
  the success rate. Only the sensor may change the reading. The range must be wide: a
  blind drone fails often, and a drone that sees everything almost always succeeds.
- The reward must state only the goal, with no shaping terms. The moving point had a
  shaped pay, the `exp` of the distance.
- In self-play, the two drones can agree on a convention ("always left") and pass
  without sensing much, as the runs above did. The known fix is Other-Play (Hu et al.
  2020, [arXiv:2003.02979](https://arxiv.org/abs/2003.02979)), a left-right mirror for
  drone B. The user decided to start without the mirror: debug runs first, and the
  mirror only if a convention shows up.

| decision | why |
|---|---|
| +1 for the arrival, −1 for a crash, 0 at every other step | It states only the goal. |
| the 10 s limit is terminal, and it counts with the task's own `steps` | A drone that has not arrived must get nothing. `vecenv.reset` gives the first episode of each env a random `time`. The first moving-point version hit exactly this: 10-step episodes with 100% success. |
| an arrived drone leaves the episode. The task gives `arrived` and `arrived_this_step`, and `vecenv` trains a drone while it is alive and has not arrived | It must stop giving training steps, but it is not a death. So `alive` and `died_this_step` keep their meaning, and `death_rate` stays right. |
| the tube grows only when every episode of the update flew the current tube | The episodes that started before a change fly the short tube for up to 16 updates (10 s = 1000 steps, 64 steps per update). Without the check, the tube can grow 5 → 10 → 15 → 20 m in three updates in a row. |
| the runner builds the update again when the tube grows | The params are constants in the compiled update. One rebuild compiled in about 2 s on the Mac (CPU), and a run has 3. `ppo.py` does not change. |
| the crash is the 0.40 m ball, and the starts are in the middle 1.6 m × 1.6 m | The ball is the rule of the tunnel and of the 0.80 m contact. In a probe of 20,000 starts where the cascade holds the start point and the start heading, 0.9% of the starts in the middle 2 m × 2 m hit a wall, and 0% in 1.6 m × 1.6 m. |
| `own` keeps the full `R`, and the tube has a random yaw | The user rejected a tilt-only `own`. |
| `core.own_obs` takes the wall distances, and `core.box_dist` measures them in any box | The tube measures its own walls with the code of the arena, with no copy in the task. |

**The cascade is no "never dodges" test here.** In both seats, the cascade flies each
drone straight to its goal. 512 episodes, `d_inf`:

| tube | heading | success | contact | wall |
|---|---|---|---|---|
| 5 m | turns to 0 (`seats.cascade_policy`) | 27.5% | 51.2% | 21.3% |
| 5 m | keeps its start heading | 13.9% | 85.0% | 1.2% |
| 20 m | turns to 0 | 4.1% | 3.7% | 92.2% |
| 20 m | keeps its start heading | 9.2% | 4.7% | 86.1% |

At 20 m, the cascade swings more than 1.1 m sideways and hits a side wall, in the
median after about 1 s (checked for the version that turns to 0).

**Local smoke run.** `d_inf`, `--set ent_coef=0`, 11 updates of 4096 envs × 64 steps,
uncommitted code on top of `2b49f01`, about 42,000 steps/s on the Mac (CPU). The code
runs, and nothing is learned yet. At update 11: 92% wall, 8% contact, 0.2% success, and
the tube is still 5 m. The entropy fell from 5.67 to 5.49. The lr sits at its cap of
1e-2 from update 1. Eval in the 20 m tube: 99.8% wall.

A small run with the grow rule forced on (`grow_at` = −1, 256 envs) checked the
curriculum: the tube grew 5 → 10 → 15 → 20 m, and it waited while episodes of the old
tube still ended.

Next: short debug runs on MetaCentrum.

## Reasons for the spec, 2026-10-25

These reasons moved here from the spec.

- The literature behind the sight range `d` is in
  [literature/avoidance.md](../literature/avoidance.md). Sight has no field of view, no
  blocking and no memory.
- The two straight paths are the same line: an exact head-on. The random start state
  breaks the tie.
- The speed is free. With `gamma` = 0.998, each second of delay multiplies the value of
  the +1 by 0.82. So the drone still has a reason to be fast.
- Runs set `ent_coef=0` against the blow-up of the action spread.
- The tube has no random starts along its length.
- The world-heading risk: `R` tells each drone its world heading, and the two drones
  always fly in opposite world directions. So a rule like "the drone that flies toward +x
  goes up" can split them without the sensor. Check it in the debug runs: compare the
  height offset at the meeting with the flight direction.
- The straight-flying test drone is there to find a half dodge. The cascade in both seats
  is not this test (see the redesign above).

## Debug runs, 2026-10-21 to 2026-10-25

15 runs on MetaCentrum, `--set ent_coef=0`, 1e9 steps (3814 updates of 4096 envs × 64
steps), 11–23 min each. `d_inf` s0 ran on `514dc38`, the rest on `c080ac8`. `swarm/` and
`run/` are the same in both. Eval: 1024 episodes, mean actions, the 20 m tube, the latest
checkpoint. For the s1 runs of `d6` to `d0`, the eval success comes from the probe
`scratch/avoidance_habits.py`, which flies the same 1024 episodes. "Training success" is
the mean of the last 300 updates.

| preset | seed | run | tube reached 20 m after update | training success | eval success | habit |
|---|---|---|---|---|---|---|
| `d_inf` | 0 | `87c03a` | 768 | 91.9% | 93.1% | wait |
| `d_inf` | 1 | `ad2036` | never, 10 m from update 319 | 0% | 0% | hover |
| `d_inf` | 2 | `13b825` | never, 10 m from update 162 | 0% | 0% | hover |
| `d_inf` | 3 | `785eb3` | 1112 | 94.7% | 95.5% | wait |
| `d6` | 0 | `bac8fc` | 765 | 98.7% | 99.3% | lanes, right |
| `d6` | 1 | `96c3a1` | 460 | 98.5% | 98.9% | lanes, left |
| `d5` | 0 | `c74fd1` | never, 10 m from update 286 | 0% | 0% | hover |
| `d5` | 1 | `4d2151` | 1000 | 98.6% | 97.5% | lanes, left |
| `d4` | 0 | `eab9e9` | 480 | 99.2% | 99.3% | lanes, left |
| `d4` | 1 | `fd3d1a` | 287 | 99.0% | 98.7% | lanes, right |
| `d3` | 0 | `7b7078` | 792 | 97.9% | 98.7% | lanes, left |
| `d3` | 1 | `134884` | 617 | 98.0% | 99.0% | lanes, left |
| `d0` | 0 | `874a5c` | 631 | 84.2% | 85.0% | lanes, right. In 51% of the episodes, one drone also holds back |
| `d0` | 1 | `56c2bb` | 530 | 94.2% | 93.7% | lanes, left |

"Lanes, left" means that each drone keeps to its own left side of the tube.

- **No run learned a dodge.** The policy learned three habits: lanes, wait and hover.
- **Lanes, 9 runs.** Each run uses the same side in 99–100% of its episodes: 6 runs keep
  left, 3 keep right. In 8 of these runs, both drones fly at about 7–12 m/s and pass at
  the same time, near the middle of the tube (median $\lvert \tilde x \rvert$ 0.3–0.9 m at
  the closest approach, median arrival gap 0.04–0.16 s). At the closest approach, they
  are 1.2–1.35 m apart across the tube. `d0` s0 passes at median
  $\lvert \tilde x \rvert$ 5.6 m, with a median arrival gap of 0.84 s.
- **The blind drones succeed in 85% and 94% of the episodes.** Lanes need no sight. So the
  success rate barely depends on `d`, and the task does not measure the sensor now.
- **Wait, `d_inf` s0 and s3.** In all successful episodes, the two arrivals are more than
  0.8 s apart (median 1.55 s and 1.46 s). One drone hovers (median speed 0.72 and
  0.89 m/s) until the other drone arrives. The first drone always passes over the waiting
  drone, at the waiting drone's end of the tube (median $\lvert \tilde x \rvert$ 8.3 and
  8.1 m). Which drone waits is random: drone 0 goes first in 50% and 51% of the episodes.
  The drone that starts lower goes first in 63%.
- **The wait pays because sight and contact skip an arrived drone.** The second drone then
  flies an empty tube. 1.6 s of waiting costs only the discount, about 27% of the +1.
- **Hover, 3 runs.** All three got stuck right after the tube grew from 5 m to 10 m. At
  10 m, 87–100% of the episodes ended at a wall. About 100–300 updates later, both drones
  hovered until 10 s (99.5% timeouts in the last 300 updates). A timeout pays 0 and a death −1, and a
  drone that never arrives never gets the +1. The entropy kept falling (`d5` s0: 2.2 to
  −9.6). In the 20 m eval of `d5` s0, the drones creep toward each other at 1–1.7 m/s and
  stop 5–17 m apart. In the 8 replay episodes, they mostly end at the ceiling.
- **The world heading does not split the drones.** The lane side and the waiting drone
  show no link to the tube heading. In some lane runs, the sign of the height offset at the
  closest approach shows a weak link: drone 0 is higher in 44% of the episodes with a
  heading below π and in 59% above π (`d0` s1). This offset is small (median 0.2–0.3 m),
  and the 1.2–1.35 m across the tube keeps the drones apart.
- Guess: only `d_inf` learns the wait because the two drones must agree which one goes
  first, so each must see the other at the start. Only `d_inf` sees 18 m.
- Guess: sight matters little at these speeds. The drones close in at 18–20 m/s, so 3 m
  of sight gives about 0.15 s of warning, and 6 m about 0.3 s.
- `d_inf` s0: the entropy fell from 5.5 to −4.1, with no blow-up. The lr fell from its cap
  of 1e-2 to 1e-3. `explained_var` fell below 0 after about update 1500 (−0.13 at the
  end). Not checked: `explained_var` also counts the steps after a drone arrived, which
  the critic does not train on.
- The first `d_inf` s1 job (2026-10-21) hung in the JAX GPU check on `alfrid1` (L40) for
  the full 1 h walltime and made no run folder. `job.sh` now stops that check after 5 min
  (`c080ac8`).

Next: the user decides the fixes (see Open).

## Other-Play, 2026-10-25

Built against the lanes. Not run yet. Drone 1 sees and acts through a random map of the
world, $F = R_z(\alpha)\, S^m$: a turn by $\alpha \sim U[0, 2\pi)$, after a mirror of the
world $y$ axis in half of the episodes. The spec has the formulas. The user chose the turn
with the mirror, the action signs for the cascade, and the first runs: `d0` and `d_inf`,
seeds 0 and 1.

| decision | why |
|---|---|
| the mirror | "keep left" puts both drones on the same side when one of them plays in the mirror. So lanes pass in about half of the episodes at most |
| the turn | the world heading is a coin that the mirror keeps. "If I fly toward world $+x$, I go up" splits the drones in every episode, because they fly in opposite directions, and $y \to -y$ keeps the $x$ part of a heading. With the turn, drone 1's heading says nothing about drone 0's |
| the mirror flips the world $y$ axis, not the tube's | it needs no tube heading. Every mirror in a vertical plane swaps left and right, so each one breaks the lanes |
| only drone 1 gets a map | the game looks the same through any map of the whole world, so only drone 1's map relative to drone 0's counts |
| the map is an exact symmetry | the X500, the rate loop, the mixer and the re-orthonormalization are their own mirror images (motors 1↔3 and 2↔4), and the tube heading is uniform. `scratch/avoidance_mirror_check.py`: drone 1's obs through its map equals its plain obs in the mapped world (gap 1e-14), and the mapped world stays the image of the true world, step after step (1e-12) |
| eval uses the map | the same game as training. Without it, a lanes policy scores about 100% in eval and hides what we test |
| the critic gets $m$, $\cos\alpha$ and $\sin\alpha$ of each drone | the critic sees the whole game, and drone 1's moves depend on its map |
| the cascade multiplies its action by the env's action signs | it reads the true state, not the obs, so the sign change of drone 1's roll and yaw rates broke it. Now its path is the same with and without the map (gap 0). The cascade in both seats, 1024 eval episodes: 4.3% success, the same in the mirrored and the plain episodes |

- A trap: eval records drone 1's action as its seat sent it, through the map. In a
  mirrored episode, the roll and yaw rates in `trajectory.npz` have the opposite sign of
  what the drone flies.
- Guess: Other-Play stops shared rules, but not a drone's own coin. A blind drone can fly
  to the tube corner nearest its random start point. Two drones pick the same corner in
  1 of 4 episodes, so a blind drone can still pass in about 75% of the episodes or more.
- Guess: a dodge by sight still works through the map. "Move away from the other drone"
  is its own mirror image.
- Local check on `9ca5f98` plus these changes: `pytest tests/` passes (53), and a 3-update
  `d0` run trains and evals.

## Open

| question | state |
|---|---|
| Other-Play against a convention such as "always left" | built: the mirror and the turn for drone 1 (see Other-Play above). Not run. Next: `d0` and `d_inf`, seeds 0 and 1 |
| a rule tied to the world heading can split the drones without the sensor | checked in the debug runs: the lanes and the wait do not use it. A weak link in the small height offset of some lane runs. The turn of Other-Play now hides drone 1's world heading |
| a blind drone can use its own random start as a coin | guess: about 75% success or more for `d0`, even with Other-Play. Check in the runs |
| the wait in `d_inf` | candidate: when the first drone arrives, the other drone must be past the middle of the tube, or the episode ends and it gets nothing. Not built. The mirror does not stop the wait. An arrival window needs a value that changes with the tube length |
| the hover after the tube grows to 10 m, in 3 of 15 runs | candidate: a timeout pays −1, the same as a death. Not built |
| a test drone that flies straight to its goal and never dodges, to find a half dodge | later |
| tests | none yet |
