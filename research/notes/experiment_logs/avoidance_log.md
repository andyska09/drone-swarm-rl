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

## Open

| question | state |
|---|---|
| Other-Play: a left-right mirror for drone B, against a convention such as "always left" | not built. Add it only if the debug runs show a convention |
| a rule tied to the world heading can split the drones without the sensor | accepted risk, not checked yet. See the reasons above |
| a test drone that flies straight to its goal and never dodges, to find a half dodge | later |
| tests | none yet |
