# Task avoidance — run log

The runs of the avoidance task, their results, and why the spec changed. The spec is in
[task_avoidance.md](../task_avoidance.md).

## Why the task exists, 2026-10-20

The task replaces the tunnel ([task_tunnel.md](../task_tunnel.md)). We trained three
tunnel runs: `d_inf`, 1e9 steps, seeds 0–2 (`7c1d6b`, `e0b72d`, `c027a4`).

- Training stopped improving after about 4e8 steps. It reached 35% success, 56% wall
  and 9% contact.
- In eval from the far ends of the legs, success was 0–0.5% and wall 93–100%. The drones
  sped up to 4.5–6 m/s and hit the first bend.
- In seed 1, 61% of the wall deaths came less than 1.5 m from a bend.

So the score mixed two skills: flying the tube and dodging. In open space there are no
walls, and only the sight range sets the warning time.

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

Next: train `d_inf` with the rule, 1e9 steps, seeds 0–2. If the spread blows up again,
try `--set ent_coef=0`.
