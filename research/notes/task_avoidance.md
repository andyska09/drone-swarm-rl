# Task avoidance — the spec

Two drones fly straight lines that meet in open space. Each drone must stay near a point
that moves along its own line at a set speed, and must not touch the other drone. Built
in [avoidance.py](../../swarm/envs/avoidance.py).

**What the task is for.** The experiment variable is what a drone senses about the
other drone. The first design is the exact relative position and velocity, inside a
sight range `d`. Later designs change only that input, and each design is scored on the
same encounters. The literature is in [lit_review_tunnel.md](lit_review_tunnel.md).

The task replaces the tunnel ([task_tunnel.md](task_tunnel.md)). In the tunnel most
drones died at the walls, so the score mixed wall flying with dodging. Here nothing but
the other drone stands in the way, and the sight range alone sets the warning.

## The encounter

Each episode picks the meeting first and then puts the lines around it, as the aircraft
encounter models do (Kochenderfer et al. 2010,
[10.2514/1.44867](https://doi.org/10.2514/1.44867)).

| | value |
|---|---|
| meeting place | uniform in a ball of `meet_radius` = 2 m around `center` |
| approach angle | the angle between the two flight directions, uniform from `angle_min` = 90° to `angle_max` = 180°, to either side |
| miss distance | how close the two lines pass, uniform from 0 to `miss` = 1 m, in a random direction across the closing direction |
| distance before the meeting | the same for both drones, uniform from `before_min` = 6 m to `before_max` = 12 m |
| distance after the meeting | `after` = 4 m |
| lines | straight and level. The first line's heading is uniform over the full circle |

Both drones start the same distance before the meeting place and fly at the same speed,
so they reach it together. The miss distance alone then decides whether they touch:
below 0.80 m between centres. Over 100,000 resets, 80% of the encounters need a dodge.
The drones start at least 8.5 m apart, and every start is at least 1.5 m inside the
arena box.

The meeting place and the distance before it are random, so a drone cannot tell from
its own position where the meeting will be. A dodge at a fixed place fails.

## The moving point

Each drone has a point that starts where the drone starts and moves along the drone's
line at `cruise` = 3 m/s. The reward pays for being close to it. So the drone is paid for
being in the right place at the right time, not for speed. The episode ends when the
points reach the ends of the lines.

The point keeps its own clock, `travelled`, in metres. `time` is only the step counter
for the time limit, because `vecenv.reset` sets `time` to a random value in the first
episode of each env to spread the timeouts.

The drones start already moving along their lines at 3 m/s. `core.spawn` adds its random
velocity (up to 0.5 m/s on each axis), a random yaw and a tilt up to 0.2 rad.

## Sight

A drone sees the other drone when it is closer than `sight` (the `d` of the lit
review), in any direction. There is no field of view, no blocking and no memory. Out of
sight, the `others` row is masked to zero. With `d_inf` the drones see each other the
whole time. With `d6` and below they start out of sight.

How much sight a head-on dodge needs at 3 m/s each, a closing speed of 6 m/s, from the
arithmetic in [lit_review_tunnel.md](lit_review_tunnel.md) section 1 with a sideways push
of 10 m/s²: 2.7 m if both drones dodge, 3.7 m if one dodges alone. So `d3` is near the
limit in self-play and past it against the line follower.

## Observation

| block | width | content |
|---|---|---|
| `own` | 20 | `core.own_obs`: body-frame velocity, `R`, `ω`, height, and the distances to the arena walls |
| `others` | 1 × 7 | the other drone: body-frame offset, body-frame velocity, and the role flag (always 1). Masked to zero out of sight |
| `target` | 6 | `core.target_obs` to the moving point: its body-frame offset and the body-frame velocity difference. The line is straight, so the point's velocity also gives the line's direction |
| `scene` | 51 | the critic's input: `core.scene_obs`, both moving points and their velocities, the metres left to the end, and whether each drone sees the other |

Other work gives a path to a policy the same way. SimpleFlight
([arXiv:2412.11764](https://arxiv.org/abs/2412.11764)) gives the next 10 points of the
path. One point is enough for a straight line.

## Action

CTBR, as in every task.

## Reward

```
r = policy_dt · exp(−(distance to the moving point) / scale)   while alive
  − crash, once
```

| `RewardConfig` | value |
|---|---|
| `scale` | 1 m |
| `crash` | 30 |

- One second on the point pays 1. A whole flight lasts 3.3–5.3 s. A crash costs as much
  as 30 s of perfect flight, plus the pay it no longer earns.
- The pay is never negative, so a crash never saves cost. It never reaches zero, so a
  drone far from its point still feels a pull back.
- Slow and fast both lose pay. A dodge loses pay only while the drone is off its point,
  so an early dodge costs more than a late one.
- No finish bonus. Surviving to the end already earns every step's pay, and the crash
  cost already widens the gap.
- The drone that dies pays `crash`. On contact both die and both pay.

## Episode

| event | ends it | how |
|---|---|---|
| the moving points reach the ends of the lines | yes | terminal, `success` if nobody died |
| a drone leaves the arena box: a wall, the ceiling or the ground | yes | `crash` to that drone, terminal for both |
| the two drones touch, 0.80 m between centres | yes | `crash` to both, terminal for both |
| `max_steps` = 600, 6 s | yes | truncation. The longest flight is 533 steps, so only a first episode shortened by `vecenv` reaches it |

A drone that flips keeps flying, so the task does not use `core.is_dead`.

## Metrics

| `info` | name | what |
|---|---|---|
| `end` | `success` | the moving points reached the ends, and nobody died |
| `end` | `contact`, `out` | how the episode ended otherwise |
| `end` | `closest` | the smallest distance between the drones' centres |
| `end` | `point_distance_unseen` | per drone: the mean distance from the moving point while the other drone was out of sight. Near 0 means the drone stays on its point until it sees the other. Higher means it flies off its line "just in case" |
| `step` | `point_distance` | per drone: the distance from the moving point |

The trainer and the eval add `death_rate`.

## Baselines and evals

- `python run/eval.py runs/<name>`: the policy flies both drones.
- `python run/eval_avoidance.py runs/<name>`: the policy flies drone 0, and a line
  follower flies drone 1. The follower flies to its moving point (`guidance.follow`) and
  never dodges. In training the other drone runs the same network, so the policy may
  learn to move only half the distance. This is the test for that.
- `python run/eval_avoidance.py runs/<name> --line-drone 0 1`: both drones only follow
  their moving points. Over 1024 eval episodes: 78.8% contact, 0% out, 21.2% success,
  0.07 m from the moving point.

## Presets

| preset | `sight` |
|---|---|
| `d_inf` | no limit |
| `d6`, `d5`, `d4`, `d3` | 6, 5, 4, 3 m |

## Where it lives

| file | what |
|---|---|
| `swarm/envs/avoidance.py` | the task |
| `run/eval_avoidance.py` | the line follower and the eval against it |

## Open

| question | state |
|---|---|
| in an exact head-on both drones see the same picture, so one network can pass every meeting with a blind "always go right" | the line follower test catches it, and angles below 180° break the symmetry |
| `run/eval.py --seat` flies a whole role, so the one-drone seat lives in `run/eval_avoidance.py`, which imports private helpers of `seats.py` and `evaluate.py` | simplify later |
| `core.is_dead` kills a flipped drone, and this task and `chase` do not | decide for all tasks later |
| tests | none yet |
| training | only a 3e6-step smoke run, to check that training and the evals run |
