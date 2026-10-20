# Task avoidance — the spec

Two drones fly straight lines that meet in open space. Each drone must stay near a point
that moves along its line at a set speed. The drones must not touch. The code is
[avoidance.py](../../swarm/envs/avoidance.py). The runs and the history of the design are
in [experiments/avoidance_log.md](experiments/avoidance_log.md).

**What the task is for.** The task tests what a drone must sense about the other drone.
The first design gives the exact position and velocity of the other drone, but only
inside a sight range `d`. Later designs change only this input. We score each design on
the same encounters. The literature is in [lit_review_tunnel.md](lit_review_tunnel.md).

## The encounter

Each episode first picks the meeting. Then it puts the two lines around it. Aircraft
encounter models do the same (Kochenderfer et al. 2010,
[10.2514/1.44867](https://doi.org/10.2514/1.44867)). "Random" below means uniform: every
value in the range is equally likely.

| what | value |
|---|---|
| meeting place | random, in a ball of `meet_radius` = 2 m around `center` |
| approach angle | the angle between the two flight directions. Random from `angle_min` = 90° to `angle_max` = 180°, to either side |
| miss distance | how close the two lines pass. Random from 0 to `miss` = 1 m. Its direction is random, at a right angle to the relative velocity of the drones |
| distance before the meeting | the same for both drones. Random from `before_min` = 6 m to `before_max` = 12 m |
| distance after the meeting | `after` = 4 m |
| lines | straight and level. The heading of the first line is random |

- Both drones reach the meeting place at the same time. So the miss distance alone
  decides if they touch. They touch below 0.80 m between centres.
- 80% of the encounters need a dodge.
- The drones start at least 8.5 m apart. Every start is at least 1.5 m inside the arena
  box.
- A drone cannot tell from its own position where the meeting will be.

## The moving point

- Each drone has a moving point. It starts where the drone starts and moves along the
  drone's line at `cruise` = 3 m/s.
- The reward pays the drone for being close to its moving point.
- The episode ends when the moving points reach the ends of the lines.
- The moving point has its own clock, `travelled`, in metres. It does not use `time`,
  because `vecenv.reset` sets `time` to a random value in the first episode of each env.
- The drones start at 3 m/s along their lines. `core.spawn` adds a random velocity of up
  to 0.5 m/s on each axis, a random yaw and a tilt of up to 0.2 rad.

## Sight

- A drone sees the other drone when it is closer than `sight`, in any direction. `sight`
  is the `d` of the lit review.
- There is no field of view, no blocking and no memory.
- When a drone does not see the other drone, the mask sets its `others` row to zero.
- With `d_inf`, the drones always see each other. With `d6` and less, they start out of
  sight.

The sight that a head-on dodge needs, at 3 m/s each and a sideways push of 10 m/s²
([lit_review_tunnel.md](lit_review_tunnel.md), section 1):

| who dodges | sight needed |
|---|---|
| both drones | 2.7 m |
| one drone | 3.7 m |

So `d3` is close to the limit when both drones dodge. Against the line follower, it is
past the limit.

## Observation

| block | width | content |
|---|---|---|
| `own` | 20 | `core.own_obs`: velocity in the body frame, `R`, `ω`, height, and the distances to the arena walls |
| `others` | 1 × 7 | the other drone: its offset and velocity in the body frame, and the role flag (always 1). Zero when out of sight |
| `target` | 6 | `core.target_obs` for the moving point: its offset and the velocity difference, in the body frame. The line is straight, so the velocity of the point also shows the direction of the line |
| `scene` | 51 | the input of the critic: `core.scene_obs`, both moving points and their velocities, the metres left to the end, and if each drone sees the other |

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

- One second on the moving point pays 1. A crash costs as much as 30 s on the moving
  point.
- The pay is never negative.
- Slow, fast and sideways flight all lose pay. So a late dodge costs less than an early
  one.
- The drone that dies pays `crash`. If the drones touch, both die and both pay.

## Episode

| event | ends the episode | how |
|---|---|---|
| the moving points reach the ends of the lines | yes | terminal. `success` if no drone died |
| a drone leaves the arena box: a wall, the ceiling or the ground | yes | `crash` for that drone, terminal for both |
| the drones touch, 0.80 m between centres | yes | `crash` for both, terminal for both |
| a drone gets more than `max_point_distance` = 2 m from its moving point | yes | `crash` for that drone, terminal for both |
| `max_steps` = 600, that is 6 s | yes | truncation. The longest flight is 533 steps, so only a first episode that `vecenv` made shorter gets there |

A drone that flips keeps flying. So the task does not use `core.is_dead`.

## Metrics

| `info` | name | what it measures |
|---|---|---|
| `end` | `success` | the moving points reached the ends, and no drone died |
| `end` | `contact`, `out`, `lost` | how the episode ended otherwise. `lost`: a drone got more than 2 m from its moving point |
| `end` | `closest` | the smallest distance between the centres of the drones |
| `end` | `point_distance_unseen` | per drone: the mean distance from the moving point while the other drone is out of sight. Near 0: the drone stays on its point until it sees the other drone. Higher: the drone flies off its line "just in case" |
| `step` | `point_distance` | per drone: the distance from the moving point |

The trainer and the eval add `death_rate`.

## Baselines and evals

| command | who flies |
|---|---|
| `python run/eval.py runs/<name>` | the policy flies both drones |
| `python run/eval_avoidance.py runs/<name>` | the policy flies drone 0, and a line follower flies drone 1 |
| `python run/eval_avoidance.py runs/<name> --line-drone 0 1` | line followers fly both drones. This is the baseline with no dodge |

- The line follower flies to its moving point (`guidance.follow`). It never dodges.
- In training, the other drone runs the same network. So the policy can learn to dodge
  only half the distance. The line follower test finds this.
- The baseline, over 1024 eval episodes: 78.8% contact, 0% out, 0% lost, 21.2% success.
  The drones stay 0.07 m from their moving points.

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
| In an exact head-on, both drones see the same picture. So one network can pass every meeting with a blind "always go right". | The line follower test catches it. Angles below 180° break the symmetry. |
| `run/eval.py --seat` flies a whole role. So the one-drone seat lives in `run/eval_avoidance.py`, which uses private helpers of `seats.py` and `evaluate.py`. | Make it simpler later. |
| `core.is_dead` kills a flipped drone. This task and `chase` do not. | Decide for all tasks later. |
| tests | None yet. |
