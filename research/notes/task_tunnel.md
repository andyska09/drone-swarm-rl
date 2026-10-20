# Task tunnel — the spec

Two drones fly opposite ways along a bent square tube. Each follows its own line and
must pass the other. The bends hide each drone from the other until late, so the
dodge happens at the last moment. Built in [tunnel.py](../../swarm/envs/tunnel.py).

**What the task is for.** The experiment variable is what a drone senses about the
other drone. The first design is the exact relative position and velocity. Later
designs change only that input, and each design is scored on the same tunnel. In a
multi-drone pursuit game, what a drone sees of the others matters, and this task
prepares that question. The literature is in
[lit_review_tunnel.md](lit_review_tunnel.md).

## The tunnel

Seen from above it is a bridge, `/‾\`: a left leg, a middle piece and a right leg.

| | value |
|---|---|
| cross-section | square, `width` = 2.4 m wide and 2.4 m tall, the same everywhere |
| inner corner | `corner` = 111°, measured through the solid, so each bend turns 69° |
| joints | mitred: the two straight pieces meet on the bisector, as in a picture frame |
| middle piece | `middle` = 2.4 m of inner wall, so a 2.4 m cube sits between the inner corners. The centre line runs 4.05 m between the bends, the outer wall 5.70 m |
| legs | `leg` = 10 m along the centre line, from the bend to the end of the line |
| ends | closed, `cap` = 1 m past each end of a line |
| centre line | 24.05 m from end to end |
| in the world | centred on `center`, at a random heading each episode. The arena box plays no part |

A drone is a ball of radius `arm_length + prop_radius` = 0.40 m, against the walls, the
floor, the ceiling and the other drone (contact at 0.80 m between centres, the MRS rule).

The tunnel frame: origin at the middle of the tube, x along the middle piece, y toward
its outer wall, z up. The outline seen from above is one closed polygon of 8 corners
(`walls`). A drone hits a wall when its centre leaves the polygon, comes within 0.40 m
of an edge, or comes within 0.40 m of the floor or the ceiling.

## Drones and lines

Two drones share the role `drone`, so one network flies both. Even drones start in the
left leg and odd drones in the right one. Each flies to the far end of the other leg.

Each drone's line is the centre line moved in parallel, by a random amount up to
`line_offset` = 0.5 m sideways and up to 0.5 m up or down. Its corners sit on the
mitre lines, so it keeps the same distance to the walls through the bends: at least
0.7 m, which leaves 0.3 m between the propellers and the wall.

## Sight

Drone i sees drone j when the segment between their centres crosses no wall, and j is
closer than `sight` (the `d` of the lit review). The floor and the ceiling are flat, so
only the walls block. Only the inner side of each bend can block the segment.

With no range limit, two drones that approach in step first see each other at:

| both drones fly | first sight |
|---|---|
| 0.4 m from the inner wall | 3.26 m |
| on the centre line | 4.97 m |
| 0.4 m from the outer wall | 6.68 m |

A drone that takes the outer side of a bend sees around the corner earlier. In a
sample of positions, no clear view across the tunnel was longer than 7.7 m. So a
`sight` above about 8 m is the same as no limit, and below about 3 m the walls stop
mattering (0.5% of the pairs within 3 m are hidden, 0.01% within 2 m).

The range decides how fast a head-on meeting can be. Our drone needs 0.42 s to move
0.4 m to the side (CTBR, roll rate capped at 4 rad/s, height held). If both drones
dodge, each by 0.4 m, with a perfect dodge from the moment of first sight, the fastest
safe speed for each drone is:

| first sight | 3 m | 4 m | 5 m | 6.69 m |
|---|---|---|---|---|
| speed, each drone | 3.2 m/s | 4.5 m/s | 5.8 m/s | 7.9 m/s |

## Observation

| block | width | content |
|---|---|---|
| `own` | 20 | body-frame velocity, `R` (9), `ω`, height above the tube floor, and the distances to the tunnel walls ahead, behind, left and right, along rays level with the floor |
| `others` | 1 × 7 | the other drone: body-frame offset, body-frame velocity, and the role flag (always 1 here). Masked to zero when the drone does not see it |
| `target` | 6 | the body-frame offset to the nearest point on the own line, and to the point `look_ahead` = 3 m further along it |
| `scene` | 44 | the critic's input: for each drone, position, velocity and `R` in the tunnel frame, `ω`, progress, line offset, and whether it sees the other drone |

**This task breaks the "same obs format" rule on purpose.** A new sensor design may
change the `others` row. The first design keeps the shared row.

Why two points on the line. The nearest point tells the drone how far it is from the
line and which way leads back. The second point tells it which way is forward, and it
goes around a bend before the drone does. Other work gives a path to a policy the same
way:

| paper | what the policy gets about its path |
|---|---|
| Meyer et al. 2020, [arXiv:1912.08578](https://arxiv.org/abs/1912.08578), Table 2 | distance from the path, the turn needed to head for a point 100 m ahead on it, and the path direction there (a boat) |
| Song et al. 2021, [arXiv:2103.08624](https://arxiv.org/abs/2103.08624), Table IV | the next gates. On 1000 random tracks: 1 gate 23% crashes, 2 gates 2.5%, 3 gates 2.3% |
| Fuchs et al. 2021, [arXiv:2008.07971](https://arxiv.org/abs/2008.07971) | the angle to the centre line, range finders to the track edges, the curvature ahead (a car) |
| SimpleFlight, [arXiv:2412.11764](https://arxiv.org/abs/2412.11764) | the next 10 points of the path, 0.05 s apart |
| Kaufmann et al. 2023 (Swift) | the pose of the next gate |

At 5 m/s, 3 m ahead is 0.6 s of warning, close to SimpleFlight's 0.5 s.

## Action

CTBR, as in every task.

## Reward

```
r = progress · (metres gained along the own line this step)
  - policy_dt · line · (distance from the own line)
  - crash, once
```

| `RewardConfig` | value |
|---|---|
| `progress` | 1.0 per metre |
| `line` | 1.0 per metre per second |
| `crash` | 10.0 |

Progress projects the drone onto its line and pays for how far that point moved, as in
Song 2021 (eq. 2) and Fuchs 2021. Moving backward pays a negative amount. At these
weights, 1 m/s along the line earns as much as 1 m off it costs. A crash costs 10 and
ends all future progress, so it never pays. There is no spin, proximity or action
term. Add one only when a problem shows up.

**Speed is not a setting.** The bonus pays per metre, not per second, so a slow flight
and a fast flight of the same line earn the same. Only the discount pays for speed: at
the default `gamma` = 0.998, a metre one second later is worth 0.82 of a metre now. At a
short `sight` a policy can slow down instead of crashing. Faster drones need a reward
change, for example a cost per second.

## Episode

| event | ends it | how |
|---|---|---|
| a drone reaches the end of its line | yes | terminal for both. It has passed the other drone by then |
| a wall, floor or ceiling hit | yes | `crash` to the drone that hit, terminal for both |
| the two drones touch | yes | `crash` to both, terminal for both |
| `max_steps` = 1500, 15 s | yes | truncation, the drones bootstrap |

The longest line is about 24 m, so 15 s needs at least 1.6 m/s on average. A drone that
flips keeps flying, as in `chase`.

## Reset

| what | how |
|---|---|
| tunnel heading | uniform over the full circle. `own` holds `R` and the body-frame velocity, so with a fixed heading the flight direction in the world would tell the drone which piece it is in |
| line offset | up to 0.5 m sideways and up or down, per drone |
| start, training | anywhere on the centre line moved up to `start_offset` = 0.5 m sideways and up or down, from the start of the own line to `start_gap` = 2 m before its end. No two drones start closer than `start_apart` = 2 m |
| start, eval | at the start of the own line, the far end of the leg, moved up to `start_offset` sideways and up or down. `eval_params` sets `start_anywhere` to false, so every eval flies the whole tunnel and holds a meeting |
| attitude and speed | `core.spawn`: random yaw, tilt up to 0.2 rad, up to 0.5 m/s on each axis |

A start anywhere on the line is the distributed initialization of Song et al. 2021,
[arXiv:2103.08624](https://arxiv.org/abs/2103.08624), Section III-D-2. Early rollouts
mostly end in a crash, so with one start point the data covers only a small part of the
track. Song et al. start the drones hovering around the centres of all path segments.
Once the policy passes the gates reliably, they start from states of earlier flights.

What the training starts do to the meetings, over 100,000 resets:

| episode | share |
|---|---|
| a meeting ahead, the drones hidden from each other at the start | 33% |
| a meeting ahead, in sight at the start | 27% |
| the drones start past each other, no meeting | 40% |

What the randomization buys. Without it, a drone that always flies 0.5 m right of the
centre passes every meeting without looking: the drones are 1.0 m apart.

| line offset up to | propeller to wall, on the line | "always 0.5 m right of my line" hits the other | the lines are 0.8 m apart already, no dodge needed |
|---|---|---|---|
| 0.4 m | 0.4 m | 22% | 2% |
| **0.5 m** | **0.3 m** | **24%** | **15%** |
| 0.6 m | 0.2 m | 24% | 29% |

One blind rule survives. A drone that always keeps to the right half of the tube, not
of its line, never collides, because the other drone runs the same network and does
the same. Random lines only make it cost: 0.51 m off the line on average, for the whole
flight. To make it fail, the other drone must sometimes break the rule, which needs a
line follower in training.

## Metrics

| `info` | name | what |
|---|---|---|
| `end` | `success` | a drone reached the end of its line and nothing crashed |
| `end` | `contact`, `wall`, `timeout` | how the episode ended otherwise |
| `end` | `closest` | the smallest distance between the drones' centres |
| `end` | `line_distance_unseen` | per drone: the mean distance from the own line while the other drone was out of sight. Near 0 means the drone flies its line until it sees the other. Near 0.5 m means a blind offset |
| `step` | `line_distance` | per drone, the distance from the own line |
| `step` | `speed` | per drone, metres per second along the own line |

The trainer and the eval add `death_rate`.

## Baselines and evals

- `python run/eval.py runs/<name>`: the policy flies both drones.
- `python run/eval_tunnel.py runs/<name>`: the policy flies drone 0, and a line follower
  flies drone 1. The follower tracks the nearest point on its line, flies along the
  line at 2 m/s, and never dodges. This is the robustness test: in training the other
  drone always moves aside too, so the policy may learn to move only half the
  distance.
- `python run/eval_tunnel.py runs/<name> --line-drone 0 1`: both drones only follow
  their lines. Over 1024 eval episodes: 80.4% contact, 0% wall hits, 19.6% success,
  1.90 m/s, 0.09 m from the line. Every eval episode holds a meeting, so four meetings
  in five need a dodge.

`run/eval.py --seat drone cascade` is not a baseline here. The cascade flies to
`reference()`, the point 3 m ahead, at up to its 6 m/s cap, and hits a wall in 59% of
the episodes (64 episodes, training starts). The follower is slow on purpose: at 3 m/s
it already hits a wall in 5% of the episodes (256 episodes, training starts).

## Presets

| preset | `sight` |
|---|---|
| `d_inf` | no limit, the walls alone decide |
| `d6`, `d5`, `d4`, `d3` | 6, 5, 4, 3 m |

## Experiment plan

1. The first sensor design: the exact relative position and velocity of the other drone.
2. Train a new policy at each `sight`, 3 seeds each.
3. Compare designs on two numbers at each `sight`: the death rate and the mean speed. A
   slow, careful policy has a low death rate, so the death rate alone ranks it too high.
4. Run the line-follower test on every run.
5. Later: more sensor designs, faster drones through the reward, a random bend angle
   and middle length.

## Where it lives

| file | what |
|---|---|
| `swarm/envs/tunnel.py` | the task: the tube, the lines, sight, the reward |
| `run/eval_tunnel.py` | the line follower and the eval against it |

Two shared files know about walls, not about the tunnel: `swarm/learn/evaluate.py` saves
the `scenery()` of a task that has one into `trajectory.npz`, and `tools/viewer/index.html`
draws those walls instead of the arena box.

## Open

| question | state |
|---|---|
| `run/eval.py --seat` flies a whole role, so the one-drone seat lives in `run/eval_tunnel.py`, which imports private helpers of `seats.py` and `evaluate.py` | unify later |
| the critic gets one `scene` for both drones, so it predicts their mean value | as in every task |
| a line follower in some training episodes | only if the robustness test fails |
| tests | none yet |
| the first CPU run, `d_inf`, 3e7 steps, 1024 envs, starts in the legs only | flies the legs at 4.8 m/s and hits a wall at the first bend in 96% of the episodes, so it never reaches a meeting yet |
