# Task tunnel — run log

The runs of the tunnel task and their results. The spec is in
[task_tunnel.md](../tasks/task_tunnel.md).

## The first CPU run

`d_inf`, 3e7 steps, 1024 envs, the drones start in the legs only. The drones flew the
legs at 4.8 m/s and hit a wall at the first bend in 96% of the episodes, so they never
reached a meeting.

## Three runs of 1e9 steps

`d_inf`, 1e9 steps, seeds 0–2 (`7c1d6b`, `e0b72d`, `c027a4`).

- Training stopped improving after about 4e8 steps. It reached 35% success, 56% wall
  and 9% contact.
- In eval from the far ends of the legs, success was 0–0.5% and wall 93–100%. The drones
  sped up to 4.5–6 m/s and hit the first bend.
- In seed 1, 61% of the wall deaths came less than 1.5 m from a bend.

So the score mixed two skills: flying the tube and dodging. This is why the avoidance
task exists ([avoidance_log.md](avoidance_log.md)).

## Reasons for the spec, 2026-10-25

These reasons, measurements and plans moved here from the spec.

### What the task is for

The experiment variable is what a drone senses about the other drone. The first design is
the exact relative position and velocity. Later designs change only that input, and each
design is scored on the same tunnel. In a multi-drone pursuit game, what a drone sees of
the others matters, and this task prepares that question. The literature is in
[literature/avoidance.md](../literature/avoidance.md).

### The tunnel and the lines

- The middle piece has 2.4 m of inner wall, so a 2.4 m cube sits between the inner
  corners. The centre line runs 4.05 m between the bends, the outer wall 5.70 m.
- A line keeps at least 0.7 m to the walls, which leaves 0.3 m between the propellers and
  the wall.

### Sight

Only the inner side of each bend can block the segment. With no range limit, two drones
that approach in step first see each other at:

| both drones fly | first sight |
|---|---|
| 0.4 m from the inner wall | 3.26 m |
| on the centre line | 4.97 m |
| 0.4 m from the outer wall | 6.68 m |

A drone that takes the outer side of a bend sees around the corner earlier. In a sample of
positions, no clear view across the tunnel was longer than 7.7 m. So a `sight` above about
8 m is the same as no limit, and below about 3 m the walls stop mattering (0.5% of the
pairs within 3 m are hidden, 0.01% within 2 m).

The range decides how fast a head-on meeting can be. Our drone needs 0.42 s to move 0.4 m
to the side (CTBR, roll rate capped at 4 rad/s, height held). If both drones dodge, each
by 0.4 m, with a perfect dodge from the moment of first sight, the fastest safe speed for
each drone is:

| first sight | 3 m | 4 m | 5 m | 6.69 m |
|---|---|---|---|---|
| speed, each drone | 3.2 m/s | 4.5 m/s | 5.8 m/s | 7.9 m/s |

### Observation

This task breaks the "same obs format" rule on purpose. A new sensor design may change the
`others` row. The first design keeps the shared row.

Why two points on the line. The nearest point tells the drone how far it is from the line
and which way leads back. The second point tells it which way is forward, and it goes
around a bend before the drone does. Other work gives a path to a policy the same way:

| paper | what the policy gets about its path |
|---|---|
| Meyer et al. 2020, [arXiv:1912.08578](https://arxiv.org/abs/1912.08578), Table 2 | distance from the path, the turn needed to head for a point 100 m ahead on it, and the path direction there (a boat) |
| Song et al. 2021, [arXiv:2103.08624](https://arxiv.org/abs/2103.08624), Table IV | the next gates. On 1000 random tracks: 1 gate 23% crashes, 2 gates 2.5%, 3 gates 2.3% |
| Fuchs et al. 2021, [arXiv:2008.07971](https://arxiv.org/abs/2008.07971) | the angle to the centre line, range finders to the track edges, the curvature ahead (a car) |
| SimpleFlight, [arXiv:2412.11764](https://arxiv.org/abs/2412.11764) | the next 10 points of the path, 0.05 s apart |
| Kaufmann et al. 2023 (Swift) | the pose of the next gate |

At 5 m/s, 3 m ahead is 0.6 s of warning, close to SimpleFlight's 0.5 s.

### Reward

Progress projects the drone onto its line and pays for how far that point moved, as in
Song 2021 (eq. 2) and Fuchs 2021. Moving backward pays a negative amount. At these
weights, 1 m/s along the line earns as much as 1 m off it costs. A crash costs 10 and ends
all future progress, so it never pays. There is no spin, proximity or action term. Add one
only when a problem shows up.

**Speed is not a setting.** The bonus pays per metre, not per second, so a slow flight and
a fast flight of the same line earn the same. Only the discount pays for speed: at the
default `gamma` = 0.998, a metre one second later is worth 0.82 of a metre now. At a short
`sight` a policy can slow down instead of crashing. Faster drones need a reward change,
for example a cost per second.

### Episode

The longest line is about 24 m, so 15 s needs at least 1.6 m/s on average. A drone that
flips keeps flying, as in `chase`.

### Start

- **The tunnel heading is random.** `own` holds `R` and the body-frame velocity, so with a
  fixed heading the flight direction in the world would tell the drone which piece it is
  in.
- **A start anywhere on the line** is the distributed initialization of Song et al. 2021,
  [arXiv:2103.08624](https://arxiv.org/abs/2103.08624), Section III-D-2. Early rollouts
  mostly end in a crash, so with one start point the data covers only a small part of the
  track. Song et al. start the drones hovering around the centres of all path segments.
  Once the policy passes the gates reliably, they start from states of earlier flights.
- **Eval starts at the start of the line**, so every eval flies the whole tunnel and holds
  a meeting.

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

One blind rule survives. A drone that always keeps to the right half of the tube, not of
its line, never collides, because the other drone runs the same network and does the
same. Random lines only make it cost: 0.51 m off the line on average, for the whole
flight. To make it fail, the other drone must sometimes break the rule, which needs a line
follower in training.

### Baselines

- The line follower is the robustness test. In training the other drone always moves
  aside too, so the policy may learn to move only half the distance.
- Both drones as line followers, over 1024 eval episodes: 80.4% contact, 0% wall hits,
  19.6% success, 1.90 m/s, 0.09 m from the line. Every eval episode holds a meeting, so
  four meetings in five need a dodge.
- The cascade is not a baseline here. It flies to the point 3 m ahead, at up to its 6 m/s
  cap, and hits a wall in 59% of the episodes (64 episodes, training starts). The follower
  is slow on purpose: at 3 m/s it already hits a wall in 5% of the episodes (256 episodes,
  training starts).

### Experiment plan

1. The first sensor design: the exact relative position and velocity of the other drone.
2. Train a new policy at each `sight`, 3 seeds each.
3. Compare designs on two numbers at each `sight`: the death rate and the mean speed. A
   slow, careful policy has a low death rate, so the death rate alone ranks it too high.
4. Run the line-follower test on every run.
5. Later: more sensor designs, faster drones through the reward, a random bend angle and
   middle length.

## Open

| question | state |
|---|---|
| a line follower in some training episodes | only if the robustness test fails |
| tests | none yet |
| `run/eval.py --seat` flies a whole role, so the one-drone seat lives in `run/eval_tunnel.py`, which imports private helpers of `seats.py` and `evaluate.py` | unify later |
