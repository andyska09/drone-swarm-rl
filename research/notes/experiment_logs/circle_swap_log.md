# Task circle swap — run log

The runs of the circle swap task, their results, and why the spec changed. The spec is
in [task_circle_swap.md](../tasks/task_circle_swap.md).

The spec of that time had a 10 m arena around (0, 0, 3). `own` had no wall distances and
was 16 wide. A `neighbors` block of (N − 1, 6) held every other drone, with no role flag.

## Why two presets

N = 4 was skipped on purpose. N = 2 came first, to get the reward right cheaply, then
N = 8.

## The first `pair` run

- `crash` was a flat 10.0. A drone that could not reach its goal gained by dying. The
  run did that in 32 of 32 eval episodes. So `crash` became 25.0 per drone in the scene.
- The best checkpoint flipped drone 1 in 32 of 32 eval episodes. A flip only cost the
  one-off 10.0, and by then the episode was over anyway. So the `orient` term came in.
- `orient` is also a small income for staying alive. With `crash` at 10.0, it raised the
  distance at which dying beats flying from 2.5 m to 3.5 m at step 307. It did not close
  that exit. Only a bigger `crash` or a real per-step survival reward does.

## The first `circle8` run

- Every drone started pointing along world x. `target` is in the body frame, so each
  drone had its own fixed goal bearing, and one shared network learned N sub-tasks with
  one geometry each. The end error ran from 0.156 m (goal straight behind) to 1.240 m
  (goal sideways), 8× across drones on the same physics.
- With a fixed heading, the policy saw `R ≈ I` in every episode it trained on. With the
  drones turned to face the centre, the same policy collapsed. Episodes ended at step 75
  with colliding pairs, 4.35 m of the 6 m leg still to go. It had relied on the absolute
  heading, not on the goal bearing.
- Pointing every drone at the centre also makes the N tasks the same, but it keeps `R`
  fixed. So the starting heading became random, for each drone.

## Reasons for the spec, 2026-10-25

These reasons moved here from the spec.

- **The 7 s episode** leaves room for a detour around a neighbour.
- **The critic gets every goal**, because the goal drives the reward.
- **`distance` and `spin`** have the ratio of Huang et al. 2024: 0.1 against 1.0.
- **`close`** has the shape and the weight of Huang et al. 2024:
  `quads_collision_smooth_max_penalty = 10.0` against `pos = 1.0`. 10.0 is both their
  code default and the value in `runs/quad_multi_mix_baseline.py`, their 8-drone room
  with no obstacles. Their obstacle run uses 4.0, for a different scene. We sum the term
  over every neighbour, not only the nearest.
- **`close_dist` = 1.5 m** is 1.9 hitbox widths. The start spacing is `2r·sin(π/N)`:
  6 m in `pair` and 2.30 m in `circle8`. `close_dist` is below 2.30 m, so the term is 0 at
  reset. At a closing speed of 2 m/s, it starts to cost about 0.35 s before contact.
- **`orient`** is Huang's `α_orient·R_i,33` at their weight of 1.0. It is positive, so it
  is also a small pay for staying alive.
- **`crash` grows with N.** The costs add up over 700 steps. A drone that stays 6 m from
  its goal pays 42 in a full episode, and at N = 8 four neighbours held at 1.0 m add 93
  more. If a crash costs less than that, a drone that cannot reach its goal gains by
  dying. That cost grows with the number of drones, so `crash` grows too. Gavin & Bronz
  2026 keep the same order: `λ_fail` = 30.0 against `λ_dist` = 0.001.
