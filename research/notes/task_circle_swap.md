# Task circle swap — the setting

N drones on a circle, each flies to the point opposite its own start. Everyone
crosses the middle, so nobody arrives without avoiding the others. Implemented in
`swarm/envs/circle_swap.py`.

## Timing

Sim at 200 Hz (`sim_dt = 0.005`), policy at 100 Hz (`steps_per_action = 2`).
`max_steps = 700` → a 7 s episode; A→B keeps 500, this one leaves room for the
detour around a neighbour.

## Action — CTBR

Unchanged from A→B: `a ∈ [-1,1]⁴`, throttle plus body-rate references, only the
rate rung of the cascade underneath.

## Observation

- `own` (16) and `target` (6) — unchanged from A→B.
- `neighbors` (N−1, 6): for each other drone `j`, `Rᵢᵀ(xⱼ - xᵢ)` and
  `Rᵢᵀ(vⱼ - vᵢ)`, sorted nearest first (`lax.top_k`). Every drone sees every
  other one, so the input width is tied to the swarm size — a policy trained at
  N = 8 cannot fly N = 32 without retraining. Sorting puts the drone that can
  kill you in slot 0 and makes the policy blind to drone identity.
- No same-side flag: one team here, so the column would be a constant. Task 5
  defines its own width.
- `neighbor_mask` is all ones, always. See Termination.

## Reset

Drone `i` at angle `2πi/N` on a circle of radius 3 m around `center = (0,0,3)`,
all at the same height, exactly on the ring — no position noise. Its goal is the
point opposite. Everything else as in A→B: velocity ±0.5 m/s, random tilt up to
0.2 rad, `ω = 0`, motors at hover RPM.

Start spacing is `2r·sin(π/N)`: 6 m at N = 2, 2.30 m at N = 8. The crossing is
6 m, the same leg as the A→B `default` preset, which a trained policy flies to
0.208 m.

## Collisions

The hitbox is a ball of radius `arm_length + prop_radius` = 0.40 m, so two drones
touch at 0.80 m. That is the MRS rule
(`multirotor_simulator.cpp:342`, `crit_dist = arm+prop+arm+prop`), and MRS ships
`collisions: {enabled: true, crash: true}` — contact kills, the elastic rebounce
is the off-by-default branch. There is no contact force in our plant; the
distance test is the whole model.

## Termination

Any death ends the episode for every drone. Death is contact with another drone,
ground contact (`z < 0`), leaving the 10 m arena, flipping (`R₂₂ < 0`), or a
non-finite state. `max_steps` is truncation.

Shared fate is Gavin & Bronz 2026's rule for inter-pursuer collisions, and it is
what keeps the observation honest: no drone ever sees a dead drone, so the
neighbour table is always full and no masked row needs an invented value. The
alternative — a drone dies, the rest fly on — forces a fill value for the empty
row, and zero there reads as "a drone at my exact position", the collision state.
Huang et al. 2024 dodge the same problem from the other side: their collisions
are elastic bounces, so nobody ever leaves.

## Reward

```
r = -policy_dt · (1.0·‖g - x‖ + 0.1·‖ω‖ + 10.0·Σⱼ max(1 - dᵢⱼ/1.5, 0) - 1.0·R₃₃) - 25.0·N · died
```

`crash` is **25.0 per drone in the scene** — 50 at N = 2, 200 at N = 8 — not A→B's
flat 10.0. Shaping accrues 700 times per episode: 6 m of error costs 42 over a full
episode, and at N = 8 four neighbours held at 1.0 m add another 37. At 10.0 a drone
that cannot reach its goal buys its way out by dying, and the first `pair` run did
exactly that in 32 of 32 eval episodes. The penalty has to stay above the bill a
living drone can run up, and that bill grows with the size of the crowd. Gavin &
Bronz 2026 keep the same order: `λ_fail = 30.0` against `λ_dist = 0.001`.

The distance and spin terms are A→B's, untouched. The near-miss term is Huang et
al. 2024's shape and weight — `quads_collision_smooth_max_penalty = 10.0` against
`pos = 1.0` — summed over every neighbour, not just the nearest. 10.0 is both the
code default and the value in `runs/quad_multi_mix_baseline.py`, their 8-drone room
with no obstacles; the 4.0 in their obstacle run is for a different scene.

`R₃₃` is Huang's `α_orient·R_i,33` at their weight of 1.0: pay for pointing up.
It was added after the first `pair` run, where the best checkpoint flipped drone 1
in 32 of 32 eval episodes — a flip only ever cost the one-off `-10.0`, and by then
the episode was over anyway. Being a *positive* term it is also a small income for
staying alive, which raises the distance at which suicide beats flying from 2.5 m
to 3.5 m at step 307. It does not close that exit; only a bigger `crash` or a real
per-step survival reward does.

`close_dist = 1.5 m` is 1.9 hitbox widths and sits below the 2.30 m start spacing
at N = 8, so the term is silent at reset and warns about 0.35 s before contact at
a 2 m/s closing speed. Both drones in a collision pay the `-10.0`, because both
are inside 0.80 m.

## Presets

- `pair` — N = 2, head-on across 6 m.
- `circle8` — N = 8, 2.30 m apart at the start.

N = 4 is skipped on purpose: 2 first to get the reward right cheaply, then 8.
