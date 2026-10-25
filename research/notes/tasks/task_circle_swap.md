# Task circle swap — the spec

N drones sit on a circle. Each drone flies to the point opposite its start. Every drone
crosses the middle, so no drone arrives without avoiding the others. The code is
[circle_swap.py](../../swarm/envs/circle_swap.py). The runs and the history of the design
are in [experiments/circle_swap_log.md](experiments/circle_swap_log.md).

## Presets

| preset | drones | `n_neighbors` | start spacing |
|---|---|---|---|
| `pair` | 2, head-on | 6, so K = 1 | 6 m |
| `circle8` | 8 | 7, so K = 7 | 2.30 m |

The start spacing is `2r·sin(π/N)`, with `radius` r = 3 m. Every flight to the opposite
point is 6 m long.

## Timing

Physics at 200 Hz, policy at 100 Hz. `max_steps` = 700, a 7 s episode. The 7 s leave
room for a detour around a neighbour.

## Action

CTBR, as in every task.

## Observation

| block | width | content |
|---|---|---|
| `own` | 20 | `core.own_obs`: body-frame velocity, `R`, `ω`, height, and the distances to the arena walls |
| `others` | K × 7 | the K nearest drones, nearest first: body-frame offset, body-frame velocity, and the role flag, always 1 here |
| `others_mask` | K | all ones: K = N − 1 fills every seat, and the episode ends at the first death |
| `target` | 6 | `core.target_obs` for the goal: `Rᵀ(g − x)` and `Rᵀ(−v)` |
| `scene` | 21N | the critic's input: `core.scene_obs` and every drone's goal. The goal drives the reward, so the critic needs it |

`K = min(n_neighbors, N − 1)`. In both presets, every drone sees every other drone.

## Reset

- Drone i starts at the angle `2πi/N` on a circle of radius 3 m around `center` =
  (0, 0, 8). All drones start at the same height, exactly on the circle. The goal of a
  drone is the point opposite its start.
- `core.spawn` adds a velocity of up to 0.5 m/s on each axis and a tilt of up to 0.2 rad.
  `ω` is 0, and the motors run at hover rpm.
- **The starting heading is random**, uniform over the full circle, for each drone.
  `target` is in the body frame. With one fixed heading, each drone gets its own fixed
  goal bearing, and the shared network learns N separate tasks. The policy also sees the
  same `R` in every episode, so it can rely on the absolute heading instead of the goal
  bearing. A random heading stops both.

## Episode

| event | ends the episode | how |
|---|---|---|
| a drone leaves the 32 × 32 × 16 m arena box, or goes below z = 0 | yes | `crash` for that drone, terminal for all |
| a drone flips over, `R[2,2] < 0` | yes | `crash` for that drone, terminal for all |
| two drones touch, 0.80 m between centres | yes | `crash` for both, terminal for all |
| the position of a drone becomes non-finite | yes | `crash` for that drone, terminal for all |
| `max_steps` = 700 | yes | truncation |

This is `core.is_dead`. Any death ends the episode for every drone. Contact at 0.80 m is
the MRS rule (`multirotor_simulator.cpp:342`, `crit_dist = arm+prop+arm+prop`). MRS ships
`collisions: {enabled: true, crash: true}`, so contact kills there too. The plant has no
contact force. The distance test is the whole contact model.

## Reward

```
r = −policy_dt · (1.0·‖g − x‖ + 0.1·‖ω‖ + 10.0·Σⱼ max(1 − dᵢⱼ/1.5, 0) − 1.0·R[2,2])   while alive
    − 25.0·N, once, when the drone dies
```

| `RewardConfig` | value | what |
|---|---|---|
| `distance` | 1.0 | per metre from the goal |
| `spin` | 0.1 | per rad/s of `ω` |
| `close` | 10.0 | near-miss cost, for each neighbour closer than `close_dist` = 1.5 m |
| `orient` | 1.0 | pay for pointing up, `R[2,2]` |
| `crash` | 25.0 | per drone in the scene: 50 at N = 2, 200 at N = 8 |

- `distance` and `spin` have the ratio of Huang et al. 2024: 0.1 against 1.0.
- `close` has the shape and the weight of Huang et al. 2024:
  `quads_collision_smooth_max_penalty = 10.0` against `pos = 1.0`. 10.0 is both their
  code default and the value in `runs/quad_multi_mix_baseline.py`, their 8-drone room
  with no obstacles. Their obstacle run uses 4.0, for a different scene. We sum the term
  over every neighbour, not only the nearest.
- `close_dist` = 1.5 m is 1.9 hitbox widths. It is below the 2.30 m start spacing at
  N = 8, so the term is 0 at reset. At a closing speed of 2 m/s, it starts to cost about
  0.35 s before contact.
- `orient` is Huang's `α_orient·R_i,33` at their weight of 1.0. It is positive, so it is
  also a small pay for staying alive.
- `crash` grows with N. The costs add up over 700 steps. A drone that stays 6 m from its
  goal pays 42 in a full episode, and at N = 8 four neighbours held at 1.0 m add 93 more.
  If a crash costs less than that, a drone that cannot reach its goal gains by dying.
  That cost grows with the number of drones, so `crash` grows too. Gavin & Bronz 2026
  keep the same order: `λ_fail` = 30.0 against `λ_dist` = 0.001.
