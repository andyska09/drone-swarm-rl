# Task chase — the spec

One drone chases an evader that the cascade flies to random waypoints. The drone
catches it with a net. Task 3 on the ladder in `goals.local.md`, and a smoke test
for task 4 — the evader is already a second drone on the drone axis, so task 4
only stops scripting it. Built in [chase.py](../../swarm/envs/chase.py).

Every value beside a Gavin column is from
[Gavin & Bronz 2026 Nv1](../papers/2607.05939v1.md) unless marked 1v1
([Gavin et al. 2026](../papers/2603.16279v1.md)).

## Observation

| | `Obs` block | ours | Gavin Nv1 | Gavin 1v1 |
|---|---|---|---|---|
| my velocity | `own` | `Rᵀv` body, 3 | `Rᵀv` body, 3 | `v` world, 3 |
| my attitude | `own` | `vec(R)`, 9 | `vec(R)`, 9 | `vec(R)`, 9 |
| my body rate | `own` | `ω` body, 3 | none | none |
| walls | `own` | height `z`, 1 | `z` + M horizontal rays | 6 wall distances |
| target position | `target` | `Rᵀ(x_other − x)`, 3 | `Rᵀ(p_j − p_i)`, 3 | `p_o − p_i` world, 3 |
| target velocity | `target` | `Rᵀ(v_other − v)`, 3 | `Rᵀ(v_j − v_i)`, 3 | `v_o − v_i` world, 3 |
| target seen | `target_mask` | always 1 | — | — |
| neighbours | `neighbors` | **same role only**, so empty | every other agent | one opponent |
| normalized | — | **no** | by view range and max speed | same |
| **total width** | | **23** | 21 + rays | 21 |

| we differ | ours | Gavin | why |
|---|---|---|---|
| body rate | `ω` in the obs | not in the obs | our rate loop sits under the policy |
| walls | height only | M horizontal rays | the evader does not run for the wall yet; walls come at task 4 |
| scaling | raw SI units | divided by view range and max speed | decided: no normalization |

`neighbors` is **same-role drones only**. The other side is already in `target`, so
counting it as a neighbour too would put the same 6 numbers in the vector twice.
The pursuer has no other pursuer, so its block is empty and the flat obs is 23 wide.

## Environment

| | ours | Gavin |
|---|---|---|
| arena | box 32 × 32 × 16 m | box 32 × 32 × 16 m |
| centre | `(0, 0, 8)` | — |
| episode | 10 s = 1000 steps | 10 s = 1000 steps |
| `sim_dt` | 0.005 (200 Hz) | not published |
| `steps_per_action` | 2 → policy at 100 Hz | 100 Hz |
| action | CTBR, `a ∈ [-1,1]⁴` | CTBR |
| `max_rate_rp` / `max_rate_yaw` | 4.0 / 2.0 rad/s | not published |
| drone | x500, 2.0 kg, footprint 0.80 m across | not published |
| drones | `roles = ("pursuer", "evader")`, `scripted = ("evader",)` | pursuers + evader |
| start | both uniform random in the box, 2 m clear of the walls | both uniform random |
| start heading | random, full circle | not published |
| evader | cascade to random waypoints | learning evader |

## The evader

The evader is a second drone on the drone axis, flown by our own cascade. Its
network action is thrown away inside `step`. At task 4 you delete `"evader"` from
`scripted` and the same env trains it.

| | value |
|---|---|
| waypoint | a random point within 3 m of the evader |
| bounds | 2 m clear of the walls |
| redraw | when the evader is within 0.2 m |
| heading reference | 0.0 |

Measured over a 10 s episode ([scratch/evader_speed.py](../../scratch/evader_speed.py)):
8 waypoints reached, mean speed 1.48 m/s, peak 3.22 m/s, only 5% of the time
under 0.5 m/s. The cascade carries momentum through each waypoint instead of
parking on it, so it is a real moving target.

Speed is capped by the cascade, not by a parameter: `position_gains(max_velocity=6.0)`
and `velocity_gains(max_acceleration=4.0)`. Pliska's test targets reach 10 m/s and
8 m/s², so ours is a slower target than the published baseline flies against.

## The net

| | ours | Gavin |
|---|---|---|
| shape | flat circle in the drone's body xy plane | rigid circular disc, aligned with the body frame |
| `net_radius` | 0.5 m | `R`, no number published |
| `net_offset` (hangs below) | 0.5 m | "mounted below each pursuer", no number |
| `capture_dist` (half slab) | 0.4 m | "a capture distance", no number |
| hitbox shape | cylinder: radius 0.5 m, height 0.8 m | not published |

| step | formula |
|---|---|
| target offset, body frame | `p = Rᵀ(x_t − x)` |
| distance from the net plane | `a = p_z + net_offset` |
| distance from the net axis | `r = ‖(p_x, p_y)‖` |
| caught | `(r <= net_radius) and (abs(a) <= capture_dist)` |
| net centre | `c_net = x + R · (0, 0, −net_offset)` |

`net_offset` must stay **above** `capture_dist`, or part of the cylinder sits above
the drone and "catch from above" stops being enforced. At 0.5 and 0.4 the cylinder
runs from 0.9 m to 0.1 m below the pursuer.

Pliska et al. 2024 hang a **2 m radius** net under the same plant. Ours is 0.5 m —
a quarter of that, and deliberately harder.

## Reward

Ours, every sign written out:

```
pursuer = + catch · caught                        one-off, ends the episode
          - crash · died                          one-off, ends the episode
          - policy_dt · distance · ‖x_e - c_net‖
          - policy_dt · step                      the clock
          - policy_dt · cmd · ‖a_ω‖               the COMMANDED body rates

evader  = - catch · caught
          - crash · died
          + policy_dt · step                      the same clock, earned
          - policy_dt · cmd · ‖a_ω‖
```

The evader's row is written and then thrown away: its loss mask is 0. It is there
so task 4 needs no new reward code.

Gavin Nv1, the pursuer. Each `r` below is a positive quantity; the signs are in
the sum:

```
rP = + rcatch                           λcatch  = 10.0   one-off
     - rdist   = λdist·‖p_e - c_net‖    λdist   = 0.001
     - rstep   = λstep                  λstep   = 0.04   the evader's survival pay
     - rcollPP = λcollPP·1(hit pursuer) λcollPP = 10.0   ends the episode
     - rcollPE = λcollPE·1(hit evader)  λcollPE = 0.1    soft, does not end
     - rfail   = λfail·1(crash or out)  λfail   = 30.0   one-off
     - rcmd    = λcmd·‖aω‖              λcmd    = 2e-4
```

Neither side has a positive per-step term. Every step costs.

### The weights

Every per-step weight is Gavin's **divided by `policy_dt`**, so that the code's
`× policy_dt` puts it back. Same reward as Gavin, and `policy_dt` still buys
independence from the control rate.

| `RewardConfig` | ours | × `policy_dt` | Gavin |
|---|---|---|---|
| `distance` | 0.1 | 0.001 | `λdist` |
| `step` | 4.0 | 0.04 | `λstep` |
| `cmd` | 0.02 | 0.0002 | `λcmd` |
| `catch` | 10.0 | not multiplied | `λcatch = 10.0` |
| `crash` | 30.0 | not multiplied | `λfail = 30.0` |

| term | ours | Gavin pursuer |
|---|---|---|
| distance | to `c_net` | to `c_net` |
| body rate | `cmd · ‖a_ω‖`, **commanded** | same. We dropped `0.1·‖ω‖` on measured `ω` |
| upright | none | none |
| velocity | none | none |
| action difference | none | none |
| contact with the evader | none, it has no body at task 3 | `λcollPE = 0.1`, soft |
| contact between pursuers | none, `N = 1` | `λcollPP = 10.0`, ends the episode |
| wall buffer | none | `λbnd`, evader only |

### The risk we accepted

Over a 1000-step episode the clock alone bills `0.04 × 1000 = 40`, and `λfail` is
only 30. So **crashing early is cheaper than surviving without a catch.** That is
in Gavin's own numbers, and it is the failure that killed the first
`circle_swap pair` run.

## Termination

| event | pays | Gavin |
|---|---|---|
| caught | `+ catch` | `λcatch` |
| outside the box | `− crash` | `λfail` |
| non-finite state | `− crash` | not published |
| step limit, 1000 | nothing | same |
| **evader** dies | nothing, to either side | "neither receives a reward when the opponent reaches a failure state" |

`crash` is charged only for a drone's own death. `info["evader_died"]` counts the
evader's.

## Hyperparameters

| field | default | task chase | why |
|---|---|---|---|
| `gamma` | 0.998 | **0.99** | Gavin's value at the same 100 Hz. 0.998 is a 5 s horizon, his is 1 s. |
| everything else | — | unchanged | — |

The default in `learn/config.py` is left alone, because `a_to_b` and `circle_swap`
were trained at 0.998. Pass it on the command line:

```bash
python run/train.py --task chase --preset default --set gamma=0.99 --steps 2e8
```

## What the code touches

| file | change |
|---|---|
| `swarm/envs/chase.py` | new |
| `swarm/learn/ppo.py` | `role_slices` skips a scripted role, so it gets no weights |
| `swarm/learn/vecenv.py` | the loss mask goes to 0 for a scripted drone |
| `swarm/envs/{a_to_b,circle_swap}.py` | `scripted = ()`, so nothing needs a `getattr` |

`EnvState` carries its own PRNG key. `evaluate.rollout` passes one key for the
whole episode, so a waypoint drawn from the caller's key would never change.

## Baseline

The cascade flown at a point `net_offset` above the evader — pure pursuit — catches
**16 of 16** episodes, median end step 489 (4.9 s)
([scratch/chase_smoke.py](../../scratch/chase_smoke.py)). No gate; this is the
number to look at beside the policy.

## Open

| question | state |
|---|---|
| eval and viewer for a moving target | `evaluate.py` records the goal once from `reset`; the viewer draws a sphere arena and no net |
| FRPN baseline | Pliska's guidance law; after the first run |
| velocity term `‖v_e − v‖` | waiting on the `a_to_b` test |
| curriculum with a sparse reward | later, after reward shaping works |
| wall distances in the obs | task 4 |
| contact between drone bodies | task 4 |
| `λbnd`, the evader's wall buffer | task 4 |
