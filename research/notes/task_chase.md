# Task chase — the spec

N drones chase an evader that the cascade flies to random waypoints. A pursuer
catches it with a net. Task 3 on the ladder in `goals.local.md`, and the seat for
tasks 4 and 5 — the pursuer count comes from `roles`, and dropping `"evader"` from
`scripted` hands the evader to a network. Built in
[chase.py](../../swarm/envs/chase.py).

| preset | roles | scripted |
|---|---|---|
| `default` | 1 pursuer, 1 evader | the evader |
| `three` | 3 pursuers, 1 evader | the evader |

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
| other drone, position | `others` | `Rᵀ(x_j − x)`, 3 | `Rᵀ(p_j − p_i)`, 3 | `p_o − p_i` world, 3 |
| other drone, velocity | `others` | `Rᵀ(v_j − v)`, 3 | `Rᵀ(v_j − v_i)`, 3 | `v_o − v_i` world, 3 |
| same role as me | `others` | 0 or 1, 1 | — | — |
| seat filled | `others_mask` | 0 or 1 per seat | — | — |
| goal point | `target` | **width 0** | — | — |
| normalized | — | **no** | by view range and max speed | same |
| **total width** | | **16 + 8K** | 21 + rays | 21 |

| we differ | ours | Gavin | why |
|---|---|---|---|
| body rate | `ω` in the obs | not in the obs | our rate loop sits under the policy |
| walls | height only | M horizontal rays | the evader does not run for the wall yet; walls come at task 4 |
| scaling | raw SI units | divided by view range and max speed | we do not normalize |

**Every other drone sits in `others`, whatever its role.** A teammate and an
opponent are told apart by the last feature, not by sitting in different blocks.
Rows are sorted by distance, nearest first, and the mask zeroes an empty seat.
`K = n_visible = min(n_neighbors, n_drones − 1)`, so `default` gives `K = 1` and a
24-wide vector, and `three` gives `K = 3` and a 40-wide one.

`target` is where a task says "fly to this point". Nobody here does — a pursuer
chases a drone and the evader runs from one — so chase gives it width 0.

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
| drones | from `roles`; `scripted = ("evader",)` | pursuers + evader |
| start | all uniform random in the box, 2 m clear of the walls | both uniform random |
| start heading | random, full circle | not published |
| evader | cascade to random waypoints | learning evader |
| contact | `collision_dist = 0.8 m` between centres | the same |

`arena`, `center`, `collision_dist` and the start spread live in `core.Common`, so
every task shares them. Only the net and the waypoints are chase's own.

## Three words, three events

Never mix them. The same name is used for the `RewardConfig` field, the function in
`chase.py`, and the key in `info` — Gavin's own names, so the paper reads straight
onto the code.

| word | what it is | ends the episode | weight |
|---|---|---|---|
| `crash` | a wall, the ground, the ceiling, or a non-finite state | yes | 30.0 = λfail |
| `collPP` | a pursuer hit another pursuer, body **or net** | yes | 10.0 = λcollPP |
| `collPE` | a pursuer hit the evader | **no** | 0.1 per step = λcollPE |

`collPE` stays soft because the pursuer must reach the evader to catch it. A rule
that ends the episode there teaches it to stand off. Gavin's words: the soft penalty
allows "gradual learning of collision avoidance while maintaining focus on the
primary task of catching with the net".

The net on another pursuer counting as `collPP` is ours; Gavin's is body to body only.

## The evader

The evader is the last drone on the drone axis, flown by our own cascade. Its
network action is thrown away inside `step` — but only while `"evader"` is in
`scripted`. Delete it from `scripted` and the same env trains the evader, with no
other change.

| | value |
|---|---|
| waypoint | a random point within 3 m of the evader |
| bounds | 2 m clear of the walls |
| redraw | when the evader is within 0.2 m |
| heading reference | 0.0 |

Speed is capped by the cascade, not by a parameter: `position_gains(max_velocity=6.0)`
and `velocity_gains(max_acceleration=4.0)`. Pliska's test targets reach 10 m/s and
8 m/s², so ours is a slower target than the published baseline flies against.

## The net

Every pursuer carries one. A catch by any of them ends the episode.

| | ours | Gavin |
|---|---|---|
| shape | square standing in the body yz plane, face along body x | rigid circular disc, aligned with the body frame |
| `net_side` | 1.0 m | `R`, no number published |
| `net_offset`, the rope | 0.4 m | "mounted below each pursuer" |
| hangs from / to | 0.4 m / 1.4 m below the drone | no number |
| `c_net`, the middle | 0.9 m below the drone | "the centre of the catching net" |
| thickness | **zero** — it is a plane | "a capture distance", no number |

**`net_offset` is the rope, not the middle.** The square starts where the rope ends
and hangs a full `net_side` below that, so the middle is
`_drop = net_offset + net_side/2` = 0.9 m down.

| step | formula |
|---|---|
| any drone in a pursuer's net frame | `p = Rᵀ(x_k − x_p) + (0, 0, _drop)` |
| on the square | `abs(p_y) <= net_side/2` and `abs(p_z) <= net_side/2` |
| touching the plane | `abs(p_x) <= body_radius` |
| caught | both, for the evader column, for any pursuer |
| `collPP` | both, for a pursuer column |
| `c_net`, world | `x_p − _drop · R_p·(0,0,1)` |

The square has **no thickness**. The drone is a ball of radius
`arm_length + prop_radius` = 0.40 m, and a ball touches a plane when its centre is
within one radius of it. That is the whole of `abs(p_x) <= body_radius` — it is the
drone's size, not a fudge factor, and there is no `capture_dist` field. At 15 m/s the
evader moves 0.15 m per policy step, well inside 0.40 m, so it cannot skip through
between two steps.

`in_net` returns `(P, N)`: every drone against every pursuer's net. `is_caught`
reads the evader column, `net_collision` reads the pursuer columns. A net hangs
0.9 m below its own drone, which is off its own square, so no pursuer catches
itself and no special case is needed.

The square reaches into the 0.8 m hitbox, so `step` checks the catch first and both
death flags carry `& ~caught`.

## Reward

Ours, every sign written out:

```
every pursuer = + catch · caught                  one-off, ends the episode.
                                                  ONE catch pays ALL pursuers
                - crash · crashed                 one-off, ends the episode
                - collPP · collided               one-off, ends the episode
                - policy_dt · collPE · touching   per step, episode goes on
                - policy_dt · distance · ‖x_e - c_net_i‖   its OWN net
                - policy_dt · step                the clock
                - policy_dt · cmd · ‖a_ω‖         the COMMANDED body rates

evader        = - catch · caught                  once, not once per pursuer
                - crash · crashed
                - policy_dt · collPE · touching   the same bill, both sides pay
                + policy_dt · step                the same clock, earned
                - policy_dt · cmd · ‖a_ω‖
```

Each pursuer pays the distance from the evader to **its own** net centre, so a
pursuer that hangs back still pays. `info["distance"]` logs the smallest of them.

The evader's row is written and then thrown away while it is scripted: its loss
mask is 0. It is there so task 4 needs no new reward code.

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
| `collPE` | 10.0 | 0.1 | `λcollPE` |
| `catch` | 10.0 | not multiplied | `λcatch = 10.0` |
| `collPP` | 10.0 | not multiplied | `λcollPP = 10.0` |
| `crash` | 30.0 | not multiplied | `λfail = 30.0` |

| term | ours | Gavin pursuer |
|---|---|---|
| distance | to `c_net`, the middle of the square | to `c_net` |
| body rate | `cmd · ‖a_ω‖`, **commanded** | same. We dropped `0.1·‖ω‖` on measured `ω` |
| upright | none | none |
| velocity | none | none |
| action difference | none | none |
| `collPE` | 0.1 per step, both sides, soft | the same |
| `collPP` | 10.0, ends it, **net counts too** | 10.0, ends it, bodies only |
| wall buffer | none | `λbnd`, evader only |

## Termination

| event | ends it | pays | Gavin |
|---|---|---|---|
| caught by any pursuer | yes | `+ catch` to every pursuer, `− catch` to the evader | `λcatch` |
| outside the box | yes | `− crash` | `λfail` |
| non-finite state | yes | `− crash` | not published |
| pursuer to pursuer, under 0.8 m or net | yes | `− collPP` to both | `λcollPP`, bodies only |
| pursuer to evader, under 0.8 m | **no** | `− policy_dt · collPE` to both, every step | the same |
| step limit, 1000 | yes | nothing | same |
| **evader** crashes | yes | nothing, to either side | "neither receives a reward when the opponent reaches a failure state" |

`crash` and `collPP` are charged only for a drone's own death. `info["evader_died"]`
counts the evader's. A drone that catches on the same step it collides is **not**
dead: both flags carry `& ~caught`.

`is_dead` is chase's own, not `core.is_dead`: Gavin ends an episode on out of
bounds and on inter-pursuer contact, and says nothing about attitude, so a pursuer
flipped upside down at altitude keeps flying.

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

## Where it lives

| file | what it holds for this task |
|---|---|
| `swarm/envs/chase.py` | the net, the waypoints, the reward, the three collision words |
| `swarm/envs/core.py` | the params, plant loop, obs blocks, hitbox, `body_radius` and spawn every task shares |
| `swarm/learn/ppo.py` | `role_slices` skips a scripted role, so it gets no weights |
| `swarm/learn/vecenv.py` | the loss mask is 0 for a scripted drone |
| every env | `reference(state, params)` → `(n, 3)`: where each drone is trying to fly |
| `swarm/learn/evaluate.py` | records `reference` per step; the cascade baseline reads it; `header.json` carries `net` |
| `tools/viewer/index.html` | box arena, the net square, the hitbox ball, per-step goal |

`reference` has two users: the cascade baseline flies to it, and the viewer draws
it as the marker sphere. A pursuer's is the evader's position; the evader's is its
waypoint.

**Nothing in `chase.py` names a drone by number.** `_sides(params)` reads `roles`
and returns the pursuer rows and the evader row. `roles` is static, so it costs
nothing at run time, and `PRESETS["three"]` needs no other change.

`EnvState` carries its own PRNG key. `evaluate.rollout` passes one key for the
whole episode, so a waypoint drawn from the caller's key would never change.

## Baseline

The cascade holding the net on the evader, 256 episodes,
[scratch/net_flight.py](../../scratch/net_flight.py). It flies the pursuer to a fixed
height above the evader, so it is an upper bound on aim, not on reflexes:

| the pursuer aims this far above the evader | caught | crashed |
|---|---|---|
| 0.9 m, `c_net` | 70.7% | 28.9% |
| 1.2 m | 84.0% | 13.7% |
| 1.4 m, the low edge of the square | 87.5% | 6.6% |

No gate. The cascade is a position controller and a weak pursuer. FRPN is the
baseline that matters.

## Open

| question | state |
|---|---|
| FRPN baseline | Pliska's guidance law; after the first run |
| velocity term `‖v_e − v‖` | waiting on the `a_to_b` test |
| curriculum with a sparse reward | later, after reward shaping works |
| wall distances in the obs | task 4 |
| `λbnd`, the evader's wall buffer | task 4 |
| the evader sees every drone, not just its nearest pursuer | task 4, when it learns |
| `rdist` aims at `c_net`, 0.9 m down; the baseline scores best aiming 1.4 m down | open |
