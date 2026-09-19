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
| target position | `target` | `Rᵀ(x_other − x)`, 3 | `Rᵀ(p_j − p_i)`, 3 | `p_o − p_i` world, 3 |
| target velocity | `target` | `Rᵀ(v_other − v)`, 3 | `Rᵀ(v_j − v_i)`, 3 | `v_o − v_i` world, 3 |
| target seen | `target_mask` | always 1 | — | — |
| neighbours | `neighbors` | **same role only**, K nearest | every other agent | one opponent |
| normalized | — | **no** | by view range and max speed | same |
| **total width** | | **23 + 7K** | 21 + rays | 21 |

| we differ | ours | Gavin | why |
|---|---|---|---|
| body rate | `ω` in the obs | not in the obs | our rate loop sits under the policy |
| walls | height only | M horizontal rays | the evader does not run for the wall yet; walls come at task 4 |
| scaling | raw SI units | divided by view range and max speed | decided: no normalization |

`neighbors` is **same-role drones only**. The other side is already in `target`, so
counting it as a neighbour too would put the same 6 numbers in the vector twice.
`K = n_visible = min(n_neighbors, largest role − 1)`, so `default` gives `K = 0`
and a 23-wide vector, and `three` gives `K = 2` and a 37-wide one.

`target` holds one drone. A pursuer's is the evader. **The evader's is the pursuer
nearest to it**, so a learned evader sees one chaser, not all of them. Gavin's
evader sees every pursuer. Open, and it only matters once the evader learns.

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
| contact | any two centres under `collision_dist = 0.8 m` kills both and ends the episode | pursuer–pursuer ends it; pursuer–evader is a soft cost |

`arena`, `center`, `collision_dist` and the start spread now live in `core.Common`,
so every task shares them. Only the net and the waypoints are chase's own.

**We differ from Gavin on contact with the evader.** He keeps it soft so the
pursuer is not afraid to close in. Ours kills both. Decided this way on purpose;
revisit if the pursuers learn to stand off.

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

Measured over a 10 s episode ([scratch/evader_speed.py](../../scratch/evader_speed.py)):
8 waypoints reached, mean speed 1.48 m/s, peak 3.22 m/s, only 5% of the time
under 0.5 m/s. The cascade carries momentum through each waypoint instead of
parking on it, so it is a real moving target.

Speed is capped by the cascade, not by a parameter: `position_gains(max_velocity=6.0)`
and `velocity_gains(max_acceleration=4.0)`. Pliska's test targets reach 10 m/s and
8 m/s², so ours is a slower target than the published baseline flies against.

## The net

Every pursuer carries one. A catch by any of them ends the episode.

| | ours | Gavin |
|---|---|---|
| shape | square standing in the body yz plane, face along body x | rigid circular disc, aligned with the body frame |
| centre | `net_offset` below the drone, body frame | "mounted below each pursuer" |
| `net_side` | 1.0 m | `R`, no number published |
| `net_offset` (hangs below) | 0.4 m | no number |
| plane thickness | the evader's own radius, `arm_length + prop_radius` = 0.4 m | "a capture distance", no number |

| step | formula |
|---|---|
| evader offset, pursuer's body frame | `p = Rᵀ(x_e − x_p)` |
| net centre, body frame | `(0, 0, −net_offset)` |
| inside the square | `abs(p_y) <= net_side/2` and `abs(p_z + net_offset) <= net_side/2` |
| crossing its plane | `abs(p_x) <= body_radius` |
| caught | both of the above, for any pursuer |
| net centre, world | `c_net = x_p + R_p · (0, 0, −net_offset)` |

There is no `capture_dist` field any more. The thickness is the evader's body
radius, so it follows the drone model: a net catches a ball, not a point. At
15 m/s the evader moves 0.15 m per policy step, well inside 0.4 m, so it cannot
skip through the plane between two steps.

The square overlaps the 0.8 m kill ball — a catch at `p = (0, 0, 0)` is 0 m from
the pursuer, while the far corner of the net is 1.10 m away and outside it. So
`step` checks the catch first and `died` carries `& ~caught`.



## Reward

Ours, every sign written out:

```
every pursuer = + catch · caught                  one-off, ends the episode.
                                                  ONE catch pays ALL pursuers
                - crash · died                    one-off, ends the episode
                - policy_dt · distance · ‖x_e - c_net_i‖   its OWN net
                - policy_dt · step                the clock
                - policy_dt · cmd · ‖a_ω‖         the COMMANDED body rates

evader        = - catch · caught                  once, not once per pursuer
                - crash · died
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
| `catch` | 10.0 | not multiplied | `λcatch = 10.0` |
| `crash` | 30.0 | not multiplied | `λfail = 30.0` |

| term | ours | Gavin pursuer |
|---|---|---|
| distance | to `c_net` | to `c_net` |
| body rate | `cmd · ‖a_ω‖`, **commanded** | same. We dropped `0.1·‖ω‖` on measured `ω` |
| upright | none | none |
| velocity | none | none |
| action difference | none | none |
| contact with the evader | `− crash`, ends the episode | `λcollPE = 0.1`, soft |
| contact between pursuers | `− crash`, ends the episode | `λcollPP = 10.0`, ends the episode |
| wall buffer | none | `λbnd`, evader only |

### The risk we accepted

Over a 1000-step episode the clock alone bills `0.04 × 1000 = 40`, and `λfail` is
only 30. So **crashing early is cheaper than surviving without a catch.** That is
in Gavin's own numbers, and it is the failure that killed the first
`circle_swap pair` run.

## Termination

| event | pays | Gavin |
|---|---|---|
| caught by any pursuer | `+ catch` to every pursuer, `− catch` to the evader | `λcatch` |
| outside the box | `− crash` | `λfail` |
| non-finite state | `− crash` | not published |
| two centres under 0.8 m | `− crash` to both | `λcollPP` ends it, `λcollPE` does not |
| step limit, 1000 | nothing | same |
| **evader** dies | nothing, to either side | "neither receives a reward when the opponent reaches a failure state" |

`crash` is charged only for a drone's own death. `info["evader_died"]` counts the
evader's. A drone that catches on the same step it touches is **not** dead: `died`
carries `& ~caught`.

`is_dead` is chase's own, not `core.is_dead`: Gavin ends an episode on out of
bounds and on contact, and says nothing about attitude, so a pursuer flipped
upside down at altitude keeps flying.

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
| `swarm/envs/core.py` | the params, plant loop, obs blocks, hitbox and spawn every task shares |
| `swarm/learn/ppo.py` | `role_slices` skips a scripted role, so it gets no weights |
| `swarm/learn/vecenv.py` | the loss mask goes to 0 for a scripted drone |
| every env | new `reference(state, params)` → `(n, 3)`: where each drone is trying to fly |
| `swarm/learn/evaluate.py` | records `reference` **per step**; the cascade baseline reads it; header gains `net` |
| `tools/viewer/index.html` | box arena, the net square, the hitbox ball, per-step goal |
| `run/plot.py` | the goal is now per step, so it slices like every other field |

`reference` has two users: the cascade baseline flies to it, and the viewer draws
it as the marker sphere. A pursuer's is the evader's position; the evader's is its
waypoint.

**Nothing in `chase.py` names a drone by number.** `_sides(params)` reads `roles`
and returns the pursuer rows and the evader row. `roles` is static, so it costs
nothing at run time, and `PRESETS["three"]` needs no other change.

`EnvState` carries its own PRNG key. `evaluate.rollout` passes one key for the
whole episode, so a waypoint drawn from the caller's key would never change.

## Baseline

The cascade flown straight at the evader — pure pursuit, measured by
[scratch/chase_smoke.py](../../scratch/chase_smoke.py). Not measured yet. No gate:
the cascade is a position controller and a weak pursuer. FRPN is the baseline that
matters.

## Open

| question | state |
|---|---|
| FRPN baseline | Pliska's guidance law; after the first run |
| velocity term `‖v_e − v‖` | waiting on the `a_to_b` test |
| curriculum with a sparse reward | later, after reward shaping works |
| wall distances in the obs | task 4 |
| `λbnd`, the evader's wall buffer | task 4 |
| the evader sees only its nearest pursuer | task 4, when it learns |
| pursuer–evader contact kills; Gavin keeps it soft | revisit if the pursuers stand off |
