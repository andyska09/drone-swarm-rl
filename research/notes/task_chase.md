# Task chase — the spec

Pursuers catch one evader with a net hung under the body. Both sides learn. The
`default` preset is the duel: one pursuer against one evader. A longer `roles` adds
pursuers with no other change, and that is task 5. Built in
[chase.py](../../swarm/envs/chase.py). How it trains: [AMSPBH.md](AMSPBH.md). The
hand-written opponents: [baselines.md](baselines.md). What the pool runs scored:
[experiments/chase_pools_log.md](experiments/chase_pools_log.md).

Every value beside a Gavin column is from
[Gavin & Bronz 2026 Nv1](../papers/2607.05939v1.md) unless marked 1v1
([Gavin et al. 2026](../papers/2603.16279v1.md)).

## Presets

| preset | arena | centre | a wall or the ceiling | the evader outside the box |
|---|---|---|---|---|
| `default` | 32 × 32 × 16 m | `(0, 0, 8)` | ends the episode | — |
| `small` | 24 × 24 × 16 m | `(0, 0, 8)` | ends the episode | — |
| `open` | 32 × 32 × 16 m | `(0, 0, 8)` | is not there | pays `outside` = 57 per second |
| `nemecek` | 20 × 20 × 10 m | `(0, 0, 5)` | is not there | pays `outside` = 57 per second |

The floor is at z = 0 in every preset, and the ground always ends the episode. Without
walls (`walls=False`) the pursuer pays nothing outside the box.

- **`open` has Němeček's arena rules.** Only the ground, or a non-finite state, ends an
  episode. The pursuer has no walls and no ceiling. The evader flies on outside the box
  and pays `outside`. Where 57 comes from is under [The weights](#the-weights).
- **`small` is `default` in a 24 × 24 × 16 m box.** Only the floor is smaller. The saved
  policies of `chase_default_s0_20261014-100847_9e4242`, put in this box, caught the
  newest evader in 0.39 of the episodes, against 0.17 in 32 × 32 m. The evader left the
  box no more often (0.03), and the pursuer left it twice as often (0.16, against 0.08).
- **`nemecek` is `open` in Němeček's box**, 20 × 20 × 10 m with the floor at 0, the box
  of his main runs (thesis, Fig. 8.10, and `arena_size_ = 10.0` in his `environment.h`).
  His 12 m/s speed cap and his 12 s episodes are not in it.

## Environment

| | ours | Gavin |
|---|---|---|
| arena | the preset's box | box 32 × 32 × 16 m |
| episode | 10 s = 1000 steps | 10 s = 1000 steps |
| `sim_dt` | 0.005 (200 Hz) | not published |
| `steps_per_action` | 2 → policy at 100 Hz | 100 Hz |
| action | CTBR, `a ∈ [-1,1]⁴` | CTBR |
| `max_rate_rp` / `max_rate_yaw` | 4.0 / 2.0 rad/s | not published |
| drone | x500, 2.0 kg, 0.80 m across the propeller tips | not published |
| drones | from `roles`, default `("pursuer", "evader")` | pursuers + evader |
| start | uniform random in the box, 2 m clear of every face | both uniform random |
| start heading | random, full circle | not published |
| start motion | up to 0.5 m/s on each axis, tilt up to 0.2 rad | not published |
| contact | `collision_dist = 0.8 m` between centres | the same |

`arena`, `center`, `collision_dist` and the start spread live in `core.Common`, so
every task shares them. The net, the waypoint, `walls` and `outside` are chase's own.

**Nothing in `chase.py` names a drone by number.** `_sides(params)` reads the pursuer
rows and the evader row from `roles`. `roles` is static, so this costs nothing at run
time. `ppo.role_slices` gives each role its own actor, critic and TrainState.

## Observation

| | `Obs` block | ours | Gavin Nv1 | Gavin 1v1 |
|---|---|---|---|---|
| my velocity | `own` | `Rᵀv` body, 3 | `Rᵀv` body, 3 | `v` world, 3 |
| my attitude | `own` | `vec(R)`, 9 | `vec(R)`, 9 | `vec(R)`, 9 |
| my body rate | `own` | `ω` body, 3 | none | none |
| height | `own` | `z`, 1 | `z`, 1 | the distance to the ground |
| walls | `own` | 4 distances, level body frame | M horizontal rays | the distance to each arena bound |
| other drone, position | `others` | `Rᵀ(x_j − x)`, 3 | `Rᵀ(p_j − p_i)`, 3 | `p_o − p_i` world, 3 |
| other drone, velocity | `others` | `Rᵀ(v_j − v)`, 3 | `Rᵀ(v_j − v_i)`, 3 | `v_o − v_i` world, 3 |
| same role as me | `others` | 0 or 1, 1 | — | — |
| seat filled | `others_mask` | 0 or 1 per seat | — | — |
| goal point | `target` | **width 0** | — | — |
| normalized | — | **no** | by view range and max speed | the same |
| **total width** | | **20 + 8K** | 13 + M + 6 per other drone | 18 + the bound distances |

| we differ | ours | Gavin | why |
|---|---|---|---|
| body rate | `ω` in the obs | not in the obs | our rate loop sits under the policy |
| walls | 4 rays in the level body frame | Nv1: M rays. 1v1: the distance to each bound | four rays are exact for a box |
| scaling | raw SI units | divided by view range and max speed | we do not normalize |

**Every other drone sits in `others`, whatever its role.** A teammate and an
opponent are told apart by the last feature, not by sitting in different blocks.
Rows are sorted by distance, nearest first, and the mask zeroes an empty seat.
`K = n_visible = min(n_neighbors, n_drones − 1)` with `n_neighbors = 6`, so the duel
gives `K = 1` and a 28-wide vector.

`target` is where a task says "fly to this point". Nobody here does — a pursuer
chases a drone and the evader runs from one — so chase gives it width 0.

**The walls are four rays level with the floor:** nose, tail, left and right. Each one
is the distance from the drone to the box along that ray, in raw metres, with no clip.
The nose ray is the body x axis flattened onto the floor. If the nose points straight
up or down, the body y axis turned by 90° takes its place, through a `jnp.where`. Roll
and pitch never turn the rays. Yaw does, like every other block. Outside the box, in
`open` and `nemecek`, some distances go negative. Eight rays would add the diagonals.

**The critic sees the whole game** (CTDE): `scene`, which is every drone's exact `x`,
`v` and `R` in the world frame and its `ω`, 18 numbers per drone, plus the opponents'
actions this step. That is 40 numbers in the duel. Gavin's value network gets the same
without `ω`, and normalized.

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
`_drop = net_offset + net_side/2` = 0.9 m down the body z axis.

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

## Three words, three events

Never mix them. `collPP` and `collPE` are Gavin's names, and each one names the
`RewardConfig` field, the function in `chase.py` and the key in `info` alike.

| word | what it is | ends the episode | weight |
|---|---|---|---|
| `fail` | leaving the box through the ground, a wall or the ceiling, or a non-finite state. Without walls, only the ground or a non-finite state | yes | 30.0 = λfail |
| `collPP` | a pursuer hit another pursuer, body **or net** | yes | 10.0 = λcollPP |
| `collPE` | a pursuer body hit the evader body | **no** | 0.1 per step = λcollPE |

The outcomes split a `fail` in two: `crash` is the ground or a non-finite state, and
`OOB` is a wall or the ceiling.

`collPE` stays soft because the pursuer must reach the evader to catch it. A rule
that ends the episode there teaches it to stand off. Gavin's words: the soft penalty
allows "gradual learning of collision avoidance while maintaining focus on the
primary task of catching with the net".

The net on another pursuer counting as `collPP` is ours. Gavin's is body to body only.

## Reward

Ours, every sign written out:

```
every pursuer = + catch · caught                    one-off, ends the episode.
                                                    ONE catch pays ALL pursuers
                - fail · failed                     one-off, its own failure only
                - collPP · collided                 one-off, ends the episode
                - policy_dt · collPE · touching     per step, the episode goes on
                - policy_dt · distance · ‖x_e - c_net_i‖   its OWN net
                - policy_dt · step                  the clock
                - policy_dt · cmd · ‖ω_cmd‖         the COMMANDED body rates, rad/s

evader        = - catch · caught                    once, not once per pursuer
                - fail · failed
                - policy_dt · collPE · touching     the same bill, both sides pay
                + policy_dt · step                  the same clock, earned
                - policy_dt · cmd · ‖ω_cmd‖
                - policy_dt · outside · outside_box   0 unless the preset sets it
```

A drone pays the per-step terms only while it is alive. Each pursuer pays the distance
from the evader to **its own** net centre, so a pursuer that hangs back still pays.
`info["step"]["distance"]` logs the smallest of them.

Gavin Nv1, the pursuer. Each `r` below is a positive quantity, and the signs are in
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

### The weights

Every per-step weight is Gavin's **divided by `policy_dt`**, so that the code's
`× policy_dt` puts it back. Same reward as Gavin, and `policy_dt` still makes the
weights independent of the control rate.

| `RewardConfig` | ours | × `policy_dt` | Gavin |
|---|---|---|---|
| `distance` | 0.1 | 0.001 | `λdist` |
| `step` | 4.0 | 0.04 | `λstep` |
| `cmd` | 0.02 | 0.0002 | `λcmd` |
| `collPE` | 10.0 | 0.1 | `λcollPE` |
| `outside` | 0.0, and 57.0 in `open` and `nemecek` | 0.57 | none |
| `catch` | 10.0 | not multiplied | `λcatch = 10.0` |
| `collPP` | 10.0 | not multiplied | `λcollPP = 10.0` |
| `fail` | 30.0 | not multiplied | `λfail = 30.0` |

`outside` = 57 is Němeček's ratio. His evader pays 0.1 per step outside the box
against a survival pay of 0.007 per step. That is 14.3 times, and 14.3 × our `step`
of 4.0 is 57.

| term | ours | Gavin pursuer |
|---|---|---|
| distance | to `c_net`, the middle of the square | to `c_net` |
| body rate | `cmd · ‖ω_cmd‖`, **commanded** | the same |
| upright | none | none |
| velocity | none | none |
| action difference | none | none |
| `collPE` | 0.1 per step, both sides, soft | the same |
| `collPP` | 10.0, ends it, **net counts too** | 10.0, ends it, bodies only |
| the evader near the bounds | none. In `open` and `nemecek`, `outside` beyond the box | `λbnd`, evader only, near the bounds |

## Termination

| event | ends it | who bootstraps | pays | Gavin |
|---|---|---|---|---|
| a net catches the evader | yes | nobody | `+ catch` to every pursuer, `− catch` to the evader | `λcatch` |
| `fail` | yes | every drone but the one that failed | `− fail` to the drone that failed | `λfail` |
| `collPP` | yes | every drone but the two that collided | `− collPP` to both | `λcollPP`, bodies only |
| `collPE` | **no** | — | `− policy_dt · collPE` to both, every step | the same |
| the step limit, 1000 | yes | every drone | nothing | the same |

Without walls, in `open` and `nemecek`, only the ground or a non-finite state is a
`fail`. A wall or the ceiling ends nothing.

**Only a catch is terminal.** `step` sets `info["truncated"]` whenever an episode ends
without a catch, so after a failure the survivors bootstrap. Neither side gains when
the other fails, which is Gavin's rule: "neither the evader nor the pursuers receive a
reward when the opponent reaches a failure state". Why we need the bootstrap for it:
[AMSPBH.md](AMSPBH.md), "After a failure".

A catch counts only if every drone was alive at the start of the step. A catch beats a
failure on the same step.

`is_dead` is chase's own, not `core.is_dead`: Gavin ends an episode on out of
bounds and on inter-pursuer contact, and says nothing about attitude, so a pursuer
flipped upside down at altitude keeps flying.

## Metrics

`info["end"]` holds one outcome per episode. In 1v1 the first seven sum to 1.

| `info` | name | what |
|---|---|---|
| `end` | `caught_evader` | a pursuer's net caught the evader |
| `end` | `crash_pursuer` | a pursuer hit the ground, or its state became non-finite |
| `end` | `OOB_pursuer` | a pursuer left through a wall or the ceiling |
| `end` | `timeout` | 1000 steps (10 s) and nothing else happened |
| `end` | `crash_evader` | the evader hit the ground, or its state became non-finite |
| `end` | `OOB_evader` | the evader left through a wall or the ceiling |
| `end` | `fail_both` | a pursuer and the evader failed on the same step |
| `end` | `time_to_catch` | seconds, catches only. Other episodes give NaN, so the mean is over the catches. This is not Gavin's right-censored time |
| `end` | `collPP` | the episode ended on a pursuer hitting a pursuer. It is not one of the seven |
| `end` | `catch_offset` | at a catch: the distance on the net from its middle to the evader, in metres |
| `step` | `distance` | the nearest net centre to the evader, averaged over the chase |
| `step` | `collPE` | the share of steps with a pursuer body on the evader |

The trainer and the eval add `ep_length`, and `return` and `death_rate` per role.

## Seats and evals

A network or a hand-written source can fly any role. Eval and training share the
code in [seats.py](../../swarm/learn/seats.py).

| source | role | what flies |
|---|---|---|
| a run directory or a `.pkl` | any | that network, with its mean action |
| `cascade` | any | the MRS cascade, flown to `reference` |
| `frpn` | pursuer | FRPN guidance |
| `apf` | evader | APF, pushed away from the pursuers and the walls |
| `paths` | evader | a hover, a circle or a figure-eight, drawn for each episode |

How each source flies, its values and its scores: [baselines.md](baselines.md).

`reference(state, params)` gives `(n, 3)`, the point each drone aims at:

- A pursuer aims `_drop` above the evader, along its own body z axis, so that its net
  centre lands on the evader. Flying its body there would hold the evader above the
  square and never catch. The cascade and FRPN both aim at this point.
- The evader aims at its waypoint: a random point within `waypoint_range` = 3 m, kept
  2 m clear of the walls, and drawn again when the evader comes within
  `waypoint_reach` = 0.2 m. Only a `cascade` evader flies to it. A learned evader
  never sees it.

`EnvState` carries its own PRNG key. `evaluate.rollout` passes one key for the whole
episode, so a waypoint drawn from the caller's key would never change.

```bash
python run/eval.py runs/<run>                         # both roles fly the run's latest weights
python run/eval.py runs/<run> --seat evader apf       # the run's pursuer against APF
python run/eval.py runs/<run> --seat pursuer runs/<run>/pool/pursuer_05.pkl
python run/matrix.py runs/<run>                       # every pursuer against every evader
```

`--seat ROLE SOURCE` is repeatable. A role that nobody names flies the run's own
checkpoint. The output lands in `runs/<run>/evals/<role>_<source>/`, for example
`evals/evader_apf/`. The viewer draws the net square and the hitbox ball, and it draws
`reference` as the goal marker only for a `cascade` seat.

## Training

```bash
python run/train.py --task chase --preset default --set gamma=0.99 --set generations=13 --steps 1.4e9
```

- `generations` above 0 trains with pools, one side at a time: [AMSPBH.md](AMSPBH.md).
  At 0, the default, both sides learn at once.
- `gamma` = 0.99 is Gavin's value at the same 100 Hz: a 1 s horizon. The default in
  `learn/config.py` stays at 0.998, a 5 s horizon, because `circle_swap` was trained
  at 0.998.

## Where it lives

| file | what it holds for this task |
|---|---|
| `swarm/envs/chase.py` | the net, the waypoint, the reward, the three words, the outcomes, the presets |
| `swarm/envs/core.py` | the params, the plant loop, the obs blocks with the wall rays, the hitbox, `body_radius` and the spawn every task shares |
| `swarm/learn/seats.py` | who flies a seat |
| `swarm/sim/guidance.py` | FRPN, APF and the paths |
| `swarm/learn/pool.py` | pool training |
| `swarm/learn/evaluate.py` | records `reference` per step. `header.json` carries the net |
| `run/matrix.py` | the win-rate matrix |
| `tools/viewer/index.html` | the box arena, the net square, the hitbox ball, the goal marker |
