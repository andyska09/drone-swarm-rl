# Comparisons

## Observation layout: ours vs. the references

Sources: [envs/__init__.py](../../../swarm/envs/__init__.py) `Obs`,
[core.py](../../../swarm/envs/core.py) `own_obs` and `others`,
[chase.py](../../../swarm/envs/chase.py) `get_obs`,
[Huang et al. 2024](../../papers/2309.13285v2.md) §III-B plus
`quad-swarm-rl/gym_art/quadrotor_multi/quadrotor_multi.py:211-230`,
[Gavin et al. 2026 1v1](../../papers/2603.16279v1.md) §III-B-1,
[Gavin & Bronz 2026 Nv1](../../papers/2607.05939v1.md) §III-B,
[Roncero v3](../../papers/2506.02849v3.md) eq. 7,
[Němeček](../../papers/F3-BP-2026-Nemecek-Vaclav-bachelor_thesis.md) §7.5 plus
`getNormalizedObservation` in his `src/env/environment.cpp`.

`R` is body → world. `Rᵀu` is the world vector `u` written in the body frame.

| | ours (`Obs`) | quad-swarm | Gavin 1v1 | Gavin Nv1 | Roncero v3 | Němeček |
|---|---|---|---|---|---|---|
| my velocity | `Rᵀv` body, 3 | `v` world, 3 | `v` world, 3 | `Rᵀv` body, 3 | `Rᵀv` body, 3 | `v` world, 3 |
| which way I point | `R`, 9 | `R`, 9 | `R`, 9 | `R`, 9 | `R`, 9 | `R`, 9 |
| how fast I turn | `ω` body, 3 | `ω` body, 3 | none | none | `ω` body, 3 | `ω` body, 3 |
| where I am in the arena | height `z` + 4 wall distances in the level body frame, 5 | altitude `h`, 1 | distance to each wall + ground, 6 | `z` + `M` horizontal rays | world position, 3 | the pursuer: `z`, 1. The evader: world position, 3 |
| where the goal is | `Rᵀ(g − x)` body, 3, only in a task with a goal | `x − g` world, 3 | — | — | — | — |
| other drones | `Rᵀ(Δp, Δv)` body + a same-role flag, `(K, 7)` | `(Δp, Δv)` world, `(K, 6)` | the opponent, `(p_o − p_i, v_o − v_i)` world, 6 | every other agent, `Rᵀ(Δp, Δv)` body, 6 each | the opponent, `Rᵀ(Δp, Δv)` body, 6 | the opponent, `Δp`, `‖Δp‖` and `Δv` world, 7 |
| how many other drones | `K = min(n_neighbors, N − 1)`, `n_neighbors = 6` | `K ≤ N − 1`, `--quads_neighbor_visible_num` | 1 | every agent | 1 | 1 |
| sort key | distance `d` | `d + (d̂ · Δv)` — distance plus closing rate | — | not stated | — | — |
| time | — | — | — | — | — | `t / 600`, 1 |
| obstacles | — | 9 SDF values | — | — | — | — |
| action | CTBR | 4 rotor thrusts | CTBR | CTBR | CTBR | world acceleration + heading rate |

We match Gavin **Nv1** almost exactly: body-frame self velocity, body-frame
relative position and velocity for every other agent. The 1v1 paper puts the
opponent in the **world** frame; the Nv1 paper, four months later, moved it to the
body frame. Our extras over Nv1 are `ω` and the same-role flag. For the walls we use
4 rays, where Nv1 uses `M`.

Roncero is close to us too: the body frame for the velocities and the opponent, and
`ω`. He gives the world position instead of wall distances. Němeček stays in the
world frame, adds the distance to the opponent and the time, and scales every value.

### Normalization

| | observation | reward |
|---|---|---|
| ours | none | divide by a running spread of the return, one per drone column |
| quad-swarm | none — raw SI units, `--normalize_input=False` | hand-tuned weights, `--reward_clip=10` |
| Gavin 1v1 | divide by two hand-picked constants: max view range `k_p`, max velocity `k_v` | not stated |
| Gavin Nv1 | same | not stated |
| Roncero v3 | not stated | not stated |
| Němeček | each value by its largest size: positions by `arena_size` = 10 m, the relative velocity by 24 m/s, his own velocity by 12 m/s, `ω` by 6.28 rad/s, `z` mapped to [−1, 1], the time by 600 steps | none: SB3 without `VecNormalize` |

## Reward layout

Per-step terms, and one-off terms marked. Our weights are per second, and the code
multiplies them by `policy_dt = 0.01`. Gavin's are per step at 100 Hz, so our chase
weights equal his. Roncero's and Němeček's are per step at 50 Hz. Sources for those
two: Roncero eq. 8–11 and §IV-A, Němeček eq. 7.22–7.23 and Table 8.8.

| term | ours `chase` | ours `circle_swap` | quad-swarm | Gavin 1v1 | Gavin Nv1 | Roncero v3 | Němeček |
|---|---|---|---|---|---|---|---|
| distance | `0.1·‖x_e − c_net‖`, pursuer | `1.0·‖g−x‖` | `αdist·‖p‖` | `λdist = 0.001` | `λdist = 0.001` | — | — |
| approach | — | — | — | — | — | `κapp = 0.25·(d_{t−1} − d_t)`, pursuer | `κa = 0.5·(d_{t−1} − d_t)`, pursuer |
| capture bonus | `catch = 10.0`, one-off | — | — | `λcatch = 10.0`, one-off | `λcatch = 10.0`, one-off | `κc = 10.0`, less above 2 m/s of closing speed | `κc = 10.0` |
| body rate | — | `0.1·‖ω‖` | `spin = 0.1` | — | — | `κbr = 0.001·‖ω‖²` | `κbr = 0.0005·‖ω‖` |
| command | `cmd = 0.02·‖ω_cmd‖` | — | `effort = 0.05` | `λcmd = 2e-4·‖aω‖` | `λcmd = 2e-4·‖aω‖` | — | — |
| command change | — | — | — | — | — | `κsm = 0.002`, on the change of the rate command | — |
| near miss | — | `10.0·Σ max(1 − d/1.5, 0)` | `αrclose`, same shape, 10.0 | — | — | — | — |
| contact | `collPP = 10.0` one-off, ends it; `collPE = 10.0`, soft | death, `25.0·N` one-off | elastic bounce, `1.0` | `λcoll = 0.1`, soft, no death | `λcollPP = 10.0`, `λcollPE = 0.1` | — | — |
| upright | — | `−1.0·R₃₃` | `orient = 1.0·R₃₃` | — | — | — | — |
| crash / out of bounds | `fail = 30.0` one-off | `25.0·N` one-off | `crash = 1.0` per step | `λfail = 30.0` | `λfail = 30.0` | `κb = 10.0`, to the one who left | the ground: `κg = 10.0` |
| survival | `step = 4.0`: the evader earns it, the pursuer pays it | — | — | — | `λstep = 0.04`: the evader earns it, the pursuers pay it | `κs = 0.0025`, the evader earns it | `κs = 0.007`, the evader earns it |
| timeout | — | — | — | — | — | — | `κτ = 10.0`: the evader earns it, the pursuer pays it |
| wall buffer | `outside = 57`, evader only, outside the box, in `open` and `nemecek` | — | — | `λbnd = 1.0`, evader only | `λbnd`, evader only | — | `κb = 0.1` per step outside the box, evader only |
| discount | 0.99 | 0.998 | not published | 0.99 | 0.99 | not stated | 0.99 |

The full chase reward, with every sign, is in [task_chase.md](../tasks/task_chase.md),
Reward. The full reward-term survey, including Hwangbo, Molchanov, Eschmann,
SimpleFlight and Swift, is in [reward_terms.md](reward_terms.md).

## Chase

The whole `chase` setup beside [Roncero v3](../../papers/2506.02849v3.md) and
[Němeček's thesis](../../papers/F3-BP-2026-Nemecek-Vaclav-bachelor_thesis.md) with his
code. What the runs scored against theirs:
[chase_pools_log.md](../experiment_logs/chase_pools_log.md).

| | ours | Roncero v3 | Němeček |
|---|---|---|---|
| **the game** | | | |
| simulator | JAX, our port of the MRS model | Isaac Lab | Flightsim (C++) with the MRS drone model |
| drone | X500, 2.0 kg | Crazyflie 2.1 brushless | X500, 2.0 kg |
| domain randomization | none | mass, inertia, thrust, moment, motor lag, drag, rate PID gains | not stated |
| arena | 20 × 20 × 10 m in `nemecek`, 32 × 32 × 16 m in `default` | 4 × 5 × 2 m | 20 × 20 × 10 m |
| a wall or the ceiling | `default`: ends the episode. `nemecek`: not there, and the evader pays 57 per second outside | ends the episode, κb = 10 to the one who left | not there, and the evader pays κb = 0.1 per step outside |
| the ground | ends the episode, `fail` = 30 | a face of the box, as above | ends the episode, κg = 10 |
| catch | the evader touches the net square under the pursuer | the centres within 0.1 m. The bonus shrinks above 2 m/s of closing speed | the centres within 0.5 m |
| episode | 10 s | 10 s | 12 s, 600 steps |
| policy rate | 100 Hz | 50 Hz | 50 Hz |
| action | CTBR | CTBR | world acceleration and heading rate |
| spawn | anywhere 2 m inside the box, any heading, tilt up to 0.2 rad, up to 0.5 m/s | any heading, tilt up to 5°, up to 0.1 m/s, 0.5 m apart and from the bounds | not stated |
| the value target at an episode end | a catch is terminal. After a failure the survivors bootstrap | a capture is terminal. Out of bounds and the timeout are truncations for every agent | every end is terminal, the timeout too: SB3 never gets `TimeLimit.truncated` |
| **training** | | | |
| method | AMSPBH, from Roncero v3 | AMSPBH, v3 (June 2026) | Roncero's [v1](https://arxiv.org/abs/2506.02849v1) (June 2025) for p_old and [v2](https://arxiv.org/abs/2506.02849v2) (September 2025) for the loop, before Hedge |
| generations | 0 to 13 | pretraining, then 1 to 13 | pretraining as generation 0, then 1 to 7 |
| steps per phase | 49.8M at 100 Hz: half of Roncero's flight time | 50M at 50 Hz | 30M at 50 Hz |
| parallel envs | 4096 | 6144 | 100 |
| who learns first | the evader | not stated: each side trains against generation k − 1 | the evader |
| pursuer g trains against evader g | no | no | yes |
| opponent mix | Hedge, γh = 0.7, ηh = 2: 70 % split evenly, 30 % mostly to the opponents that beat the learner | the same as ours | p_old = 0.75: the newest opponent gets 25 %, and the older ones share 75 % evenly, whatever their results |
| win rates for Hedge | 600 episodes, mean actions, the fixed eval seeds | 600 episodes under training conditions | — |
| opponent picked | once per phase, for each block of 64 envs | not stated | at each new episode |
| a frozen opponent flies | its mean action | not stated | its mean action |
| pursuer pool seed | FRPN, G = 6.677, W = 0.337, an acceleration | FRPN, G = 5.0, W = 0.20, a velocity command | FRPN. Thesis Table 8.1: G = 6.677, W = 0.337. His pool code: G = 0.665, W = 0.321 |
| evader pool seeds | `paths`, one member that draws a hover, a circle or a figure-eight for each episode, 8 × Roncero's size and √8 × his speed | hover, circle and figure-eight, radius 0.6–1.0 m, 1–2 m/s | 200 recorded trajectories, and APF |
| network | 2 × 256, tanh, one std per action | actor and critic [512, 256, 256, 128], ELU | actor and critic 3 × 256, ReLU |
| critic input | the whole scene and the opponent's action | the actor's observation | the actor's observation |
| PPO epochs × minibatches | 4 × 32 | 5 × 8 | 10 × 5 |
| learning rate | 2e-4, KL-adaptive | 2e-4, KL-adaptive, target 0.016 | 3e-4, fixed |
| entropy / clip / value loss | 0.001 / 0.2 / 0.5 | 0.001 / 0.2 / 1.0 | 0.001 / 0.1 / 1.0 |
| γ | 0.99 | not stated | 0.99 |
| **evaluation** | | | |
| final eval | random spawns on `EVAL_SEED`, 1024 episodes, one seed per run | fixed starts at (0, ±1.5, 1) m, 1024 episodes × 3 seeds, no randomization | every generation against every generation, Fig. 8.12 |
| a pursuer wins on | a catch, or a failure of the evader. `fail_both` is half a win to each side | a capture, or the evader out of bounds | a catch |

The other PPO settings are ours, not the paper's: rollout 64, GAE 0.95, grad clip 0.5,
reward normalization and raw observations.

## Chase beside Gavin & Bronz 2026

Every Gavin value here is from [Gavin & Bronz 2026 Nv1](../../papers/2607.05939v1.md),
unless marked 1v1 ([Gavin et al. 2026](../../papers/2603.16279v1.md)). These tables moved
here from the chase spec on 2026-10-25.

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

Gavin's value network gets the same input as our critic, without `ω`, and normalized.

The net:

| | ours | Gavin |
|---|---|---|
| shape | square standing in the body yz plane, face along body x | rigid circular disc, aligned with the body frame |
| `net_side` | 1.0 m | `R`, no number published |
| `net_offset`, the rope | 0.4 m | "mounted below each pursuer" |
| hangs from / to | 0.4 m / 1.4 m below the drone | no number |
| `c_net`, the middle | 0.9 m below the drone | "the centre of the catching net" |
| thickness | **zero** — it is a plane | "a capture distance", no number |

The net on another pursuer counting as `collPP` is ours. Gavin's is body to body only.

Gavin Nv1, the pursuer. Each `r` below is a positive quantity, and the signs are in the
sum:

```
rP = + rcatch                           λcatch  = 10.0   one-off
     - rdist   = λdist·‖p_e - c_net‖    λdist   = 0.001
     - rstep   = λstep                  λstep   = 0.04   the evader's survival pay
     - rcollPP = λcollPP·1(hit pursuer) λcollPP = 10.0   ends the episode
     - rcollPE = λcollPE·1(hit evader)  λcollPE = 0.1    soft, does not end
     - rfail   = λfail·1(crash or out)  λfail   = 30.0   one-off
     - rcmd    = λcmd·‖aω‖              λcmd    = 2e-4
```

Our per-step weights are Gavin's divided by `policy_dt`:

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

The episode ends:

| event | ours | Gavin |
|---|---|---|
| a net catches the evader | ends it, terminal | `λcatch` |
| `fail` | ends it, the survivors bootstrap | `λfail` |
| `collPP` | ends it, the survivors bootstrap | `λcollPP`, bodies only |
| `collPE` | does not end it | the same |
| the step limit, 1000 | ends it | the same |
