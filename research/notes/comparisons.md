# Comparisons

## Observation layout: ours vs. the references

Sources: [envs/__init__.py](../../swarm/envs/__init__.py) `Obs`,
[core.py](../../swarm/envs/core.py) `own_obs` and `others`,
[chase.py](../../swarm/envs/chase.py) `get_obs`,
[Huang et al. 2024](../papers/2309.13285v2.md) §III-B plus
`quad-swarm-rl/gym_art/quadrotor_multi/quadrotor_multi.py:211-230`,
[Gavin et al. 2026 1v1](../papers/2603.16279v1.md) §III-B-1,
[Gavin & Bronz 2026 Nv1](../papers/2607.05939v1.md) §III-B.

`R` is body → world. `Rᵀu` is the world vector `u` written in the body frame.

| | ours (`Obs`) | quad-swarm | Gavin 1v1 | Gavin Nv1 |
|---|---|---|---|---|
| my velocity | `Rᵀv` body, 3 | `v` world, 3 | `v` world, 3 | `Rᵀv` body, 3 |
| which way I point | `R`, 9 | `R`, 9 | `R`, 9 | `R`, 9 |
| how fast I turn | `ω` body, 3 | `ω` body, 3 | none | none |
| walls and ground | height `z` + 4 wall distances in the level body frame, 5 | altitude `h`, 1 | distance to each wall + ground, 6 | `z` + `M` horizontal rays |
| where the goal is | `Rᵀ(g − x)` body, 3, only in a task with a goal | `x − g` world, 3 | — | — |
| other drones | `Rᵀ(Δp, Δv)` body + a same-role flag, `(K, 7)` | `(Δp, Δv)` world, `(K, 6)` | the opponent, `(p_o − p_i, v_o − v_i)` world, 6 | every other agent, `Rᵀ(Δp, Δv)` body, 6 each |
| how many other drones | `K = min(n_neighbors, N − 1)`, `n_neighbors = 6` | `K ≤ N − 1`, `--quads_neighbor_visible_num` | 1 | every agent |
| sort key | distance `d` | `d + (d̂ · Δv)` — distance plus closing rate | — | not stated |
| obstacles | — | 9 SDF values | — | — |
| action | CTBR | 4 rotor thrusts | CTBR | CTBR |

We match Gavin **Nv1** almost exactly: body-frame self velocity, body-frame
relative position and velocity for every other agent. The 1v1 paper puts the
opponent in the **world** frame; the Nv1 paper, four months later, moved it to the
body frame. Our extras over Nv1 are `ω` and the same-role flag. For the walls we use
4 rays, where Nv1 uses `M`.

### Our own blocks, exactly

`Obs(own, others, others_mask, target, scene)`, per drone:

| block | width | contents | `chase` |
|---|---|---|---|
| `own` | 20 | `Rᵀv` (3), `vec(R)` (9), `ω` (3), `z` (1), 4 wall distances | the same |
| `others` | `(K, 7)` | `Rᵀ(x_j − x_i)`, `Rᵀ(v_j − v_i)`, the same-role flag | `K = 1`: the opponent, flag 0 |
| `others_mask` | `K` | 1 if the seat is filled | 1 |
| `target` | 0 or 6 | `Rᵀ(g − x)`, `Rᵀ(v_g − v)` | width 0 |
| `scene` | `18N` and more | the critic's input: every drone's exact `x`, `v`, `R`, `ω` | 36, plus the opponent's action |

Two notes on this table:

- **The actor reads `own`, `others`, `others_mask` and `target`.** `flat` zeroes an
  empty seat, so the width is always `20 + 8K + t`: 28 in chase.
- **The `target` block already holds a relative velocity.** `Rᵀ(−v)` is
  `Rᵀ(v_g − v)` with `v_g = 0`. A moving target changes the value, not the shape.

### Normalization

| | observation | reward |
|---|---|---|
| ours | none | divide by a running spread of the return, one per drone column |
| quad-swarm | none — raw SI units, `--normalize_input=False` | hand-tuned weights, `--reward_clip=10` |
| Gavin 1v1 | divide by two hand-picked constants: max view range `k_p`, max velocity `k_v` | not stated |
| Gavin Nv1 | same | not stated |

## Reward layout

Per-step terms, and one-off terms marked. Our weights are per second, and the code
multiplies them by `policy_dt = 0.01`. Gavin's are per step at 100 Hz, so our chase
weights equal his.

| term | ours `chase` | ours `circle_swap` | quad-swarm | Gavin 1v1 | Gavin Nv1 |
|---|---|---|---|---|---|
| distance | `0.1·‖x_e − c_net‖`, pursuer | `1.0·‖g−x‖` | `αdist·‖p‖` | `λdist = 0.001` | `λdist = 0.001` |
| capture bonus | `catch = 10.0`, one-off | — | — | `λcatch = 10.0`, one-off | `λcatch = 10.0`, one-off |
| body rate | — | `0.1·‖ω‖` | `spin = 0.1` | — | — |
| command | `cmd = 0.02·‖ω_cmd‖` | — | `effort = 0.05` | `λcmd = 2e-4·‖aω‖` | `λcmd = 2e-4·‖aω‖` |
| near miss | — | `10.0·Σ max(1 − d/1.5, 0)` | `αrclose`, same shape, 10.0 | — | — |
| contact | `collPP = 10.0` one-off, ends it; `collPE = 10.0`, soft | death, `25.0·N` one-off | elastic bounce, `1.0` | `λcoll = 0.1`, soft, no death | `λcollPP = 10.0`, `λcollPE = 0.1` |
| upright | — | `−1.0·R₃₃` | `orient = 1.0·R₃₃` | — | — |
| crash / out of bounds | `fail = 30.0` one-off | `25.0·N` one-off | `crash = 1.0` per step | `λfail = 30.0` | `λfail = 30.0` |
| survival | `step = 4.0`: the evader earns it, the pursuer pays it | — | — | — | `λstep = 0.04`: the evader earns it, the pursuers pay it |
| wall buffer | `outside = 57`, evader only, outside the box, in `open` and `nemecek` | — | — | `λbnd = 1.0`, evader only | `λbnd`, evader only |
| discount | 0.99 | 0.998 | not published | 0.99 | 0.99 |

The full chase reward, with every sign, is in [task_chase.md](task_chase.md), Reward.
The full reward-term survey, including Hwangbo, Molchanov, Eschmann, SimpleFlight
and Swift, is at the end of [archive/task_a_to_b.md](archive/task_a_to_b.md).
