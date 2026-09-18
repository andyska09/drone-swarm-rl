# Comparisons

## Observation layout: ours vs. the references

Sources: [circle_swap.py](../../swarm/envs/circle_swap.py) `get_obs`,
[a_to_b.py](../../swarm/envs/a_to_b.py) `get_obs`,
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
| walls and ground | height `z`, 1 | altitude `h`, 1 | distance to each wall + ground, 6 | `z` + `M` horizontal rays |
| where the goal / opponent is | `Rᵀ(g − x)` body, 3 | `x − g` world, 3 | `p_o − p_i` world, 3 | `Rᵀ(p_j − p_i)` body, 3 |
| how it moves relative to me | `Rᵀ(v_g − v)` body, 3 | — | `v_o − v_i` world, 3 | `Rᵀ(v_j − v_i)` body, 3 |
| neighbours | `Rᵀ(Δp, Δv)` body, `(K, 6)` | `(Δp, Δv)` world, `(K, 6)` | one opponent only | same block as the opponent |
| how many neighbours | `K = N − 1`, every one | `K ≤ N − 1`, `--quads_neighbor_visible_num` | 1 | every agent |
| neighbour sort key | distance `d` | `d + (d̂ · Δv)` — distance plus closing rate | — | not stated |
| obstacles | — | 9 SDF values | — | — |
| action | CTBR | 4 rotor thrusts | CTBR | CTBR |

We match Gavin **Nv1** almost exactly: body-frame self velocity, body-frame
relative position and velocity for every other agent. The 1v1 paper puts the
opponent in the **world** frame; the Nv1 paper, one year later, moved it to the
body frame. Our two extras over Nv1 are `ω` and the fact that the goal block is
separate from the neighbour block. Our one gap is the walls: they get `M` rays,
we get height only.

### Our own blocks, exactly

`Obs(own, neighbors, neighbor_mask, target, target_mask)`, per drone:

| block | width | contents | `a_to_b` | `circle_swap` |
|---|---|---|---|---|
| `own` | 16 | `Rᵀv` (3), `vec(R)` (9), `ω` (3), `z` (1) | same | same |
| `neighbors` | `(K, F)` | `Rᵀ(x_j − x_i)`, `Rᵀ(v_j − v_i)` | `K = 0`, `F = 7`, all zero | `K = N−1`, `F = 6` |
| `neighbor_mask` | `K` | 1 if the row is real | all zero | all one |
| `target` | 6 | `Rᵀ(g − x)`, `Rᵀ(v_g − v)` | `v_g = 0` | `v_g = 0` |
| `target_mask` | 1 | 1 if the goal is seen | always 1 | always 1 |

Three notes on this table:

- **The `target` block already holds a relative velocity.** `Rᵀ(−v)` is
  `Rᵀ(v_g − v)` with `v_g = 0`. A moving target changes the value, not the shape.
- **`a_to_b` and `circle_swap` disagree on `F`** — `NEIGHBOR_FEATURES = 7` against
  `6`. `a_to_b`'s block is all zeros and all masked, so nothing reads it, but the
  7 is left over from a same-side flag that no task writes.

### Normalization

| | observation | reward |
|---|---|---|
| ours | none (removed) | divide by running spread of the return |
| quad-swarm | none — raw SI units, `--normalize_input=False` | hand-tuned weights, `--reward_clip=10` |
| Gavin 1v1 | divide by two hand-picked constants: max view range `k_p`, max velocity `k_v` | not stated |
| Gavin Nv1 | same | not stated |

## Reward layout

Per-step terms only; one-off terms are marked.

| term | ours `a_to_b` | ours `circle_swap` | quad-swarm | Gavin 1v1 | Gavin Nv1 |
|---|---|---|---|---|---|
| distance | `1.0·‖g−x‖` | `1.0·‖g−x‖` | `αdist·‖p‖` | `λdist = 0.001` | `λdist = 0.001` |
| capture bonus | — | — | — | `λcatch = 10.0`, one-off | `λcatch = 10.0`, one-off |
| body rate | `0.1·‖ω‖` | `0.1·‖ω‖` | `spin = 0.1` | — | — |
| command | — | — | `effort = 0.05` | `λcmd = 2e-4·‖aω‖` | `λcmd = 2e-4·‖aω‖` |
| near miss | — | `10.0·Σ max(1 − d/1.5, 0)` | `αrclose`, same shape, 10.0 | — | — |
| contact | death | death, `25.0·N` one-off | elastic bounce, `1.0` | `λcoll = 0.1`, soft, no death | `λcollPP = 10.0`, `λcollPE = 0.1` |
| upright | — | `−1.0·R₃₃` | `orient = 1.0·R₃₃` | — | — |
| crash / out of bounds | `10.0` one-off | `25.0·N` one-off | `crash = 1.0` per step | `λfail = 30.0` | `λfail = 30.0` |
| survival | — | — | — | — | `λstep = 0.04`, evader only |
| wall buffer | — | — | — | `λbnd = 1.0`, evader only | `λbnd`, evader only |
| discount | 0.998 | 0.998 | not published | 0.99 | 0.99 |

The full reward-term survey, including Hwangbo, Molchanov, Eschmann, SimpleFlight
and Swift, is in [parking_error.md](parking_error.md).
