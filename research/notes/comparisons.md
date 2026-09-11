# Comparisons

## Observation layout: ours vs. the two references

Sources: [a_to_b.py](../../swarm/envs/a_to_b.py) `get_obs`,
[Huang et al. 2024](../papers/2309.13285v2.md) §III-B (the `quad-swarm-rl` paper),
[Gavin et al. 2026](../papers/2603.16279v1.md) §II-B (JAX interception, CTBR).

| | ours `a_to_b` | quad-swarm | Gavin 2026 |
|---|---|---|---|
| where the goal / opponent is | `(goal − x)` body, 3 | `(x − goal)` world, 3 | `(p_o − p_i)` world, 3 |
| how it moves relative to me | `−v` body, 3 | — | `(v_o − v_i)` world, 3 |
| my velocity | `v` body, 3 | `v` world, 3 | `v`, 3 |
| which way I point | `R`, 9 | `R`, 9 | `R`, 9 |
| how fast I turn | `ω` body, 3 | `ω` body, 3 | none |
| walls and ground | height `z`, 1 | altitude `h`, 1 | distance to each wall + ground, 6 |
| neighbours | `(K, 7)`, empty so far | `(Δp, Δv)` × K | one opponent only |
| obstacles | — | 9 SDF values | — |
| action | CTBR | 4 rotor thrusts | CTBR |

### Normalization

| | observation | reward |
|---|---|---|
| ours | none (removed) | divide by running spread of the return |
| quad-swarm | none — raw SI units, `--normalize_input=False` | hand-tuned weights, `--reward_clip=10` |
| Gavin 2026 | divide by two hand-picked constants: max view range, max velocity | not stated |

Nobody uses min-max. Two of three do nothing at all, and get away with it because a
10 m room bounds every quantity to within one order of magnitude.
