# 11_gamma

**Question:** Do three pursuers catch more with a reward horizon of 2 s, in place of 1 s?
**Differs from** `09_critic_input`, the two runs with `normalize_obs=false` (`2d0b98`, `6fce56`): `gamma=0.995`.
**Runs:** MetaCentrum, 2 runs, 2026-11-04, commit `11324cb`: `714794` (seed 0), `4180cc` (seed 1).

- These runs have the adaptive lr. With the constant lr, `03_breadth` of `chase_3v1` gets another result for `gamma=0.995`.

## Results

Eval of the last checkpoint, 1024 episodes. "Training" is the mean of the last 100 updates.

| run | gamma | seed | `caught_e` eval | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | time to catch | `distance` | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `2d0b98` | 0.99 | 0 | 0.183 | 0.097 | 0.009 | 0.015 | 0.70 | 6.1 s | 5.2 m | 0.173 | 0.04–0.05 | 0.60 |
| `714794` | 0.995 | 0 | 0.157 | 0.053 | 0.006 | 0.020 | 0.76 | 5.9 s | 4.7 m | 0.172 | 0.02–0.05 | 0.57 |
| `6fce56` | 0.99 | 1 | 0.242 | 0.032 | 0.011 | 0.001 | 0.71 | 6.0 s | 4.5 m | 0.233 | 0.03 | 0.66 |
| `4180cc` | 0.995 | 1 | 0.176 | 0.080 | 0.027 | 0.027 | 0.69 | 5.7 s | 5.0 m | 0.174 | 0.03–0.04 | 0.57 |

- The eval catch is lower in both seeds, by 0.026 and 0.066. The noise on one `caught_e` is about ±0.012.
- Both seeds end at the same training catch, 0.17.
- The critic's explained variance is 0.57 in both, against 0.60 and 0.66.

The same 1024 episodes, flown again. "Closest net" is the nearest that any net comes to the evader in an episode.

| | `2d0b98` | `714794` | `6fce56` | `4180cc` |
|---|---|---|---|---|
| catch, `hover` | 0.18 | 0.14 | 0.25 | 0.20 |
| catch, moving | 0.18 | 0.16 | 0.24 | 0.16 |
| catch, moving, under 3 m from a face | 0.13 | 0.15 | 0.21 | 0.16 |
| catch, moving, over 5 m from a face | 0.25 | 0.25 | 0.31 | 0.21 |
| closest net under 2 m | 0.74 | 0.76 | 0.76 | 0.79 |
| closest net under 1 m | 0.37 | 0.35 | 0.45 | 0.39 |
| catch, of the episodes under 1 m | 0.49 | 0.45 | 0.54 | 0.46 |
| mean net distance after 4 s | 3.9 m | 3.4 m | 2.9 m | 3.4 m |

- A net comes under 2 m in 0.74–0.79 of the episodes, and under 1 m in 0.35–0.45.
- About half of the episodes under 1 m end in a catch: 0.45–0.54.
