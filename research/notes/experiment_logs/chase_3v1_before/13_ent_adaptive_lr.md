# 13_ent_adaptive_lr

**Question:** Does a larger entropy bonus hold the std up, so that the policy keeps moving?
**Differs from** `09_critic_input`, the two runs with `normalize_obs=false` (`2d0b98`, `6fce56`): `ent_coef=0.003`.
**Runs:** MetaCentrum, 2 runs, 2026-11-05, commit `11324cb`: `2cda6a` (seed 0), `503bac` (seed 1).

- Why: in `12_freeze` the pull down on the std is about 0.0015, and the entropy weight of 0.001 is smaller.

## Results

Eval of the last checkpoint, 1024 episodes. "Training" is the mean of the last 100 updates. The std is its value at the end for each action.

| run | `ent_coef` | seed | `caught_e` eval | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | time to catch | `distance` | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `2d0b98` | 0.001 | 0 | 0.183 | 0.097 | 0.009 | 0.015 | 0.70 | 6.1 s | 5.2 m | 0.173 | 0.04–0.05 | 0.60 |
| `2cda6a` | 0.003 | 0 | 0.006 | 0.282 | 0.479 | 0.000 | 0.23 | 2.1 s | 9.0 m | 0.040 | 328–18010 | 0.64 |
| `6fce56` | 0.001 | 1 | 0.242 | 0.032 | 0.011 | 0.001 | 0.71 | 6.0 s | 4.5 m | 0.233 | 0.03 | 0.66 |
| `503bac` | 0.003 | 1 | 0.027 | 0.228 | 0.281 | 0.005 | 0.46 | 4.1 s | 8.0 m | 0.047 | 124–679 | 0.62 |

- The std blows up in both seeds: it ends at 124–18010.
- The eval catch falls to 0.006 and 0.027, against 0.183 and 0.242.

The mean std and the training catch over the updates:

| update | 100 | 190 | 400 | 600 | 1000 | 1500 | 1890 |
|---|---|---|---|---|---|---|---|
| std, `2cda6a` | 1.35 | 1.78 | 3.2 | 5.9 | 31 | 213 | 1178 |
| std, `503bac` | 1.38 | 1.96 | 3.0 | 3.3 | 5.2 | 41 | 198 |
| `caught_e`, `2cda6a` | 0.05 | 0.06 | 0.09 | 0.14 | 0.05 | 0.05 | 0.03 |
| `caught_e`, `503bac` | 0.05 | 0.07 | 0.11 | 0.21 | 0.14 | 0.04 | 0.05 |

- The std grows from the first updates.
- Up to update 600 the catch keeps pace, at 0.14 and 0.21. Then it falls to 0.03–0.05.

The adaptive lr rises to its cap of 0.01 at about update 1000 and 1300. With the constant lr, `18_ent_const_lr` gets another result for the same `ent_coef`.
