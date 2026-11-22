# 06_beta

**Question:** Do three pursuers catch more when each action comes from a Beta, which has no clip?
**Differs from** `03_cost30_slot0`: every policy draws its actions from a Beta, as in Chou et al. 2017. The network gives α and β for each action, each a softplus plus 1. The action is 2x − 1 with x ~ Beta(α, β), so it lies in [−1, 1] with no clip. A new actor starts at α = β ≈ 1.69, a std of about 0.48. Eval flies the mean of the Beta.
**Runs:** MetaCentrum, 8 runs, 2026-11-03, commit `c90447f`: `499a41`, `5d3d2a` (no block), `dcae69`, `05297b` (attitude), `9766d3`, `5fe3b4` (rpm), `f62c2b`, `8cae56` (both), seeds 0 and 1 of each. After these runs the code went back to the clipped Gaussian. So these runs load only on `c90447f`.

## Results

Eval with `--seat evader paths`, 1024 episodes, the last checkpoint. Each cell is the lowest and the highest value of the 8 runs of a batch. The batches are `04_paths_sense` of `chase_1v1`, `03_cost30_slot0`, `04_ent0` and this one. "The last 5 %" is the mean of the last 5 % of the updates.

| | 1v1, 2026-10-28 | 3v1, `ent_coef` 0.001 | 3v1, `ent_coef` 0 | 3v1 Beta, `ent_coef` 0.001 |
|---|---|---|---|---|
| the pursuer's std per action, at the end | 0.15–0.33 | 18–87 | 5.9–14 | 0.12–0.28 |
| mean action at the edge, eval | 1–25 % | 87–99 % | 75–95 % | 0 % |
| the adaptive lr, the last 5 % | 0.0004–0.0006 | 0.01 | 0.002–0.01 | 0.0003–0.0006 |
| `caught_e`, training | 0.88–1.00 | 0.03–0.05 | 0.04–0.05 | 0.02–0.04 |
| `caught_e`, eval | 0.91–1.00 | 0.004–0.030 | 0.004–0.035 | 0.020–0.040 |
| the critic's explained variance, the last 5 % | 0.87–0.96 | 0.31–0.53 | 0.38–0.53 | 0.54–0.73 |

- The std of the Beta is its mean over the steps of the 8 saved eval episodes of each run. For the Beta, "at the edge" means a mean action of 0.9 or more in size, because its mean cannot reach ±1.
- The std ends at 0.12–0.28, as with one pursuer, and the adaptive lr stays low.
- The catch does not change: 0.020–0.040 in the eval.

The eval outcomes of the three batches with three pursuers:

| | Gaussian, `ent_coef` 0.001 | Gaussian, `ent_coef` 0 | Beta, `ent_coef` 0.001 |
|---|---|---|---|
| `caught_e` | 0.004–0.030 | 0.004–0.035 | 0.020–0.040 |
| `coll_pp` | 0.32–0.76 | 0.18–0.59 | 0.07–0.17 |
| `crash_p` | 0.22–0.65 | 0.33–0.80 | 0.01–0.07 |
| `OOB_p` | 0.00–0.02 | 0.00–0.03 | 0.09–0.18 |
| `time_limit` | 0.00–0.04 | 0.00–0.15 | 0.57–0.75 |

- With the Beta, 0.57–0.75 of the episodes run to the time limit. The pursuers stay alive.

Other numbers of these runs, from the 64 saved eval episodes:

| what | value |
|---|---|
| the nearest net to the evader, for all 10 s | 8–11 m |
| the other two pursuers to the evader | 10–20 m |
| the speed of the pursuers, and of the evader | about 3.7 m/s, 3.1–3.6 m/s |
| the eval return of a pursuer | −47 to −50. FRPN gets −13 in `05_frpn_noise` |
| `caught_e` in training | rises to about 0.03 in the first 20–30 % of the run, then stays flat |

- The pursuers hold back. One pursuer of `04_paths_sense` flies at 5.5–8.9 m/s and closes from 14 m to 2.5 m in 2 s.
