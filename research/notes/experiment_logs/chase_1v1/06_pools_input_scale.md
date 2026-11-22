# 06_pools_input_scale

**Question:** Does the input scale lift the pursuer's win rate in a pool run in `default`?
**Differs from** `05_pools_sense`: no role senses a block, and `ent_coef=0.001`. Two runs scale the inputs of each network by a running mean and std, and two run with `normalize_obs=false`. Seeds 0 and 1.
**Runs:** MetaCentrum, 4 runs, 2026-11-04, commit `636ab7a`, the first commit with the input scale.

## Results

The columns of `02_pools`:

| run | input scale | seed | lower triangle | older evaders | diagonal | last pair |
|---|---|---|---|---|---|---|
| `8a62bf` | on | 0 | 0.61 | 0.67 | 0.27 | 0.16 |
| `94c6c1` | on | 1 | 0.67 | 0.72 | 0.32 | 0.39 |
| `5084e8` | off | 0 | 0.66 | 0.69 | 0.40 | 0.36 |
| `e2eb68` | off | 1 | 0.59 | 0.63 | 0.34 | 0.56 |

- The lower triangle is 0.64 with the scale and 0.62 without it. The two seeds of one setting differ by 0.06 and 0.07.
- The diagonal and the last pair are lower with the scale: 0.29 and 0.28, against 0.37 and 0.46.
- Without the scale, the lower triangle (0.66 and 0.59) is close to the 0.61 of `9e4242` in `02_pools`.

By setting, seeds 0 and 1. The last column is from the opponent mix of generation 13. The pursuer's std ends at 0.54–0.62 in all four runs.

| input scale | `pursuer_13` against paths, APF and the cascade | the evader's win rate against the older pursuers |
|---|---|---|
| on | 0.89–0.95 | 0.94, 0.89 |
| off | 0.94–0.97 | 0.80, 0.76 |

- A matrix plays a pursuer only against the evaders of its own run, and the scale changes both sides. With the scale, the evaders win more against the older pursuers.
