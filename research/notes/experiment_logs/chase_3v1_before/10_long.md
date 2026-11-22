# 10_long

**Question:** Do three pursuers catch more with 10 times the steps?
**Differs from** `09_critic_input`, the two runs with `normalize_obs=false` (`2d0b98`, `6fce56`): 5e9 steps.
**Runs:** MetaCentrum, 2 runs, 2026-11-04, commit `2d15eb5`: `76ea05` (seed 0), `4e80e4` (seed 1). They took 1 h 58 min and 1 h 31 min.

## Results

Eval of the last checkpoint, 1024 episodes. "Training" is the mean of the last 100 updates.

| run | steps | seed | `caught_e` eval | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | time to catch | `distance` | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `2d0b98` | 5e8 | 0 | 0.183 | 0.097 | 0.009 | 0.015 | 0.70 | 6.1 s | 5.2 m | 0.173 | 0.04–0.05 | 0.60 |
| `76ea05` | 5e9 | 0 | 0.184 | 0.028 | 0.002 | 0.001 | 0.79 | 5.9 s | 4.4 m | 0.188 | 0.005–0.008 | 0.63 |
| `6fce56` | 5e8 | 1 | 0.242 | 0.032 | 0.011 | 0.001 | 0.71 | 6.0 s | 4.5 m | 0.233 | 0.03 | 0.66 |
| `4e80e4` | 5e9 | 1 | 0.265 | 0.045 | 0.007 | 0.018 | 0.67 | 6.3 s | 4.3 m | 0.258 | 0.005–0.007 | 0.66 |

- The eval catch moves by 0.00 and 0.02. The noise on one `caught_e` is about ±0.013.
- The std ends at 0.005–0.008, against 0.03–0.05.

The 5e9 runs over training, as the mean of each 5e8 steps up to that mark:

| steps | 5e8 | 1e9 | 2e9 | 3e9 | 4e9 | 5e9 |
|---|---|---|---|---|---|---|
| `caught_e`, s0 | 0.124 | 0.153 | 0.173 | 0.180 | 0.183 | 0.186 |
| `caught_e`, s1 | 0.209 | 0.251 | 0.248 | 0.251 | 0.250 | 0.256 |
| mean std, s0 | 0.265 | 0.033 | 0.012 | 0.008 | 0.007 | 0.006 |
| mean std, s1 | 0.216 | 0.018 | 0.008 | 0.006 | 0.006 | 0.006 |

- The catch goes flat early: seed 1 stops at about 1e9 steps, and seed 0 almost stops at about 2.5e9.
- `lr_p` reaches its floor of 1e-5 at about 2e9 steps in seed 1, and at 3.5e9 in seed 0.

The catch by what the evader does, from a second flight of the same 1024 episodes. `hover` means a mean evader speed under 0.5 m/s, 333 episodes. The wall rows use the evader's closest approach to a face of the box. FRPN flew the pursuers on the same commit.

| catch rate | `2d0b98` | `76ea05` | `6fce56` | `4e80e4` | FRPN |
|---|---|---|---|---|---|
| all | 0.183 | 0.184 | 0.242 | 0.265 | 0.510 |
| `hover` | 0.18 | 0.10 | 0.25 | 0.23 | 0.59 |
| moving | 0.18 | 0.22 | 0.24 | 0.28 | 0.47 |
| moving, under 3 m from a face | 0.13 | 0.21 | 0.21 | 0.24 | 0.32 |
| moving, over 5 m from a face | 0.25 | 0.26 | 0.31 | 0.31 | 0.55 |

- The pursuers catch a `hover` no more often than a moving evader. FRPN catches 0.59 of the `hover` paths.
- The catch is only a little lower near a face of the box.
- FRPN catches 0.510 here.
