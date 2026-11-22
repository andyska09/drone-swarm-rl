# 18_ent_const_lr

**Question:** Do three pursuers catch more with `ent_coef=0.003`, when the lr is constant?
**Differs from** `17_const_lr_netpass`, run `be848f`: `ent_coef=0.003`. Seeds 0 and 1.
**Runs:** MetaCentrum, 2 runs, 2026-11-11, commit `fba6aec`: `730968` (seed 0), `9eb432` (seed 1). `be848f` is seed 0.

## Results

Eval with 1024 episodes and mean actions. The flights are 4096 episodes, of the last checkpoint and of update 850 of `9eb432`.

Training, the mean of each 200 updates:

| updates | 1–200 | 201–400 | 401–600 | 601–800 | 801–1000 | 1001–1200 | 1201–1400 | 1401–1600 | 1601–1800 | 1801–1907 |
|---|---|---|---|---|---|---|---|---|---|---|
| catch, `be848f` | 0.24 | 0.47 | 0.47 | 0.45 | 0.47 | 0.55 | 0.58 | 0.56 | 0.51 | 0.50 |
| catch, `9eb432` | 0.36 | 0.69 | 0.85 | 0.90 | 0.90 | 0.83 | 0.80 | 0.82 | 0.83 | 0.83 |
| `coll_pp`, `be848f` | 0.36 | 0.30 | 0.41 | 0.42 | 0.38 | 0.28 | 0.28 | 0.33 | 0.35 | 0.30 |
| `coll_pp`, `9eb432` | 0.33 | 0.26 | 0.13 | 0.08 | 0.08 | 0.16 | 0.19 | 0.16 | 0.15 | 0.15 |
| std, `be848f` | 0.669 | 0.256 | 0.136 | 0.094 | 0.078 | 0.066 | 0.059 | 0.054 | 0.051 | 0.050 |
| std, `9eb432` | 0.862 | 0.549 | 0.363 | 0.263 | 0.210 | 0.189 | 0.182 | 0.176 | 0.166 | 0.160 |
| `return_p`, `be848f` | 5.27 | 13.76 | 13.18 | 12.61 | 13.19 | 16.19 | 17.35 | 16.27 | 14.55 | 13.74 |
| `return_p`, `9eb432` | 9.08 | 21.29 | 26.91 | 28.78 | 28.86 | 26.31 | 25.06 | 26.06 | 26.25 | 26.15 |
| time to catch, `9eb432` (s) | 4.79 | 3.76 | 3.08 | 2.75 | 2.45 | 2.19 | 2.08 | 2.14 | 2.17 | 2.12 |

Over updates 1001–1907, and the std of each action at the end (throttle, roll rate, pitch rate, yaw rate):

| run | `approx_kl_p` | `clipfrac_p` | explained variance | std of each action |
|---|---|---|---|---|
| `be848f` | 0.022 | 0.28 | 0.77 | 0.055, 0.046, 0.044, 0.059 |
| `9eb432` | 0.014 | 0.20 | 0.81 | 0.134, 0.111, 0.112, 0.381 |

- The std does not blow up: it ends at 0.11–0.13 for the throttle and the roll and pitch rates, and at 0.38 for the yaw rate.
- The KL and the clip fraction are lower with 0.003: 0.014 and 0.20, against 0.022 and 0.28.

The first updates, beside the two runs of `13_ent_adaptive_lr`. The lr is the median over updates 1–100, then over 101–200:

| update | 10 | 50 | 100 | 200 | 400 | lr, 1–100 | lr, 101–200 |
|---|---|---|---|---|---|---|---|
| std, `9eb432` | 0.996 | 0.969 | 0.868 | 0.687 | 0.437 | 5.0e-4 | 5.0e-4 |
| std, `503bac` | 1.017 | 1.127 | 1.391 | 2.014 | 3.043 | 3.0e-3 | 1.9e-3 |
| std, `2cda6a` | 1.016 | 1.078 | 1.357 | 1.841 | 3.201 | 3.0e-3 | 1.6e-3 |

- With the adaptive lr the std is 1.36–1.39 at update 100, and the lr of the first 100 updates is 3.0e-3, 6 times the constant lr.

Eval of `9eb432` at its checkpoints, 1024 episodes:

| update | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 850 | 900 | 1000 | 1100 | 1200 | 1300 | 1500 | 1700 | 1907 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catch | 0.554 | 0.766 | 0.778 | 0.832 | 0.875 | 0.907 | 0.910 | 0.914 | 0.880 | 0.869 | 0.844 | 0.785 | 0.810 | 0.786 | 0.818 | 0.850 |
| `coll_pp` | 0.392 | 0.161 | 0.192 | 0.149 | 0.108 | 0.070 | 0.070 | 0.070 | 0.109 | 0.115 | 0.145 | 0.206 | 0.181 | 0.192 | 0.165 | 0.125 |
| time to catch (s) | 3.99 | 3.90 | 3.16 | 2.92 | 2.84 | 2.79 | 2.65 | 2.56 | 2.39 | 2.26 | 2.19 | 2.12 | 2.07 | 2.07 | 2.05 | 2.16 |

- The catch falls after its best checkpoint: 0.914 at update 850, 0.785 at 1200 and 0.850 at 1907.
- The time to catch keeps falling, from 2.56 s at update 850 to 2.07 s at 1300, while `coll_pp` rises from 0.070 to 0.181–0.206.

The outcomes, 4096 episodes, with the 95 % Wilson interval of the catch. A death is a `coll_pp`, an OOB or a crash:

| run | catch | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | deaths | `coll_pp` share of the deaths | median time to catch |
|---|---|---|---|---|---|---|---|---|
| `be848f` | 0.498 [0.482, 0.513] | 0.278 | 0.062 | 0.093 | 0.069 | 0.433 | 0.64 | 4.1 s |
| `9eb432`, last | 0.837 [0.825, 0.848] | 0.136 | 0.020 | 0.007 | 0.001 | 0.163 | 0.83 | 1.9 s |
| `9eb432`, update 850 | 0.910 [0.901, 0.919] | 0.074 | 0.006 | 0.010 | 0.000 | 0.090 | 0.82 | 2.4 s |
| FRPN | 0.524 [0.509, 0.539] | 0.322 | 0.012 | 0.142 | 0.000 | 0.476 | 0.68 | 1.5 s |

- `ent_coef=0.003` with the constant lr lifts the catch from 0.498 to 0.837. Update 850 catches 0.910, and FRPN 0.524.

The catch by the path of the evader, and by the start distance of the nearest net (the quarters of all episodes, with edges at 6.8, 9.4 and 12.3 m):

| | `be848f` | `9eb432`, last | `9eb432`, update 850 | FRPN |
|---|---|---|---|---|
| hover | 0.511 | 0.874 | 0.931 | 0.603 |
| circle | 0.501 | 0.854 | 0.925 | 0.559 |
| figure-eight | 0.481 | 0.784 | 0.876 | 0.413 |
| nearest quarter | 0.57 | 0.85 | 0.93 | 0.60 |
| second quarter | 0.52 | 0.83 | 0.92 | 0.53 |
| third quarter | 0.46 | 0.84 | 0.90 | 0.50 |
| farthest quarter | 0.44 | 0.83 | 0.88 | 0.46 |

- The catch does not depend on the start distance: 0.83–0.85 in each quarter. In `be848f` it falls from 0.57 to 0.44.
- The figure-eight is the hardest path: 0.784, against 0.874 for the hover.

The ends per second, among the episodes still running, in the bins 1–2, 2–3 and 3–4 s:

| | `be848f` | `9eb432`, last | `9eb432`, update 850 | FRPN |
|---|---|---|---|---|
| catch | 0.059 / 0.105 / 0.138 | 0.443 / 0.553 / 0.534 | 0.278 / 0.551 / 0.662 | 0.359 / 0.333 / 0.419 |
| `coll_pp` | 0.041 / 0.073 / 0.070 | 0.065 / 0.106 / 0.101 | 0.024 / 0.037 / 0.051 | 0.213 / 0.219 / 0.304 |

The behaviour:

| | `be848f` | `9eb432`, last | `9eb432`, update 850 | FRPN |
|---|---|---|---|---|
| nets at 2 s, median (m) | 2.7 / 4.4 / 6.6 | 1.4 / 2.6 / 4.7 | 1.5 / 2.8 / 4.8 | 2.2 / 4.3 / 7.0 |
| nets at 4 s, median (m) | 2.1 / 3.6 / 5.6 | 1.3 / 2.7 / 4.5 | 1.1 / 1.9 / 3.4 | 2.4 / 3.9 / 6.0 |
| closing at 1–2 / 2–3 / 3–5 m (m/s) | 0.44 / 0.45 / 0.69 | 0.80 / 1.46 / 3.10 | 0.81 / 1.46 / 2.95 | 4.79 / 5.82 / 6.63 |
| closing at 5–8 / 8–12 / 12–40 m (m/s) | 1.40 / 4.20 / 5.66 | 6.05 / 7.62 / 6.69 | 5.75 / 7.39 / 6.66 | 7.70 / 8.19 / 6.21 |
| net 2–5 m away: toward / around the evader (m/s) | 0.71 / 2.81 | 2.63 / 2.70 | 2.47 / 2.00 | 6.60 / 2.78 |
| pursuer speed, median / top 10 % (m/s) | 5.26 / 10.70 | 6.84 / 13.38 | 5.85 / 12.06 | 9.34 / 12.70 |
| height over the evader, net within 3 m, q10 / q50 / q90 (m) | −0.67 / 0.60 / 1.69 | −0.54 / 0.64 / 1.51 | −0.37 / 0.63 / 1.54 | −0.21 / 0.60 / 1.39 |
| no catch: closest net, median; share within 1 m | 1.12 m; 0.41 | 0.91 m; 0.59 | 0.97 m; 0.52 | 0.90 m; 0.62 |
| `coll_pp`: nearest net to the evader, median; share under 2 m | 2.10 m; 0.46 | 1.26 m; 0.80 | 1.26 m; 0.80 | 1.32 m; 0.74 |
| `coll_pp`: ranks 1+2 / 2+3 / 1+3 | 0.68 / 0.26 / 0.06 | 0.75 / 0.18 / 0.07 | 0.77 / 0.17 / 0.06 | 0.85 / 0.11 / 0.04 |
| `coll_pp`: share with the two bodies under 0.8 m apart | 0.44 | 0.30 | 0.28 | 0.66 |
| `coll_pp`: the pair's distance 0.5 s before, median | 2.94 m | 3.83 m | 2.63 m | 6.01 m |
| `coll_pp`: speed of the pair at the hit, median (m/s) | 5.10 | 5.92 | 4.73 | 11.92 |
| angle of the two nearest at the evader, median | 89° | 92° | 93° | 78° |

- The pursuers no longer hold on a ring: with a net 2–5 m away they fly 2.63 m/s toward the evader, against 0.71.
- At a `coll_pp` the nearest net is a median 1.26 m from the evader, and the two bodies are under 0.8 m apart in 0.30 of the hits.

The payoff of closing in with the own net 2–5 m from the evader, as in `16_probes`:

| run | G, < 0 | 0–0.5 | 0.5–1 | 1–2 | 2–4 | > 4 m/s | own catch within 1 s, < 0 / > 4 | own `coll_pp` within 1 s, < 0 / > 4 |
|---|---|---|---|---|---|---|---|---|
| `be848f` | 2.21 | 2.92 | 3.01 | 3.02 | 3.10 | 3.05 | 0.011 / 0.089 | 0.045 / 0.067 |
| `9eb432`, last | 9.46 | 10.72 | 11.07 | 11.08 | 11.03 | 11.81 | 0.061 / 0.268 | 0.051 / 0.081 |
| `9eb432`, update 850 | 11.20 | 12.17 | 12.51 | 12.19 | 11.46 | 11.08 | 0.048 / 0.218 | 0.025 / 0.032 |

Seed 0 beside seed 1, and beside `be848f`. The best checkpoint of `730968`, of the 18 that were checked, is update 1050:

| | `be848f` (0.001, seed 0) | `730968` (0.003, seed 0) | `9eb432` (0.003, seed 1) |
|---|---|---|---|
| training catch, updates 801–1000 / 1001–1200 / 1801–1907 | 0.470 / 0.548 / 0.497 | 0.615 / 0.671 / 0.619 | 0.901 / 0.831 / 0.828 |
| training `coll_pp`, the same updates | 0.384 / 0.284 / 0.303 | 0.349 / 0.284 / 0.345 | 0.085 / 0.156 / 0.146 |
| std at the end, each action | 0.055, 0.046, 0.044, 0.059 | 0.157, 0.116, 0.120, 0.206 | 0.134, 0.111, 0.112, 0.381 |
| eval catch at the best checkpoint, 1024 episodes | 0.573 (1500) | 0.684 (1050) | 0.914 (850) |
| eval `coll_pp` at that checkpoint | 0.334 | 0.278 | 0.070 |
| eval catch at the last checkpoint, 1024 episodes | 0.487 | 0.628 | 0.850 |
| catch, 4096 episodes, last | 0.498 [0.482, 0.513] | 0.605 [0.590, 0.620] | 0.837 [0.825, 0.848] |
| `coll_pp` / `crash_p` / `OOB_p` / `time_limit`, last | 0.278 / 0.062 / 0.093 / 0.069 | 0.360 / 0.018 / 0.016 / 0.001 | 0.136 / 0.020 / 0.007 / 0.001 |
| catch, 4096 episodes, best checkpoint | — | 0.682 [0.668, 0.696] | 0.910 [0.901, 0.919] |
| `coll_pp`, 4096 episodes, best checkpoint | — | 0.275 | 0.074 |
| median time to catch, last | 4.1 s | 1.8 s | 1.9 s |
| hover / circle / figure-eight, last | 0.511 / 0.501 / 0.481 | 0.685 / 0.602 / 0.530 | 0.874 / 0.854 / 0.784 |
| net 2–5 m away: toward / around the evader (m/s), last | 0.71 / 2.81 | 1.81 / 3.08 | 2.63 / 2.70 |
| closing at 1–2 / 2–3 / 3–5 m (m/s), last | 0.44 / 0.45 / 0.69 | 0.80 / 1.13 / 2.06 | 0.80 / 1.46 / 3.10 |
| `coll_pp`: nearest net to the evader, median; share under 2 m | 2.10 m; 0.46 | 1.60 m; 0.66 | 1.26 m; 0.80 |
| `coll_pp`: speed of the pair at the hit, median (m/s) | 5.10 | 6.70 | 5.92 |
| against APF, 1024 episodes: catch, last / best checkpoint | 0.454 / — | 0.477 / 0.612 | 0.719 / 0.856 |

The eval catch of `730968` at its checkpoints, 1024 episodes:

| update | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 850 | 900 | 1000 | 1050 | 1100 | 1200 | 1300 | 1400 | 1500 | 1700 | 1907 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catch | 0.287 | 0.472 | 0.599 | 0.557 | 0.477 | 0.540 | 0.572 | 0.587 | 0.618 | 0.645 | 0.684 | 0.659 | 0.671 | 0.671 | 0.609 | 0.604 | 0.617 | 0.628 |
| `coll_pp` | 0.455 | 0.402 | 0.348 | 0.393 | 0.439 | 0.415 | 0.397 | 0.383 | 0.345 | 0.311 | 0.278 | 0.310 | 0.294 | 0.290 | 0.341 | 0.348 | 0.329 | 0.337 |

- The two seeds differ by 0.232 in the catch: 0.605 and 0.837. At their best checkpoints they catch 0.682 and 0.910.
- With the same seed 0, 0.003 catches 0.107 more than 0.001: 0.605, against 0.498.
- Seed 0 hits more: `coll_pp` is 0.360, against 0.136.
