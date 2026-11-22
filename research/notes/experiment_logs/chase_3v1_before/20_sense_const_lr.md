# 20_sense_const_lr

**Question:** Do three pursuers catch more when each one senses the attitude or the motor speeds of the other drones, now that they learn to fly in?
**Differs from** `17_const_lr_netpass`, run `be848f`: the presets `paths_3v1_attitude`, `paths_3v1_rpm` and `paths_3v1_attitude_rpm`. Seeds 0 and 1. `ent_coef` is 0.001.
**Runs:** MetaCentrum, 6 runs, 2026-11-11, commit `fba6aec`. The base `be848f` has seed 0 only.

| run | the pursuer senses | seed |
|---|---|---|
| `be848f` | none | 0 |
| `b503f3` | attitude | 0 |
| `740128` | attitude | 1 |
| `873189` | rpm | 0 |
| `22b282` | rpm | 1 |
| `6e196e` | attitude, rpm | 0 |
| `1390aa` | attitude, rpm | 1 |

## Results

The training catch, the mean of each 200 updates, and the std at the end:

| updates | 1–200 | 201–400 | 401–600 | 601–800 | 801–1000 | 1001–1200 | 1201–1400 | 1401–1600 | 1601–1800 | 1801–1907 | std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `be848f` | 0.24 | 0.47 | 0.47 | 0.45 | 0.47 | 0.55 | 0.58 | 0.56 | 0.51 | 0.50 | 0.051 |
| `b503f3` | 0.18 | 0.34 | 0.34 | 0.37 | 0.39 | 0.37 | 0.37 | 0.36 | 0.37 | 0.42 | 0.064 |
| `740128` | 0.22 | 0.36 | 0.31 | 0.36 | 0.39 | 0.42 | 0.42 | 0.42 | 0.37 | 0.37 | 0.061 |
| `873189` | 0.24 | 0.41 | 0.49 | 0.57 | 0.58 | 0.58 | 0.61 | 0.57 | 0.55 | 0.55 | 0.049 |
| `22b282` | 0.17 | 0.37 | 0.44 | 0.53 | 0.56 | 0.56 | 0.54 | 0.49 | 0.50 | 0.50 | 0.052 |
| `6e196e` | 0.27 | 0.44 | 0.55 | 0.56 | 0.60 | 0.64 | 0.62 | 0.63 | 0.64 | 0.64 | 0.049 |
| `1390aa` | 0.30 | 0.54 | 0.58 | 0.62 | 0.62 | 0.60 | 0.64 | 0.62 | 0.61 | 0.60 | 0.046 |

The eval catch at the checkpoints, 1024 episodes each. The mean is over the seven checkpoints from 1000 to 1907:

| update | 200 | 400 | 600 | 800 | 900 | 1000 | 1100 | 1200 | 1300 | 1500 | 1700 | 1907 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `be848f` | 0.392 | 0.512 | 0.476 | 0.373 | 0.492 | 0.502 | 0.547 | 0.565 | 0.562 | 0.573 | 0.498 | 0.487 | 0.533 |
| `b503f3` | 0.238 | 0.353 | 0.386 | 0.385 | 0.371 | 0.389 | 0.361 | 0.399 | 0.383 | 0.367 | 0.385 | 0.461 | 0.392 |
| `740128` | 0.337 | 0.344 | 0.322 | 0.383 | 0.391 | 0.388 | 0.378 | 0.403 | 0.438 | 0.426 | 0.368 | 0.360 | 0.394 |
| `873189` | 0.368 | 0.463 | 0.546 | 0.600 | 0.581 | 0.572 | 0.604 | 0.606 | 0.636 | 0.530 | 0.541 | 0.522 | 0.573 |
| `22b282` | 0.298 | 0.377 | 0.446 | 0.577 | 0.597 | 0.563 | 0.543 | 0.530 | 0.549 | 0.467 | 0.472 | 0.542 | 0.524 |
| `6e196e` | 0.424 | 0.552 | 0.549 | 0.560 | 0.607 | 0.630 | 0.665 | 0.661 | 0.604 | 0.647 | 0.637 | 0.638 | 0.640 |
| `1390aa` | 0.500 | 0.522 | 0.630 | 0.624 | 0.604 | 0.604 | 0.591 | 0.588 | 0.626 | 0.589 | 0.593 | 0.550 | 0.592 |

- The eval catch falls after its best checkpoint in four of the six runs: `740128`, `873189`, `22b282` and `1390aa`.

The outcomes, 4096 episodes of the last checkpoint, with the 95 % Wilson interval of the catch:

| run | catch | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | deaths | `coll_pp` share of the deaths | median time to catch |
|---|---|---|---|---|---|---|---|---|
| `be848f` | 0.498 [0.482, 0.513] | 0.278 | 0.062 | 0.093 | 0.069 | 0.433 | 0.64 | 4.1 s |
| `b503f3` | 0.440 [0.425, 0.456] | 0.412 | 0.031 | 0.060 | 0.056 | 0.504 | 0.82 | 4.0 s |
| `740128` | 0.366 [0.351, 0.381] | 0.477 | 0.029 | 0.057 | 0.072 | 0.563 | 0.85 | 3.5 s |
| `873189` | 0.549 [0.533, 0.564] | 0.288 | 0.049 | 0.067 | 0.047 | 0.404 | 0.71 | 3.9 s |
| `22b282` | 0.548 [0.533, 0.564] | 0.314 | 0.018 | 0.034 | 0.085 | 0.366 | 0.86 | 4.0 s |
| `6e196e` | 0.642 [0.627, 0.656] | 0.290 | 0.025 | 0.021 | 0.023 | 0.335 | 0.87 | 3.4 s |
| `1390aa` | 0.580 [0.565, 0.595] | 0.334 | 0.031 | 0.053 | 0.002 | 0.418 | 0.80 | 2.2 s |

- With the attitude alone the pursuers catch less than the base seed: 0.440 and 0.366, against 0.498.
- With the rpm alone they catch 0.549 and 0.548. With both they catch 0.642 and 0.580.
- `coll_pp` is 0.71–0.87 of the deaths in all six runs.

The catch by the path of the evader, and the behaviour:

| | `be848f` | `b503f3` | `740128` | `873189` | `22b282` | `6e196e` | `1390aa` |
|---|---|---|---|---|---|---|---|
| hover | 0.511 | 0.435 | 0.368 | 0.544 | 0.529 | 0.688 | 0.607 |
| circle | 0.501 | 0.417 | 0.382 | 0.579 | 0.611 | 0.663 | 0.622 |
| figure-eight | 0.481 | 0.468 | 0.347 | 0.523 | 0.505 | 0.576 | 0.512 |
| nets at 4 s, median (m) | 2.1 / 3.6 / 5.6 | 2.0 / 3.2 / 4.6 | 2.6 / 4.0 / 5.6 | 2.1 / 3.8 / 6.1 | 2.3 / 3.9 / 6.1 | 1.5 / 2.6 / 4.0 | 1.6 / 2.7 / 4.4 |
| closing at 1–2 / 2–3 / 3–5 m (m/s) | 0.44 / 0.45 / 0.69 | 0.29 / 0.35 / 0.66 | 0.35 / 0.36 / 0.63 | 0.54 / 0.66 / 0.90 | 0.50 / 0.55 / 0.76 | 0.37 / 0.49 / 1.00 | 0.62 / 0.95 / 2.00 |
| closing at 5–8 / 8–12 m (m/s) | 1.40 / 4.20 | 2.00 / 5.66 | 1.85 / 5.95 | 1.75 / 4.78 | 1.49 / 3.96 | 3.87 / 7.37 | 4.58 / 6.98 |
| net 2–5 m away: toward / around the evader (m/s) | 0.71 / 2.81 | 0.54 / 2.90 | 0.64 / 3.25 | 0.88 / 2.81 | 0.75 / 2.70 | 0.88 / 2.57 | 1.64 / 3.27 |
| `coll_pp`: nearest net to the evader, median; share under 2 m | 2.10 m; 0.46 | 2.06 m; 0.48 | 2.36 m; 0.37 | 2.12 m; 0.46 | 2.23 m; 0.43 | 1.69 m; 0.63 | 1.55 m; 0.68 |
| `coll_pp`: ranks 1+2 / 2+3 / 1+3 | 0.68 / 0.26 / 0.06 | 0.62 / 0.29 / 0.09 | 0.60 / 0.33 / 0.08 | 0.68 / 0.27 / 0.05 | 0.66 / 0.27 / 0.07 | 0.59 / 0.31 / 0.09 | 0.69 / 0.22 / 0.09 |

- In five of the six runs the pursuers still hold back at 2–5 m: they fly 0.54–0.88 m/s toward the evader. `1390aa` flies 1.64 m/s toward it.

A probe on the last checkpoints: the sensed columns of some drones are set to a fixed value in the input of the pursuer, and the drones fly as before. A fixed attitude is the identity, and a fixed rpm is 0.55, the hover value. 1024 eval episodes. Each cell is the catch, then `coll_pp`, then `crash_p`:

| run | nothing fixed | attitude of the evader | attitude of the teammates | rpm of the evader | rpm of the teammates |
|---|---|---|---|---|---|
| `b503f3` | 0.461 / 0.412 / 0.023 | 0.336 / 0.415 / 0.106 | 0.401 / 0.449 / 0.046 | — | — |
| `740128` | 0.360 / 0.473 / 0.030 | 0.243 / 0.450 / 0.099 | 0.301 / 0.474 / 0.064 | — | — |
| `873189` | 0.522 / 0.304 / 0.044 | — | — | 0.524 / 0.303 / 0.043 | 0.536 / 0.296 / 0.050 |
| `22b282` | 0.542 / 0.316 / 0.015 | — | — | 0.544 / 0.312 / 0.015 | 0.548 / 0.307 / 0.015 |
| `6e196e` | 0.638 / 0.293 / 0.020 | 0.406 / 0.437 / 0.096 | 0.553 / 0.393 / 0.025 | 0.632 / 0.300 / 0.020 | 0.649 / 0.294 / 0.021 |
| `1390aa` | 0.550 / 0.357 / 0.026 | 0.260 / 0.412 / 0.164 | 0.548 / 0.324 / 0.049 | 0.551 / 0.359 / 0.025 | 0.566 / 0.351 / 0.022 |

- No policy reads the rpm: with it fixed, the catch changes by −0.006 to +0.016.
- The policies read the attitude of the evader: with it fixed, the catch falls by 0.125, 0.117, 0.232 and 0.290.
- They read the attitude of the teammates less: the catch falls by 0.060, 0.059, 0.085 and 0.002.
