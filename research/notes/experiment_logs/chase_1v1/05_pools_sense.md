# 05_pools_sense

**Question:** Does a pursuer that senses more of the evader win more often against learned evaders?
**Differs from** `02_pools`, run `9e4242`: the pursuer senses the attitude, the motor speeds or both of the evader, in the presets `default_p_attitude`, `default_p_rpm` and `default_p_attitude_rpm`. The evader senses no block. Both sides learn at once, so a run has 14 phases. Seeds 0 and 1. Settings: `gamma=0.99`, `generations=13`, `ent_coef=0`, 1.4e9 steps.
**Runs:** MetaCentrum, 6 runs, 2026-10-28, commit `edea2f4`.

- The base `9e4242` is one seed on older code, where the sides learned in turn.

## Results

The columns of `02_pools`. Each cell is the mean of the two seeds, then seeds 0 and 1:

| config | lower triangle | older evaders | diagonal | last pair |
|---|---|---|---|---|
| `9e4242`, no block | 0.61 | 0.66 | 0.27 | 0.20 |
| default_p_attitude | 0.73 [0.64 0.81] | 0.78 [0.70 0.87] | 0.37 [0.28 0.46] | 0.31 [0.18 0.45] |
| default_p_rpm | 0.77 [0.73 0.81] | 0.81 [0.77 0.85] | 0.52 [0.47 0.57] | 0.59 [0.58 0.61] |
| default_p_attitude_rpm | 0.88 [0.87 0.89] | 0.93 [0.92 0.93] | 0.60 [0.57 0.63] | 0.66 [0.44 0.88] |

- Every run is above `9e4242` in every column, but seed 0 of `default_p_attitude`: its diagonal is 0.28 against 0.27, and its last pair 0.18 against 0.20.
- In all four columns the order is attitude, then rpm, then both. With both blocks the pursuer wins 0.93 against the older evaders, against 0.66.
- The last pair is one cell, and its two seeds spread from 0.44 to 0.88 with both blocks.

Evals of the last checkpoint, 1024 episodes. Each cell is the pursuer's win rate, seeds 0 and 1:

| config | last pursuer against last evader | FRPN against last evader | last pursuer against cascade |
|---|---|---|---|
| `9e4242`, no block | 0.20 | 0.04 | 0.95 |
| default_p_attitude | 0.18, 0.45 | 0.04, 0.12 | 0.99, 0.99 |
| default_p_rpm | 0.58, 0.61 | 0.19, 0.03 | 0.97, 0.97 |
| default_p_attitude_rpm | 0.44, 0.88 | 0.16, 0.08 | 0.99, 0.92 |

- FRPN wins 0.03–0.19 against the last evader of every run.
- The last pursuer wins 0.92–0.99 against the cascade evader in every run.
