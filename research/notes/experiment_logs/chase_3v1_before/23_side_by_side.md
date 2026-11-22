# 23_side_by_side

**Question:** Which setup of `paths_3v1` catches the most at its last update?
**Differs from** the files before it: no new run. The table puts their runs side by side.
**Runs:** none, 2026-11-14.

- "Same code and reward" is yes for a run with the progress reward and the critic input for each drone: the constant-lr runs (`fba6aec`), and `c6de8e` and `398c2a` (`18a4c40`).
- It is no for the adaptive-lr sense runs of `03_cost30_slot0`, and for the runs of `13_ent_adaptive_lr`.

## Results

The eval at the last update (1907) of each run with `gamma=0.99` and 5e8 steps: 1024 episodes, mean actions. The setups are sorted by their average catch over the two seeds. Each sense run has `ent_coef=0.001`.

| the pursuer senses | lr | average catch | run | seed | catch | hits between pursuers | same code and reward |
|---|---|---|---|---|---|---|---|
| nothing, `ent_coef=0.003` | constant | 0.739 | `730968` | 0 | 0.628 | 33.7 % | yes |
| | | | `9eb432` | 1 | 0.850 | 12.5 % | yes |
| attitude and rpm | constant | 0.594 | `6e196e` | 0 | 0.638 | 29.3 % | yes |
| | | | `1390aa` | 1 | 0.550 | 35.7 % | yes |
| rpm | constant | 0.533 | `873189` | 0 | 0.522 | 30.4 % | yes |
| | | | `22b282` | 1 | 0.542 | 31.6 % | yes |
| nothing, `ent_coef=0.001` | constant | 0.487 (one seed) | `be848f` | 0 | 0.487 | 28.0 % | yes |
| | | | no run | 1 | — | — | — |
| attitude | constant | 0.411 | `b503f3` | 0 | 0.461 | 41.2 % | yes |
| | | | `740128` | 1 | 0.360 | 47.3 % | yes |
| nothing, `ent_coef=0.001` | adaptive | 0.225 | `c6de8e` | 0 | 0.194 | 56.1 % | yes |
| | | | `398c2a` | 1 | 0.255 | 32.5 % | yes |
| attitude | adaptive | 0.023 | `cabc36` | 0 | 0.030 | 53.0 % | no |
| | | | `05fb55` | 1 | 0.017 | 75.9 % | no |
| attitude and rpm | adaptive | 0.023 | `396719` | 0 | 0.017 | 32.4 % | no |
| | | | `7c45e0` | 1 | 0.030 | 59.0 % | no |
| nothing, `ent_coef=0.003` | adaptive | 0.017 | `2cda6a` | 0 | 0.006 | 28.2 % | no |
| | | | `503bac` | 1 | 0.027 | 22.8 % | no |
| rpm | adaptive | 0.008 | `150658` | 0 | 0.009 | 43.8 % | no |
| | | | `fade20` | 1 | 0.007 | 62.8 % | no |

- Every constant-lr setup catches more than every adaptive-lr setup: 0.411–0.739, against 0.008–0.225.
- One pair of setups differs only in the lr: no sensed block with `ent_coef=0.001` catches 0.487 with the constant lr (one seed), against 0.225 with the adaptive lr.
- The first row, no sensed block with `ent_coef=0.003` and the constant lr, is where `01_base_seeds` of `chase_3v1` starts.
