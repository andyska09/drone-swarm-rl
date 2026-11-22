# 02_sense

**Question:** Do three pursuers hit each other less, and catch the paths evader more, when each one senses the attitude or the motor speeds of the other drones?
**Differs from** `01_pool`: the paths pilot flies the evader, so only the pursuers learn, in one phase. The task was `chase_modal`, with the presets `3v1`, `3v1_attitude`, `3v1_rpm` and `3v1_attitude_rpm`. They are `paths_3v1` to `paths_3v1_attitude_rpm` of `chase` now. Settings: `gamma=0.99`, 5e8 steps (1907 updates), the adaptive lr, `ent_coef=0.001`, seeds 0 to 2.
**Runs:** MetaCentrum, 12 runs, 2026-10-28, commit `23b65a8`. The code of today cannot load these runs. The 12 runs of the same batch with one pursuer are in [04_paths_sense](../chase_1v1/04_paths_sense.md).

## Results

Eval of the last checkpoint with `--seat evader paths`, 1024 episodes. Each cell is the mean of the three seeds, then seeds 0, 1 and 2.

| preset | `caught_e` | `coll_pp` | `time_to_catch` |
|---|---|---|---|
| `3v1` | 0.05 [0.03 0.05 0.07] | 0.63 [0.79 0.65 0.45] | — |
| `3v1_attitude` | 0.05 [0.05 0.03 0.07] | 0.61 [0.59 0.46 0.78] | — |
| `3v1_rpm` | 0.06 [0.07 0.07 0.04] | 0.57 [0.55 0.73 0.44] | — |
| `3v1_attitude_rpm` | 0.07 [0.07 0.08 0.07] | 0.66 [0.67 0.67 0.62] | — |

- The catch is 0.05–0.07 in every preset. One pursuer catches 0.96–0.99.
- The seeds of one preset spread more in `coll_pp` (0.46–0.78 with the attitude) than the presets differ (0.57–0.66).

Other numbers of these runs:

| what | value |
|---|---|
| `caught_e` in training, at 25 % of the run and at the end | 0.03–0.06, 0.05–0.08 |
| `crash_p` + `OOB_p` in training, at the start and at the end | 0.96, 0.02–0.15 |
| `coll_pp` in training, at 10 % of the run and at the end | 0.02–0.16, 0.33–0.75 |
| the std of the actions at the end | 0.4–1.3, and 3.2 in `3v1` seed 0. With one pursuer: 0.17–0.23 |
| the time of the median `coll_pp`, eval of `3v1` seed 0 | 4.6 s. 4 % come before 2 s |
| the slot of the evader, eval of `3v1` seed 0 | the last of the 3 slots in 71 % of the steps |

- In training, the crashes and the exits fall, and `coll_pp` rises in their place.
- In 4 of the 8 saved episodes of `3v1` seed 0, two pursuers fly into each other at 4–9 m/s, 5–8 m from the evader.
