# 03_cost30_slot0

**Question:** Do three pursuers catch more when a `coll_pp` costs 30 and the evader has a fixed slot in the input?
**Differs from** `02_sense`: a `coll_pp` costs 30, the same as a crash. The evader is in slot 0 of the others. The task is `chase`, with the presets `paths_3v1` to `paths_3v1_attitude_rpm`. Seeds 0 and 1.
**Runs:** MetaCentrum, 8 runs, 2026-11-01, commit `de90325`: `a761e9`, `ab4e23` (no block), `cabc36`, `05fb55` (attitude), `150658`, `fade20` (rpm), `396719`, `7c45e0` (both), seeds 0 and 1 of each.

- Why 30: at `gamma=0.99`, the rest of an episode is worth about −5 to a pursuer that lives on. So a `coll_pp` at 10 cost only about 5 more than to live on, and a crash at 30 about 25 more.
- Why slot 0: in `02_sense` a teammate sat in slot 0 in most steps.

## Results

Eval of the last checkpoint, 1024 episodes. Each cell is the mean of the two seeds. The `old` columns are `02_sense`, with three seeds.

| preset | `caught_e` | old | `coll_pp` | old | `crash_p` |
|---|---|---|---|---|---|
| `paths_3v1` | 0.005 | 0.05 | 0.44 | 0.63 | 0.53 |
| `paths_3v1_attitude` | 0.024 | 0.05 | 0.64 | 0.61 | 0.30 |
| `paths_3v1_rpm` | 0.008 | 0.06 | 0.53 | 0.57 | 0.45 |
| `paths_3v1_attitude_rpm` | 0.024 | 0.07 | 0.46 | 0.66 | 0.51 |

- The catch is 0.005–0.024, below the 0.05–0.07 of `02_sense`.

Training, the last 5 % of the updates, against seeds 0 and 1 of `02_sense`:

| | old | new |
|---|---|---|
| `caught_e` | 0.05–0.08 | 0.03–0.05 |
| `coll_pp` | 0.33–0.70 | 0.20–0.33 |
| `crash_p` + `OOB_p` | 0.03–0.16 | 0.33–0.56 |
| the pursuer's entropy at the end | 1.7–10.3 | 18–22 |

- The entropy rises for the whole run. The std ends at about 20–60 for each action, so almost every action in training is a random full push.
- `coll_pp` falls, and `crash_p` + `OOB_p` rise in its place. A pursuer still dies in about 3 of 4 episodes.

Other numbers of these runs:

| what | value |
|---|---|
| the adaptive lr | rises to its cap of 0.01 in the second half of each run. In `02_sense` it stays near 0.001 |
| `OOB_p`, in training and in the eval | 0.18–0.39, 0.001–0.019 |

- The eval flies the mean action, which the drones almost never fly in training. So the eval tells little here.
