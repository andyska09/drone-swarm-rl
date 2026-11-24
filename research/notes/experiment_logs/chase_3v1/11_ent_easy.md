# 11_ent_easy

**Question:** Does `ent_coef` 0.001 keep the catch (`caught_e`) of the three pursuers in `paths_3v1` and `apf_3v1`?
**Differs from** `04_new_base`, and from `06_apf_one_pursuer`, config `apf_3v1`: `ent_coef` 0.001. `paths_3v1` and `apf_3v1`, one config each. Seeds 0 and 1. The rows with `ent_coef` 0.003 do not run again: `paths_3v1` is in `04_new_base`, and `apf_3v1` is in `06_apf_one_pursuer`.
**Runs:** RCI, `tools/rci/sweeps/11_ent_easy.sh`, 4 jobs.

- With `ent_coef` 0.003 the three pursuers catch 0.975 in `paths_3v1` and 0.940 in `apf_3v1`.
- In `09_apf_x8_free_ent`, `ent_coef` 0.001 lifts the catch in `apf_x8_free_3v1` from 0.451 to 0.706.
- Beside the catch, the result is the std of each action at the end of the run.

## Results

| config | seeds | caught_e | min | max | coll_pp | death_rate_p |
|---|---|---|---|---|---|---|
| paths_3v1 | 2/2 | 0.979 | 0.979 | 0.979 | 0.019 | 0.014 |
| apf_3v1 | 2/2 | 0.953 | 0.949 | 0.958 | 0.045 | 0.031 |

- In `paths_3v1` the three pursuers catch 0.979 with `ent_coef` 0.001. With 0.003 they catch 0.975, in `04_new_base`.
- In `apf_3v1` they catch 0.953 with `ent_coef` 0.001. With 0.003 they catch 0.940, in `06_apf_one_pursuer`.

The spread of the pursuers' actions in training, as the mean of the two seeds. The std of each action starts at 1.0, and the table gives its value at the end of the run. The rows of 0.003 are from `04_new_base` and from `06_apf_one_pursuer`.

| config | `ent_coef` | caught_e in training, last tenth | entropy_p, first tenth | entropy_p, last tenth | std thrust | std roll | std pitch | std yaw |
|---|---|---|---|---|---|---|---|---|
| paths_3v1 | 0.003 | 0.963 | 5.87 | 7.19 | 1.15 | 1.05 | 1.02 | 4.07 |
| paths_3v1 | 0.001 | 0.971 | 5.51 | 2.61 | 0.48 | 0.34 | 0.35 | 0.73 |
| apf_3v1 | 0.003 | 0.932 | 5.87 | 6.91 | 1.03 | 0.98 | 0.96 | 3.71 |
| apf_3v1 | 0.001 | 0.950 | 5.51 | 2.51 | 0.41 | 0.34 | 0.35 | 0.75 |
