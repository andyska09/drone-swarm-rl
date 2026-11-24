# 09_apf_x8_free_ent

**Question:** Does a smaller entropy weight stop the growth of the std of the pursuers' actions, and lift the catch (`caught_e`), in `apf_x8_free_3v1`?
**Differs from** `08_apf_x8_free`, config `apf_x8_free_3v1`: `ent_coef` 0.001 and 0. Seeds 0 and 1. The row with `ent_coef` 0.003 does not run again: it is `apf_x8_free_3v1` in `08_apf_x8_free`.
**Runs:** RCI, `tools/rci/sweeps/09_apf_x8_free_ent.sh`, 4 jobs.

- In `08_apf_x8_free`, the std of the thrust grows from 1.0 to 2.58 in training, and the three pursuers catch 0.451.
- Beside the catch, the result is the std of each action at the end of the run.

## Results

| config | seeds | caught_e | min | max | coll_pp | death_rate_p |
|---|---|---|---|---|---|---|
| ent_coef0.001 | 2/2 | 0.706 | 0.687 | 0.726 | 0.136 | 0.094 |
| ent_coef0 | 2/2 | 0.445 | 0.414 | 0.476 | 0.073 | 0.052 |

- With `ent_coef` 0.001 the three pursuers catch 0.706: 0.726 and 0.687 in the two seeds.
- With `ent_coef` 0 they catch 0.445. With 0.003 they catch 0.451, in `08_apf_x8_free`.

The spread of the pursuers' actions in training, as the mean of the two seeds. The std of each action starts at 1.0, and the table gives its value at the end of the run. The row of 0.003 is `apf_x8_free_3v1` of `08_apf_x8_free`.

| `ent_coef` | caught_e in training, last tenth | entropy_p, first tenth | entropy_p, last tenth | std thrust | std roll | std pitch | std yaw |
|---|---|---|---|---|---|---|---|
| 0.003 | 0.373 | 5.88 | 9.48 | 2.58 | 1.41 | 1.38 | 10.77 |
| 0.001 | 0.635 | 5.56 | 2.53 | 0.44 | 0.32 | 0.33 | 0.79 |
| 0 | 0.422 | 5.39 | −0.41 | 0.23 | 0.14 | 0.14 | 0.38 |
