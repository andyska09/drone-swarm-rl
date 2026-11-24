# 09_apf_x8_free_ent

**Question:** Does a smaller entropy weight stop the growth of the std of the pursuers' actions, and lift the catch (`caught_e`), in `apf_x8_free_3v1`?
**Differs from** `08_apf_x8_free`, config `apf_x8_free_3v1`: `ent_coef` 0.001 and 0. Seeds 0 and 1. The row with `ent_coef` 0.003 does not run again: it is `apf_x8_free_3v1` in `08_apf_x8_free`.
**Runs:** RCI, `tools/rci/sweeps/09_apf_x8_free_ent.sh`, 4 jobs.

- In `08_apf_x8_free`, the std of the thrust grows from 1.0 to 2.58 in training, and the three pursuers catch 0.451.
- Beside the catch, the result is the std of each action at the end of the run.
