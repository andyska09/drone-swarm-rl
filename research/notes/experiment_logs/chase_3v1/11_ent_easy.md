# 11_ent_easy

**Question:** Does `ent_coef` 0.001 keep the catch (`caught_e`) of the three pursuers in `paths_3v1` and `apf_3v1`?
**Differs from** `04_new_base`, and from `06_apf_one_pursuer`, config `apf_3v1`: `ent_coef` 0.001. `paths_3v1` and `apf_3v1`, one config each. Seeds 0 and 1. The rows with `ent_coef` 0.003 do not run again: `paths_3v1` is in `04_new_base`, and `apf_3v1` is in `06_apf_one_pursuer`.
**Runs:** RCI, `tools/rci/sweeps/11_ent_easy.sh`, 4 jobs.

- With `ent_coef` 0.003 the three pursuers catch 0.975 in `paths_3v1` and 0.940 in `apf_3v1`.
- In `09_apf_x8_free_ent`, `ent_coef` 0.001 lifts the catch in `apf_x8_free_3v1` from 0.451 to 0.706.
- Beside the catch, the result is the std of each action at the end of the run.
