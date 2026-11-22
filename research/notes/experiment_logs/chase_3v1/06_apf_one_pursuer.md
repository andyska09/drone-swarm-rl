# 06_apf_one_pursuer

**Question:** What catch (`caught_e`) do three pursuers give against the APF evader, and what catch does one pursuer give against the APF evader and against the paths evader?
**Differs from** `04_new_base`: the preset. `apf_3v1`, `apf_1v1` and `paths_1v1`, one config each. Seeds 0 and 1. `paths_3v1` does not run again: its row is in `04_new_base`.
**Runs:** RCI, `tools/rci/sweeps/06_apf_one_pursuer.sh`, 6 jobs.

- The pursuers learn against the evader of the preset, and the eval flies them against the same evader.
- The catch of one pursuer is the number that the catch of three pursuers must beat, against the same evader.
