# 06_apf_one_pursuer

**Question:** What catch (`caught_e`) do three pursuers give against the APF evader, and what catch does one pursuer give against the APF evader and against the paths evader?
**Differs from** `04_new_base`: the preset. `apf_3v1`, `apf_1v1` and `paths_1v1`, one config each. Seeds 0 and 1. `paths_3v1` does not run again: its row is in `04_new_base`.
**Runs:** RCI, `tools/rci/sweeps/06_apf_one_pursuer.sh`, 6 jobs.

- The pursuers learn against the evader of the preset, and the eval flies them against the same evader.
- The catch of one pursuer is the number that the catch of three pursuers must beat, against the same evader.

## Results

| config | seeds | caught_e | min | max | coll_pp | death_rate_p |
|---|---|---|---|---|---|---|
| apf_1v1 | 2/2 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| paths_1v1 | 2/2 | 0.999 | 0.998 | 0.999 | 0.000 | 0.001 |
| apf_3v1 | 2/2 | 0.940 | 0.932 | 0.947 | 0.059 | 0.040 |

- One pursuer catches 1.000 against the APF evader and 0.999 against the paths evader.
- Three pursuers catch 0.940 against the APF evader: 0.932 and 0.947 in the two seeds. Against the paths evader they catch 0.975, in `04_new_base`.
- With three pursuers, `coll_pp` is 0.059 against the APF evader. With one pursuer it is 0.
