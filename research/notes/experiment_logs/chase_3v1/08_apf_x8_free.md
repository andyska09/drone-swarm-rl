# 08_apf_x8_free

**Question:** What catch (`caught_e`) do one pursuer and three pursuers give when they learn against the free APF×8 evader?
**Differs from** `06_apf_one_pursuer`: the evader. `apf_x8_free_1v1` and `apf_x8_free_3v1`, one config each. Seeds 0 and 1.
**Runs:** RCI, `tools/rci/sweeps/08_apf_x8_free.sh`, 4 jobs.

- The free APF×8 evader has 8 times the push from the pursuers, and only the motors limit it. It is `apf_x8_free` of `07_apf_agile_eval`.

## Results

| config | seeds | caught_e | min | max | coll_pp | death_rate_p |
|---|---|---|---|---|---|---|
| apf_x8_free_1v1 | 2/2 | 0.977 | 0.969 | 0.985 | 0.000 | 0.005 |
| apf_x8_free_3v1 | 2/2 | 0.451 | 0.446 | 0.456 | 0.082 | 0.057 |

The spread of the pursuers' actions in training, for the configs of this experiment and of `06_apf_one_pursuer`, as the mean of the two seeds. The std of each action starts at 1.0, and the table gives its value at the end of the run.

| config | caught_e in training, last tenth | entropy_p, first tenth | entropy_p, last tenth | std thrust | std roll | std pitch | std yaw |
|---|---|---|---|---|---|---|---|
| apf_1v1 | 1.000 | 5.71 | 3.08 | 0.57 | 0.40 | 0.41 | 0.76 |
| apf_3v1 | 0.932 | 5.87 | 6.91 | 1.03 | 0.98 | 0.96 | 3.71 |
| apf_x8_free_1v1 | 0.977 | 5.64 | 4.03 | 0.61 | 0.56 | 0.52 | 1.07 |
| apf_x8_free_3v1 | 0.373 | 5.87 | 9.48 | 2.58 | 1.41 | 1.38 | 10.76 |
