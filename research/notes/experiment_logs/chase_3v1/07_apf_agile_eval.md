# 07_apf_agile_eval

**Question:** Which change to the APF evader lowers the catch (`caught_e`) of the policies of `06_apf_one_pursuer` the most: a stronger push from the pursuers, no limits, or both?
**Differs from** `06_apf_one_pursuer`: no training. The four policies of `apf_3v1` and `apf_1v1` fly against six variants of the APF evader. The push from the pursuers is times 2, 4 or 8. The limits of 20 m/s² and 12 m/s stay, or the evader flies without them.
**Runs:** the Mac, eval only, 24 evals of 4096 episodes each, by a probe script in `scratch/`.

- The policies learned against the APF evader of the spec. No policy met a variant in training.
- Without the limits, only the motors limit the evader, as they limit the pursuers.

## Results

Each cell is the mean of the two seeds. `apf` is the evader of `06_apf_one_pursuer`. `x4` is the push from the pursuers times 4, and `free` is the evader without the limits.

| evader | 1v1 caught_e | 1v1 time_to_catch, s | 1v1 time_limit | 1v1 OOB_e | 1v1 OOB_p | 3v1 caught_e | 3v1 time_to_catch, s | 3v1 coll_pp | 3v1 time_limit | 3v1 OOB_e | 3v1 OOB_p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| apf | 1.000 | 1.51 | 0.000 | 0.000 | 0.000 | 0.940 | 2.54 | 0.059 | 0.000 | 0.000 | 0.000 |
| apf_x2 | 1.000 | 1.59 | 0.000 | 0.000 | 0.000 | 0.918 | 2.90 | 0.080 | 0.000 | 0.000 | 0.000 |
| apf_x2_free | 0.998 | 1.63 | 0.000 | 0.000 | 0.002 | 0.918 | 2.95 | 0.073 | 0.001 | 0.001 | 0.008 |
| apf_x4 | 0.999 | 2.14 | 0.001 | 0.000 | 0.000 | 0.885 | 3.42 | 0.108 | 0.003 | 0.000 | 0.002 |
| apf_x4_free | 0.969 | 2.41 | 0.019 | 0.003 | 0.009 | 0.840 | 3.38 | 0.105 | 0.030 | 0.004 | 0.019 |
| apf_x8 | 0.965 | 3.24 | 0.020 | 0.004 | 0.011 | 0.620 | 4.20 | 0.300 | 0.044 | 0.004 | 0.028 |
| apf_x8_free | 0.566 | 3.02 | 0.083 | 0.226 | 0.123 | 0.497 | 3.72 | 0.212 | 0.058 | 0.124 | 0.107 |

- One pursuer catches 0.965 or more against every variant but `apf_x8_free`, where it catches 0.566.
- Three pursuers catch less than one pursuer against every variant. Their catch falls from 0.940 to 0.620 as the push grows to 8 times, and `coll_pp` rises from 0.059 to 0.300.
- Against `apf_x8_free` the evader leaves the box in 0.226 of the episodes with one pursuer, and in 0.124 with three.
