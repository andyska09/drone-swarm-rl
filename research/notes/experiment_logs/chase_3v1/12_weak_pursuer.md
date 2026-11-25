# 12_weak_pursuer

**Question:** With the weak drone, what catch (`caught_e`) does one pursuer give against the APF evader, and what catch do three pursuers give?
**Differs from** `11_ent_easy`, config `apf_3v1`: the pursuers fly the weak drone. `apf_1v1_weak` and `apf_3v1_weak` × three weak drones. Seeds 0 and 1.
**Runs:** RCI, `tools/rci/sweeps/12_weak_pursuer.sh`, 12 jobs. The script skips a config that has runs.

- The three weak drones, in level flight:

| label | top motor speed | drag value | top speed | top thrust to the side | the evader is |
|---|---|---|---|---|---|
| none | 5528 rpm | 2.122 | 8.0 m/s | 13.3 m/s² | 1.5 times faster |
| `v6_a13` | 5528 rpm | 3.773 | 6.0 m/s | 13.3 m/s² | 2 times faster |
| `v4_a7` | 4679 rpm | 4.244 | 4.0 m/s | 6.7 m/s² | 3 times faster |

- The APF evader flies the base drone under its rule of 12 m/s and 20 m/s².
- `ent_coef` is 0.001 for both presets. No run before this one tested 0.001 with one pursuer.
- The catch of one pursuer is the number that the catch of three pursuers must beat.

## Results

| config | seeds | caught_e | min | max | coll_pp | death_rate_p | time_to_catch |
|---|---|---|---|---|---|---|---|
| apf_1v1_weak | 2/2 | 0.999 | 0.999 | 0.999 | 0.000 | 0.000 | 3.661 |
| apf_1v1_weak_v6_a13 | 2/2 | 0.998 | 0.997 | 1.000 | 0.000 | 0.000 | 4.123 |
| apf_1v1_weak_v4_a7 | 2/2 | 0.702 | 0.628 | 0.775 | 0.000 | 0.000 | 6.791 |
| apf_3v1_weak | 2/2 | 0.748 | 0.695 | 0.802 | 0.144 | 0.096 | 5.155 |
| apf_3v1_weak_v6_a13 | 2/2 | 0.721 | 0.675 | 0.767 | 0.140 | 0.094 | 5.469 |
| apf_3v1_weak_v4_a7 | 2/2 | 0.148 | 0.116 | 0.180 | 0.173 | 0.116 | 6.669 |

- Three pursuers catch less than one pursuer with each weak drone: 0.748, 0.721 and 0.148.
- The time to catch grows from 3.66 s to 6.79 s for one pursuer, and from 5.16 s to 6.67 s for three. With the base drone it is 1.51 s, in `06_apf_one_pursuer`, and 2.58 s, in `11_ent_easy`.
