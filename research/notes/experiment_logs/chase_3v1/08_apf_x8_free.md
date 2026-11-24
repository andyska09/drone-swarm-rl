# 08_apf_x8_free

**Question:** What catch (`caught_e`) do one pursuer and three pursuers give when they learn against the free APF×8 evader?
**Differs from** `06_apf_one_pursuer`: the evader. `apf_x8_free_1v1` and `apf_x8_free_3v1`, one config each. Seeds 0 and 1.
**Runs:** RCI, `tools/rci/sweeps/08_apf_x8_free.sh`, 4 jobs.

- The free APF×8 evader has 8 times the push from the pursuers, and only the motors limit it. It is `apf_x8_free` of `07_apf_agile_eval`.
- In `07_apf_agile_eval`, with no training against it, one pursuer catches 0.566 and three pursuers catch 0.497. The evader leaves the box in 0.226 and 0.124 of the episodes.

## Results

| config | seeds | caught_e | min | max | coll_pp | death_rate_p |
|---|---|---|---|---|---|---|
| apf_x8_free_1v1 | 2/2 | 0.977 | 0.969 | 0.985 | 0.000 | 0.005 |
| apf_x8_free_3v1 | 2/2 | 0.451 | 0.446 | 0.456 | 0.082 | 0.057 |

- One pursuer catches 0.977: 0.985 and 0.969 in the two seeds.
- Three pursuers catch 0.451: 0.446 and 0.456 in the two seeds.
- With three pursuers, `coll_pp` is 0.082. With one pursuer it is 0.
