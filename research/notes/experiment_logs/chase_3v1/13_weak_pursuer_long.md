# 13_weak_pursuer_long

**Question:** With 2e9 steps, do three weak pursuers reach the catch (`caught_e`) of one weak pursuer?
**Differs from** `12_weak_pursuer`, configs `apf_3v1_weak` and `apf_3v1_weak_v6_a13`: 2e9 steps. Seeds 0 and 1.
**Runs:** RCI, `tools/rci/sweeps/13_weak_pursuer_long.sh`, 4 jobs.

- The catch of three pursuers in training still rises at the end of `12_weak_pursuer`. From 2.5e8 to 5e8 steps it goes from 0.51 to 0.82 in `apf_3v1_weak`, and from 0.53 to 0.79 in `apf_3v1_weak_v6_a13`. These are means of the two seeds.
- The catch of one pursuer in training is 0.99 after 2.5e8 steps.
- A job of three pursuers with 5e8 steps took 16 minutes. So each job has a time limit of 2 hours.

## Results

| config | seeds | caught_e | min | max | coll_pp | death_rate_p | time_to_catch |
|---|---|---|---|---|---|---|---|
| apf_3v1_weak_v6_a13 | 2/2 | 0.952 | 0.947 | 0.957 | 0.038 | 0.025 | 4.015 |
| apf_3v1_weak | 2/2 | 0.942 | 0.931 | 0.952 | 0.052 | 0.035 | 3.616 |

- Three pursuers do not reach one pursuer, which catches 0.999 and 0.998 in `12_weak_pursuer`.
