# 13_weak_pursuer_long

**Question:** With 2e9 steps, do three weak pursuers reach the catch (`caught_e`) of one weak pursuer?
**Differs from** `12_weak_pursuer`, configs `apf_3v1_weak` and `apf_3v1_weak_v6_a13`: 2e9 steps. Seeds 0 and 1.
**Runs:** RCI, `tools/rci/sweeps/13_weak_pursuer_long.sh`, 4 jobs.

- In `12_weak_pursuer`, with 5e8 steps, three pursuers catch 0.748 in `apf_3v1_weak` and 0.721 in `apf_3v1_weak_v6_a13`. One pursuer catches 0.999 and 0.998.
- The catch of three pursuers in training still rises at the end of `12_weak_pursuer`. From 2.5e8 to 5e8 steps it goes from 0.51 to 0.82 in `apf_3v1_weak`, and from 0.53 to 0.79 in `apf_3v1_weak_v6_a13`. These are means of the two seeds.
- The catch of one pursuer in training is 0.99 after 2.5e8 steps.
- A job of three pursuers with 5e8 steps took 16 minutes. So each job has a time limit of 2 hours.

## Results
