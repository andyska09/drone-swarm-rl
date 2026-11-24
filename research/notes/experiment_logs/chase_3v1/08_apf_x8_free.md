# 08_apf_x8_free

**Question:** What catch (`caught_e`) do one pursuer and three pursuers give when they learn against the free APF×8 evader?
**Differs from** `06_apf_one_pursuer`: the evader. `apf_x8_free_1v1` and `apf_x8_free_3v1`, one config each. Seeds 0 and 1.
**Runs:** RCI, `tools/rci/sweeps/08_apf_x8_free.sh`, 4 jobs.

- The free APF×8 evader has 8 times the push from the pursuers, and only the motors limit it. It is `apf_x8_free` of `07_apf_agile_eval`.
- In `07_apf_agile_eval`, with no training against it, one pursuer catches 0.566 and three pursuers catch 0.497. The evader leaves the box in 0.226 and 0.124 of the episodes.
