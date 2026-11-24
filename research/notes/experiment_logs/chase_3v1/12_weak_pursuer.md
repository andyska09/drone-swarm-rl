# 12_weak_pursuer

**Question:** With the weak drone, what catch (`caught_e`) does one pursuer give against the APF evader, and what catch do three pursuers give?
**Differs from** `11_ent_easy`, config `apf_3v1`: the pursuers fly the weak drone. `apf_1v1_weak` and `apf_3v1_weak`, one config each. Seeds 0 and 1.
**Runs:** RCI, `tools/rci/sweeps/12_weak_pursuer.sh`, 4 jobs.

- The weak drone has a top motor speed of 5528 rpm and a drag value of 2.122. In level flight its top speed is 8.0 m/s, and its top thrust to the side is 13.3 m/s².
- The APF evader flies the base drone under its rule of 12 m/s and 20 m/s². So the evader is 1.5 times faster than a pursuer.
- With the base drone, one pursuer catches 1.000 (`06_apf_one_pursuer`), and three pursuers catch 0.953 (`11_ent_easy`).
- `ent_coef` is 0.001 for both presets. No run before this one tested 0.001 with one pursuer.
- The catch of one pursuer is the number that the catch of three pursuers must beat.
