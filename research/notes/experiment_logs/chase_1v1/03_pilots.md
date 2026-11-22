# 03_pilots

**Question:** What does each hand-written pursuer catch against each hand-written evader in `default`?
**Differs from** the code of today: `hover`, `circle` and `figure8` each fly as an evader of their own. Today the paths pilot draws one of the three for each episode. No policy trains here, so the reward does not matter.
**Runs:** eval only, 1024 episodes for each pair, commit `06eac95`, by a script in `scratch/`. The evals are in `runs/chase_baselines/`.

## Results

| pursuer | evader | caught | crash P | OOB P | timeout | time to catch |
|---|---|---|---|---|---|---|
| frpn | hover | 0.768 | 0.007 | 0.225 | 0.001 | 2.64 s |
| frpn | circle | 0.855 | 0.013 | 0.114 | 0.018 | 3.51 s |
| frpn | figure8 | 0.677 | 0.015 | 0.175 | 0.134 | 4.03 s |
| frpn | apf | 0.794 | 0 | 0.062 | 0.144 | 5.23 s |
| frpn | cascade | 0.762 | 0.017 | 0.216 | 0.006 | 3.16 s |
| cascade | hover | 1.000 | 0 | 0 | 0 | 3.91 s |
| cascade | circle | 0.100 | 0 | 0 | 0.900 | 2.95 s |
| cascade | figure8 | 0.306 | 0 | 0 | 0.694 | 5.45 s |
| cascade | apf | 0.421 | 0 | 0 | 0.579 | 4.94 s |
| cascade | cascade | 0.997 | 0 | 0 | 0.003 | 4.83 s |

No evader crashed or left the box, so the table has no column for it.

- FRPN catches 0.677–0.855 of every evader. It leaves the box in 0.062–0.225 of the episodes.
- The figure-eight is the hardest evader for FRPN (0.677). Against APF, FRPN takes the longest: 5.23 s.
- The cascade pursuer catches the hover (1.000) and the cascade evader (0.997). It catches 0.100 of the circles.
