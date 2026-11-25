# 01_pool_steps

**Question:** Do three pursuers learn to catch the evaders of their own pool run, and do they catch more with 4 times the steps?
**Differs from** `11_ent_easy` of `chase_3v1`, config `paths_3v1`: preset `3v1`, where the evader learns too, and `generations=13`. 1.4e9 steps and 5.6e9 steps, one config each. Seeds 0 and 1.
**Runs:** RCI, `tools/rci/sweeps/01_pool_steps.sh`, 4 jobs.

- A pool run has 14 phases. In each phase, each side learns against the frozen policies that the other side made in the phases before. A side has 50M steps in a phase with 1.4e9 steps, and 200M with 5.6e9.
- The only pool run with three pursuers before this one is `01_pool` of `chase_3v1_before`, on older code. In it, no pursuer wins more than 0.07 against the paths evader.
- The result is the table of `run/matrix.py`, in the four columns of `02_pools` of `chase_1v1`: lower triangle, older evaders, diagonal and last pair. Beside it, the win rate of `pursuer_13` against the paths evader and the APF evader.
- The run of `01_pool` took 41 minutes for 1.4e9 steps. So a job has a time limit of 1:30:00 with 1.4e9 steps, and of 3:59:00 with 5.6e9.
