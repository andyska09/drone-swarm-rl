# 01_pool_steps

**Question:** Do three pursuers learn to catch the evaders of their own pool run, and do they catch more with 4 times the steps?
**Differs from** `11_ent_easy` of `chase_3v1`, config `paths_3v1`: preset `3v1`, where the evader learns too, and `generations=13`. 1.4e9 steps and 5.6e9 steps, one config each. Seeds 0 and 1.
**Runs:** RCI, `tools/rci/sweeps/01_pool_steps.sh`, 4 jobs.

- A pool run has 14 phases. In each phase, each side learns against the frozen policies that the other side made in the phases before. A side has 50M steps in a phase with 1.4e9 steps, and 200M with 5.6e9.
- The only pool run with three pursuers before this one is `01_pool` of `chase_3v1_before`, on older code. In it, no pursuer wins more than 0.07 against the paths evader.
- The result is the table of `run/matrix.py`, in the four columns of `02_pools` of `chase_1v1`: lower triangle, older evaders, diagonal and last pair. Beside it, the win rate of `pursuer_13` against the paths evader and the APF evader.
- The run of `01_pool` took 41 minutes for 1.4e9 steps. So a job has a time limit of 1:30:00 with 1.4e9 steps, and of 3:59:00 with 5.6e9.
- The config `1v1` is a line to compare to. It is the same pool run with one pursuer: preset `default`, 1.4e9 steps, seeds 0 and 1.

## Results

Only the config with 1.4e9 steps ran. Its row is the eval of the last pursuers against the last evader of the same run:

| config | seeds | caught_e | min | max | coll_pp | death_rate_p | time_to_catch |
|---|---|---|---|---|---|---|---|
| steps1.4e9 | 2/2 | 0.643 | 0.546 | 0.740 | 0.075 | 0.067 | 3.905 |

- The last pursuers catch the last evader in 0.643 of the episodes: 0.740 in seed 0 and 0.546 in seed 1.

`run/matrix.py` flies every pursuer against every evader of its run: 1024 episodes for each pair, mean actions. A cell is the win rate of the pursuers, and the columns are those of `02_pools` of `chase_1v1`:

| seed | lower triangle | older evaders | diagonal | last pair | `pursuer_13` against paths | `pursuer_13` against APF |
|---|---|---|---|---|---|---|
| 0 | 0.879 | 0.927 | 0.568 | 0.839 | 0.917 | 0.889 |
| 1 | 0.837 | 0.887 | 0.511 | 0.643 | 0.895 | 0.835 |

- The pursuers win 0.927 and 0.887 against the evaders that they trained on, and 0.568 and 0.511 against the evader of their own generation.
- A win is a catch, or an evader that crashes or leaves the box. So the last pair is above the `caught_e` of the first table.
