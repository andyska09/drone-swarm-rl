# 08_relu

**Question:** Do three pursuers catch more with ReLU units, which do not saturate as tanh units do?
**Differs from** `07_input_scale`, the two runs with `normalize_obs=false`: `activation=relu`.
**Runs:** MetaCentrum, 2 runs, 2026-11-04, commit `636ab7a`: `061ccd` (seed 0), `3d59a8` (seed 1).

## Results

Training, the mean of the last 100 updates. The tanh column is the two runs of `07_input_scale` with `normalize_obs=false`.

| | ReLU, s0 | ReLU, s1 | tanh |
|---|---|---|---|
| `caught_e` | 0.005 | 0.004 | 0.029–0.045 |
| `OOB_p` | 0.84 | 0.89 | 0.26–0.36 |
| `coll_pp` | 0.007 | 0.008 | 0.12–0.24 |
| `time_limit` | 0 | 0 | 0.17–0.23 |
| `distance` | 9.9 | 9.9 | 7.4–7.9 |
| explained variance | 0.98 | 0.99 | 0.55–0.59 |
| final std: thrust, roll, pitch, yaw | 1.2, 0.007, 0.007, 9.2 | 0.16, 0.008, 0.006, 0.84 | 12–29 |

- The pursuers do not chase: the catch is 0.005 and 0.004, and the net stays 9.9 m from the evader.
- A pursuer leaves the box in 0.84 and 0.89 of the episodes.
- The std does not blow up. It ends at 0.006–0.008 for the roll and the pitch, and at 9.2 and 0.84 for the yaw.

Other numbers, from 512 episodes of each run with sampled actions, the last checkpoint:

| what | value |
|---|---|
| the yaw action sits at the clip | in 85–91 % of the steps |
| exactly one pursuer dies | in 98–99 % of the episodes |
| the median end of an episode | at 3.5–4 s, at about 4 m/s, level |
| the eval of seed 0, with mean actions | the mean thrust action is −0.97, so `crash_p` is 0.999 at about 1.1 s |
