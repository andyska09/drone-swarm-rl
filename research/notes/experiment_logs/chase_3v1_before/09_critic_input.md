# 09_critic_input

**Question:** Do three pursuers learn to fly in when the critic gives each pursuer its own value?
**Differs from** `07_input_scale`: the critic reads each drone's own actor input, after the scene and the actions of the opponents. In `paths_3v1` that is 120 numbers, up from 76. The critic weights stay shared, and the actor does not change. MAPPO calls this input the Agent-Specific Global State.
**Runs:** MetaCentrum, 4 runs, 2026-11-04 at 15:37, commit `2d15eb5`. The hashes are those of `07_input_scale`.

- Why: with one input for all three pursuers, the critic cannot show which of them flies into a hit. So the blame falls on all three.

## Results

Eval of the last checkpoint, 1024 episodes. "Training" is the mean of the last 100 updates. The std is its value at the end for each action.

| run | scaling | seed | `caught_e` eval | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | time to catch | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `a2abcb` | on | 0 | 0.099 | 0.019 | 0.007 | 0.009 | 0.87 | 5.4 s | 0.093 | 0.04–0.07 | 0.62 |
| `3849a5` | on | 1 | 0.112 | 0.091 | 0.003 | 0.017 | 0.78 | 5.9 s | 0.109 | 0.07–0.10 | 0.69 |
| `2d0b98` | off | 0 | 0.183 | 0.097 | 0.009 | 0.015 | 0.70 | 6.1 s | 0.173 | 0.04–0.05 | 0.60 |
| `6fce56` | off | 1 | 0.242 | 0.032 | 0.011 | 0.001 | 0.71 | 6.0 s | 0.233 | 0.03 | 0.66 |

- These are the first runs with three pursuers that learn to fly in: the catch is 0.099–0.242, against 0.009–0.035 in `07_input_scale`.
- The std no longer blows up: it ends at 0.03–0.10.
- Without the input scale the catch is about twice as high, in both seeds: 0.183 and 0.242, against 0.099 and 0.112.

Other numbers of these runs:

| what | value |
|---|---|
| the nearest pursuer, in the 8 saved eval episodes of each run | goes from 13 m to about 3–5 m by 4–6 s. Most episodes reach the step limit |
| the catch in training from update 1300 to 1900, without the scale | 0.16 to 0.18, and 0.22 to 0.24 |
| the catch in training from update 1300 to 1900, with the scale | flat, 0.09–0.11 |
| three FRPN pursuers | catch 0.44–0.49 |
