# 07_input_scale

**Question:** Do three pursuers catch more when each network scales its input by a running mean and std?
**Differs from** `03_cost30_slot0`: only `paths_3v1`, with no sensed block. Two runs scale the inputs, and two run with `normalize_obs=false` on the same code.
**Runs:** MetaCentrum, 4 runs, 2026-11-04 at 10:56, commit `636ab7a`, the first commit with the input scale. The four hashes repeat in `09_critic_input`, because a hash covers only the config.

## Results

Eval of the last checkpoint, 1024 episodes. "Training" is the mean of the last 100 updates. The std is its value at the end for each action.

| run | scaling | seed | `caught_e` eval | `coll_pp` eval | `crash_p` eval | `time_limit` eval | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|
| `a2abcb` | on | 0 | 0.024 | 0.88 | 0.03 | 0.06 | 0.028 | 2520–4544 | 0.42 |
| `3849a5` | on | 1 | 0.025 | 0.55 | 0.40 | 0.02 | 0.035 | 1290–1423 | 0.25 |
| `2d0b98` | off | 0 | 0.035 | 0.34 | 0.56 | 0.03 | 0.045 | 18–29 | 0.59 |
| `6fce56` | off | 1 | 0.009 | 0.44 | 0.39 | 0.16 | 0.029 | 12–27 | 0.55 |

- The catch stays at 0.009–0.035, with the scale and without it.
- The std blows up in all four runs, and much more with the scale: 1290–4544, against 12–29.
- The critic fits worse with the scale: its explained variance is 0.25–0.42, against 0.55–0.59.

The std of each run over training, the mean over the four actions:

| update | 100 | 250 | 400 | 600 | 900 | 1300 | 1900 |
|---|---|---|---|---|---|---|---|
| on, s0 | 1.17 | 1.63 | 3.55 | 10.3 | 69.8 | 622 | 3238 |
| on, s1 | 1.18 | 1.38 | 2.61 | 7.39 | 32.1 | 185 | 1289 |
| off, s0 | 1.08 | 1.28 | 1.91 | 3.79 | 8.90 | 13.4 | 24.4 |
| off, s1 | 1.15 | 1.38 | 1.64 | 3.20 | 7.42 | 9.39 | 17.5 |

Other numbers of these runs:

| what | value |
|---|---|
| the adaptive lr | reaches its cap of 0.01 in all four runs |
| `ep_length` around update 140, with the scale and without it | 600–720, 250–300 |
