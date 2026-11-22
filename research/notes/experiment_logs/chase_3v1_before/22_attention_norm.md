# 22_attention_norm

**Question:** Does the attention net learn with a norm on its Perceiver output?
**Differs from** `21_attention`: a LayerNorm with no scale and no bias on the output of each query.
**Runs:** MetaCentrum, 2 runs, 2026-11-11, commit `7a9878f`: `91b18c` (seed 0), `2cdecf` (seed 1). The norm has no weights, so the checkpoints of `21_attention` also load on this commit, but the network is then another one.

## Results

A test on the Mac before the runs: `paths_3v1`, seed 0, 80 updates of 1024 envs, the flags of the runs. The numbers are the mean of updates 73–80:

| network | lr | `distance` | catch | `policy_loss_p` |
|---|---|---|---|---|
| MLP | adaptive | 5.84 m | 0.107 | −0.0044 |
| attention, no norm | adaptive | 9.94 m | 0.006 | +0.0048 |
| attention, no norm | 3e-4, constant | 5.82 m | 0.131 | −0.0055 |
| attention, a layer of 32 with tanh after the Perceiver | adaptive | 5.55 m | 0.110 | −0.0023 |
| attention, a LayerNorm with scale and bias | adaptive | 5.99 m | 0.073 | −0.0046 |
| attention, the LayerNorm of `7a9878f` | adaptive | 6.53 m | 0.071 | −0.0043 |

- The test shows the failure in 80 updates: with no norm and the adaptive lr, `distance` stays at 9.94 m and the catch at 0.006.
- Each of three changes makes the net learn: a constant lr, a layer of 32 with tanh, and a LayerNorm.

At update 80 of the test, on 1000 states that `b087ac` flew:

| network | Perceiver output | shuffle: own block | shuffle: the evader | shuffle: the teammates |
|---|---|---|---|---|
| attention, no norm | 135.8 | 0.002 | 0.002 | 0.347 |
| attention, the LayerNorm of `7a9878f` | 0.63 | 0.277 | 0.393 | 0.401 |

- With the norm the policy reads all three parts of its input.

Eval of the last checkpoint, 4096 episodes, with the 95 % Wilson interval of the catch:

| run | network | seed | `caught_e` | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | mean time to catch | `distance` |
|---|---|---|---|---|---|---|---|---|---|
| `c6de8e` | MLP | 0 | 0.213 [0.201, 0.226] | 0.538 | 0.031 | 0.013 | 0.206 | 4.5 s | 5.25 m |
| `398c2a` | MLP | 1 | 0.252 [0.239, 0.266] | 0.326 | 0.030 | 0.034 | 0.357 | 5.3 s | 4.76 m |
| `91b18c` | attention, norm | 0 | 0.326 [0.312, 0.341] | 0.448 | 0.035 | 0.026 | 0.164 | 4.5 s | 4.77 m |
| `2cdecf` | attention, norm | 1 | 0.267 [0.253, 0.280] | 0.307 | 0.021 | 0.012 | 0.394 | 4.9 s | 4.62 m |

- With the norm the attention net learns: the catch is 0.326 and 0.267.
- It catches more than the MLP in both seeds, by 0.113 and 0.015.

Training, the mean of each range of updates, and of the last 100:

| updates | 1–100 | 101–200 | 201–500 | 501–1000 | 1001–1500 | 1501–1907 | last 100 |
|---|---|---|---|---|---|---|---|
| catch, `91b18c` | 0.091 | 0.157 | 0.245 | 0.319 | 0.338 | 0.337 | 0.338 |
| catch, `2cdecf` | 0.085 | 0.160 | 0.182 | 0.242 | 0.259 | 0.265 | 0.269 |
| `distance`, `91b18c` (m) | 6.51 | 4.92 | 4.38 | 4.16 | 4.10 | 4.11 | 4.12 |
| `distance`, `2cdecf` | 6.58 | 4.98 | 4.70 | 4.25 | 4.19 | 4.16 | 4.15 |
| std, `91b18c` | 0.914 | 0.685 | 0.303 | 0.070 | 0.027 | 0.017 | 0.015 |
| std, `2cdecf` | 0.905 | 0.653 | 0.305 | 0.076 | 0.031 | 0.020 | 0.017 |
| `lr_p`, `91b18c` | 3.0e-3 | 1.4e-3 | 5.5e-4 | 7.8e-5 | 2.6e-5 | 1.5e-5 | 1.4e-5 |
| `lr_p`, `2cdecf` | 2.7e-3 | 1.3e-3 | 5.3e-4 | 9.3e-5 | 3.5e-5 | 2.1e-5 | 1.8e-5 |
| `policy_loss_p`, `91b18c` | −0.0006 | −0.0021 | −0.0040 | −0.0059 | −0.0058 | −0.0056 | −0.0055 |
| `policy_loss_p`, `2cdecf` | −0.0011 | −0.0026 | −0.0034 | −0.0046 | −0.0048 | −0.0048 | −0.0048 |

- The training catch is flat from update 1000 on.
- The lr ends near its floor of 1e-5: 1.4e-5 and 1.8e-5 over the last 100 updates.

Over the checkpoints, on 1000 states that `c6de8e` flew. The shuffle rows are the own block / the evader / the teammates:

| update | 50 | 200 | 500 | 1000 | 1900 |
|---|---|---|---|---|---|
| attention output before the norm, `91b18c` | 15.3 | 30.1 | 35.1 | 35.4 | 35.3 |
| attention output before the norm, `2cdecf` | 15.0 | 27.6 | 32.4 | 32.7 | 32.8 |
| Perceiver output, `91b18c` | 0.58 | 0.47 | 0.45 | 0.45 | 0.45 |
| Perceiver output, `2cdecf` | 0.50 | 0.41 | 0.41 | 0.41 | 0.41 |
| first layer saturated, `91b18c` | 0.878 | 0.900 | 0.904 | 0.904 | 0.904 |
| first layer saturated, `2cdecf` | 0.867 | 0.886 | 0.893 | 0.893 | 0.893 |
| shuffle, `91b18c` | 0.42 / 0.27 / 0.29 | 0.49 / 0.44 / 0.35 | 0.55 / 0.57 / 0.45 | 0.57 / 0.61 / 0.46 | 0.57 / 0.61 / 0.46 |
| shuffle, `2cdecf` | 0.38 / 0.23 / 0.38 | 0.47 / 0.42 / 0.39 | 0.52 / 0.50 / 0.50 | 0.53 / 0.54 / 0.50 | 0.53 / 0.55 / 0.49 |

- The attention output before the norm still grows, from 15 to 33–35. What the trunk gets stays at 0.41–0.58.

One pursuer with the 3v1 weights: the last checkpoints fly in `paths_1v1` with no new training, 4096 episodes:

| run | seed | `caught_e` in 1v1 | `caught_e` in 3v1 | `crash_p` | `OOB_p` | `time_limit` | mean time to catch |
|---|---|---|---|---|---|---|---|
| `91b18c` | 0 | 0.322 [0.308, 0.337] | 0.326 | 0.033 | 0.004 | 0.640 | 5.8 s |
| `2cdecf` | 1 | 0.143 [0.132, 0.154] | 0.267 | 0.001 | 0.001 | 0.855 | 5.8 s |

- The same weights fly one pursuer: `OOB_p` is 0.004 and 0.001.
- Alone, seed 0 catches as often as three pursuers do: 0.322, against 0.326. Seed 1 catches 0.143, against 0.267.

One update takes 1.25 s, against 0.76 s for the MLP. So a run takes 43 min, against 27.
