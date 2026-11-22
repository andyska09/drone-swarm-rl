# 21_attention

**Question:** Do three pursuers catch as much with the attention net as with the MLP?
**Differs from** `15_progress`: `net=attention`, after the encoder of Geles et al. 2026. The size of its input does not depend on the number of teammates. It reads its inputs unscaled, so it needs `normalize_obs=false`.
**Runs:** MetaCentrum, 2 runs, 2026-11-10, commit `79e1f0e`: `b087ac` (seed 0), `da9e33` (seed 1). The MLP runs are `c6de8e` and `398c2a` of `15_progress`.

| part of the input | how the network reads it | size |
|---|---|---|
| own block | straight into the trunk | 20 |
| the evader: offset and velocity | straight into the trunk | 6 |
| each teammate: offset and velocity | a Perceiver: a layer of 128 with tanh on each teammate, then 4 learned queries that attend to the teammates with 4 heads of 32 | 4 × 128 = 512 |

- The trunk is 2 × 256 with tanh, as in the MLP, and it reads 538 values. The MLP reads 44 values.
- The critic has the same shape. Each part also holds the world state of its drone, and the part of the evader holds its action.

## Results

Eval of the last checkpoint, 4096 episodes, with the 95 % Wilson interval of the catch:

| run | network | seed | `caught_e` | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | `distance` |
|---|---|---|---|---|---|---|---|---|
| `c6de8e` | MLP | 0 | 0.213 [0.201, 0.226] | 0.538 | 0.031 | 0.013 | 0.206 | 5.25 m |
| `398c2a` | MLP | 1 | 0.252 [0.239, 0.266] | 0.326 | 0.030 | 0.034 | 0.357 | 4.76 m |
| `b087ac` | attention | 0 | 0.008 [0.005, 0.011] | 0.058 | 0.153 | 0.781 | 0.000 | 10.04 m |
| `da9e33` | attention | 1 | 0.009 [0.007, 0.013] | 0.088 | 0.074 | 0.822 | 0.007 | 10.21 m |

- The attention runs do not learn: the catch is 0.008 and 0.009, against 0.213 and 0.252.
- A pursuer leaves the box in 0.781 and 0.822 of the episodes.

Training, the mean of each range of updates:

| updates | 1–100 | 101–200 | 201–500 | 501–1000 | 1001–1500 | 1501–1907 |
|---|---|---|---|---|---|---|
| catch, `c6de8e` | 0.094 | 0.156 | 0.201 | 0.214 | 0.206 | 0.205 |
| catch, `398c2a` | 0.093 | 0.176 | 0.224 | 0.251 | 0.258 | 0.258 |
| catch, `b087ac` | 0.006 | 0.008 | 0.007 | 0.009 | 0.009 | 0.009 |
| catch, `da9e33` | 0.007 | 0.008 | 0.008 | 0.009 | 0.009 | 0.010 |
| `distance`, `c6de8e` (m) | 6.57 | 5.13 | 4.75 | 4.67 | 4.70 | 4.70 |
| `distance`, `398c2a` | 6.52 | 5.04 | 4.54 | 4.37 | 4.33 | 4.33 |
| `distance`, `b087ac` | 9.92 | 9.99 | 9.91 | 9.96 | 10.06 | 9.98 |
| `distance`, `da9e33` | 9.97 | 10.03 | 10.02 | 10.04 | 9.99 | 10.13 |
| `OOB_p`, `c6de8e` | 0.179 | 0.109 | 0.063 | 0.023 | 0.016 | 0.016 |
| `OOB_p`, `398c2a` | 0.182 | 0.054 | 0.054 | 0.031 | 0.031 | 0.031 |
| `OOB_p`, `b087ac` | 0.563 | 0.571 | 0.685 | 0.621 | 0.712 | 0.766 |
| `OOB_p`, `da9e33` | 0.534 | 0.646 | 0.642 | 0.693 | 0.698 | 0.708 |
| std, `c6de8e` | 0.920 | 0.751 | 0.358 | 0.086 | 0.035 | 0.023 |
| std, `398c2a` | 0.919 | 0.715 | 0.326 | 0.080 | 0.032 | 0.021 |
| std, `b087ac` | 0.909 | 0.744 | 0.558 | 0.344 | 0.201 | 0.120 |
| std, `da9e33` | 0.912 | 0.754 | 0.546 | 0.326 | 0.178 | 0.120 |
| `lr_p`, `c6de8e` | 3.3e-3 | 1.9e-3 | 6.9e-4 | 1.9e-4 | 8.4e-5 | 5.3e-5 |
| `lr_p`, `398c2a` | 3.3e-3 | 1.9e-3 | 7.0e-4 | 1.8e-4 | 7.0e-5 | 4.4e-5 |
| `lr_p`, `b087ac` | 5.1e-3 | 3.2e-3 | 2.5e-3 | 2.0e-3 | 1.7e-3 | 1.1e-3 |
| `lr_p`, `da9e33` | 4.4e-3 | 3.2e-3 | 2.7e-3 | 2.5e-3 | 1.7e-3 | 1.2e-3 |
| `policy_loss_p`, `c6de8e` | −0.0016 | −0.0033 | −0.0045 | −0.0045 | −0.0046 | −0.0045 |
| `policy_loss_p`, `398c2a` | −0.0019 | −0.0032 | −0.0038 | −0.0039 | −0.0039 | −0.0039 |
| `policy_loss_p`, `b087ac` | +0.0040 | +0.0043 | +0.0047 | +0.0049 | +0.0040 | +0.0046 |
| `policy_loss_p`, `da9e33` | +0.0038 | +0.0045 | +0.0050 | +0.0043 | +0.0050 | +0.0052 |

- `distance` stays at 9.9–10.1 m in every range of updates. In the MLP runs it falls to 4.3–4.7 m.
- `policy_loss_p` is above zero in every range, +0.0038 to +0.0052. In the MLP runs it is below zero.

The flights of the pursuers, 512 eval episodes of each run. The bearing row is the R² of the roll and pitch rate commands of the first 0.3 s against the direction to the evader:

| | `b087ac` | `da9e33` | `c6de8e` | `398c2a` |
|---|---|---|---|---|
| yaw rate command, mean ± std | −0.66 ± 0.03 | −0.65 ± 0.14 | −0.41 ± 0.65 | +0.30 ± 0.57 |
| change of the action from step to step | 0.001–0.004 | 0.002–0.004 | 0.03–0.06 | 0.03–0.05 |
| the command follows the bearing, roll / pitch | 0.14 / 0.15 | 0.01 / 0.23 | 0.65 / 0.49 | 0.46 / 0.54 |
| episodes that end with a pursuer out of the arena | 398 | 410 | 6 | 17 |
| the exit: median time | 4.1 s | 5.0 s | — | — |
| the exit: median distance to the evader | 19.0 m | 19.9 m | — | — |
| the exit: the pursuer flies away from the evader | 92 % | 94 % | — | — |

- The pursuers give almost one action: it changes by 0.001–0.004 from step to step, against 0.03–0.06.
- At its exit a pursuer is 19–20 m from the evader, and in 92–94 % of the exits it flies away from it.

The size of the Perceiver output over the updates of `b087ac`: its std over 1000 states that `b087ac` flew, the mean over the 512 outputs. In `da9e33` it is 2890 at update 1900:

| update | start | 50 | 500 | 1000 | 1900 |
|---|---|---|---|---|---|
| Perceiver output | 0.59 | 170 | 1356 | 2061 | 2022 |

The first layer of the trunk in `b087ac` at the end, by the part of the input. The sum is the std of what the part adds to one unit before the tanh. The shuffle is the RMS change of the mean action when the part is shuffled across the states:

| part of the input | its sum in a unit of the first layer | the action moves on a shuffle |
|---|---|---|
| own block | 2.8 | 0.000 |
| the evader | 2.7 | 0.000 |
| the teammates, through the Perceiver | 4.6e5 | 0.112 |

- The Perceiver output grows from 0.59 to about 2000. From update 50 on, every unit of the first layer is saturated.
- The policy reads neither its own block nor the evader: a shuffle of either moves the action by 0.000.

Other numbers:

| what | value |
|---|---|
| the adaptive lr | reaches its cap of 1e-2 in updates 1–2 |
| the Perceiver output over 12 updates from the start, with a constant lr of 2e-4, 1024 envs | stays at 0.62–0.69 |
| the same with the adaptive lr | goes from 0.62 to 2.4 |
| one update | 3.41 s |
