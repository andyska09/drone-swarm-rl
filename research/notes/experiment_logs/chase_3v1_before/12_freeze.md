# 12_freeze

**Question:** Why does the catch of three pursuers stop rising, while FRPN catches twice as much?
**Differs from** `10_long`: no training. These are checks on the runs of `09_critic_input`, `10_long` and `11_gamma`.
**Runs:** flights and probes only, 2026-11-05, by scripts in `scratch/freeze/`. The 5e9 runs flew on a copy of their own commit, `2d15eb5`.

## Results

The std and the training catch over the updates. Each cell is the mean of the updates within 10 of the column. `b4511f` is `paths_1v1` seed 0 of `04_paths_sense` in `chase_1v1`, which has 762 updates.

| update | 100 | 190 | 400 | 762 | 1000 | 3000 | 10000 | 19000 |
|---|---|---|---|---|---|---|---|---|
| std, `76ea05` | 1.01 | 0.92 | 0.42 | 0.14 | 0.10 | 0.029 | 0.008 | 0.006 |
| std, `4e80e4` | 0.97 | 0.81 | 0.31 | 0.11 | 0.070 | 0.016 | 0.006 | 0.006 |
| std, `b4511f` | 0.79 | 0.58 | 0.33 | 0.18 | — | — | — | — |
| `caught_e`, `76ea05` | 0.05 | 0.08 | 0.17 | 0.11 | 0.11 | 0.16 | 0.17 | 0.18 |
| `caught_e`, `4e80e4` | 0.05 | 0.13 | 0.20 | 0.25 | 0.25 | 0.25 | 0.26 | 0.26 |
| `caught_e`, `b4511f` | 0.14 | 0.20 | 0.79 | 0.96 | — | — | — | — |

- The std falls as about c / update, with c about 50–100, from update 1000 until the lr reaches its floor.
- The lr follows the std: lr / std stays at 1.1–1.7e-3 from update 190 until the lr reaches its floor. With one pursuer it is 1.4–2.7e-3.
- One pursuer has a similar curve, but it learned in time: its catch rose to 0.79 by update 400.

What the pursuers of the 5e9 runs do, last checkpoint, 1024 eval episodes, mean actions:

| what | value |
|---|---|
| the net distances after 6 s, nearest first, medians | 2.6–3.0, 4.2–4.6 and 5.7–6.3 m |
| the closing speed of the nearest pursuer, from 5 s on | 0.25 m/s or less |
| the distance between the two closest pursuers, median | 4.1–4.8 m |
| the nearest net from 4 to 9 s, with the second net 7–10 m away | 3.6 m (`4e80e4`), 4.3 m (`76ea05`) |
| the nearest net from 4 to 9 s, with the second net 3–4 m away | 2.6 m |

- The pursuers fly in a formation behind the evader, and the nearest does not go in.

From the same eval episodes, the pursuers fly on from 4 s with the action (1 − β) × learned + β × FRPN. "Nearest only" changes the pursuer whose net is nearest at 4 s. The score is the discounted return from 4 s, the mean of the three pursuers, at γ = 0.99. Only the episodes count in which every drone flies and the nearest net is 2–6 m away at 4 s: 779 and 773 of 1024. Each cell is `4e80e4`, then `76ea05`.

| β | nearest only: score | nearest only: catch | all three: score | all three: catch | all three: `coll_pp` |
|---|---|---|---|---|---|
| 0 (learned) | −4.33, −4.35 | 0.21, 0.13 | −4.33, −4.35 | 0.21, 0.13 | 0.04, 0.03 |
| 0.10 | −4.25, −4.30 | 0.24, 0.16 | −4.22, −4.29 | 0.27, 0.20 | 0.03, 0.04 |
| 0.25 | −4.13, −4.24 | 0.33, 0.19 | −4.02, −4.22 | 0.41, 0.29 | 0.10, 0.10 |
| 0.50 | −3.57, −3.96 | 0.62, 0.36 | −3.97, −4.25 | 0.56, 0.40 | 0.40, 0.41 |
| 0.75 | −2.70, −2.97 | 0.80, 0.70 | −5.16, −5.27 | 0.45, 0.41 | 0.53, 0.59 |
| 1 (FRPN) | −2.32, −2.15 | 0.84, 0.81 | −5.63, −6.36 | 0.44, 0.38 | 0.53, 0.60 |

- A step toward FRPN raises the score: at β = 0.25 for all three, from −4.33 to −4.02 and from −4.35 to −4.22. So the learned policy is not at a top.
- The score of the nearest alone rises up to β = 1. Its catch rises from 0.21 to 0.84, and from 0.13 to 0.81.
- For all three, the score is highest at β = 0.25–0.5. `coll_pp` jumps from 0.10 to 0.40 between β = 0.25 and 0.50.

The score of all three at β = 1, at other settings, from the same flights:

| setting | learned | all three on FRPN |
|---|---|---|
| γ = 0.99, `coll_pp` cost 30 | −4.33, −4.35 | −5.63, −6.36 |
| γ = 0.99, `coll_pp` cost 10 | −4.27, −4.33 | −3.05, −3.35 |
| γ = 0.995, `coll_pp` cost 30 | −8.22, −8.45 | −8.69, −9.92 |
| γ = 0.998, `coll_pp` cost 30 | −19.00, −20.23 | −13.16, −15.02 |

- A `coll_pp` cost of 10, or γ = 0.998, puts all three on FRPN above the learned policy. γ = 0.995 does not.

The gradient of the loss on `log_std`, for each epoch. `6fce56` was restored at update 1907, at a std of 0.03, and ran 4 updates. Each cell is the mean over the 4 actions and the 4 updates.

| epoch | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| gradient of the loss on `log_std` | −0.00089 | +0.00050 | +0.00099 | +0.00140 |

- A positive gradient lowers the std. The entropy weight alone gives −0.001.
- In epoch 1, on new samples, the reward gives almost no pull on the std. The pull down comes in epochs 2–4, which use the same samples again.
