# 02_pools

**Question:** Which box, and which std reset, lr and phase length, give the pursuer the highest win rate against the evaders of its own pool run?
**Differs from** the code of today: the sides learn in turn, the evader first, so a run has 28 phases. The reward is the one before the progress reward. Settings: `gamma=0.99`, `generations=13`, 1.4e9 steps, seed 0. The table names what else differs for each run.
**Runs:** 8 runs, 2026-10-14 to 2026-10-17, one commit for each run.

- The run before these, `797201` (commit `dfc87ee`), had a std for each state and a tanh squash. The pursuer's roll and pitch std fell from 0.43 and 0.33 to 0.06 and 0.16, and from generation 6 no pursuer beat a new evader. A new std head on `pursuer_13` lifted its win rate against `evader_07` from 0.20 to 0.95 in 50 updates.
- So these runs have one learned std per action with a clip, and most of them set the std back to 1.0 at each phase.

## Results

`run/matrix.py` flies every pursuer against every evader of the run: 1024 episodes for each pair, mean actions. A cell is the pursuer's win rate. Each column is the mean of these cells, for pursuer g against evader e:

| column | cells | count |
|---|---|---|
| lower triangle | e ≤ g | 105 |
| older evaders | e < g. Pursuer g trains against all of them | 91 |
| diagonal | e = g. Pursuer g never meets this evader in training | 14 |
| last pair | g = e = 13 | 1 |

| run | commit | preset | std reset | `ent_coef` | lr | steps per phase | lower triangle | older evaders | diagonal | last pair |
|---|---|---|---|---|---|---|---|---|---|---|
| `9e4242` | `b2b9b11` | `default` | yes | 0 | adaptive | 50M | 0.61 | 0.66 | 0.27 | 0.20 |
| `a583c4` | `368de74` | `open` | yes | 0 | adaptive | 50M | 0.38 | 0.42 | 0.16 | 0.13 |
| `d9cc42` | `4bf19c9` | `small` | yes | 0 | adaptive | 50M | 0.74 | 0.78 | 0.47 | 0.44 |
| `b9225b` | `7c241a2` | `nemecek` | yes | 0 | adaptive | 50M | 0.91 | 0.95 | 0.71 | 0.51 |
| `411ea3`, seed 1 | `456e45a` | `nemecek` | yes | 0 | adaptive | 50M | 0.87 | 0.92 | 0.59 | 0.43 |
| `9a36ce` | `456e45a` | `nemecek` | yes | 0 | adaptive | 100M | 0.89 | 0.93 | 0.65 | 0.23 |
| `937cda` | `15273f3` | `nemecek` | no | 0.001 | adaptive | 50M | 0.71 | 0.75 | 0.45 | 0.19 |
| `cc7173` | `456e45a` | `nemecek` | no | 0.001 | fixed 3e-4 | 50M | 0.45 | 0.50 | 0.13 | 0.22 |

- A smaller box gives the pursuer more wins: the lower triangle is 0.61 in `default` against 0.74 in `small`, and 0.38 in `open` against 0.91 in `nemecek`.
- In `nemecek` with 50M steps per phase, the lower triangle is 0.87 and 0.91 with the std reset, 0.71 without it (`937cda`), and 0.45 without it and with a fixed lr (`cc7173`).
- The two seeds differ by 0.04 on the lower triangle and by 0.12 on the diagonal. With 100M steps per phase, the lower triangle (0.89) lies between the two seeds, and the last pair is 0.23.

The win rate of `pursuer_13` against single evaders of its own run:

| `pursuer_13` of | evaders 0 to 9 | `evader_05` to `evader_07` | `evader_13` |
|---|---|---|---|
| `b9225b`, `411ea3`, `9a36ce` | 0.88–0.98 | — | 0.51, 0.43, 0.23 |
| `411ea3` | — | 0.93–0.96 | 0.43 |
| `411ea3`, after 52M more steps against `evader_13` alone | — | 0.41–0.56 | 0.91 |

- The last pursuer wins 0.88–0.98 against the evaders 0 to 9, and 0.23–0.51 against the last evader.
- Trained against the last evader alone, it wins 0.91 against it, and it falls to 0.41–0.56 against `evader_05` to `evader_07`.

Against the papers. Ours is the lowest and the highest of `b9225b` and `411ea3`:

| | ours | [Roncero v3](../../../papers/2506.02849v3.md) | [Němeček](../../../papers/F3-BP-2026-Nemecek-Vaclav-bachelor_thesis.md) |
|---|---|---|---|
| older evaders, generations 0 to 7 (28 cells) | 0.93–0.98 | a heatmap, no numbers | 0.97 |
| against the hand-written opponents | 0.94–0.95 | 0.93 | — |

- The second row is the mean of three win rates of generation 13: `pursuer_13` against paths, `pursuer_13` against APF, and `evader_13` against FRPN. Roncero's 0.93 is from his Fig. 5.
- A catch means something different in each setup, so the win rates compare only roughly.

The opponent mix of the generation-3 pursuer:

| opponent | ours (`411ea3`) | Němeček |
|---|---|---|
| paths | 18 % | 15 % |
| APF | not in the pool | 15 % |
| evader 0 | 23 % | 15 % |
| evader 1 | 23 % | 15 % |
| evader 2 | 37 % | 15 % |
| evader 3 | not in the pool | 25 % |
