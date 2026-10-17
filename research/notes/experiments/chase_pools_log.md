# Task 4 — pool runs, run log

What each pool run was and what it scored. The task is in
[task_chase.md](../task_chase.md), and the training in [AMSPBH.md](../AMSPBH.md).

## Pool training results

Every run here trains both sides against pools of frozen opponents
([AMSPBH.md](../AMSPBH.md)), with `generations=13` and `gamma=0.99`. So each run has
14 generations, 0 to 13. `run/matrix.py` flies every pursuer against every evader,
1024 episodes for each pair. It writes the pursuer's win rate to
`runs/<run>/matrix.csv`.

- The **lower triangle** is pursuer g against evaders 0 to g.
- The **older evaders** are evaders 0 to g − 1. Pursuer g trains against all of them.
- The **diagonal** is pursuer g against evader g. Pursuer g never meets this evader in training.
- The **last pair** is `pursuer_13` against `evader_13`.

| run | preset | std reset | `ent_coef` | lr | steps per phase | lower triangle | older evaders | diagonal | last pair |
|---|---|---|---|---|---|---|---|---|---|
| `9e4242` | `default` | yes | 0 | adaptive | 50M | 0.61 | 0.66 | 0.27 | 0.20 |
| `a583c4` | `open` | yes | 0 | adaptive | 50M | 0.38 | 0.42 | 0.16 | 0.13 |
| `d9cc42` | `small` | yes | 0 | adaptive | 50M | 0.74 | 0.78 | 0.47 | 0.44 |
| `b9225b` | `nemecek` | yes | 0 | adaptive | 50M | **0.91** | **0.95** | **0.71** | **0.51** |
| `411ea3`, seed 1 | `nemecek` | yes | 0 | adaptive | 50M | 0.87 | 0.92 | 0.59 | 0.43 |
| `9a36ce` | `nemecek` | yes | 0 | adaptive | 100M | 0.89 | 0.93 | 0.65 | 0.23 |
| `937cda` | `nemecek` | no | 0.001 | adaptive | 50M | 0.71 | 0.75 | 0.45 | 0.19 |
| `cc7173` | `nemecek` | no | 0.001 | fixed 3e-4 | 50M | 0.45 | 0.50 | 0.13 | 0.22 |

Every run is seed 0 unless the row says otherwise. `default` is the 32 × 32 × 16 m
box, and `small` is 24 × 24 × 16 m. `open` is `default` with open walls, as in
[Termination](../task_chase.md#termination). `nemecek` is `open` in a 20 × 20 × 10 m box.

### What the runs show

- **A smaller box helps the pursuer.** The evader must turn more, so it flies slower. Part of this gain is an easier game, not better learning.
- **The std reset matters.** Without it, the pursuer's std falls from 0.59 to 0.03, and the pursuer stops learning after generation 7 (`937cda`).
- **A fixed lr of 3e-4 learns too slowly at the start** (`cc7173`). The adaptive rule picks about 5e-3 in generation 0.
- **Two seeds show the size of luck.** `b9225b` and `411ea3` differ by 0.04 on the lower triangle and by 0.12 on the diagonal. So a smaller gap between two one-seed runs can be luck.
- **The last evaders win.** In the three `nemecek` runs with the std reset, `pursuer_13` beats evaders 0 to 9 in 88–98 % of the episodes. It beats `evader_13` in only 23–51 %. These evaders win mostly by timeout, not by crashes.
- **One pursuer can beat the last evader, but then it forgets the others.** Trained against `evader_13` alone for 52M steps, `pursuer_13` of `411ea3` goes from 0.43 to 0.91 against it. Against `evader_05` to `evader_07`, it falls from 0.93–0.96 to 0.41–0.56.
- **Longer phases do not help.** With 100M steps per phase (`9a36ce`), the triangle numbers stay in the range of the two seeds, and the last pair falls to 0.23. In the last phase, Hedge gave `evader_12` a weight of 0.24. But the pursuer's win against it went only from 0.38 to 0.41.

Roncero names this limit in his §V. Feed-forward MLPs limit "memory and capacity in
the partially observable mixture setting", and "recurrent architectures may improve
both robustness and behavioral diversity". Partially observable means here that the
pursuer cannot see which evader of the pool it flies against.

### Against the papers

How the three setups differ: [task_chase.md](../task_chase.md), Against Roncero and
Němeček.

| | ours | [Roncero v3](../../papers/2506.02849v3.md) | [Němeček](../../papers/F3-BP-2026-Nemecek-Vaclav-bachelor_thesis.md) |
|---|---|---|---|
| older evaders, generations 0 to 7 | 0.93–0.98 | a heatmap, no numbers | 0.97 |
| against the hand-written opponents | 0.94–0.95 | 0.93 | — |

- Němeček's diagonal does not compare with ours, because his pursuer g trains against evader g. So the fair number is the older evaders.
- A catch means something different in each setup. So the win rates compare only roughly.
- Our hand-written number is the mean of three pairs: the pursuer against `paths` and against APF, and the evader against FRPN. Roncero's 0.93 is from his Fig. 5. He does not say which pairs his mean uses.
- Roncero runs 3 seeds. We ran 2 seeds of one setup.

### The opponent mix

The mix of the generation-3 pursuer, ours under Hedge and Němeček's under p_old:

| opponent | ours (`411ea3`) | Němeček |
|---|---|---|
| `paths` / Traj | 18 % | 15 % |
| APF | not in the pool | 15 % |
| evader 0 | 23 % | 15 % |
| evader 1 | 23 % | 15 % |
| evader 2 | 37 % | 15 % |
| evader 3 | not in the pool | 25 % |

## Open

| question | state |
|---|---|
| the last evaders win | try Roncero's layers first, `--set hidden=512,256,256,128`. If that fails, try a recurrent network |
