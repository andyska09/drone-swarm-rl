# Chase — pool runs, run log

What each pool run was and what it scored, and why the setup changed. The task is in
[task_chase.md](../task_chase.md), the training in
[task_chase_training.md](../task_chase_training.md), and the hand-written opponents in
[task_chase_baselines.md](../task_chase_baselines.md).

## Design decisions

- **Pools replace the duel's swap** ([chase_duel_log.md](chase_duel_log.md)).
- **After a failure, the survivors bootstrap.** A failure used to be terminal for every
  drone. So the pursuer gained when the evader failed, because it stopped paying the
  clock. Now only a catch is terminal ([task_chase.md](../task_chase.md#termination)).
- **The adaptive lr replaced `anneal_lr`.**
- **One learned std per action, with a clip.** It replaced Gavin's per-state std with a
  `tanh` squash after the pool run `chase_default_s0_20261011-103639_797201`. There the
  pursuer's roll/pitch std fell from 0.43/0.33 to 0.06/0.16, and from generation 6 no
  pursuer beat a new evader. A new std head on `pursuer_13` lifted its win rate against
  `evader_07` from 0.20 to 0.95 in 50 updates.
- **`ent_coef` is 0.001, not Gavin's 0.01.** Of fourteen `chase` runs, the two with
  `ent_coef = 0.01` are the only two with a broken std: 112 to 60784 under the clip, 371
  to 942 under the squash. The other twelve all end between 0.02 and 0.51, and the two
  runs that solve `chase default` (`caught` 0.996 and 1.000) end at 0.08 to 0.13. The
  squash run at 0.01 ended at `distance` 11.40 m, where an untrained policy starts,
  against 0.5 to 4.0 m for every trained one. Why: [05-RL.md](../learning/05-RL.md).
- **The path picks where the evader starts.** The first version put the path around the
  evader's random spawn. Near a corner no full-size path fits there, so the path shrank:
  15% of circles ended up below half size. A shrunk path kept its lap time, so it was
  also slower.
- **FRPN and APF fly under speed limits.** Without them FRPN reached about 19 m/s.
  Against a hover evader it then caught 0.59 and left the arena 0.39. This was measured
  before the path change.
- **The `small` preset.** The saved policies of
  `chase_default_s0_20261014-100847_9e4242`, put in a 24 × 24 × 16 m box, caught the
  newest evader in 0.39 of the episodes, against 0.17 in 32 × 32 m. The evader left the
  box no more often (0.03), and the pursuer left it twice as often (0.16, against 0.08).

## Pool training results

Every run here trains both sides against pools of frozen opponents
([task_chase_training.md](../task_chase_training.md)), with `generations=13` and
`gamma=0.99`. So each run has 14 generations, 0 to 13. `run/matrix.py` flies every
pursuer against every evader, 1024 episodes for each pair. It writes the pursuer's win
rate to `runs/<run>/matrix.csv`.

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

How the three setups differ: [comparisons.md](../literature/comparisons.md), Chase.

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

## Baselines against each other

`runs/chase_baselines/` (gitignored) holds every hand-written pursuer against every
hand-written evader: 1024 episodes each, `EVAL_SEED = 1_000_000`, made after the path
change. It was made at commit `06eac95`, in the `default` preset, when `hover`, `circle`
and `figure8` were still sources of their own. `guidance.py` has changed only a name since
then. `scratch/baseline_evals.py` rebuilds it in about a minute, but it still names the
three shapes as sources, so it needs an update before it runs again. Rebuild it after
any change to `guidance.py`.

| pursuer | evader | caught | crash P | OOB P | timeout | time to catch |
|---|---|---|---|---|---|---|
| frpn | hover | 0.768 | 0.007 | 0.225 | 0.001 | 2.64 s |
| frpn | circle | 0.855 | 0.013 | 0.114 | 0.018 | 3.51 s |
| frpn | figure8 | 0.677 | 0.015 | 0.175 | 0.134 | 4.03 s |
| frpn | apf | 0.794 | 0 | 0.062 | 0.144 | 5.23 s |
| frpn | cascade | 0.762 | 0.017 | 0.216 | 0.006 | 3.16 s |
| cascade | hover | 1.000 | 0 | 0 | 0 | 3.91 s |
| cascade | circle | 0.100 | 0 | 0 | 0.900 | 2.95 s |
| cascade | figure8 | 0.306 | 0 | 0 | 0.694 | 5.45 s |
| cascade | apf | 0.421 | 0 | 0 | 0.579 | 4.94 s |
| cascade | cascade | 0.997 | 0 | 0 | 0.003 | 4.83 s |

No evader crashed or left the arena, and no `fail_both` happened, so the table leaves out
those columns.

- **FRPN catches 68–86% of every evader.** It loses by leaving the arena (6–23%), and
  against the figure-eight and APF also by timeout (13–14%). FRPN knows nothing about
  walls, and our walls end the episode. In Němeček's env only the ground ends it for the
  pursuer (`environment.cpp:242`).
- **The figure-eight is the hardest evader for FRPN to catch** (68%). Against APF, FRPN
  takes the longest: 5.2 s on average.
- **The cascade pursuer catches a target that stands still or moves slowly** (hover 100%,
  cascade evader 99.7%). It fails against a moving path (circle 10%), because it chases
  from behind.

## Open

| question | state |
|---|---|
| the last evaders win | try Roncero's layers first, `--set hidden=512,256,256,128`. If that fails, try a recurrent network |
