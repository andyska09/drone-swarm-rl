# Chase — pool runs, run log

What each pool run was and what it scored, and why the setup changed. The task, its pool
training and the hand-written opponents are in [task_chase.md](../tasks/task_chase.md).

## Design decisions

- **Pools replace the duel's swap** ([chase_duel_log.md](chase_duel_log.md)).
- **After a failure, the survivors bootstrap.** A failure used to be terminal for every
  drone. So the pursuer gained when the evader failed, because it stopped paying the
  clock. Now only a catch is terminal ([task_chase.md](../tasks/task_chase.md#episode-end)).
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
([task_chase.md](../tasks/task_chase.md#pools)), with `generations=13` and
`gamma=0.99`. So each run has 14 generations, 0 to 13.

`run/matrix.py` flies every pursuer against every evader and writes
`runs/<run>/matrix.csv`. Each cell is one pair. It holds the share of 1024 episodes that
the pursuer wins. Both sides fly their mean action, on the fixed eval seeds.
[task_chase.md](../tasks/task_chase.md#the-opponent-mix) says what counts as a win.

The last four columns of the table leave out the seeds `frpn` and `paths`, APF and the
cascade. Each column is the plain mean of these cells, for pursuer g against evader e,
with g and e from 0 to 13:

| column | cells | count |
|---|---|---|
| lower triangle | e ≤ g. That is the older evaders plus the diagonal | 105 |
| older evaders | e < g. Pursuer g trains against all of them | 91 |
| diagonal | e = g. Pursuer g never meets this evader in training | 14 |
| last pair | g = e = 13. One cell, not a mean | 1 |

Every cell counts once. So in the lower triangle, `pursuer_13` with its 14 cells weighs
14 times as much as `pursuer_00` with 1. In this section, a range such as 0.88–0.98 is
the lowest and the highest value of a set, not a mean.

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
[Episode end](../tasks/task_chase.md#episode-end). `nemecek` is `open` in a 20 × 20 × 10 m box.

### What the runs show

- **A smaller box helps the pursuer.** The evader must turn more, so it flies slower. Part of this gain is an easier game, not better learning.
- **The std reset matters.** Without it, the pursuer's std falls from 0.59 to 0.03, and the pursuer stops learning after generation 7 (`937cda`).
- **A fixed lr of 3e-4 learns too slowly at the start** (`cc7173`). The adaptive rule picks about 5e-3 in generation 0.
- **Two seeds show the size of luck.** `b9225b` and `411ea3` differ by 0.04 on the lower triangle and by 0.12 on the diagonal. So a smaller gap between two one-seed runs can be luck.
- **The last evaders win.** In the three `nemecek` runs with the std reset, `pursuer_13` wins 0.88–0.98 against evaders 0 to 9 (30 cells, 10 in each run). Against `evader_13` it wins only 0.23–0.51 (the last pair of each run). These evaders win mostly by timeout, not by crashes.
- **One pursuer can beat the last evader, but then it forgets the others.** Trained against `evader_13` alone for 52M steps, `pursuer_13` of `411ea3` goes from 0.43 to 0.91 against it. Against `evader_05` to `evader_07` (3 cells), it falls from 0.93–0.96 to 0.41–0.56.
- **Longer phases do not help.** With 100M steps per phase (`9a36ce`), the lower triangle, the older evaders and the diagonal stay between the two seeds, and the last pair falls to 0.23. In the last phase, Hedge gave `evader_12` a weight of 0.24. But the pursuer's win rate against it went only from 0.38 (Hedge, 600 episodes before the phase) to 0.41 (the matrix cell after it).

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

- Ours is the lowest and the highest of the two seeds, `b9225b` and `411ea3`. Roncero runs 3 seeds.
- The first row is the older-evaders mean cut to generations 0 to 7, so 28 cells.
- The second row is the mean of three win rates of the last generation: `pursuer_13` against `paths`, `pursuer_13` against APF, and `evader_13` against FRPN. Each is the win rate of the trained side. Roncero's 0.93 is from his Fig. 5. He does not say which pairs his mean uses.
- Němeček's diagonal does not compare with ours, because his pursuer g trains against evader g. So the fair number is the older evaders.
- A catch means something different in each setup. So the win rates compare only roughly.

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

## Reasons for the spec, 2026-10-25

These reasons, sources and measurements moved here from the spec and from the two old
notes on pool training and on the hand-written opponents.

### The task

- **`open` has Němeček's arena rules.** Only the ground, or a non-finite state, ends an
  episode. The pursuer has no walls and no ceiling. The evader flies on outside the box
  and pays `outside`.
- **`nemecek` is `open` in Němeček's box**, 20 × 20 × 10 m with the floor at 0, the box of
  his main runs (thesis, Fig. 8.10, and `arena_size_ = 10.0` in his `environment.h`). His
  12 m/s speed cap and his 12 s episodes are not in it.
- **`outside` = 57 is Němeček's ratio.** His evader pays 0.1 per step outside the box
  against a survival pay of 0.007 per step. That is 14.3 times, and 14.3 × our `step` of
  4.0 is 57.
- **The observation differs from Gavin in three places.** `ω` is in it, because our rate
  loop sits under the policy. The walls are 4 rays, because four rays are exact for a
  box. We do not normalize.
- **The net has no thickness.** The drone is a ball of radius `arm_length + prop_radius`
  = 0.40 m, and a ball touches a plane when its centre is within one radius of it. That is
  the whole of `abs(p_x) <= body_radius`. It is the drone's size, not a fudge factor, and
  there is no `capture_dist` field. At 15 m/s the evader moves 0.15 m per policy step,
  well inside 0.40 m, so it cannot skip through between two steps.
- **No pursuer catches itself.** A net hangs 0.9 m below its own drone, which is off its
  own square, so no special case is needed.
- **A capture beats a death on the same step**, because the square reaches into the
  0.8 m hitbox.
- **`collPE` stays soft** because the pursuer must reach the evader to catch it. A rule
  that ends the episode there teaches it to stand off. Gavin's words: the soft penalty
  allows "gradual learning of collision avoidance while maintaining focus on the primary
  task of catching with the net".
- **The weights are Gavin's, per second.** Every per-step weight is Gavin's divided by
  `policy_dt`, so that the code's `× policy_dt` puts it back. Same reward as Gavin, and
  `policy_dt` still makes the weights independent of the control rate.
- **Only a catch is terminal.** Neither side gains when the other fails, which is Gavin's
  rule: "neither the evader nor the pursuers receive a reward when the opponent reaches a
  failure state". Without the bootstrap, the pursuer gains when the evader fails, because
  it stops paying the clock. At γ = 0.99 that is worth about 4, against a catch of 10.
  Roncero also bootstraps past an out-of-bounds (§III-B). A NaN bootstrap value is
  dropped, because a diverged opponent puts NaN into the survivor's observation.
- **A flipped drone keeps flying.** Gavin ends an episode on out of bounds and on
  inter-pursuer contact, and says nothing about attitude, so a pursuer flipped upside down
  at altitude keeps flying.
- **The pursuer's reference point** is the net drop above the evader. Flying its body
  there would hold the evader above the square and never catch.
- **`gamma` = 0.99** is Gavin's value at the same 100 Hz: a 1 s horizon. The default stays
  at 0.998, a 5 s horizon, because `circle_swap` was trained at 0.998.

### Pool training

- **The method is AMSPBH** from [Sánchez Roncero 2026](../../papers/2506.02849v3.md):
  §III-C for the training and the Hedge sampler (eq. 12), §IV-A for the settings. We keep
  to the paper. Where it is silent, we follow Němeček's code,
  `research/code_sources/rl_tracking_eagle-cleaned-up-logic/examples/deep_eagle/examples/train_pipeline.py`.
- **Each phase continues the learner's TrainState.** The paper calls this full-agent
  loading.
- **The std reset is ours.** The std shrinks within a phase, and the learner must explore
  again against the new pool. `reset_std=false` turns it off.
- **The seeds stay in the pools in every generation**, as in the paper: Fig. 6 and
  §IV-C, "heuristic opponents (FRPN, trajectories) retain non-trivial weight even in later
  generations". Fig. 2's `Π0P = {πP0}` reads otherwise, but the results do not.
- **Each member gets at least 3 blocks** with `γh = 0.7` and at most 14 members.
- **A frozen network flies its mean action.** Němeček does the same
  (`deterministic=True`, `train_pipeline.py:228`).
- **The run with `--steps 1.4e9` and `generations=13`** has 14 phases of 190 updates,
  49.8M steps per side each. The few updates left over are not run.

The files of a pool run:

| file | what it holds |
|---|---|
| `metrics.csv` | one row per update. Adds `generation`, `training` (the side that learns) and an `lr` per role. The frozen side's learner stats stay empty: losses, entropy, KL, clip fraction, explained variance |
| `hedge.csv` | one row per generation, side and pool member: the win rate and the weight |
| `pool/<role>_<generation>.pkl` | one policy per side per generation. Loads as a `--seat` source |
| `latest.pkl` | the training state at the last checkpoint |
| `checkpoints/` | numbered policy snapshots, as in every run |

`python run/train.py --resume runs/<run>` continues from the last checkpoint, with the
same pool.

### Baselines

- **The lineup is Roncero v3's** ([2506.02849v3](../../papers/2506.02849v3.md) §IV-B): an
  FRPN pursuer, an APF evader and three path evaders. FRPN and APF use Němeček's code and
  values ([thesis](../../papers/F3-BP-2026-Nemecek-Vaclav-bachelor_thesis.md)). Roncero
  does not publish all of his. Němeček flew our drone: the X500, with the same mass, `kf`,
  `km` and RPM range.
- **A baseline keeps its heading**, and `control.py` is not changed, because it must
  match the C++ goldens.
- **Němeček's limits sit in the law, not in the physics.** His simulator cuts every
  drone's speed to 12 m/s after each step (`mrs_uav.cpp:89-98`). We cannot do that,
  because the network flies the same physics.
- **The cascade is the easy opponent.** It keeps MRS's defaults.
- **FRPN values:** Němeček, Table 8.1. He tuned them under 12 m/s and 20 m/s² (§8.1), so
  the speed limit applies both. Roncero's FRPN is different: its output is a velocity
  command, clipped to the top speed, with G = 5.0 and W = 0.20, and a cascaded PID flies
  it. Without the limits our FRPN leaves the arena: it reaches about 19 m/s.
- **APF values:** Němeček's eval set (`APF_PARAMS` in `eval_rl_pur.py`, Table 8.6). All six
  faces closer than `r_w` push, as in his code. His thesis text says that only the nearest
  face on each axis pushes. APF adds up the push of every pursuer, so it works for N v 1.
  Roncero's APF is different: `k_P = k_W = 1.0`, `α_P = 3`, `α_W = 2`, and a velocity
  controller tracks the force. APF never hit a wall, and its top speed was about 9.7 m/s.
  The limits change almost nothing for it.

The paths beside Roncero's:

| | Roncero v3 | ours |
|---|---|---|
| arena | 4 × 5 × 2 m | 32 × 32 × 16 m |
| circle radius | 0.6–1.0 m | 4.8–8.0 m (× 8) |
| figure-eight radius | 0.45–0.9 m | 3.6–7.2 m (× 8) |
| speed | 1.0–2.0 m/s | 2.83–5.66 m/s (× √8) |
| ellipse | "randomized elliptical deformation" | each axis × 0.85–1.0, from Roncero v2 |
| height | 0.5–1.9 m | random, 2 m clear of the floor and the ceiling |
| start point | "random phase" | random |
| direction | not given | random, half of them clockwise |
| hover | holds its start point | holds a random point |

- The sizes scale by 8 and the speeds by √8. That keeps each shape's turn acceleration,
  `v²/r`. The figure-eight is Gerono's: `(A sin θ, B/2 sin 2θ)`.
- The point turns at a steady rate, like Roncero v2's angular velocity, so the speed along
  the path changes. Over the 1024 eval paths, a circle changes a little, because the
  ellipse is mild: 2.67–6.06 m/s. A figure-eight is fastest where its loops cross:
  1.79–8.00 m/s. The peak acceleration of a circle is 3.1 m/s² (median), and 7.2 m/s² at
  most. For a figure-eight it is 7.7 m/s² (median), and 20.0 m/s² at most. The drone
  gives about 31 m/s² sideways, so it can fly all of them.
- **The path picks the start.** The env spawns the evader at a random spot. Near a corner,
  no full-size path through that spot fits, and a shrunk path keeps its lap time, so it
  is also slower. Němeček does the same: his evader starts at row 0 of its trajectory file
  (`target_uav.cpp:57`).
- **Němeček's trajectories.** He does not make shapes. He loads 200 recorded trajectories
  from CSV files, in two profiles of 100 each (thesis §3.1, Table 3.1). At each step the
  evader's state is set to the next row, so it is replayed, not flown
  (`environment.cpp:184-198`). In this mode the evader cannot leave the arena
  (`environment.cpp:217`). We fly Roncero's shapes instead. From Němeček we take only one
  thing: the path decides where the evader starts.
- **Not copied.** Roncero v2 also turns each path to a random heading and tilts it by up
  to 0.05π. v3 does not say this, so we do not do it. Roncero's code is not public.
  `KTH-RPL/AMSPB_PEG`, the repo that his paper page links, gave 404 on 2026-10-10.

Sources:

- Roncero v3: [2506.02849v3](../../papers/2506.02849v3.md) §IV-A, §IV-B. Roncero v2, for
  the path numbers: <https://arxiv.org/html/2506.02849v2>, §V-B.
- Němeček: [thesis](../../papers/F3-BP-2026-Nemecek-Vaclav-bachelor_thesis.md) §3.1, §3.2,
  §8.1, Tables 8.1 and 8.6.
- Němeček's code, under
  `research/code_sources/rl_tracking_eagle-cleaned-up-logic/examples/deep_eagle/`:
  `examples/agents.py` (`OriginalFRPN`, `AdvancedAPFEvader`), `examples/eval_rl_pur.py`
  (`APF_PARAMS`), `src/agent/mrs_uav.cpp`, `src/agent/target_uav.cpp`,
  `src/env/environment.cpp`.

## Reasons for the spec, 2026-10-27

- **Both sides learn at once.** Each side trains only against the other side's pool up
  to generation k − 1, so neither side needs the other's new weights. The phases go from
  28 to 14, and each side still gets 49.8M steps per phase. The runs before 2026-10-27
  used the old schedule: the sides learned in turn, the evader first.
- **One outcome for any N**, in the order caught, fail_both, pursuer failure, evader
  failure, time_limit. A pursuer failure is a crash, an OOB or a collPP, so a collPP is
  an evader win.
- **The start rule is our own: 2 m apart.** A drone touches a net only within 1.54 m of
  the pursuer's centre: the farthest touching centre is at (0.40, 0.5, 1.4) m in the net
  frame. 2 m is above both that and the 0.8 m contact distance, so no catch and no
  collision can happen at step 0. The paths evader is placed first, so its path keeps
  its full size, and then the pursuers are drawn again.

## Open

| question | state |
|---|---|
| the last evaders win | try Roncero's layers first, `--set hidden=512,256,256,128`. If that fails, try a recurrent network |
