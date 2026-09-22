# Task 4 — duel

`chase` with nothing scripted. Both sides learn, one at a time, in one run.

Task 3 stands at `caught 0.999`, `1.34 s` to catch, `0.42 m` off the net centre.
Read [task_chase.md](task_chase.md) first — everything there still holds.

## What is new

| | |
|---|---|
| obs | four wall distances in `own` |
| preset | `duel`, nothing scripted |
| trainer | one role trains at a time, the swap is a score threshold |
| eval | `--seat ROLE SOURCE` replaces `--preset` and `--policy` |

## Walls in the observation

The evader runs from the pursuer and cannot see a wall coming. `own` carries
height and nothing else about the arena.

Four distances from the drone to the arena bounds, along the **level body
frame**: nose, tail, left, right. Raw metres. No clip, no scaling — nothing else
in the obs is scaled either.

Level frame: flatten `R[:, 0]` onto the horizontal plane and normalize. If its
length is near zero the nose points at the ceiling and has no compass direction,
so use the flattened `R[:, 1]` instead — `R` is orthonormal, so that axis is flat
whenever the nose is not. A `jnp.where`, no Python branch, no NaN.

Roll and pitch never turn the rays. Yaw does, like every other block.

It lives in `core.own_obs`, so `own` goes 16 → 20 for every task. Consequences:

- `tests/test_env.py` asserts 16 in two places
- every existing checkpoint stops loading, task 3's included
- `a_to_b` and `circle_swap` carry four numbers they do not use

The arena is a box, so four rays are exact. Rays only earn more than that with
obstacles.

## Preset

```python
"duel": EnvParams(scripted=()),
```

The evader's waypoint stays. `reference` still returns it, which is what feeds
the goal marker and a cascade in the evader's seat. The viewer already draws no
marker for it — `header.json` carries `goal` per drone, false for both here.

## ρ, the score the swap reads

Gavin's performance metric, one number per episode, 0 to 1, the pursuer's side.

| episode ended with | ρP |
|---|---|
| a catch at step `t` | `1 - t / max_steps` |
| the evader alive at the step limit | 0 |
| the pursuer crashed | 0 |
| the evader crashed | 1 |

`ρE = 1 - ρP`. It separates a catch at 1.3 s from a catch at 9 s; `caught` does
not, and that gap is what improves late in training. Goes in `info` as `rho`.

Anchors. Both papers right-censor time-to-catch — a crash or a timeout is written
down as the full 10 s ([Nv1](../papers/2607.05939v1.md) §V-A,
[1v1](../papers/2603.16279v1.md) §V-B). So a published table gives ρP directly:
`ρP = (1 - mean_time / 10) + evader_crash_alone`.

| setup | source | time | evader crash | ρP |
|---|---|---|---|---|
| task 3, our pursuer against the scripted evader | ours | 1.34 s | — | **0.87** |
| 1v1, 8 × 8 × 5 arena, DRL vs DRL | 1v1 Table III | 3.78 s | 4.1% | **0.66** |
| 3v1, evader boxed in 12 × 12 × 6, PFSP-CTBR vs PFSP | Nv1 Table II | 3.71 s | 0.3% | **0.63** |
| 1v1, 40 × 40 × 14 arena, DRL vs DRL | 1v1 Table III | 6.62 s | 1.5% | **0.35** |

**The last row is the only published 1v1 in a big arena, and it is the nearest
match to `duel`.** Every setup that passes 0.6 either adds pursuers or shrinks the
space the evader may use.

**Read `ρP` against theirs only by its catch part.** Their evaders crash in 0.3% to
4% of episodes, so their `ρP` is almost all catches. Ours crashes in 33%, so a
third of our `ρP` is the crash branch. At update 404 we read `ρP = 0.495`, of which
0.137 is catches — against 0.34 of catches for the 1v1 big-arena row. The headline
number flatters us by about a third.

Trap: `7.36 s` in Nv1 Table II is the **APF pursuer against the APF evader** — two
heuristics, not the learned pursuers. The first version of this note read that cell
and set `swap_hi` from it.

## The swap

One `TrainState` per role already exists. The frozen role is held by selecting
between the stepped and the unstepped state after `apply_gradients` — params,
Adam moments and the step count all stay put. One compiled update.

- both roles start random, the pursuer trains first
- the pursuer's turn ends when `ρP > swap_hi`
- the evader's turn ends when `ρP < swap_lo`
- neither turn ends before `swap_min` updates
- the frozen side keeps sampling its actions, so it is not one fixed behaviour

`TrainConfig`: `swap_min`, `swap_hi = 0.6`, `swap_lo = 0.3`. The swap is off at
`swap_min = 0`, the default, so every other task runs unchanged. The turn order is
`train_roles`, or every learned role in `roles` order when it is empty — and `rho`
scores the first of them, which is why the pursuer is named first.

**No cap on a phase.** Gavin does not publish one and a stuck phase should be
visible, not hidden. `metrics.csv` gains a `training` column; a stuck phase reads
as one long flat block.

Every swap writes a checkpoint, as Gavin freezes a snapshot at every swap. That
is also the start of a policy pool, if one is ever needed.

## Eval

`--preset` and `--policy` are gone, and so is the `three` preset. One flag
replaces both:

```bash
python run/eval.py runs/<duel> --seat pursuer cascade
```

Two arguments, role then who flies it. Repeatable. The source is `cascade`, a run
directory, or a `.pkl`. Any seat nobody names flies this run's own checkpoint,
which is what a bare `run/eval.py runs/<duel>` does. The output lands in
`evals/pursuer_cascade/`.

The cascade is the fixed opponent for both gates. In the duel, `reference`
returns the evader's waypoint, so a cascade evader flies exactly what the
scripted evader of task 3 flew.

## The gate

| what | command | pass |
|---|---|---|
| the evader against a fixed pursuer | `--seat pursuer cascade` | `caught < 0.5` |
| the pursuer against a fixed evader | `--seat evader cascade` | `caught > 0.9` |
| no win by crashing | either | `evader_crashed < 0.05` |
| it converges, not cycles | `metrics.csv` | the swing in `ρP` narrows across phases |

The cascade caught the scripted evader 0.996 of the time, and task 3's pursuer
caught it 0.999 of the time. Those are the numbers the first two rows move away
from.

## The run

```bash
conda run -n drone-swarm python run/train.py --task chase --preset duel \
    --set gamma=0.99 --set swap_min=10 --steps 4e8
```

About 1526 updates. Task 3 ran 629 updates in 42 minutes, so this is near 100
minutes.

## What the first run showed

`runs/chase_duel_s0_20260920-153504_737629`, all 1525 updates, 4e8 steps, 2.3 h.
The plan from here is in [handoff_duel.md](handoff_duel.md).

Final, from the three evals:

| eval | pursuer / evader | `caught` | `evader_crashed` | episode |
|---|---|---|---|---|
| `evals/latest` | ours / ours | 0.7041 | 0.0469 | 5.0 s |
| `evals/evader_cascade` | ours / cascade | 0.9609 | 0.0000 | 3.3 s |
| `evals/pursuer_cascade` | cascade / ours | 0.0156 | 0.2793 | 7.5 s |

Gate rows 1 and 2 pass. `evader_crashed < 0.05` fails. The convergence row is
not measurable — the third pursuer turn ran 754 updates and never ended, so the
evader trained for 45 updates of 1525.

**Same evader policy, two crash rates: 4.7% against our fast pursuer, 27.9% against
the slow cascade pursuer.** The difference is episode length, not skill. A caught
evader never reaches a wall. The evader learned to dodge, not to fly.

The rest of this section is the diagnosis as it was read at update 404.

**The pursuer's fourth turn is long, and that is normal.** Four swaps at updates
25, 35, 45 and 58, then none for 340 updates. The flat block from 60 to 240 is the
pursuer learning to fly and not crash; the catch comes after. 1v1 §V-A describes
the same curve: episode length "first increases as both agents learn to hover and
avoid crashes, but soon falls sharply as the pursuer discovers a quick capture
strategy". Ours rose to 900, held, then fell to 650.

Do not call a long turn a stuck turn before the episode length starts to fall.

| update | `caught` | `distance` | `ep_length` | `ρP` |
|---|---|---|---|---|
| 240 | 0.046 | 6.98 m | 851 | 0.300 |
| 404 | **0.441** | **3.72 m** | **650** | **0.495** |

**`ρP` started as a wall-crash meter and is slowly becoming a chase score:**

| updates | `ρP` | from catches | from evader crashes |
|---|---|---|---|
| 70-255 | 0.286 | 0.009 | 0.277 |
| 256-340 | 0.392 | 0.050 | 0.342 |
| 341-404 | 0.465 | 0.137 | 0.328 |

Until update 65 `crashed` sat at 0.99 — somebody hit a wall in almost every
episode — and the first four swaps rode that number, not any chase skill.

**`evader_crashed` never moves.** It holds 0.28-0.34 across the whole run. The evader
is frozen, so this is one fixed policy walking into a wall in a third of episodes.
The gate wants `evader_crashed < 0.05`.

### Why: our evader may use the whole arena

Neither paper allows that, and this note missed it.

| | ours | Nv1 | 1v1 |
|---|---|---|---|
| arena | 32 × 32 × 16 | 32 × 32 × 16 | 40 × 40 × 14 and 8 × 8 × 5 |
| where the evader may fly | **all of it** | **12 × 12 × 6 in the centre** (§V-A) | all of it, but `λbnd` pays it to stay off the walls |
| evader crash rate | **0.28** | 0.003 | 0.015 |

Both papers say the same sentence, in their reward sections: "neither the evader nor the
pursuer receive a reward when the opponent reaches a failure state… rather than
forcing the opponent to crash." Our **reward** obeys it. Our **swap trigger** does
not, because `ρP = 1` on an evader crash.

In Nv1 that branch of `ρP` only weights PFSP opponent sampling, where a
crash-prone evader is sampled **less**. We made it drive the curriculum.

## Open

| | |
|---|---|
| `swap_hi`, `swap_lo` | still guessed, but 0.6 now looks reachable: `ρP` passed 0.49 at update 404 and is still climbing |
| `swap_min` | 10 still guessed. Neither paper publishes a turn length |
| no phase cap | a turn ran 340 updates and was **still learning**. Do not add a cap on turn length alone — it would have cut this one off |
| `λbnd` | **1v1 Table I publishes `λbnd = 1.0`**, evader only. The buffer width and the ϕbnd curve are still unpublished. Nv1 publishes no weight and boxes the evader into 12 × 12 × 6 instead |
| confine the evader | Nv1's hard box or 1v1's soft `λbnd`. Two different answers, no clear winner |
| what the swap reads | `ρP` as Nv1 defines it, or `ρP` with the evader-crash branch dropped from the trigger only |
| four rays or eight | four are exact for a box. Eight give the diagonals for free |
| hyperparameters | 1v1 Table II: entropy 0.01, no lr decay, 15 epochs, 1 minibatch, 4e9 steps. Ours: 0.0, decay on, 4 epochs, 32 minibatches, 4e8 steps. Nv1 publishes no table |
| the first phase | the evader starts as a fresh policy: near hover, drifting, sometimes crashing |

## PPO settings against 1v1

Theirs is [1v1](../papers/2603.16279v1.md) Table II. Ours is the `config.json` of
`runs/chase_duel_s0_20260920-153504_737629`. Nv1 publishes no hyperparameter table.

| | 1v1 Table II | ours |
|---|---|---|
| discount factor γ | 0.99 | 0.99 |
| GAE λ | 0.95 | 0.95 |
| clip | 0.2 | 0.2 |
| critic weight | 0.5 | 0.5 |
| max gradient norm | 0.5 | 0.5 |
| hidden layers | 2 × 256 | 2 × 256 |
| entropy coefficient | **0.01** | **0.0** |
| learning rate | **5e-4** | **3e-4** |
| decay the learning rate | **False** | **True** |
| PPO epochs per batch | **15** | **4** |
| minibatches per epoch | **1** | **32** |
| parallel environments | **1024** | **4096** |
| rollout length | **128** | **64** |
| total environment steps | **4e9** | **4e8** |
| activation | **ReLU** | **tanh** |
| samples in one update | **131,072** | **262,144** |
| samples in one gradient step | **131,072** | **8,192** |
| gradient steps per update | **15** | **128** |
| number of updates | **about 30,500** | **1,525** |
| action spread | **network outputs it, `tanh` squash** | **one free `log_std`, clipped** |
| observation scaling | **positions by view range, velocity by max speed** | **raw SI units** |
| reward normalization | **not published** | **on** |

1v1 §IV-C says 2e9 steps and its Table II says 4e9. The table above uses 4e9.
