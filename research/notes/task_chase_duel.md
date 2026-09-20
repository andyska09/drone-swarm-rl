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

Anchors: task 3 scores `ρP ≈ 0.87`. Gavin's best 3v1 pursuers against a learned
evader need 7.36 s of 10, so `ρP ≈ 0.26` on a catch and near 0.2 on average.

## The swap

One `TrainState` per role already exists. The frozen role is held by selecting
between the stepped and the unstepped state after `apply_gradients` — params,
Adam moments and the step count all stay put. One compiled update.

- both roles start random, the pursuer trains first
- the pursuer's turn ends when `ρP > swap_hi`
- the evader's turn ends when `ρP < swap_lo`
- neither turn ends before `swap_min` updates
- the frozen side keeps sampling its actions, so it is not one fixed behaviour

`TrainConfig`: `swap_hi = 0.6`, `swap_lo = 0.3`, `swap_min = 10`. The swap is off
when `swap_min = 0`, so every other task runs unchanged.

**No cap on a phase.** Gavin does not publish one and a stuck phase should be
visible, not hidden. `metrics.csv` gains a `training` column; a stuck phase reads
as one long flat block.

Every swap writes a checkpoint, as Gavin freezes a snapshot at every swap. That
is also the start of a policy pool, if one is ever needed.

## Eval

`--preset` and `--policy` are gone. One flag replaces both:

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
| the evader against a fixed pursuer | `--seat pursuer cascade` | `caught_final < 0.5` |
| the pursuer against a fixed evader | `--seat evader cascade` | `caught_final > 0.9` |
| no win by crashing | either | `evader_died_final < 0.05` |
| it converges, not cycles | `metrics.csv` | the swing in `ρP` narrows across phases |

The cascade caught the scripted evader 0.996 of the time, and task 3's pursuer
caught it 0.999 of the time. Those are the numbers the first two rows move away
from.

## The run

```bash
conda run -n drone-swarm python run/train.py --task chase --preset duel \
    --set gamma=0.99 --steps 4e8
```

About 1526 updates. Task 3 ran 629 updates in 42 minutes, so this is near 100
minutes.

## Open

| | |
|---|---|
| `swap_hi`, `swap_lo`, `swap_min` | guessed. Retune once `metrics.csv` shows the real range of `ρP` |
| no phase cap | a stuck phase is possible on purpose. Watch the `training` column |
| `λbnd` | only if the evader parks at a wall. Gavin publishes no weight, no threshold, no curve |
| four rays or eight | four are exact for a box. Eight give the diagonals for free |
| the first phase | the evader starts as a fresh policy: near hover, drifting, sometimes crashing |
