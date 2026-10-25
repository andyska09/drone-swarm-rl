# Task chase — the hand-written opponents

Fixed opponents for eval. A learned pursuer or evader from a run flies against them:

```bash
python run/eval.py runs/<run> --seat evader apf      # the run's pursuer against APF
python run/eval.py runs/<run> --seat pursuer frpn    # the run's evader against FRPN
```

FRPN and `paths` also start the two training pools
([task_chase_training.md](task_chase_training.md)). APF and the cascade never enter
training. The code is in [guidance.py](../../swarm/sim/guidance.py) (the laws and the
paths) and in `frpn_policy`, `apf_policy` and `paths_policy` in
[seats.py](../../swarm/learn/seats.py). What they score against each other, and the
history of their design: [experiments/chase_pools_log.md](experiments/chase_pools_log.md).

## The lineup

We copy Roncero v3's lineup ([2506.02849v3](../papers/2506.02849v3.md) §IV-B): an FRPN
pursuer, an APF evader and three path evaders. FRPN and APF use Němeček's code and values
([thesis](../papers/F3-BP-2026-Nemecek-Vaclav-bachelor_thesis.md)). Roncero does not
publish all of his. Němeček flew our drone: the X500, with the same mass, `kf`, `km` and
RPM range.

The `--seat` sources and their roles are in [task_chase.md](task_chase.md), Seats and
evals. The three shapes of `paths`, drawn for each episode with equal chance:

- `hover` holds one point.
- `circle` flies a circle, stretched into a mild ellipse.
- `figure8` flies a figure-eight.

`SOURCES` in `seats.py` lists the sources. A source in the wrong role raises an error.
The three shapes are not sources of their own.

## How a baseline flies

Every drone flies the same physics, the same rate loop and the same CTBR limits: throttle
0–1, ±4 rad/s roll and pitch, ±2 rad/s yaw. No drone has a speed clamp in the physics.

FRPN, APF and the paths go through this chain at every step:

1. The law gives a world acceleration.
2. FRPN and APF only: `speed_limit` allows at most 20 m/s², and no speeding up past 12 m/s.
3. `drone_limit` keeps the thrust between idle and full RPM. Height comes first and
   sideways gets the rest, so it never asks for a flip.
4. MRS `acceleration_controller` turns the acceleration into an attitude and a throttle.
   A NaN throttle becomes 0.
5. MRS `attitude_controller` turns the attitude into body rates. Yaw is capped at the
   env's 2 rad/s, not MRS's 1.
6. `command_to_action` makes the CTBR action. The env's rate loop flies it, as it flies a
   network's action.

The heading stays where it is. `control.py` is not changed, because it must match the C++
goldens.

**Němeček's limits sit in the law, not in the physics.** His simulator cuts every drone's
speed to 12 m/s after each step (`mrs_uav.cpp:89-98`). We cannot do that, because the
network flies the same physics.

**The `cascade` seat is the one exception.** It keeps MRS's defaults: ±6 m/s and ±4 m/s²
per axis, yaw 1 rad/s, heading 0. It is the easy opponent. As a pursuer it flies its net
onto the evader. As an evader it flies to random waypoints.

## FRPN

`frpn(dp, dv) = G·((1 − W)·(dp + dv·t_go) / t_go² + W·dp)`, with `t_go = |dp| / |dv|`.
`dp` and `dv` are the evader minus the pursuer. The output is an acceleration.

- **Values:** `G = 6.677`, `W = 0.337` (Němeček, Table 8.1). He tuned them under 12 m/s
  and 20 m/s² (§8.1), so `speed_limit` applies both.
- **It aims its net centre at the evader**, not its body: `dp = chase.reference − x`.
- **Roncero's FRPN is different.** Its output is a velocity command, clipped to the top
  speed, with `G = 5.0` and `W = 0.20`. A cascaded PID flies it.
- **Without the limits it leaves the arena.** It reaches about 19 m/s.

## APF

Each pursuer pushes the evader away from the pursuer's body:
`(k_p + c_p·closing) / dist^e_p`. Each arena face closer than `r_w` also pushes:
`(k_w + c_w·toward)·(1/gap − 1/r_w)^e_w`.

- **Values:** Němeček's eval set (`APF_PARAMS` in `eval_rl_pur.py`, Table 8.6):
  `k_p = 2.135`, `k_w = 57.89`, `c_p = 1.370`, `c_w = 29.71`, `r_w = 9.997 m`,
  `e_p = 1.032`, `e_w = 0.5652`.
- **It adds up the push of every pursuer**, so it works for N v 1.
- **All six faces closer than `r_w` push**, as in his code. His thesis text says that only
  the nearest face on each axis pushes.
- **Roncero's APF is different:** `k_P = k_W = 1.0`, `α_P = 3`, `α_W = 2`, and a velocity
  controller tracks the force.
- APF never hit a wall, and its top speed was about 9.7 m/s. The limits change almost
  nothing for it.

## Paths

### How a path is made

At the start of each episode, `start(state)` in `seats.py` does three things:

1. It draws one path with `draw_path`: a size, a speed, a start point and a direction.
2. It puts the path at a random place where all of it fits 2 m inside the walls, at a
   random height.
3. It moves the evader to the start point and gives it the path's velocity there.

So the evader always starts on its path, moving with it, and every path keeps its full
size. At each step, `follow` flies the evader after the moving point.

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

The sizes scale by 8 and the speeds by √8. That keeps each shape's turn acceleration,
`v²/r`. The figure-eight is Gerono's: `(A sin θ, B/2 sin 2θ)`.

### The speed is not constant

The point turns at a steady rate, `w = speed / mean radius`, like Roncero v2's angular
velocity. So the speed along the path changes. Over the 1024 eval paths:

- A circle changes a little, because the ellipse is mild: 2.67–6.06 m/s.
- A figure-eight is fastest where its loops cross: 1.79–8.00 m/s.

The peak acceleration of a circle is 3.1 m/s² (median), and 7.2 m/s² at most. For a
figure-eight it is 7.7 m/s² (median), and 20.0 m/s² at most. The drone gives about
31 m/s² sideways, so it can fly all of them.

### Why the path picks the start

The env spawns the evader at a random spot. Near a corner, no full-size path through that
spot fits, and a shrunk path keeps its lap time, so it is also slower. So the path picks
where the evader starts, not the spawn.

Němeček does the same. His evader starts at row 0 of its trajectory file
(`target_uav.cpp:57`).

### Němeček's trajectories

He does not make shapes. He loads 200 recorded trajectories from CSV files, in two
profiles of 100 each (thesis §3.1, Table 3.1). At each step the evader's state is set to
the next row, so it is replayed, not flown (`environment.cpp:184-198`). In this mode the
evader cannot leave the arena (`environment.cpp:217`). We fly Roncero's shapes instead.
From Němeček we take only one thing: the path decides where the evader starts.

### Not copied

- Roncero v2 also turns each path to a random heading and tilts it by up to 0.05π. v3
  does not say this, so we do not do it.
- Roncero's code is not public. `KTH-RPL/AMSPB_PEG`, the repo that his paper page links,
  gave 404 on 2026-10-10.

## Tests

`tests/test_eval.py` checks three things. The seven outcomes sum to 1 (FRPN against APF,
64 episodes). FRPN catches the `paths` evader in more than 60% of 64 episodes. The
cascade catches the cascade evader in more than 90% of 32 episodes.

## Sources

- Roncero v3: [2506.02849v3](../papers/2506.02849v3.md) §IV-A, §IV-B. Roncero v2, for the
  path numbers: <https://arxiv.org/html/2506.02849v2>, §V-B.
- Němeček: [thesis](../papers/F3-BP-2026-Nemecek-Vaclav-bachelor_thesis.md) §3.1, §3.2,
  §8.1, Tables 8.1 and 8.6.
- Němeček's code, under
  `research/code_sources/rl_tracking_eagle-cleaned-up-logic/examples/deep_eagle/`:
  `examples/agents.py` (`OriginalFRPN`, `AdvancedAPFEvader`), `examples/eval_rl_pur.py`
  (`APF_PARAMS`), `src/agent/mrs_uav.cpp`, `src/agent/target_uav.cpp`,
  `src/env/environment.cpp`.
