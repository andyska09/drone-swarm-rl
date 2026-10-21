# Task avoidance — the spec

Two drones fly head-on through a closed square tube. Each drone must reach the start
point of the other drone. The drones must not touch, and they must not hit the tube. The
code is [avoidance.py](../../swarm/envs/avoidance.py). The runs and the history of the
design are in [experiments/avoidance_log.md](experiments/avoidance_log.md).

**What the task is for.** The task is a measuring instrument. The input is what a drone
senses about the other drone. The reading is the success rate. Only the sensor may change
the reading. The first design gives the exact position and velocity of the other drone,
but only inside a sight range `d`. The literature is in
[literature/avoidance.md](literature/avoidance.md).

## The tube

| what | value |
|---|---|
| inside | 3 m wide and 3 m high (`width`), closed at both ends |
| length | 5 m at the start of training, 20 m at the end (see Curriculum). Eval always flies 20 m (`eval_params`) |
| in the world | the floor is the ground (z = 0), and the ceiling is at 3 m. The middle is at `center` in x and y. Each episode turns the tube to a random yaw (rotation about z). The arena box plays no part |

A drone is a ball of radius 0.40 m (arm + propeller). It hits the tube when this ball
touches a side wall, an end wall, the floor or the ceiling.

## Drones and goals

- Two drones share the role `drone`, so one network flies both. Both learn (self-play).
- Each drone starts 1 m from an end wall (`start_gap`), the two at opposite ends. Across
  the tube, the start is a random point in the middle 1.6 m × 1.6 m (`start_box`).
  `core.spawn` adds a random velocity of up to 0.5 m/s on each axis, a random yaw and a
  tilt of up to 0.2 rad.
- The goal of a drone is the start point of the other drone. So the two straight paths
  are the same line: an exact head-on. The random start state breaks the tie.
- A drone arrives when it is closer than 1 m to its goal (`arrive_radius`). It then leaves
  the episode: it stays frozen where it arrived, and the other drone can neither see it
  nor touch it. The other drone flies on.
- The speed is free.
- A drone that flips keeps flying. So the task does not use `core.is_dead`.

## Sight

- A drone sees the other drone when it is closer than `sight`, in any direction. `sight`
  is the `d` of the lit review. There is no field of view, no blocking and no memory.
- When a drone does not see the other drone, the mask sets its `others` row to zero.
- With `d_inf`, the drones always see each other. With `d0`, they never do.

## Observation

| block | width | content |
|---|---|---|
| `own` | 20 | `core.own_obs`: velocity in the body frame, the full `R`, `ω`, height above the floor, and the distances to the tube walls ahead, behind, left and right, level with the floor |
| `others` | 1 × 7 | the other drone: its offset and velocity in the body frame, and the role flag (always 1). Zero when out of sight |
| `target` | 6 | `core.target_obs` for the goal, with goal velocity 0: the offset to the goal and the own velocity with a minus sign, in the body frame |
| `scene` | 46 | the input of the critic, in the tube frame. For each drone: position, velocity, `R`, `ω`, goal, and if it arrived. Then the tube length and the time left |

The tube frame has its origin in the middle of the tube, x along the tube and z up.

## Action

CTBR, as in every task.

## Reward

```
r = +1   at the step the drone arrives
    −1   at the step the drone crashes
     0   at every other step
```

| `RewardConfig` | value |
|---|---|
| `arrive` | 1 |
| `crash` | 1 |

- The reward states only the goal. It has no shaping terms.
- On contact, both drones get −1. When a drone hits the tube, only that drone gets −1.
- With `gamma` = 0.998, each second of delay multiplies the value of the +1 by 0.82. So
  the drone still has a reason to be fast.

## Episode

| event | how |
|---|---|
| both drones arrived | terminal. `success` |
| a drone hits the tube | terminal for both drones. `wall` |
| the drones come closer than 0.80 m between centres | terminal for both drones, both get −1. `contact` |
| 10 s pass (`time_limit`) | terminal, no bootstrap. A drone that has not arrived gets nothing. `timeout` |
| `max_steps` = 1100 steps | truncation. Only the first episode of each env gets there, because `vecenv.reset` gives it a random `time` |

The 10 s limit counts with the task's own step counter, `steps`, not with `time`.

## Curriculum

- One run. The tube starts at 5 m (`length`). After an update, it gets 5 m longer
  (`length_step`) if both of these are true:
  - every episode that ended in the update flew the current tube, that is, the metric
    `length` equals the current length;
  - more than 80% of these episodes succeeded (`grow_at`).
- It stops at 20 m (`length_max`).
- The first condition is there because the episodes that started before a change still
  fly the short tube, for up to 16 updates.
- No random starts along the tube.
- The task gives `curriculum(params, metrics)`. `runner.train` calls it after every update,
  and it builds the compiled update again when the tube grows.

## Metrics

| `info` | name | what it measures |
|---|---|---|
| `end` | `success` | both drones arrived |
| `end` | `wall`, `contact`, `timeout` | how the episode ended otherwise |
| `end` | `closest` | the smallest distance between the centres while both drones fly |
| `end` | `time_to_arrive` | per drone, the time from the start to the arrival. The average over the two drones covers only the episodes where both arrived |
| `end` | `length` | the tube length of the episode. It shows when the tube grew |

The trainer and the eval add `death_rate`. An arrived drone is not a death.

## Training and eval

| command | what |
|---|---|
| `python run/train.py --task avoidance --preset d_inf --set ent_coef=0` | train. `ent_coef=0` is against the blow-up of the action spread |
| `python run/eval.py runs/<name>` | the policy flies both drones in the 20 m tube |

## Presets

| preset | `sight` |
|---|---|
| `d_inf` | no limit |
| `d6`, `d5`, `d4`, `d3` | 6, 5, 4, 3 m |
| `d0` | 0 m: blind, for the calibration |

## Open

| question | state |
|---|---|
| In self-play, the two drones can agree on a convention ("always left") and pass without sensing much. The known fix is Other-Play (Hu et al. 2020, [arXiv:2003.02979](https://arxiv.org/abs/2003.02979)): a left-right mirror for drone B. | Not built. Add it only if the debug runs show a convention. |
| `R` tells each drone its world heading, and the two drones always fly in opposite world directions. So a rule like "the drone that flies toward +x goes up" can split them without the sensor. | Accepted risk. Check it in the debug runs: compare the height offset at the meeting with the flight direction. |
| A test drone that flies straight to its goal and never dodges, to find a half dodge. | Later. The cascade in both seats is not this test (see the log). |
| `core.is_dead` kills a flipped drone. This task and `chase` do not. | Decide for all tasks later. |
| tests | None yet. |
