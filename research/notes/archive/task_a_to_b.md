# Task A→B — the setting

Archived: `swarm/envs/a_to_b.py` was deleted on 2026-10-11 (commit `6d892a6`). This
note keeps the task as it was. The survey of why its learned policy did not park is in
[literature/reward_terms.md](../literature/reward_terms.md).

One drone, one stationary goal. Fly there and stay.

## Timing

Sim at 200 Hz (`sim_dt = 0.005`), policy at 100 Hz (`steps_per_action = 2`).
`max_steps = 500` → a 5 s episode.

## Action — CTBR

`a ∈ [-1,1]⁴`, clipped. `a₀` maps to collective throttle `[0,1]`, `a₁:₄` to body
rate references, scaled by `±4 rad/s` roll/pitch and `±2 rad/s` yaw. Only the
rate rung of the cascade runs underneath, at sim rate, feeding the mixer —
the policy replaces everything above it.

## Observation

- `own` (16): velocity in the body frame, `R` flattened, `ω`, and height `z`.
  Everything except `z` is frame-free, so the drone cannot tell where in the
  world it is — only which way is up and how far off the ground.
- `target` (6): goal offset `g - x` and relative velocity `-v`, both rotated
  into the body frame.
- `neighbors`: empty, `K = 0` for `N = 1`.

## Reset

One `center = (0,0,3)` holds the start, the goal box and the arena. Position
uniform in ±0.5 m around it, velocity ±0.5 m/s, a random tilt up to 0.2 rad
about a random axis, `ω = 0`, motors at hover RPM. The goal is drawn in
`±goal_range` about the same centre.

## Termination

Death is ground contact (`z < 0`), leaving the 10 m arena, flipping over
(`R₂₂ < 0`), or a non-finite state. The episode ends there (`N = 1`, so no one
is left) and a dead drone scores nothing further. Hitting `max_steps` is
truncation, not death.

## Reward

Distance and spin costs integrated over `policy_dt`, plus a one-off crash penalty:

```
r = -policy_dt · (1.0·‖g - x‖ + 0.1·‖ω‖) - 10.0 · died
```

All three weights are in `RewardConfig`. The `spin` term was added after the
first trained `default` policy never settled: it orbited the goal at 0.21 m and
0.4 m/s and still did so after 15 s. Pure distance pays the same for circling at
0.21 m as for parking there, so nothing asked it to stop. `0.1` against a
distance weight of `1.0` is Huang et al. 2024's ratio, whose reward has the same
`-dt · (weighted sum)` shape.

Tilt, effort and action-rate shaping stay absent — add one only when a failure
mode asks for it.

## Presets

- `default` — goal drawn in a ±(4,4,2) m box: fly there and hold.
- `hover` — goal at the centre: hold against the reset perturbation.

## Decisions, moved from choices.md

- **Why `spin` is in.** The first trained `default` policy beat the cascade on
  the objective (return −3.39 against −3.89) and still never settled. It orbited
  the goal at 0.21 m with 0.28 m/s of tangential speed, and a 15 s rollout showed
  a permanent limit cycle, not slow arrival. A pure distance cost pays the same
  for circling at 0.21 m as for parking at 0.21 m, so nothing asked it to stop.
  Both reference works carry a motion penalty at 0.1–0.2 of their own distance
  weight: Huang et al. 2024 uses `0.1·‖ω‖` against `1.0·‖p‖`; Gavin et al. 2026
  uses `2e-4·‖a_ω‖` against `1e-3·‖p‖`. We copy Huang's, because our reward has
  the same `−dt · (weighted sum)` shape and the same distance weight, so the
  number transfers with no rescaling. Gavin's term penalises the *command* and is
  aimed at sim-to-real feasibility, not at settling.
- **Discount factor `gamma = 0.998`.** Resolved; it was open until the learner
  ran. The effective horizon is `1/(1−gamma)` policy steps, and at `policy_dt =
  0.01 s` that is 500 steps, exactly one episode. The first real run used `0.995`
  = 2 s, which was enough for `hover` (worst leg 0.87 m) and too short for
  `default`, where the worst leg is about 6.9 m and takes 3 to 4 s to fly. A
  policy cannot plan for a payoff it discounts away before it arrives.
