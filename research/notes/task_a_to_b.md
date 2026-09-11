# Task A→B — the setting

One drone, one stationary goal. Fly there and stay. Implemented in
`swarm/envs/a_to_b.py`; the shared interface is in `plan_t1t2.md`.

## Timing

Sim at 200 Hz (`sim_dt = 0.005`), policy at 100 Hz (`steps_per_action = 2`).
`max_steps = 500` → a 5 s episode.

## Action — CTBR

`a ∈ [-1,1]⁴`, clipped. `a₀` maps to collective throttle `[0,1]`, `a₁:₄` to body
rate references, scaled by `±4 rad/s` roll/pitch and `±2 rad/s` yaw. Only the
rate rung of the M2 cascade runs underneath, at sim rate, feeding the mixer —
the policy replaces everything above it.

## Observation

- `own` (16): velocity in the body frame, `R` flattened, `ω`, and height `z`.
  Everything except `z` is frame-free, so the drone cannot tell where in the
  world it is — only which way is up and how far off the ground.
- `target` (6): goal offset `g - x` and relative velocity `-v`, both rotated
  into the body frame.
- `neighbors`: empty, `K = 0` for `N = 1`.

## Reset

Position uniform in ±0.5 m around `(0,0,3)`, velocity ±0.5 m/s, a random tilt up
to 0.2 rad about a random axis, `ω = 0`, motors at hover RPM. The goal is drawn
around the same centre — `±(4,4,2)` m for `default`, a single point for `hover`.

## Termination

Death is ground contact (`z < 0`), leaving the 10 m arena, flipping over
(`R₂₂ < 0`), or a non-finite state. A dead drone freezes and the episode ends
(`N = 1`, so no one is left). Hitting `max_steps` is truncation, not death.

## Reward

Per-step cost integrated over `policy_dt`, plus a one-off crash penalty:

```
r = -policy_dt · (1.0·‖g - x‖ + spin·‖ω‖ + tilt·(1 - R₂₂) + effort·‖a‖ + smooth·‖a - a_prev‖)
    - 10.0 · died
```

Only the distance and crash terms are on by default; the shaping weights are
knobs in `RewardConfig`, all zero until a failure mode asks for one.

## Presets

- `default` — goal drawn in a ±(4,4,2) m box: fly there and hold.
- `hover` — goal at the start centre: hold against the reset perturbation.
- `recover` — start tilted up to 1.05 rad, `ω` ±2 rad/s, `v` ±2 m/s: catch it
  first, then hold.
