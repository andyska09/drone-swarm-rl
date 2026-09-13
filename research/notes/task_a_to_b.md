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
