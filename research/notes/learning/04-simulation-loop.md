# The simulation loop

## The whole thing

```
loop forever, every dt = 0.01 s:        # simulation_rate: 100 Hz
    actuators = cascade(state, command) # command → 4 throttles
    model.setInput(actuators)           # throttles → target rpm
    model.step(dt)                      # integrate physics, update state
```

`makeStep()` at `uav_system.hpp:304` is exactly those three lines. `command` is
set once and just sits there. The loop runs 100 times a second regardless.

## A controller is a reflex, not a planner

**The position controller does not fly anywhere.** It does this and only this:

> "You are at x. You want to be at x_ref. Here is a velocity." → 3 numbers → done.

No memory of a plan, no trajectory, no idea it is being called repeatedly. It
is a **spring**, not a driver. Stretch it (position error), it pulls (velocity
output). 10 ms later it is called again with fresh state and produces a fresh
number.

Flying is not something any component does. Flying is what *emerges* from
running six springs 100 times a second while physics integrates in between.

Every controller reads the **live state** every tick. That is the feedback.

## Entry level

You set a command at any level; it enters there and falls through everything
below (`uav_system.hpp:314-373`, a chain of
`if (active_input == X) { …; active_input = X_below; }`):

```
POSITION_CMD          → position_controller      ↓
VELOCITY_HDG_CMD      → velocity_controller      ↓
ACCELERATION_HDG_CMD  → acceleration_controller  ↓
ATTITUDE_CMD          → attitude_controller      ↓
ATTITUDE_RATE_CMD     → rate_controller          ↓
CONTROL_GROUP_CMD     → mixer                    ↓
ACTUATOR_CMD          → model.setInput() → model.step(dt)
```

Enter at `ATTITUDE_RATE_CMD` and the top three controllers never execute.

## One tick, traced

Drone hovering level at (0,0,2). `POSITION_CMD = (5, 0, 2)`, heading 0.

| Level | Input | Output |
|---|---|---|
| position | error = (5, 0, 0) m | PID gives 85 → capped → **vel = (6, 0, 0) m/s** |
| velocity | error = (6, 0, 0) m/s | PID gives 42 → capped → **acc = (4, 0, 0) m/s²** |
| acceleration | want (4,0,0) | `fd = m(a + gẑ)` = (8, 0, 19.62) N → **tilt 22.2°**, throttle **0.465** |
| attitude | Rd vs R = I | error vector (0, 0.378, 0) → **rates (0, 4.16, 0) rad/s** |
| rate | error = 4.16 rad/s | PID gives 1.095 → capped → **pitch = 1.0** |
| mixer | (0, 1.0, 0, 0.465) | raw (−0.24, 1.17, −0.24, 1.17) → desaturate → **(0, 0.93, 0, 0.93)** |
| model | throttles | rpm target (1170, 7337, 1170, 7337); lag → **(3381, 5129, 3381, 5129)** |

Three things to notice.

**The saturations do most of the work early on.** Position wanted 85 m/s and
got 6. Velocity wanted 42 m/s² and got 4. Far from the target the whole stack
just runs at its limits — the gains only matter near the end.

**22.2° is not arbitrary.** `atan(4 / 9.81) = 22.2°`. To accelerate at 4 m/s²
you must tilt exactly that much. Physics, not tuning.

**Throttle came out 0.465 — exactly hover** — because `fd` is projected onto
the *current* up-axis and the drone is still level. And then the mixer's
desaturation quietly scaled pitch from 1.0 down to 0.658 to protect the
throttle. The drone gets less pitch than asked for, on tick 1.

## Over many ticks

- **0–0.1 s:** rpm ramps (30 ms motor lag). Drone starts pitching. Barely moves.
- **0.1–1 s:** pitch reaches ~22°, accelerates to the 6 m/s cap. Position
  controller stays saturated.
- **cruise:** error still large → still saturated → attitude roughly constant.
- **last ~1 s:** error drops below 3 m, the position PID comes off its cap,
  commands shrink, drone pitches back to level and decelerates.
- **settled:** all six levels output near-zero, throttle sits at 0.465, hover.

Nothing planned that. Six springs and a clock.

## Who sets the command

**Nothing inside this repo.** `UavSystem` only exposes `setInput()` overloads;
somebody outside has to call one.

That somebody is the ROS wrapper `src/uav_system_ros.cpp`, which subscribes to
one topic per cascade level — `<uav>/position_cmd`, `<uav>/velocity_hdg_cmd`,
`<uav>/attitude_rate_cmd`, `<uav>/actuators_cmd`, and so on. Whoever publishes
picks the entry level. In the lab stack that publisher is the MRS control
manager / trajectory tracker, in a different repo. **This repo is the drone,
not the pilot.**

What happens when nobody publishes is revealing. `timeoutInput()` fires and
does:

```cpp
cmd.position = state.x;          // where you are right now
uav_system_.setInput(cmd);
```

It sets the target to the drone's current position. Error goes to zero, the
cascade commands nothing, the drone hovers. The whole safety behaviour is
"no orders → stay put".

For our project **we set it** — a test script in M2, `reset_env` in M3, and
eventually the policy itself publishing to the attitude-rate level.
