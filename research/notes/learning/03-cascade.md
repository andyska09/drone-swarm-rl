# The control cascade — six controllers

`.../uav_system/controllers/`

Turns a human-level command ("go to (5,0,2)") into 4 motor throttles.

## Commands are types

`references.hpp` defines one small struct per level. **A controller is nothing
but a converter from one type to the one below it.** That is the whole
architecture.

| Type | Contents |
|---|---|
| `Position` | position (3), heading |
| `VelocityHdg` | velocity (3), heading |
| `AccelerationHdg` | acceleration (3), heading |
| `Attitude` | orientation `R` (3×3), throttle |
| `AttitudeRate` | rate_x/y/z, throttle |
| `ControlGroup` | roll, pitch, yaw, throttle — all in [−1, 1] |
| `Actuators` | 4 motor throttles in [0, 1] |

Throttle only appears from `Attitude` downward. Above that, thrust is implied
by the acceleration you asked for.

**Heading is not yaw.** MRS defines heading as the `atan2` of the body-x axis
*projected onto the ground plane*. Yaw is an Euler angle. They agree only when
level. Heading stays meaningful while tilted, which is why the outer levels use
it — and why `getYawRateIntrinsic()` is the ugliest function in the repo.

## PID (`pid.hpp`, 30 lines)

```
out = kp·e  +  kd·(e − e_prev)/dt  +  ki·∫e
```

then saturate. Anti-windup: only accumulate the integral while the output is
not saturated. The derivative is on the *error*, so a step change in the
reference gives a kick.

Four of the six controllers are this. Two are not.

## 1. Position → Velocity

`vel = PID(pos_ref − pos)` per world axis, capped at 6 m/s.
`kp=2, kd=0.15, ki=0.2`.

## 2. Velocity → Acceleration

`acc = PID(vel_ref − vel)` per world axis, capped at 4 m/s².
`kp=2, kd=0.05, ki=0.01`.

Both are three independent scalar PIDs. No coupling, no cleverness.

## 3. Acceleration → Attitude — geometry, no PID

This is "tilt to move" written down.

You want acceleration `a`. You need force `fd = m·(a + g·ẑ)` — what you asked
for **plus** enough to cancel gravity. The drone can only push along its own
up-axis, so point that axis at `fd`:

```
Rd.col(2) = normalize(fd)
```

Body-x is then chosen to hit the requested heading; body-y = z × x closes the
frame.

The throttle is the subtle bit (`:91`):

```
thrust_force = fd · R.col(2)        ← the CURRENT up-axis, not the desired one
throttle = (√(F/(kf·n)) − min_rpm) / (max_rpm − min_rpm)
```

Projecting onto the *current* attitude is deliberate: while still rotating,
do not command thrust you cannot point yet. The second line is the thrust curve
inverted.

## 4. Attitude → Attitude rate

Problem: "how far apart are two orientations?" You cannot subtract rotation
matrices. The trick:

```
E = ½·(Rdᵀ·R − Rᵀ·Rd)
```

`E` comes out skew-symmetric, and a skew-symmetric matrix is secretly a
3-vector. Pull that vector out, run each component through a PID, get three
body rates. Capped at 10 rad/s roll/pitch, 1 rad/s yaw.

This is the one place where "rotations are not a vector space" actually bites.

## 5. Attitude rate → Control group

`torque = PID(ω_ref − ω)`, with the gains **multiplied by the inertia `J`**
(`:62`) so the output lands in torque units, then normalized to [−1, 1].
`kp=4, kd=0.04`.

Innermost loop. This is the one CTBR hands to a policy.

## 6. Control group → Actuators (`mixer.hpp`)

Nominally `A⁻¹`. In practice MRS computes the pseudo-inverse then **overwrites
most of it**: normalize the roll/pitch columns, snap yaw to ±1 or 0, set the
throttle column to all 1s. It throws away physical units on purpose, to match
PX4's control-group convention.

Then **desaturation**, because a motor cannot go below 0 or above 1:

- any motor negative → add a constant to all four (shifts thrust, preserves torques)
- any motor above 1 → scale roll/pitch/yaw down, keep throttle

That second rule says **prefer altitude over attitude when you run out of
motor**. A real design decision, and the kind of thing you would never think to
ask about.
