# ODEs, Euler, RK4, and the motor lag

The math the simulator runs on. Four parts: what an ODE is, why the naive
integrator is wrong, what RK4 does about it, and where `exp(-dt/τ)` comes from.

## 1. What an ODE is

Sometimes you cannot write down *where* something is. But you can always write
down **how fast it is changing right now, given what you know right now**.

A **derivative** is a rate of change. The dot means "per second":

- `x` = position (m) → `ẋ` = m/s = velocity
- `v` = velocity (m/s) → `v̇` = m/s² = acceleration

An **ordinary differential equation** has this shape:

```
ẋ = f(x)
```

*"Tell me the current state, and I will tell you how fast it is changing."*

- **Differential** — an equation about derivatives.
- **Ordinary** — the state depends on one variable only, time. If it also
  varied across space it would be a *partial* differential equation (heat,
  fluids). Not our problem.
- **Equation** — you are handed a constraint; the unknown you want is a
  *function* `x(t)`.

### The falling ball

State = position and velocity. Two numbers.

```
ẋ = v
v̇ = −9.81
```

Out loud: *position changes at the rate of velocity; velocity changes at −9.81
per second.*

### What makes "state" click

You **had** to put velocity in the state. With position alone you could not
write `ẋ` — how fast the ball moves is not determined by where it is. Two balls
at the same height can move at different speeds.

> **The state is exactly the set of numbers you need so that the derivative is
> fully determined.** That is the definition of state.

This is why the drone carries 18 numbers instead of 3. `R` alone does not say
how it is rotating — you need `ω`. `x` alone does not say how it is moving —
you need `v`.

### The drone's ODE

Same shape, bigger `f`. 18 numbers in, 18 numbers out:

```
ẋ = v
v̇ = −g·ẑ + (T/m)·R.col(2) − drag
Ṙ = R·[ω]×
ω̇ = J⁻¹(τ − ω × Jω)
```

`f` does not depend on time explicitly, only on the state. That is why MRS's
`operator()` takes a `t` argument marked `[[maybe_unused]]` — Boost demands the
signature, the physics ignores it.

### Solving it

The ball has an **analytic solution** — exact, for every `t`, forever:

```
x(t) = x₀ + v₀·t − ½·9.81·t²
```

The drone does not. Its equations are nonlinear: rotation matrices multiplying,
a cross product, drag going as `|v|²`. There is no formula for where a
quadrotor is at `t = 7.3 s` and there never will be.

So you do the only other thing: start from a known state and **take small steps
forward**, using the derivative to guess each one. That is *numerical
integration*. RK4 is one recipe for it.

## 2. Euler's method, and exactly how it fails

You are at `x`, your rate of change is `f(x)`, so after a small time `dt`:

```
x_new = x + dt · f(x)
```

Picture it: you stand on an invisible curve. You cannot see the curve but you
can feel its slope where you stand. So you walk in a straight line at that
slope for `dt` seconds and declare yourself back on the curve.

**Why it is wrong:** the slope changes as you move. You used the slope from the
*start* of the interval for the *whole* interval.

### Watch it fail

Free fall, `dt = 0.1 s`, one second. Rules: `x_new = x + 0.1·v`,
`v_new = v + 0.1·(−9.81)`.

| step | t | x | v |
|---:|---:|---:|---:|
| 0 | 0.0 | 0.000 | 0.000 |
| 1 | 0.1 | 0.000 | −0.981 |
| 2 | 0.2 | −0.098 | −1.962 |
| 3 | 0.3 | −0.294 | −2.943 |
| ⋮ | | | |
| 10 | 1.0 | **−4.415** | −9.810 |

Exact: `−½·9.81·1²` = **−4.905 m**. Off by 0.49 m — **10%, in one second, on
the easiest problem in physics.**

Look at step 1: it claims the ball has not moved, because it started at `v = 0`
and used that zero for the whole 100 ms. The ball was already accelerating.
Every step under-counts the same way and the errors pile up.

### The scaling problem

Halve the step to `dt = 0.05` (20 steps): you get `−4.660`, error `0.245` —
exactly half. **Euler's error shrinks in proportion to `dt`.** Want 100× the
accuracy, take 100× the steps. That is called *first-order*, and it is a bad
deal.

## 3. RK4

**The core trick:** the slope at the start is a poor representative of the
interval, so sample the slope in several places and average.

### Warm-up — the midpoint method

1. Take a throwaway **half-step** with the starting slope, just to peek at where
   the middle is.
2. Evaluate the slope **there**.
3. Go back to the start and take the **full** step using that midpoint slope.

```
k1 = f(x)                  # slope at the start
k2 = f(x + (dt/2)·k1)      # slope at the estimated midpoint
x_new = x + dt·k2          # full step using the midpoint slope
```

The half-step is discarded. It exists only to find a better slope. **That
probe-then-commit pattern is the whole idea** — RK4 just does it more times.

### RK4 itself

```
k1 = f(x)                    # slope at the start
k2 = f(x + (dt/2)·k1)        # slope at the midpoint, probed with k1
k3 = f(x + (dt/2)·k2)        # slope at the midpoint again, probed with k2
k4 = f(x + dt·k3)            # slope at the end, probed with k3

x_new = x + (dt/6)·(k1 + 2·k2 + 2·k3 + k4)
```

Four probes: start, middle, middle again, end.

### The weights 1, 2, 2, 1

They sum to 6 and you divide by 6, so it is a **weighted average of four
slopes**. The midpoints get double weight because a slope from the middle
represents the interval better than one from either edge.

If you have seen Simpson's rule for areas — `(1/6)(f_start + 4·f_mid + f_end)` —
this is the same shape, with `k2` and `k3` together playing the role of
`4·f_mid`.

Where the numbers actually come from: expand both `x_new` and the true
`x(t+dt)` as Taylor series, choose coefficients so they agree through the `dt⁴`
term. A page of algebra with no insight in it. The result is what matters.

### Watch it work

Same free fall. **One single RK4 step of `dt = 1.0 s`.** State `(x, v)`,
`f(x,v) = (v, −9.81)`.

```
k1 = f(0, 0)                              = (0,      −9.81)
k2 = f(0 + 0.5·0,       0 + 0.5·(−9.81))  = (−4.905, −9.81)
k3 = f(0 + 0.5·(−4.905), 0 + 0.5·(−9.81)) = (−4.905, −9.81)
k4 = f(0 + 1·(−4.905),   0 + 1·(−9.81))   = (−9.81,  −9.81)

x_new = 0 + (1/6)·[0 + 2(−4.905) + 2(−4.905) + (−9.81)]
      = (1/6)·(−29.43)
      = −4.905          ← exact
```

**One RK4 step at `dt = 1.0` — four function calls — beat ten Euler steps at
`dt = 0.1`, and landed exactly right.**

Exact here because the true solution is a quadratic, and RK4 handles
polynomials up to degree 4 perfectly. On the drone it will not be exact, just
very close.

### The trade

| method | `f` calls per step | global error |
|---|---:|---|
| Euler | 1 | ∝ dt |
| midpoint (RK2) | 2 | ∝ dt² |
| **RK4** | **4** | **∝ dt⁴** |

Halve `dt`: Euler gets 2× better, RK4 gets **16×** better. You pay 4× per step
but can take far bigger steps for the same accuracy, so RK4 usually wins on
total work.

RK4 is a choice, not a law. Most RL quadrotor sims use plain Euler at a smaller
`dt` because it is 4× cheaper and they run billions of steps.

### For the drone

`x` is 18 numbers, every `k` is 18 numbers. `f` runs **4 times per tick**, and
each run unpacks `R`, re-orthonormalizes it, squares four rpms, multiplies by
the allocation matrix, computes quadratic drag, does a cross product, and
inverts `J`.

**`motor_rpm` is not part of the 18.** At `multirotor_model.hpp:332` the ODE
reads `state_.motor_rpm` from the object, not from the vector being integrated,
so all four probes see the same rpm — frozen for the step. Which is the next
section.

## 4. The motor delay

**The physical fact:** a motor and propeller have mass. Command a new speed and
it takes time to get there — about 30 ms on this airframe.

**The model:** a first-order lag, written as an ODE.

```
ṙ = (r_cmd − r) / τ
```

*"The rpm moves toward the commanded rpm at a rate proportional to how far away
it currently is."* Big gap, moves fast. Small gap, crawls. Approaches but never
quite arrives. `τ` is the **time constant**, `motor_time_constant = 0.03 s`.

This is the most reused equation in engineering: a hot cup cooling, a capacitor
charging, a bucket draining through a hole. Same equation every time.

### This one you can solve on paper

Hold `r_cmd` constant across one step of length `dt`:

```
r(t + dt) = r_cmd + ( r(t) − r_cmd )·e^(−dt/τ)
```

Rearranged into old-and-new form:

```
α = e^(−dt/τ)
r_new = α·r + (1 − α)·r_cmd
```

Which is literally `multirotor_model.hpp:244-246`:

```cpp
double filter_const = exp((-dt) / (params_.motor_time_constant));
state_.motor_rpm = filter_const * state_.motor_rpm + (1.0 - filter_const) * input_;
```

### Why this is not inside the RK4

Because it does not need to be. **RK4 is for equations you cannot solve.** This
one has an exact solution, so MRS applies the exact answer directly — faster
*and* more accurate than integrating it numerically.

> General principle: integrate numerically only what you must.

### What α means

With `dt = 0.01`, `τ = 0.03`:

```
α = e^(−0.01/0.03) = e^(−0.3333) = 0.7165
```

Each tick the rpm closes **28.35%** of the remaining gap. A motor slammed from
hover (4255 rpm) to full (7800 rpm):

| tick | t (s) | rpm | gap closed |
|---:|---:|---:|---|
| 0 | 0.00 | 4255 | 0 % |
| 1 | 0.01 | 5260 | 28.3 % |
| 2 | 0.02 | 5980 | 48.7 % |
| 3 | 0.03 | 6496 | **63.2 %** ← one τ |
| 6 | 0.06 | 7322 | 86.5 % ← 2τ |
| 9 | 0.09 | 7623 | 95.0 % ← 3τ |

The 63.2% is not a coincidence — it is `1 − 1/e`. **That is what a time constant
means:** after exactly one τ you have closed 63% of the gap; after 3τ, 95%. The
curve never actually arrives, it just gets boring.

### Why it matters

This 30 ms is the **hard bandwidth limit of the whole vehicle**. The rate
controller runs at 100 Hz and can demand anything it likes; the motors still
need ~30 ms to deliver. A controller tuned to act faster than its actuators
just fights the lag and oscillates.

It is also the single most important number for sim-to-real. Get `τ` wrong and
you train a perfect policy for a drone that does not exist.
