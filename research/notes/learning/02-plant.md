# The plant — the physics

`mrs_multirotor_simulator/include/mrs_multirotor_simulator/uav_system/multirotor_model.hpp`

Takes 4 motor throttles, moves the drone. That is all it does.

## State

18 numbers get integrated, 4 do not:

| | | |
|---|---|---|
| `x` | 3 | position, **world** frame |
| `v` | 3 | velocity, **world** frame |
| `R` | 9 | attitude, a 3×3 rotation matrix |
| `ω` | 3 | angular velocity, **body** frame |
| `motor_rpm` | 4 | *not* integrated — updated by the lag filter after the step |

Two frames. **World** is nailed to the ground, z up. **Body** is glued to the
drone and tumbles with it. `R` converts body → world.

The one fact that drives everything: **`R.col(2)` is the drone's own up-axis,
written in world coordinates.**

## The ODE (`:344-350`)

```
ẋ = v
v̇ = −g·ẑ  +  (T/m)·R.col(2)  +  F_ext/m  −  drag·v̂/m
Ṙ = R·[ω]×
ω̇ = J⁻¹·( τ  −  ω × Jω  +  M_ext )
```

- **`v̇`** — gravity points down in *world*. Thrust points along the drone's
  *own* up-axis. This is the entire reason a quadrotor **must tilt to move
  sideways**: it can only push one direction, so it aims itself.
- **`Ṙ = R[ω]×`** — `[ω]×` is the skew-symmetric matrix that turns a cross
  product into a matrix multiply. Just "orientation changes at rate ω".
- **`ω̇`** — Newton for rotation. `J` is inertia (how hard to spin about each
  axis). `ω × Jω` is gyroscopic coupling: spin about two axes and you get
  torque about the third for free.

Drag is quadratic: `c·π·L²·|v|²`, opposing travel.

## Where thrust and torque come from (`:334`)

One matrix, one line:

```
[τx, τy, τz, T]ᵀ  =  A · [rpm₁², rpm₂², rpm₃², rpm₄²]ᵀ
```

`A` is the **allocation matrix**. It bakes in the arm geometry plus `kf`
(thrust per rpm² — force goes with rpm **squared**) and `km` (yaw torque a
spinning rotor drags out of the air).

That is the physical heart of the simulator: 4 numbers in, 4 out, one linear
map applied to the squares.

## Throttle vs rpm vs thrust

Three different things, easy to confuse:

```
throttle ∈ [0, 1]                                  ← what the mixer outputs
rpm      = min_rpm + throttle·(max_rpm − min_rpm)  ← linear
thrust   = kf · rpm²                               ← quadratic
```

Hover needs `√(mg / (n·kf))` = **4255 rpm** = throttle **0.465**.

## Motor lag lives outside the ODE (`:244`)

```
α = exp(−dt / 0.03)
rpm ← α·rpm + (1−α)·rpm_commanded
```

30 ms time constant. See 01 for where this comes from and why it is not
integrated.

Note the ordering in `step()`: RK4 runs **first**, then the lag filter updates
`motor_rpm`. So the physics on tick *N* uses the rpm from the end of tick
*N−1* — a deliberate one-tick delay.

## Two hygiene details

`R` drifts off-orthonormal from floating-point error. MRS re-orthonormalizes
every step with a Cholesky-based polar decomposition (`:249`). Skip it and `R`
quietly stops being a rotation.

NaNs are caught and the step is rolled back (`:228`).
