# Němeček 2026

One pursuer chases one evader in a box. A learned or a classical pursuer plays a learned
or a classical evader. In the main setup, both sides learn in turns, each against a pool
of the other side's past policies. Thesis: Václav Němeček, "Strategies for UAV
Pursuit-Evasion Under Active Avoidance", bachelor thesis, CTU FEL, May 2026
([transcript](../../../papers/F3-BP-2026-Nemecek-Vaclav-bachelor_thesis.md)). A value from
his code names its file and line. Each code path starts in the `examples/deep_eagle/`
folder of his repository `rl_tracking_eagle`, branch `cleaned-up-logic`.

## Setup

- One drone has the role `pursuer` (his interceptor), and one drone has the role `evader`
  (his target). So $N = 2$. A role has its own policy, or a pilot flies it. (§1.1, §7.5)
- $p$ is the pursuer, and $e$ is the evader.
- Both roles fly the same drone. (§7.3)
- The world is an arena box above a floor, with no obstacles. (§1.2, §8.4, §9)
- The pursuer tries to come within the capture radius of the evader. The evader tries to
  stay free until the time runs out. (§7.6.1)
- The evader must stay inside the box. (§8.4)
- Both roles have the same top speed and the same top acceleration. (§1.2)

| what | value | source |
|---|---|---|
| arena box $L$ | 20 × 20 × 10 m | §8.4 |
| floor height | $z = 0$ | eq. 7.21 |
| centre $c$ | (0, 0, 5) m | `src/env/environment.cpp:219-221`, with the half-width 10 m of `examples/train_pipeline.py:44` |
| step limit $T$ | 600 steps (12 s at the code's $\Delta t$) | §7.6.1. 12 s: `src/agent/mrs_uav.cpp:87-88` |
| top speed $v_{\max}$, each role | 12.0 m/s | §1.2, §8.1 |
| top acceleration $a_{\max}$, each role | 20.0 m/s² | §1.2, §8.1 |

## Start

- Training draws a new start for each episode. It draws the positions and the
  velocities of both drones. (§3.1)
- CMA-ES and every benchmark fly a fixed set of starts, drawn once before. (§3.1)
- With the recorded paths, the evader starts at the first point of its path, with the
  velocity of the path. (`src/env/environment.cpp:436-453`, `src/agent/target_uav.cpp:57`)

| what | value | source |
|---|---|---|
| position | uniform in a ball of radius 5.0 m | Tables 8.2, 8.3, 8.4 and 8.7. Code: $p_e$ draws $U[-5, 5]$ m on each axis, and $p_p = p_e + U[-5, 5]^3$ m. Each $z$ below 3 m is raised to 3 m (`src/env/environment.cpp:457-488`, `config/config.yaml:56-57`) |
| velocity | $U[-3, 3]^3$ m/s. A $v_z$ below −2 m/s is raised to −2 m/s | §3.1, `src/env/environment.cpp:466-467, 489-490`, `config/config.yaml:58-59` |
| attitude | heading $\psi \sim U[0, 2\pi)$. The body $z$ axis is uniform over the directions within 0.5 rad of vertical | `src/env/environment.cpp:64-83`, `config/config.yaml:60` |
| body rates $\omega$ | $U[-0.5, 0.5]^3$ rad/s | `src/env/environment.cpp:475, 495`, `config/config.yaml:62` |
| motor speeds | 0 rpm | `include/agent/uav_system/uav_system.hpp:170` |

## Drone

The quadrotor of the Flightsim simulator, set to the values of the Holybro X500 V2. Both
roles fly it. (§7.1, §7.2, §7.3)

$$
\dot p = v, \qquad \dot q = \tfrac{1}{2}\, q \otimes \begin{pmatrix} 0 \\ \omega \end{pmatrix}, \qquad m \dot v = R \left( f_T + f_D \right) - m g\, e_3, \qquad J \dot\omega = \tau - \omega \times J \omega
$$

$$
f_T = \begin{pmatrix} 0 \\ 0 \\ f_1 + f_2 + f_3 + f_4 \end{pmatrix}, \qquad f_k = k_f n_k^2, \qquad f_D = -\operatorname{diag}\left(c_{D,x},\ c_{D,y},\ c_{D,z}\right) R^\top v
$$

$$
\tau = \begin{pmatrix} \sqrt{1/2}\; l \left(f_1 - f_2 - f_3 + f_4\right) \\ \sqrt{1/2}\; l \left(-f_1 - f_2 + f_3 + f_4\right) \\ k_m \left(f_1 - f_2 + f_3 - f_4\right) \end{pmatrix}, \qquad \dot n_k = \frac{n^{\text{cmd}}_k - n_k}{\tau_m}, \qquad \tau_m = \begin{cases} \tau_{\text{up}} & \text{if } n^{\text{cmd}}_k > n_k \\ \tau_{\text{down}} & \text{else} \end{cases}
$$

- $q$ is the unit quaternion of $R$, and $\otimes$ is the quaternion product. (eq. 7.2)
- $n_k$ (his $\Omega_k$) is the speed of motor $k$ in rpm, and $n^{\text{cmd}}_k$ (his
  $\Omega_c$) is its command. (eq. 7.5)
- $f_D$ is the drag. It grows linearly with the velocity in the body frame. $c_D$ (his
  drag coefficients) sets it on each body axis. (eq. 7.9)
- The thrust of each motor stays in its range. (eq. 7.8)
- The integrator is RK4. (`include/agent/uav_system/multirotor_model.hpp:171-173`)
- The motor speeds stay fixed during a physics step. Then each motor speed moves by the
  exact solution of its lag over the step.
  (`include/agent/uav_system/multirotor_model.hpp:195-204`)
- After each physics step, $q$ is made a unit quaternion again.
  (`include/agent/uav_system/multirotor_model.hpp:206`)
- After each policy step, a speed above $v_{\max}$ is scaled down to $v_{\max}$. (§7.5.
  The method: `src/agent/mrs_uav.cpp:89-98`)
- The physics values are the same in every episode. (`config/config.yaml:48-53`)

| symbol | value | source |
|---|---|---|
| $m$ | 2.0 kg | Table 7.1 |
| $g$ | 9.81 m/s² | Table 7.1 |
| $l$ | 0.25 m, the arm length | Table 7.1 |
| $r_p$ | 0.15 m, the propeller radius | Table 7.1 |
| $J$ | diag(0.0317, 0.0317, 0.0625) kg m² | Table 7.1 |
| $k_f$ (his $c_f$) | 2.7087e−7 N/rpm² | Table 7.1 |
| $k_m$ (his $\kappa$, the torque constant) | 0.07 | Table 7.1 |
| $\tau_{\text{up}}$, $\tau_{\text{down}}$ (his $k_{\text{up}}$, $k_{\text{down}}$) | 0.0125 s, 0.025 s | Table 7.1 |
| $n_{\min}$, $n_{\max}$ | 1170 rpm, 7800 rpm | Table 7.1 |
| $c_{D,x}$, $c_{D,y}$ | 0.2, with the unit "−" | Table 7.1. Code: the drag force is $c_D$ times the velocity, so the unit is kg/s (`include/agent/uav_system/multirotor_model.hpp:255-280`) |
| $c_{D,z}$ | 0.3, with the unit "−" | Table 7.1 |
| physics step $\delta$ | 0.001 s | `include/agent/uav_system/uav_system.hpp:499-512` |
| torque signs | as in the $\tau$ formula above | eq. 7.10. Code: other motor numbers, with the yaw row $k_m k_f (-n_1^2 - n_2^2 + n_3^2 + n_4^2)$ (`config/config.yaml:34-38`, `include/agent/uav_system/uav_system.hpp:212-217`) |

### Action and control

- The action is $a \in [-1, 1]^4$: a world acceleration and a heading rate. (§7.5)
- The simulator flies the acceleration $a_{\max} (a_1, a_2, a_3)$ and the heading rate
  $\dot\psi_{\max}\, a_4$. (§7.5) Each entry is first clipped to $[-1, 1]$.
  (`src/env/environment.cpp:140-149`)
- A pilot gives a wanted acceleration $a^d$. This becomes the action by: (eq. 7.17)

$$
(a_1, a_2, a_3) = \begin{cases} a^d / a_{\max} & \text{if } \lVert a^d \rVert \le a_{\max} \\ a^d / \left(\lVert a^d \rVert + \epsilon\right) & \text{else} \end{cases}
$$

- The MRS controllers fly the action. The acceleration controller gives a tilt and a
  throttle. The attitude controller gives body rates. The rate controller gives the
  torques, and the mixer gives the motor commands.
  (`include/agent/uav_system/uav_system.hpp:468-511`)
- The attitude controller uses a quaternion error: $q_e = q_d^{*} \otimes q$, and the
  error vector is $e_R = -2 s\, (q_{e,x}, q_{e,y}, q_{e,z})$. $q_d$ is the wanted attitude,
  and $s$ is 1 when $q_{e,w} \ge 0$, else −1. (§7.4, eq. 7.12–7.14)

| what | value | source |
|---|---|---|
| policy step $\Delta t$ | 0.02 s (50 Hz) | `src/agent/mrs_uav.cpp:87-88`, `examples/eval_rl_pur.py:59` |
| $\dot\psi_{\max}$ | 6.28 rad/s | `config/config.yaml:174` |
| $\epsilon$ | 1e−6 | §3.2.1 gives it for APF. For eq. 7.17: `examples/agents.py:51, 114` |
| acceleration and attitude controllers | every 0.01 s (100 Hz) | `include/agent/uav_system/uav_system.hpp:468-482`, `src/agent/mrs_uav.cpp:87-88` |
| rate controller and mixer | every 0.005 s (200 Hz) | `include/agent/uav_system/uav_system.hpp:496-511` |
| attitude PID gains $k_p$, $k_d$, $k_i$ | 6.0, 0.05, 0.0 | `config/config.yaml:41-44` |
| attitude output limits | 15 rad/s roll and pitch, 1.0 rad/s yaw | `config/config.yaml:45-46` |
| rate PID gains $k_p$, $k_d$, $k_i$ | 4.0, 0.04, 0.0, each times $J$ | `include/agent/uav_system/controllers/rate_controller.hpp:16-18, 62-64` |

## Observation

The actor input of drone $i$.

| block | content | size | source |
|---|---|---|---|
| time | $t / T$, with $t$ the step of the episode | 1 | eq. 7.15, eq. 7.16 |
| opponent | $\Delta p = p_e - p_p$, $d_{pe}$, $\Delta v = v_e - v_p$ | 7 | eq. 7.15, eq. 7.16 |
| arena, pursuer | $z_p$ | 1 | eq. 7.15 |
| arena, evader | $p_e$ | 3 | eq. 7.16 |
| own | $\operatorname{vec}(R_i)$, $v_i$, $\omega_i$ | 15 | eq. 7.15, eq. 7.16 |

- The pursuer reads 24 values, and the evader reads 26. (§7.5) Code: each network reads 3
  frames, this step and the 2 steps before, so 72 and 78 values.
  (`include/env/environment.h:59`, `examples/train_pipeline.py:89-91`)
- Each drone sees only its opponent, so $K = 1$. (§7.5)
- Both roles read $\Delta p$ and $\Delta v$ with the same sign, from the pursuer to the
  evader. (eq. 7.15, eq. 7.16)
- $\Delta p$, $\Delta v$, $p_e$ and $v_i$ are in the world frame. $\omega_i$ is in the body
  frame. (`src/env/environment.cpp:355-401`)
- The env scales each input by its largest value. (§7.5) The network reads the inputs as
  the env gives them. (`examples/train_pipeline.py:465-479`)

| input | scaled to | source |
|---|---|---|
| $\Delta p$, $d_{pe}$, and the $x$ and $y$ of $p_e$ | divided by the half-width $L_x / 2$ of the box, at most 200 m | `src/env/environment.cpp:106, 113-114, 125` |
| $z_p$, $z_e$ | $z / (L_z / 2) - 1$, which maps $[0, L_z]$ to $[-1, 1]$ | `src/env/environment.cpp:118, 126` |
| $\Delta v$ | divided by $v_{\max,p} + v_{\max,e}$ | `src/env/environment.cpp:107, 115` |
| $v_i$ | divided by the drone's own $v_{\max}$ | `src/env/environment.cpp:120, 128` |
| $\omega_i$ | divided by $\dot\psi_{\max}$ | `src/env/environment.cpp:121, 129` |
| $t / T$ | as it is | `src/env/environment.cpp:122, 130` |
| $\operatorname{vec}(R_i)$ | as it is | `src/env/environment.cpp:119, 127` |

The critic input is the actor input. (`examples/train_pipeline.py:464-472`)

## Events

- **Capture:** the distance $d_{pe}$ between the two centres falls to the capture radius
  $r_{\text{cap}}$ or below. (eq. 7.18)
- **Crash** (his ground collision): a drone reaches the floor, $z \le 0$. (eq. 7.21)
- **Outside** (his boundary box collision): the evader is outside the box. The episode
  goes on. (§8.4, §8.6.1)
- **Timeout:** $T$ steps pass with no capture. (eq. 7.19)
- **End:** the episode ends at a capture, at a crash of either drone, or at the timeout.
  (§7.6.1)
- At one step, the code keeps the last of these that holds: capture, crash of the
  evader, crash of the pursuer, timeout. (`src/env/environment.cpp:210-249, 337-339`)

| what | value | source |
|---|---|---|
| capture radius $r_{\text{cap}}$ | 0.5 m | §7.6.1 |
| the box of Outside | $\lvert x \rvert \le L_x / 2$, $\lvert y \rvert \le L_y / 2$, $z \le L_z$ | `src/env/environment.cpp:219-221` |
| which ends are terminal | every end, the timeout included | `src/env/simulation.cpp:85-91`, and `examples/train_pipeline.py:240-272` gives no truncation flag |

## Reward

$$
\begin{aligned}
r_p ={}& \kappa_a \left(d^{\,t-1}_{pe} - d^{\,t}_{pe}\right) - \kappa_{br} \lVert \omega_p \rVert + \kappa_c\, [\text{capture}] - \kappa_b\, [\text{outside}_p] - \kappa_g\, [\text{crash}_p] - \kappa_\tau\, [\text{timeout}] \\
r_e ={}& \kappa_s - \kappa_{br} \lVert \omega_e \rVert - \kappa_c\, [\text{capture}] - \kappa_b\, [\text{outside}_e] - \kappa_g\, [\text{crash}_e] + \kappa_\tau\, [\text{timeout}]
\end{aligned}
$$

- Each term is paid at each step at which it holds. No term has a factor $\Delta t$.
  (§7.6.3)
- $d^{\,t}_{pe}$ is the distance at step $t$. So the pursuer earns $\kappa_a$ for each metre
  it closes. (eq. 7.22)
- $\lVert \omega \rVert$ is the drone's own body rate. (eq. 7.22) Code: the rate the drone
  flies, not its command. (`src/env/environment.cpp:331-332`)
- Each drone pays $\kappa_g$ for its own crash. (eq. 7.21–7.23)
- Eq. 7.22 also charges the pursuer for Outside. §8.4 and §8.6.1 bind only the evader to
  the box. Code: only the evader pays $\kappa_b$. (`src/env/environment.cpp:87-95`)
- The thesis gives the weights for the adversarial training. (Table 8.8) Code: every PPO
  run uses the same weights. (`src/env/environment.cpp:85-96`,
  `config/config.yaml:185, 195-196`)

| symbol | value | source |
|---|---|---|
| $\kappa_a$ (his approach reward) | 0.5 per m | Table 8.8 |
| $\kappa_s$ (his step reward) | 0.007 per step | Table 8.8 |
| $\kappa_{br}$ (his body-rate penalty) | 0.0005 per rad/s, per step | Table 8.8 |
| $\kappa_c$ (his capture bonus or penalty) | 10.0 | Table 8.8 |
| $\kappa_\tau$ (his timeout, $\kappa_t$ in Table 8.8) | 10.0 | Table 8.8 |
| $\kappa_b$ (his out-of-bounds penalty) | 0.1 per step | Table 8.8 |
| $\kappa_g$ (his ground collision penalty) | 10.0 | Table 8.8 |

## Metrics

| name | when | definition | source |
|---|---|---|---|
| `caught_e` (his interception success rate) | at the end | the pursuer captures the evader before the timeout | §8.1, Fig. 8.10 |
| `time_limit` (his evasion success rate) | at the end | the evader stays free until the timeout | §8.2, Table 8.3 |
| `OOB_e` (his collision with the arena limits) | at the end | the evader was outside the box | §8.4 |
| `time_to_catch` (his average interception time) | at the end | the time of the capture, in s. Only captures count | Tables 8.2 and 8.7. Only captures: `examples/eval_rl_pur.py:43-60` |
| `ep_length` (his average episode length) | at the end | the number of steps in the episode | Fig. 8.2, Fig. 8.11 |
| `return` (his average reward) | at the end | the sum of the rewards of a drone | Fig. 8.2, Fig. 8.11 |

- `return` and `ep_length` come from the training episodes. (Fig. 8.2–8.4, 8.11, 8.13)
- Fig. 8.14 gives `time_to_catch` in steps. (Fig. 8.14)
- For each pair of evader limits, Tables 8.3 and 8.4 give a time in s, headed "average
  interception time". §8.2 calls it the average survival time. Which episodes it counts:
  not published.

## Training

PPO of the Stable-Baselines3 library (SB3), with one actor and one critic for each role.
The roles learn in turns. (§5.3, §6.3, §7.1)

| setting | value | source |
|---|---|---|
| actor | MLP 3 × 256, ReLU. A Gaussian with one learned std per action, the same in every state. The std starts at 1.0. The drawn action is clipped to $[-1, 1]$ | `examples/train_pipeline.py:464-472`, with the SB3 defaults of its `MlpPolicy` |
| critic | MLP 3 × 256, ReLU, a network of its own | `examples/train_pipeline.py:464` |
| $\gamma$ | 0.99 | Table 7.3 |
| GAE $\lambda$ | 0.95 | Table 7.3 |
| PPO clip | 0.1 | Table 7.3 |
| learning rate | 3e−4, fixed | Table 7.3. Fixed: `examples/train_pipeline.py:472` |
| entropy weight | 0.001 | Table 7.3 |
| value loss weight | 1.0 | Table 7.3 |
| gradient norm clip | 5.0 | Table 7.3 |
| batch | 100 environments × 600 steps (his horizon $n_{\text{steps}}$) per update | Table 7.3. Environments: `examples/train_pipeline.py:51`, `config/config.yaml:216` |
| minibatch | 12,000 steps (his batch size), so 5 minibatches | Table 7.3 |
| epochs | 10 | `examples/train_pipeline.py:472` sets none, so the SB3 default holds |
| optimizer | Adam, with the SB3 defaults | `examples/train_pipeline.py:472` |
| reward scale | the reward as it is | `examples/train_pipeline.py:465-479` |
| advantage scale | centred and scaled in each minibatch | `examples/train_pipeline.py:472` sets none, so the SB3 default holds |
| steps | the evader: 8 phases, 240M steps. The pursuer: 9 phases, 270M steps | Fig. 8.11, Fig. 8.13. Phases: `examples/train_pipeline.py:485-528` |
| hardware | CPU | `examples/train_pipeline.py:2, 472` |
| training time | not published | — |

- Table 7.3 holds the values of the adversarial training, for both roles. (§7.7)
- Each phase starts the learner from its own last weights. (Algorithm 2) Code: it also
  keeps the optimizer state. (`examples/train_pipeline.py:472-528`)

### Pools

His AMSPB: asynchronous multi-stage population-based training. (§6.3)

- Each role keeps a pool of frozen opponents. A pool only grows. (§6.3, Algorithm 2)
- The first members, his generation −1, are pilots. The pursuer pool starts with FRPN.
  The evader pool starts with the recorded paths, and APF joins it later. (§8.6.2,
  Algorithm 2)
- Generation 0, his pretraining: the pursuer learns from random weights against the
  recorded paths. Then it learns more, against the recorded paths and APF. The evader learns
  from random weights against FRPN. Then each copy joins its own pool. (Algorithm 2
  lines 1–6, §8.6.1)
- Each generation after 0 has two phases. First the evader learns against the pursuer
  pool, and a copy joins the evader pool. Then the pursuer learns against the evader
  pool, the new evader included. A copy joins the pursuer pool. (Algorithm 2 lines 7–12)
- Each phase runs for a fixed number of steps. (§8.6.3) Code: the pretraining of the
  pursuer has one phase for each set of opponents. (`examples/train_pipeline.py:491, 503`)
- At the start of each episode, the learner draws its opponent. If the pool has one
  member, it gets that member. Else, with the chance $p_{\text{old}}$, it gets a uniform
  draw from the pool without its newest member. Else it gets the newest member. (§6.3,
  eq. 6.1, Algorithm 2)
- Each environment draws its own opponent. (`examples/train_pipeline.py:153-176, 260`)
- A frozen network flies its mean action. (`examples/train_pipeline.py:228`)

| what | value | source |
|---|---|---|
| generations after 0 | 7 | Fig. 8.10, Fig. 8.12 |
| steps per phase | 30M | §8.6.3 |
| $p_{\text{old}}$ | 0.75 | §8.6.2 |

### CMA-ES

CMA-ES (covariance matrix adaptation evolution strategy) tunes the pilots. Each
generation, it draws a set of parameter values from a Gaussian. Then it moves the
Gaussian toward the best ones. (§4)

- Each parameter set flies the same fixed set of starts. (§3.1)
- The fitness $F$ counts the captures, and then the time to each capture: (eq. 7.20)

$$
F = \sum_i [\text{capture}_i] \left( A + B\, \frac{T - t_i}{T} \right)
$$

- $i$ runs over the episodes, and $t_i$ is the step of the capture in episode $i$. A
  capture always counts more than any gain in time. (§7.6.2)
- For FRPN, CMA-ES looks for the values with the highest $F$. (§7.6.2) The APF with walls
  uses the fitness of §7.6 too. (§8.4)

| what | value | source |
|---|---|---|
| $A$ | 10 | Table 7.2 |
| $B$ | 1 | Table 7.2 |
| FRPN: the search | $G$ and $W$, 25 generations against the recorded paths | §8.1, Fig. 8.1. Code: 20 parameter sets in each generation, at most 500 in all, on 200 starts (`examples/cmaes_pokus.py:245, 258-264`) |
| FRPN: the fitness in the code | $100\, F$ | `examples/cmaes_pokus.py:125-129` |
| APF without walls: the search | $k_{sp}$, $w_{vp}$, $k_{dp}$ against FRPN, once for each pair of evader limits | §8.2, Table 8.5. Code: 12 parameter sets in each generation, at most 1000 in all, on 200 starts (`examples/cmaes_apf_cluster.py:166-174`) |
| APF without walls: the fitness in the code | CMA-ES looks for the lowest value of: 1000 per capture, less 1/6 for each step that the evader stays free | `examples/cmaes_apf_cluster.py:96-132` |
| APF with walls: the search | all seven values against FRPN, in the box | §8.4, Table 8.6. Code: 15 parameter sets in each generation, at most 500 in all, on 200 starts (`examples/cmaes_evader.py:267, 302-308`) |
| APF with walls: the fitness in the code | CMA-ES looks for the lowest value of: 100 per capture, plus $(600 - t_i)/60$ per capture, plus 1000 for each evader that leaves the box or crashes | `examples/cmaes_evader.py:225-235` |

## Evaluation

- Every benchmark flies one fixed set of starts. (§3.1)
- A network flies its mean action. (`examples/eval_rl_pur.py:31`,
  `examples/test_arenas_all.py:39`)
- Eval flies the training arena. (Fig. 8.10, Fig. 8.12) In the larger arenas, $T$ stays
  600 steps. (§8.6.4)
- The matrix of generations: each pursuer generation $k$ plays the recorded paths, APF,
  and each evader generation from 0 to $k$. A cell holds `caught_e`. (Fig. 8.10,
  Fig. 8.12)
- The arena test: in each arena, the pursuer of generation 7 plays each evader generation
  from 0 to 7 of its own run. The test gives the mean `caught_e` and the mean
  `time_to_catch` over these 8 evaders. (§8.6.4, Fig. 8.14)

| benchmark | episodes | source |
|---|---|---|
| FRPN against the recorded paths | 1,000 | Table 8.2. §8.1 gives 10 starts for each of the 200 paths |
| the evaders, for each pair of limits | 1,000 | Tables 8.3, 8.4 |
| the pursuer of `ppo_pursuer` against the recorded paths and APF | 2,000 | Table 8.7 |
| the matrix of generations | not published | — |
| the arena test | 100 for each evader generation | `examples/test_arenas_all.py:25-38` |

A pilot reads the true positions and velocities of both drones. (§2)

| pilot | role | what flies | source |
|---|---|---|---|
| FRPN | pursuer | fast response proportional navigation: it aims at where the evader will be, plus a pull straight at the evader | §2.3 |
| APF | evader | an artificial potential field: the pursuer pushes the evader away, and in the box the walls push it back in | §3.2 |
| recorded paths | evader | the evader replays one of 200 recorded agile flights | §3.1 |

### FRPN

$$
a^d = G \left( (1 - W)\, \frac{\Delta p + \Delta v\, t_{\text{go}}}{t_{\text{go}}^2} + W\, \Delta p \right), \qquad t_{\text{go}} = \frac{\lVert \Delta p \rVert}{\lVert \Delta v \rVert}
$$

- $\Delta p = p_e - p_p$ and $\Delta v = v_e - v_p$. (eq. 2.4, eq. 2.5)
- The heading rate is 0. (`examples/agents.py:86`)

| symbol | value | source |
|---|---|---|
| $G$, the gain | 6.677 | Table 8.1. The pool in the code: 6.65, as 0.665 × 10 (`examples/train_pipeline.py:462`, `examples/agents.py:60`) |
| $W$, the weight of the pull | 0.337 | Table 8.1. The pool in the code: 0.321 (`examples/train_pipeline.py:462`) |

### APF

$$
f_p = \frac{k_{sp} + w_{vp}\, v^{\text{clos}}_p}{\lVert r \rVert^{k_{dp}} + \epsilon}\, \hat r, \qquad r = p_e - p_p, \qquad \hat r = \frac{r}{\lVert r \rVert + \epsilon}, \qquad v^{\text{clos}}_p = \max\left( (v_p - v_e) \cdot \hat r,\ 0 \right)
$$

$$
f_{w,k} = \begin{cases} \left( k_{sw} + w_{vw}\, v^{\text{clos}}_{w,k} \right) \left( \dfrac{1}{d_{w,k} + \epsilon} - \dfrac{1}{R_w} \right)^{k_{dw}} & \text{if } d_{w,k} < R_w \\ 0 & \text{else} \end{cases}, \qquad a^d = f_p + \sum_{k \in \{x, y, z\}} f_{w,k}
$$

- $d_{w,k}$ is the distance to the nearest wall along axis $k$, and $v^{\text{clos}}_{w,k}$
  is the evader's speed toward it. $f_{w,k}$ pushes along axis $k$, away from that wall.
  (eq. 3.2) Code: each of the two walls of an axis pushes when it is nearer than $R_w$.
  (`examples/agents.py:134-150`)
- The heading rate is 0. (`examples/agents.py:158-159`)
- Without walls, the APF has only $f_p$. (§3.2, Table 8.5)

| symbol | with walls | without walls, at 12 m/s and 20 m/s² | source |
|---|---|---|---|
| $k_{sp}$, the push of the pursuer | 2.14 | 160.17 | Tables 8.5, 8.6 |
| $w_{vp}$, its gain on the closing speed | 1.37 | 13.61 | Tables 8.5, 8.6 |
| $k_{dp}$, its power of the distance | 1.03 | 2.15 | Tables 8.5, 8.6 |
| $k_{sw}$, the push of a wall | 57.89 | — | Table 8.6 |
| $w_{vw}$, its gain on the closing speed | 29.7 | — | Table 8.6 |
| $R_w$, the reach of a wall | 10.0 m | — | Table 8.6 |
| $k_{dw}$, its power | 0.57 | — | Table 8.6 |

### Recorded paths

- The evader copies the full recorded state of its path at each step. (§3.1) Code: one
  sample for each step, back to the start at the end of the path.
  (`src/agent/target_uav.cpp:72-79`, `src/env/environment.cpp:184-198`)
- Profile A holds 100 paths, and profile B holds 100. (§3.1)
- Code: each environment holds one path of each profile, and each episode draws the
  profile. (`src/env/simulation.cpp:45-54`, `examples/train_pipeline.py:158-160`)

| profile | top speed | lowest top speed of a path | mean speed | top acceleration | lowest top acceleration of a path | mean acceleration | source |
|---|---|---|---|---|---|---|---|
| A | 9.97 m/s | 0.00 m/s | 1.69 m/s | 3.63 m/s² | 0.00 m/s² | 0.61 m/s² | Table 3.1 |
| B | 6.52 m/s | 2.20 m/s | 1.85 m/s | 7.00 m/s² | 3.66 m/s² | 2.98 m/s² | Table 3.1 |

## Presets

Training describes `pool`. Each other setup changes who learns, the opponent, the arena,
or the evader's limits. The names are ours. (§8)

| preset | who learns, and how | opponent | arena | evader limits | source |
|---|---|---|---|---|---|
| `frpn_tune` | the FRPN pursuer, by CMA-ES | the recorded paths | open space | those of the paths | §8.1, §8.4. Code: a half-width of 200 m, and a crash ends nothing (`examples/cmaes_pokus.py:230-231`) |
| `apf_grid` | the APF evader without walls, by CMA-ES, for each pair of limits | FRPN | open space, with no Outside cost | $v_{\max}$ of 3, 6, 9 or 12 m/s, with $a_{\max}$ of 5, 10, 15 or 20 m/s², each pair | §8.2, Table 8.3. Code: a half-width of 200 m, and a crash ends nothing (`examples/cmaes_apf_cluster.py:155-157`) |
| `ppo_evader_grid` | the evader, by PPO, for 50M steps for each pair of limits | FRPN | open space | as `apf_grid` | §8.3, Fig. 8.4. Code: a half-width of 200 m. A crash ends nothing, and the evader under the floor pays $\kappa_b$. PPO has the SB3 defaults: clip 0.2, value loss weight 0.5, gradient norm clip 0.5. The entropy weight falls linearly from 0.01 to 0.005 (`examples/grid_search_frpn_rl.py:77-79, 96-97`, `src/env/environment.cpp:228-229`) |
| `apf_box` | the APF evader with walls, by CMA-ES | FRPN | the box | as Setup | §8.4 |
| `ppo_pursuer` | the pursuer, by PPO: generation 0 of the pools | the recorded paths, then the recorded paths and the APF of `apf_box` | the box | as Setup | §8.5, §8.6.1 |
| `pool` | both roles, by PPO, in generations | the pool of the other role, as in Training | the box | as Setup | §8.6.2 |
| `no_pool` | as `pool`, with $p_{\text{old}}$ = 0.0 (his adversarial learning without an opponent pool) | the newest member of the other role's pool | the box | as Setup | §8.6.2 |
| `pool_<L>` | as `pool`, from random weights | as `pool` | each box of Table 8.9: 20 × 20 × 10, 30 × 30 × 15, 40 × 40 × 20, 60 × 60 × 30, 80 × 80 × 40 and 160 × 160 × 80 m | as Setup | §8.6.4, Table 8.9. Code: the 20 × 20 × 10 m run is the `pool` run (`examples/test_arenas_all.py:91`) |

- The thesis calls `pool` adversarial learning with an opponent pool. (§8.6.2)
- In `frpn_tune`, `apf_grid` and `ppo_evader_grid`, the pursuer has the $v_{\max}$ and
  the $a_{\max}$ of Setup. (§8.1, §8.2)
