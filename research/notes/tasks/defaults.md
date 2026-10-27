# Defaults

The model that every task shares. A task spec has the same sections and lists only what
it changes. `Default.` in a task spec means that the section here holds.

## Words

| word | meaning |
|---|---|
| episode | one run of a task, from the start state to the end |
| step | one policy step of $\Delta t$ |
| role | a group of drones that share one policy, for example `pursuer` |
| `<name>_<r>` | a name for one role: the name, then the first letter of the role, for example `return_p` or `crash_e`. Two letters name an event of two drones, for example `coll_pp`. The code uses the same names |
| policy | the network that flies the drones of one role. It sees only the actor input |
| critic | the network that estimates the reward still to come. Only training uses it |
| touch | two drones touch when $d_{ij} < d_c$. A drone touches a wall when its ball of radius $\rho$ meets the wall |
| death | a drone fails and stops. Its last reward comes at the step it dies in |
| terminal | an end after which the future value is 0 |
| truncation | an end by the clock. The critic estimates the future value |
| future value | what the learner adds for the time after the end: 0, or the critic's estimate |
| preset | a named set of task values |
| pilot | a hand-written controller that can fly any role in place of its network |
| $[X]$ | 1 when $X$ is true, else 0 |

## Symbols

| symbol | meaning |
|---|---|
| $N$ | the number of drones |
| $p_i$, $v_i$ | position and velocity of drone $i$, world frame |
| $z_i$ | height of drone $i$, the third entry of $p_i$ |
| $R_i$ | rotation of drone $i$, body to world. $R_{i,33}$ is 1 when the drone is level and below 0 when it is upside down |
| $\omega_i$ | body rates of drone $i$ |
| $e_3$ | $(0, 0, 1)$ |
| $d_{ij}$ | $\lVert p_i - p_j \rVert$, the distance between two centres |
| $a_i$ | action of drone $i$ |
| $\omega^{\text{cmd}}_i$ | body-rate command of drone $i$ |
| $c$, $L$ | centre and size of the arena box |
| $B$ | the arena box: $\lvert p_k - c_k \rvert \le L_k / 2$ on each axis $k$ |
| $\rho$ | body radius, arm plus propeller: 0.40 m |
| $d_c$ | contact distance, two body radii: 0.80 m |
| $\delta$ | physics step: 0.005 s |
| $\Delta t$ | policy step: 0.01 s |
| $T$ | the step limit of an episode |
| $K$ | the number of other drones in the actor input |
| $\gamma$ | discount |

## Setup

- The world is the arena box $B$: $L$ = 32 × 32 × 16 m around $c$ = (0, 0, 8) m. The
  floor is at $z = 0$.
- Each task sets its roles and $N$.

## Start

Each task places the drones. Then each drone gets:

- the velocity $v_i \sim U[-0.5, 0.5]^3$ m/s,
- the attitude $R_i = R_z(\psi_i)\, R^{\text{tilt}}_i$. The heading is
  $\psi_i \sim U[0, 2\pi)$. $R^{\text{tilt}}_i$ turns by an angle $\sim U[0, 0.2]$ rad
  about a random axis,
- $\omega_i = 0$, and each motor at hover speed.

## Drone

The X500 quadrotor of the MRS simulator. The physics runs every $\delta$ (200 Hz).

$$
\dot p = v, \qquad m \dot v = -m g\, e_3 + f R e_3 - k_d \lVert v \rVert v, \qquad \dot R = R\, [\omega]_\times, \qquad J \dot\omega = \tau - \omega \times J \omega
$$

The four motor speeds $n_k$, in rpm, give the torque $\tau$ and the thrust $f$:

$$
\begin{pmatrix} \tau \\ f \end{pmatrix} = \operatorname{diag}\left(0.707\, l k_f,\ 0.707\, l k_f,\ 3 k_m r_p k_f,\ k_f\right) \begin{pmatrix} -1 & 1 & 1 & -1 \\ -1 & 1 & -1 & 1 \\ -1 & -1 & 1 & 1 \\ 1 & 1 & 1 & 1 \end{pmatrix} \begin{pmatrix} n_1^2 \\ n_2^2 \\ n_3^2 \\ n_4^2 \end{pmatrix}
$$

Each motor follows its throttle $u_k \in [0, 1]$ with a lag. Over one physics step:

$$
n_k \leftarrow e^{-\delta / \tau_m}\, n_k + \left(1 - e^{-\delta / \tau_m}\right) \left(n_{\min} + (n_{\max} - n_{\min})\, u_k\right)
$$

- The integrator is RK4. The motor speeds stay fixed during a physics step.
- After each physics step, $R$ is made orthonormal again.

| symbol | value |
|---|---|
| $m$ | 2.0 kg |
| $g$ | 9.81 m/s² |
| $l$ | 0.25 m, the arm length |
| $r_p$ | 0.15 m, the propeller radius |
| $J$ | diag(0.0329, 0.0329, 0.0625) kg m² |
| $k_f$ | 2.7087e−7 N/rpm² |
| $k_m$ | 0.07 |
| $k_d$ | $0.30\, \pi l^2$ = 0.0589 kg/m |
| $\tau_m$ | 0.03 s |
| $n_{\min}$, $n_{\max}$ | 1170 rpm, 7800 rpm |
| hover | 4255 rpm, a throttle of 0.4654 |

### Action and control

- The action is $a \in [-1, 1]^4$. A larger value is clipped to this range.
- The throttle is $u = (a_1 + 1)/2$. The body-rate command is
  $\omega^{\text{cmd}} = (4 a_2,\ 4 a_3,\ 2 a_4)$ rad/s.
- The policy acts every $\Delta t$ (100 Hz). Each action holds for 2 physics steps.
- At each physics step, the rate controller gives the torque command
  $\tau^{\text{cmd}} = J \left(4.0\, (\omega^{\text{cmd}} - \omega) - 0.04\, \Delta\omega / \delta\right)$.
  $\Delta\omega$ is the change of $\omega$ over the last physics step.
- The MRS mixer turns $\tau^{\text{cmd}}$ and $u$ into the motor throttles
  $u_1, \dots, u_4$.

## Observation

The actor input of drone $i$. Each vector is in the body frame of drone $i$.

| block | content | size |
|---|---|---|
| own | $R_i^\top v_i$, $\operatorname{vec}(R_i)$, $\omega_i$, $z_i$, and the wall distances $w_i$ | 20 |
| others | for each of the $K$ nearest other drones $j$, nearest first: $R_i^\top (p_j - p_i)$, $R_i^\top (v_j - v_i)$, and $[\text{drone } j \text{ has the role of drone } i]$ | $7K$ |
| mask | 1 for each seat in others that holds a drone, else 0. A seat with mask 0 holds zeros | $K$ |
| goal | set by the task | 0 |

- $K = \min(6, N - 1)$.
- $w_i$ holds the distances from $p_i$ to the walls of $B$ along four rays, level with
  the floor: nose, tail, left and right. The nose ray is the body $x$ axis, projected on
  the floor. If the nose points straight up or down, the body $y$ axis, projected and
  turned by −90°, takes its place. Outside $B$, some distances are negative.
- The inputs are raw SI units.

The critic input is the same for every drone of a role. So the critic gives one value
for all of them.

| block | content | size |
|---|---|---|
| scene | $p_j$, $v_j$, $\operatorname{vec}(R_j)$ and $\omega_j$ of every drone, world frame | $18N$ |
| opponents | this step's action $a_j$ of each drone in another role | 4 per drone |

## Events

- **Death:** drone $i$ dies when it leaves the box ($p_i \notin B$, the floor included),
  turns upside down ($R_{i,33} < 0$), or touches another drone.
- **End:** the episode ends at the first death, which is terminal for every drone, or
  after $T$ steps, which is a truncation for every flying drone. If the task ends the
  episode at step $T$, the task's end holds.
- In every task, a NaN or infinite position kills the drone. The task counts it as its
  crash or wall event.

## Reward

- Each task gives its own reward.
- A per-second term is multiplied by $\Delta t$. A one-off term is paid at its event.
- A drone gets the per-step terms while it is alive at the start of a step.

## Metrics

| name | definition |
|---|---|
| `ep_length` | the number of steps in the episode |
| `return` | the sum of the rewards of a drone, mean over the drones of a role |
| `death_rate` | the share of the drones of a role that are dead at the end |
| `time_limit` | 1 if the episode ended by truncation after $T$ steps, else 0 |

- With more than one role, a per-drone metric has one column for each role, named by
  `<name>_<r>`, for example `return_p`.
- A task metric is read at the last step of the episode, or it is the mean over the
  steps.
- If a metric is NaN in an episode, the mean leaves that episode out.

## Training

PPO, with one actor and one critic for each role. Each role has its own optimizer,
learning rate, reward scale and advantage scale.

| setting | value |
|---|---|
| actor | MLP 2 × 256, tanh. A Gaussian with one learned std per action, the same in every state. The std starts at 1.0. The drawn action is clipped to $[-1, 1]$ |
| critic | MLP 2 × 256, tanh |
| $\gamma$ | 0.998 |
| GAE $\lambda$ | 0.95 |
| PPO clip | 0.2 |
| value loss weight | 0.5 |
| entropy weight | 0.001 |
| gradient norm clip | 0.5 |
| optimizer | Adam, $\epsilon$ = 1e−5 |
| learning rate | starts at 2e−4. At each minibatch, KL > 0.032 divides it by 1.5, and 0 < KL < 0.008 multiplies it by 1.5. It stays in [1e−5, 1e−2] |
| batch | 4096 environments × 64 steps per update |
| epochs × minibatches | 4 × 32 |

- Each drone column divides its reward by the running std of its discounted return.
- The advantages are centred and scaled for each role.
- At a truncation, each flying drone adds $\gamma V$ of the final state to its last
  reward. A NaN or infinite $V$ counts as 0.
- The loss uses the steps where a drone flies, and the step where it dies or arrives.
- The first episode of each environment has a random step limit in $[1, T]$. Its time
  still starts at 0.

## Evaluation

- 1024 episodes, on the fixed seed 1,000,000, far from the training seeds.
- A network flies its mean action.
- Each episode runs once, to its own end.
- Eval flies the task values that the run started with, and the task's eval values on
  top.
- Eval can also fly another preset of the same task, for example a bigger arena.
- A pilot can fly a role in place of its network.

### The cascade

The cascade is a pilot for any role in any task. The MRS cascade flies drone $i$ to
the reference point of the task, with heading 0. It runs every $\Delta t$, and its rate
command goes through the action limits.

| stage | PID gains $k_p$, $k_d$, $k_i$ | output limit |
|---|---|---|
| position | 2.0, 0.15, 0.2 | 6 m/s |
| velocity | 2.0, 0.05, 0.01 | 4 m/s² |
| acceleration | geometry: the attitude and the throttle that give the wanted force | — |
| attitude | 6.0, 0.05, 0.01 | 10 rad/s roll and pitch, 1 rad/s yaw |

### The acceleration chain

A pilot that gives a world acceleration $a$ flies it through these steps, at each
step:

1. The motors limit $a$. The vertical part gets thrust first, and the sideways part gets
   what is left. The thrust stays between the thrust at $n_{\min}$ and at $n_{\max}$.
2. The MRS acceleration controller turns $a$ into an attitude and a throttle. It keeps
   the drone's heading. A NaN throttle becomes 0.
3. The MRS attitude controller, with the cascade's gains and a yaw rate limit of
   2 rad/s, turns the attitude into body rates.
4. The throttle and the body rates become the action, clipped to $[-1, 1]$.
