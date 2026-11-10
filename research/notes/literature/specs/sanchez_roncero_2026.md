# Sánchez Roncero et al. 2026, v3

One pursuer chases one evader. Both sides learn. Each side trains in generations, against
a pool of the other side's past policies. Paper:
[arXiv:2506.02849v3](https://arxiv.org/abs/2506.02849v3).

## Setup

- One drone has the role `pursuer`, and one drone has the role `evader`. So $N = 2$.
  (§III-B)
- Both roles fly the same drone. (§III-B)
- The world is an arena box. (Table I)
- The pursuer tries to come within the capture radius of the evader. The evader tries to
  stay free until the time runs out. (§III-B)
- $p$ is the pursuer, and $e$ is the evader. (§III-B)

| what | value | source |
|---|---|---|
| arena box $L$ | 4 × 5 × 2 m | Table I |
| centre $c$ and floor height | not published | — |
| step limit $T$ (his horizon) | 10 s | Table I |

## Start

- In training, each drone starts at a random point of the box. (§IV-A-a, §IV-A-b)
- Eval uses its own start points. See Evaluation. (§IV-A-c)

| what | value | source |
|---|---|---|
| clearance from the faces | at least 0.5 m | §IV-A-a |
| clearance between the drones | at least 0.5 m | §IV-A-a |
| attitude | roll and pitch in ±5°, yaw in ±180° | §IV-A-a |
| velocity | in ±0.1 m/s | §IV-A-a |
| the distributions, and if the velocity bound holds per axis | not published | — |
| body rates and motor speeds | not published | — |

## Drone

- The Crazyflie 2.1 brushless quadrotor. Its values are coarsely adapted from the
  identification of the brushed Crazyflie in Förster 2015. (§III-A)
- Isaac Lab simulates it, with physics on the GPU. (§IV-A-a)
- The state is $(p, q, v, \omega, n)$. $q$ is the attitude quaternion, body to world, and
  $R = R(q)$. $n_k$ (his $\Omega_k$) are the four rotor speeds. (§III-A)

The motion: (eq. 1)

$$
\dot p = v, \qquad \dot q = \tfrac{1}{2}\, q \otimes (0, \omega), \qquad m \dot v = -m g\, e_3 + f R e_3 + f_{\text{aero}}, \qquad J \dot\omega = \tau - \omega \times J \omega
$$

Rotor $k$ gives the thrust $f_k = k_f n_k^2$ (his $k_\eta \Omega_k^2$) and the moment
$k_\tau n_k^2$ (his $k_m \Omega_k^2$). They give the thrust $f$, the torque $\tau$ and the
drag $f_{\text{aero}}$: (eq. 2, §III-A)

$$
f = \sum_k f_k, \qquad \tau = M \left(f_1, f_2, f_3, f_4\right)^\top, \qquad f_{\text{aero}} = -\Big(\sum_k n_k\Big) K_{\text{aero}} R^\top v, \qquad K_{\text{aero}} = \operatorname{diag}(k_{xy}, k_{xy}, k_z)
$$

- $M$ is the 3 × 4 allocation matrix of the X frame. It depends on the arm length $l$ and
  on the ratio $k_\tau / k_f$. (§III-A)
- $f_{\text{aero}}$ is the drag, in the body frame. (§III-A)

Each motor follows its speed command $n^{\text{cmd}}_k$ (his $\Omega^{\text{cmd}}$) with a
lag. Over one physics step: (eq. 3)

$$
n_k \leftarrow e^{-\delta / \tau_m}\, n_k + \left(1 - e^{-\delta / \tau_m}\right) n^{\text{cmd}}_k
$$

- Then a clip keeps $n_k$ in $[0, n_{\max}]$. So the thrust is at most
  $4 k_f n_{\max}^2$. (§III-A)

| what | value | source |
|---|---|---|
| $m$, $J$, $l$ | not published | — |
| $k_f$, $k_\tau$ | not published | — |
| $\tau_m$ (his $\tau_{\text{mot}}$) | not published | — |
| $n_{\max}$ (his $\Omega_{\max}$), and the unit of $n_k$ | not published | — |
| $k_{xy}$, $k_z$ | not published | — |
| physics step $\delta$ | 0.002 s (500 Hz) | §IV-A-a, Table I |
| policy step $\Delta t$ | 0.02 s (50 Hz) | §IV-A-a, Table I |
| integrator | not published | — |

### Action and control

- The action is $a \in [-1, 1]^4$. (§III-A-a, §III-B)
- The action gives the body-rate command and the thrust command $f^{\text{cmd}}$ (his
  $T^{\text{cmd}}$): (eq. 4)

$$
\omega^{\text{cmd}} = \left(a_1\, \omega^{xy}_{\max},\ a_2\, \omega^{xy}_{\max},\ a_3\, \omega^{z}_{\max}\right), \qquad f^{\text{cmd}} = \frac{a_4 + 1}{2}\, f_{\max}
$$

- The policy acts every $\Delta t$. (§III-A-a, §IV-A-a)
- A PID rate controller runs every $\delta$ and tracks $\omega^{\text{cmd}}$. It gives a
  wanted angular acceleration. (§III-A-a, Table I)
- The angular acceleration and $f^{\text{cmd}}$ form a wrench: a force and a torque
  together. The allocation matrix maps the wrench to four rotor thrusts. $k_f$ turns
  them into the speed commands $n^{\text{cmd}}_k$. (§III-A-a)

| what | value | source |
|---|---|---|
| $\omega^{xy}_{\max}$, $\omega^{z}_{\max}$ | not published | — |
| $f_{\max}$ (his $T_{\max}$) | not published | — |
| gains of the rate PID | not published | — |

### Domain randomization

- Training draws the drone values from these ranges, through all of training. Eval flies
  the nominal values. (§IV-A-a, §IV-A-c)
- The ranges have no unit. Each one scales the nominal value. (Table I)

| value | range | source |
|---|---|---|
| $m$ | [0.9, 1.1] | Table I |
| $J$ | [0.8, 1.2] | Table I |
| $k_f$ | [0.8, 1.2] | Table I |
| $k_\tau$ | [0.8, 1.2] | Table I |
| $\tau_m$ | [0.8, 1.2] | Table I |
| drag | [0.5, 2.0] | Table I |
| rate PID $K_p$ | [0.85, 1.15] | Table I |
| rate PID $K_i$ | [0.85, 1.15] | Table I |
| rate PID $K_d$ | [0.7, 1.2] | Table I |
| when the values are drawn, and from what distribution | not published | — |

## Observation

The actor input of drone $i$ (his $o_t$). Drone $j$ is its opponent. (eq. 7)

| block | content | size | source |
|---|---|---|---|
| own | $\operatorname{vec}(R_i)$ (his $R_{\text{flat}}$), $R_i^\top v_i$, $\omega_i$, and $p_i$ in the world frame | 18 | eq. 7 |
| others | $R_i^\top (p_j - p_i)$, $R_i^\top (v_j - v_i)$ | 6 | eq. 7 |

- Both roles use the same layout, each in its own frame. (§III-B)
- $K = 1$. The input has 24 entries. (eq. 7)
- How the inputs are scaled before a network reads them: not published.

The critic input is the actor input $o_t$ (his symmetric actor-critic). (§III-C-c)

## Events

- **Capture:** the distance between the centres falls to the capture radius:
  $d_{pe} \le r_c$. (eq. 5)
- **OOB** (his out-of-bounds): a drone leaves the arena box. (§III-B)
- **End:** the episode ends at a capture, at an OOB, or after $T$ (his escape, a
  timeout). (§III-B)
- A capture is terminal. An OOB and the step limit are truncations: the critic estimates
  the future value. (§III-B-b)

| what | value | source |
|---|---|---|
| capture radius $r_c$ | 0.1 m | Table I |
| contact between the drones | not published | — |

## Reward

$$
\begin{aligned}
r_p ={}& \lambda_{\text{catch}}\, \beta\, [\text{capture}] + \lambda_{\text{app}} \left(d_{t-1} - d_t\right) + r^{\text{reg}}_p \\
r_e ={}& \lambda_{\text{step}} - \lambda_{\text{catch}}\, [\text{capture}] + r^{\text{reg}}_e \\
r^{\text{reg}}_i ={}& -\lambda_{\text{rate}} \lVert \omega_i \rVert^2 - \lambda_{\text{smooth}} \lVert a^\omega_{i,t} - a^\omega_{i,t-1} \rVert - \lambda_{\text{fail}}\, [\text{OOB}_i]
\end{aligned}
$$

$$
\beta = \exp\left(-\alpha_{\text{close}} \max\left(0,\ v_{\text{close}} - v_{\text{safe}}\right)\right), \qquad v_{\text{close}} = -\hat u^\top \left(v_e - v_p\right)
$$

- $d_t$ is $d_{pe}$ at step $t$. (eq. 5, eq. 9)
- $\lambda_{\text{app}}$ pays the pursuer for each metre that it closes on the evader.
  (eq. 9)
- $\beta$ shrinks the capture bonus of the pursuer when the closing speed
  $v_{\text{close}}$ passes $v_{\text{safe}}$. $\hat u$ is the unit vector along the line
  of sight. (eq. 9, eq. 10)
- The evader pays the full $\lambda_{\text{catch}}$ at a capture. (eq. 11)
- Both drones pay $r^{\text{reg}}$ (his $r^{\text{reg}}_t$). (eq. 8)
- $\lambda_{\text{rate}}$ costs the body rates $\omega_i$. $\lambda_{\text{smooth}}$
  costs the change of $a^\omega_i = (a_1, a_2, a_3)$, the body-rate part of the action,
  from one step to the next. (eq. 8)
- Each drone pays $\lambda_{\text{fail}}$ only for its own OOB. (§III-B-a)
- The terms are per step. The paper writes no $\Delta t$ in them. (eq. 8–11)

| symbol | value | source |
|---|---|---|
| $\lambda_{\text{catch}}$ (his $\kappa_c$) | 10.0 | §IV-A-c |
| $\lambda_{\text{app}}$ (his $\kappa_{\text{app}}$) | 0.25 | §IV-A-c |
| $\lambda_{\text{step}}$ (his $\kappa_s$) | 0.0025 | §IV-A-c |
| $\lambda_{\text{rate}}$ (his $\kappa_{\text{br}}$) | 0.001 | §IV-A-c |
| $\lambda_{\text{smooth}}$ (his $\kappa_{\text{sm}}$) | 0.002 | §IV-A-c |
| $\lambda_{\text{fail}}$ (his $\kappa_b$) | 10.0 | §IV-A-c |
| $v_{\text{safe}}$ | 2 m/s | §IV-A-c |
| $\alpha_{\text{close}}$ | 3.0 | §IV-A-c |
| units of the weights | not published | — |

## Metrics

| name | when | definition | source |
|---|---|---|---|
| `caught_e` (his C, capture) | at the end | the episode ends in a capture | §III-B, Table II |
| `time_limit` (his E, escape) | at the end | the evader stays free for $T$ | §III-B, Table II |
| `OOB_p` (his P-OOB) | at the end | the pursuer leaves the box | §III-B, Table II |
| `OOB_e` (his E-OOB) | at the end | the evader leaves the box | §III-B, Table II |
| pursuer win rate | at the end | the share of episodes with `caught_e` or `OOB_e` | §III-C-b |
| evader win rate | at the end | the share of episodes with `time_limit` or `OOB_p` | §III-C-b |
| balanced | at the end | for each seed, the lower of the two population scores, of the pursuer and of the evader | Fig. 4 |
| change against `AMSPB` | at the end | the win rate of a Hedge setup minus the win rate of `AMSPB` | Fig. 5 |
| win rate against the pilots | at the end | the mean win rate against the pilots | Fig. 5 |

- The first four metrics are the outcomes. Each episode ends in exactly one of them.
  (§III-B)
- What a population score is: not published.

On hardware, Table II also gives these for each pair:

| name | definition | source |
|---|---|---|
| `ep_length` (his $T$) | the length of the episode, in s: mean and top value | Table II |
| distance | $d_{pe}$, in m: mean and least value | Table II |
| closing speed | $v_{\text{close}}$, in m/s, over the captures only: mean and top value | Table II |
| speed | $\lVert v_i \rVert$ of each drone, in m/s: mean and top value | Table II |
| angular speed | $\lVert \omega_i \rVert$ of each drone, in rad/s: mean and top value | Table II |

- If a mean runs over the steps or over the trials: not published.

## Training

PPO, with one actor and one critic for each role. (§III-C-c)

| setting | value | source |
|---|---|---|
| actor | MLP [512, 256, 256, 128], ELU. It gives a Gaussian over the actions | §IV-A-c, §III-C-c |
| critic | MLP [512, 256, 256, 128], ELU. It gives one value | §IV-A-c, §III-C-c |
| the std of the Gaussian, and how the action is kept in $[-1, 1]$ | not published | — |
| environments | 6144 | §IV-A-a |
| steps (his frames) | 50M for each role in each generation. 650M for each role over the 13 generations | §IV-A-a |
| steps of the pretraining | not published | — |
| learning rate | 2e−4, KL-adaptive, with the KL threshold 0.016 | §IV-A-c |
| how the KL schedule changes the learning rate | not published | — |
| epochs × minibatches | 5 × 8 | §IV-A-c |
| GAE $\lambda$ | 0.95 | §IV-A-c |
| PPO clip | 0.2 | §IV-A-c |
| entropy weight | 0.001 | §IV-A-c |
| value loss weight | 1.0 | §IV-A-c |
| $\gamma$ | not published | — |
| steps per environment per update | not published | — |
| optimizer and gradient norm clip | not published | — |
| input scale, reward scale and advantage scale | not published | — |
| training seeds | 3 for each preset | §IV-A-b |
| training time and hardware | not published | — |

- In the value target, a capture adds 0 as the future value. A truncation adds the
  critic's value of the next actor input, $V(o_{t+1})$. (§III-B-b)

### Pools

His AMSPBH: asynchronous multi-stage population-based training with Hedge sampling. It
builds on PSRO (policy-space response oracles). Hedge sampling gives more weight to the
opponents that are hard to beat. (§I, §III-C)

- Each role keeps a pool (his population) of frozen policies. A pool only grows.
  (§III-C)
- Generation 0 pretrains each role against pilots: the pursuer against the trajectory
  evaders, and the evader against the FRPN pursuer. (§III-C-a, §IV-A-b)
- At each generation $k = 1, \dots, K$, each role trains against the other role's pool
  as it was after generation $k - 1$. So each learner (his oracle) meets only policies of
  earlier generations. (§III-C-a)
- While a role trains, its opponents stay frozen. (§III-B)
- At the end of generation $k$, each learner joins its own pool. (§III-C-a)
- Each generation starts from the full agent of the role's last generation: the
  policy, the critic and the optimizer state. (§III-C-a, §III-C-c)
- The learner trains as one policy against the whole mix. It sees its opponent only
  through the actor input. (§III-C-b)

| what | value | source |
|---|---|---|
| generations $K$ | 13, after the pretraining | §IV-A-a, §IV-A-b, Fig. 2 |
| if the two roles train at the same time or in turn | not published | — |

| pool | first members | then | source |
|---|---|---|---|
| pursuers | FRPN, and the pursuer of generation 0 | the pursuer of each generation $k$ | Fig. 6, §III-C-a |
| evaders | the trajectory evaders, and the evader of generation 0 | the evader of each generation $k$ | Fig. 6, §III-C-a |

- The pilots stay in their pool in every generation. (Fig. 6, §IV-C-c)

### The opponent mix

His Hedge sampling, before each generation $k$:

- The learner of generation $k - 1$ plays each member $j$ of the opposing pool for about
  600 episodes. (§III-C-b, §IV-A-b)
- These episodes run as in training: with domain randomization and random starts.
  (§IV-A-b)
- $w_j$ is the learner's win rate, and $c_j = 1 - w_j$ is the challenge of member $j$.
  (§III-C-b)

| the pursuer wins | the evader wins | source |
|---|---|---|
| `caught_e`, `OOB_e` | `time_limit`, `OOB_p` | §III-C-b |

The weight $\mu_j$ (his $p_j$) of member $j$, in a pool of $M$ members (his $N$): (eq. 12)

$$
\mu_j = \frac{\gamma_h}{M} + (1 - \gamma_h) \frac{c_j^{\eta_h}}{\sum_l c_l^{\eta_h}}
$$

- $\gamma_h$ gives every member a least share. $\eta_h$ sets how much the mix leans to the
  hard members. (§III-C-b)
- If every $c_j$ is 0, the second term is uniform. (§III-C-b)

| symbol | value | source |
|---|---|---|
| $\gamma_h$ | 0.7 | §IV-A-b |
| $\eta_h$ | 2 | §IV-A-b |
| how the mix gives an opponent to an episode or to an environment | not published | — |
| a frozen opponent flies its mean action, or draws one | not published | — |

## Evaluation

- The final eval flies 1024 episodes for each pair (his matchup) and each seed. So each
  pair flies 3072 episodes over the 3 seeds. (§IV-A-c)
- The pursuer starts at (0, −1.5, 1) m, and the evader at (0, 1.5, 1) m. A random offset
  in a 0.1 m sphere moves each one. (§IV-A-c)
- Eval flies the nominal drone values. A network flies its mean action (his
  deterministic actor). (§IV-A-c)
- Each generation-13 policy plays the generation-13 policies of the other two presets,
  and the pilots. (Fig. 4)

| what | value | source |
|---|---|---|
| attitude and velocity at the start | not published | — |
| episodes for each cell of the win-rate matrix | not published | — |

| pilot | role | what flies | in training | source |
|---|---|---|---|---|
| FRPN | pursuer | fast-response proportional navigation: a guidance law that aims at where the evader will be | the first member of the pursuer pool | §IV-B-a, §IV-A-b |
| trajectories | evader | a hover, a circle or a figure-eight (his lemniscate), the same whatever the pursuer does | the first member of the evader pool | §IV-B-b, §IV-A-b |
| APF | evader | a potential field that pushes away from the pursuer and the walls | eval only | §IV-B-c, §IV-C-b |
| manual | either | a person with a joystick | hardware only | §IV-C-d, Table II |

- FRPN and APF give a velocity command. A cascaded PID controller and the same 500 Hz
  rate controller as the policies fly it. (§IV-B)

### FRPN

$$
v^{\text{cmd}} = G \left( (1 - W) \frac{\Delta p + \Delta v\, t_{\text{go}}}{t_{\text{go}}^2} + W\, \Delta p \right), \qquad t_{\text{go}} = \frac{\lVert \Delta p \rVert}{\lVert \Delta v \rVert + \epsilon}
$$

- $\Delta p = p_e - p_p$ and $\Delta v = v_e - v_p$ (his relative position and
  velocity). (eq. 13, eq. 7)
- A clip keeps $v^{\text{cmd}}$ under the top speed. (§IV-B-a)

| symbol | value | source |
|---|---|---|
| $G$ | 5.0 | §IV-B-a |
| $W$ | 0.20 | §IV-B-a |
| $\epsilon$ and the top speed | not published | — |
| gains of the cascaded PID | not published | — |

### Trajectories

| what | value | source |
|---|---|---|
| shape | hover, circle or figure-eight | §IV-B-b |
| how the shape is drawn | not published | — |
| circle radius | in [0.6, 1.0] m | §IV-B-b |
| circle speed | in [1.0, 2.0] m/s | §IV-B-b |
| circle height | in [0.5, 1.9] m | §IV-B-b |
| circle stretch | a random stretch into an ellipse | §IV-B-b |
| start phase | random | §IV-B-b |
| figure-eight | radius in [0.45, 0.9] m. The other ranges are similar to those of the circle | §IV-B-b |
| the size of the stretch, the centre of the path, and the hover point | not published | — |
| how the drone follows the path | not published | — |

### APF

- The pursuer pushes the evader with $f_P = k_P\, \Delta p / \lVert \Delta p \rVert^{\alpha_P}$,
  inside its activation radius. $\Delta p$ is the offset between the drones, with the
  sign that pushes the evader away from the pursuer. (§IV-B-c)
- Each wall pushes the evader with a term of the same form, with $k_W$ and $\alpha_W$.
  (§IV-B-c)
- The evader tracks the sum of the pushes, normalized, as a velocity reference.
  (§IV-B-c)

| symbol | value | source |
|---|---|---|
| $k_P$, $\alpha_P$ | 1.0, 3 | §IV-B-c |
| $k_W$, $\alpha_W$ | 1.0, 2 | §IV-B-c |
| activation radii, and the speed of the velocity reference | not published | — |

### The win-rate matrix

Every pursuer generation of a run plays every evader generation of that run. Cell
$(i, j)$ holds the win rate of pursuer generation $i$ against evader generation $j$, for
$i, j = 1, \dots, 13$. (Fig. 3)

### Hardware

- Two Crazyflie 2.1 brushless drones fly. (§IV-C-d, §V)
- A motion-capture system gives the position and the attitude at 100 Hz. (§IV-C-d)
- Both drones take off with the cascaded PID and hover at the eval start points. Then the
  policy or the pilot takes over. (§IV-C-d)
- An offboard computer sends the commands at 50 Hz, for an episode of 10 s. (§IV-C-d)
- Every trial starts with full batteries. The policies fly as they left training.
  (§IV-C-d)
- The tests are 42 flights. (§IV-C-d, §V)
- The AMSPBH pursuer flies against the hover, circle, APF, manual, `AMSPBH`, `AMSPB` and
  `AMSDRL` evaders. The AMSPBH evader flies against the `AMSPB`, manual, `AMSDRL` and FRPN
  pursuers. (Table II)

| what | value | source |
|---|---|---|
| how the velocity and the body rates are measured | not published | — |

## Presets

The main setup is `AMSPBH`. Each other setup changes how a learner picks its opponents.
(§III-C-b, §IV-B)

| preset | the opponents of a learner | $\gamma_h$ | $\eta_h$ | source |
|---|---|---|---|---|
| `AMSPBH` | the whole pool, by the Hedge mix | 0.7 | 2 | §IV-A-b |
| `AMSPB` | the whole pool, with the same share for each member | — | 0 | §III-C-b, §IV-B-e |
| `AMSDRL` | the newest frozen opponent only: a pool of size one | — | — | §III-C-b, §IV-B-d |
| ablation | the whole pool, by the Hedge mix | 0, 0.3 or 0.7 | 0.5, 1 or 2 | Fig. 5 |

- The ablation trains each of the 9 pairs of $\gamma_h$ and $\eta_h$. (Fig. 5)
- `AMSDRL` flies the same environment and the same body-rate action. (§IV-B-d)
- If the newest opponent of `AMSDRL` comes from generation $k - 1$ or from generation $k$:
  not published.
