# de Souza et al. 2021

Pursuers chase one evader, up to twice as fast, in a round arena in the plane. Each
pursuer moves at a fixed speed, and its policy sets its turn rate. All pursuers fly one
shared policy, trained with TD3. A hand-written pilot flies the evader.
Paper: [doi:10.1109/LRA.2021.3068952](https://doi.org/10.1109/LRA.2021.3068952).

## Setup

- $n$ drones (his agents, or pursuers) have the role `pursuer`, and one drone (his
  target) has the role `evader`. So $N = n + 1$. The experiments use $n$ from 1 to 8.
  (§III, §V)
- In the simulation, a pursuer is a point in the plane with a heading, and the evader is
  a point. (§III, eq. 1)
- The pursuers have one policy, and every pursuer flies it. Each $n$ has its own policy.
  (§IV-A)
- A pilot flies the evader. Only the pursuers learn. (§V-A, §VI-A)
- The world is a round arena of radius $R_{\text{arena}}$, with no obstacles. (§III)
- The pursuers try to capture the evader in the shortest time. A capture by any one
  pursuer is a success. (§III)
- The paper gives lengths in pixels (px) and times in steps. (§V)

| what | value | source |
|---|---|---|
| arena radius $R_{\text{arena}}$ | 430 px | §V |
| step limit $T$ (his $T_{\text{timeout}}$) | 500 steps | §V |
| length of one px, in m | not published | — |
| length of one step, in s | not published | — |

## Start

- Each pursuer starts at a random point in a circle of radius 100 px. (§V)
- The evader starts at a random point between a circle of radius 300 px and the border
  of the arena. (§V)
- Each pursuer starts with a random heading. (Fig. 10, Fig. 11)

| what | value | source |
|---|---|---|
| centre of the two circles | not published | — |
| distribution of the points and of the headings | not published | — |
| clearance between drones | not published | — |

## Drone

The pursuer is a unicycle (his unicycle kinematic model). It moves forward along its
heading $\psi_i$ at the speed $v$, and it turns at the rate $\omega_i$. (§III, eq. 1)

$$
\dot x_i = v \cos\psi_i, \qquad \dot y_i = v \sin\psi_i, \qquad \dot\psi_i = \omega_i
$$

- $p_i = (x_i, y_i)$ is the position of drone $i$ in the plane. (§III)
- The simulation steps a discrete form of eq. 1. Which form: not published. (§III)
- If a step ends outside the arena, the drone moves to the nearest point of the border.
  This holds for the pursuers and for the evader. (§III)
- The evader moves in any direction (his omnidirectional), at the speed $v_T$. Its pilot
  sets the direction (Evaluation). (§III, §V)

### Action and control

- The action is the turn rate: $a_i = \omega_i$. The policy output is clipped to
  $[\omega_{\min}, \omega_{\max}]$. (§IV)
- The speed stays at $v = v_p$. (§III, §V)
- The policy acts at each step. (§IV-A)

| symbol | value | source |
|---|---|---|
| pursuer speed $v_p$ | 10 px per step | §V |
| $\omega_{\max}$ | $\pi/10$ rad per step | §V |
| $\omega_{\min}$ | not published | — |
| evader speed $v_T$ | 0 to 20 px per step | §V |

## Observation

The actor input of pursuer $i$ (his state $s_i$). (§IV-B, Fig. 1)

| block | content | size | source |
|---|---|---|---|
| own | $\psi_i$, the heading in a fixed world frame, and $\dot\psi_i$ | 2 | §IV-B |
| evader (his $s_{i,T}$) | $d_{ie}$, $\dot d_{ie}$, $\alpha_{ie}$, $\dot\alpha_{ie}$ | 4 | §IV-B |
| others (his $s_{i,j}$) | for each other pursuer $j$: $d_{ij}$, $\alpha_{ij}$ | $2(n - 1)$ | §IV-B |

- $\alpha_{ij}$ (his heading error) is the angle between the heading of pursuer $i$ and
  the vector from $i$ to $j$. $\alpha_{ie}$ is the same angle for the evader. (§IV-B,
  Fig. 1)
- $\dot d_{ie}$ and $\dot\alpha_{ie}$ are the rates of change of $d_{ie}$ and
  $\alpha_{ie}$. (§IV-B)
- The input holds $2n + 4$ numbers. (§IV-B)
- Each pursuer sees every other pursuer, so $K = n - 1$. (§V)
- The others are in the order of their angle $\alpha_{ij}$ (his sweeping-angle
  ordering). (§IV-B, abstract)
- Where the order starts, and the sign and the range of $\alpha$: not published.
- The sensors give the true values. Nothing blocks their view. (§III)
- The actor input is scaled (his normalized). How: not published. (§VII)
- The critic input: not published.

## Events

- **Capture:** a pursuer comes within the capture distance of the evader:
  $d_{ie} < d_{\text{cap}}$ for some pursuer $i$. That pursuer is the captor. (§III,
  §IV-C)
- **coll_pp:** two pursuers touch. The episode goes on. (§III)
- **End:** the episode ends at a capture, or after $T$ steps (his timeout). (§III)

| what | value | source |
|---|---|---|
| capture distance $d_{\text{cap}}$ (his $R_{\text{cap}}$) | 30 px in eval. The curriculum changes it in training | §V |
| contact distance of a coll_pp | not published | — |
| the captor, if two pursuers come within $d_{\text{cap}}$ at the same step | not published | — |
| which ends are terminal, and which are a truncation | not published | — |

## Reward

Each pursuer $i$ gets, at each step: (§IV-C)

$$
r_i = \begin{cases} r_{\text{captor}} & \text{if } d_{ie} \le d_{\text{cap}} \\ r_{\text{helper}} & \text{if } d_{je} \le d_{\text{cap}} \text{ for some } j \ne i \\ -w_q\, q - w_d\, d_{ie} & \text{else} \end{cases}
$$

The formation score $q$ (his q-score) is the same for every pursuer. It lies in
$[0, 2]$, and a lower $q$ is a better formation: (eq. 2)

$$
q = \frac{1}{n} \sum_{i=1}^{n} \left( \hat u_0 \cdot \hat u_i + 1 \right)
$$

- $\hat u_i$ (his $\hat d_{iT}$) is the unit vector from pursuer $i$ to the evader.
  Pursuer 0 is the pursuer nearest to the evader. (§IV-C)
- $q$ is low when the pursuers come at the evader from different directions. (§IV-C)
- With $n = 1$, the per-step term is $-w_d\, d_{ie}$. (§IV-C)
- The per-step terms come at each step with no capture. The paper writes no $\Delta t$ in
  the terms. (§IV-C)
- At the capture step, the captor gets $r_{\text{captor}}$, and each other pursuer gets
  $r_{\text{helper}}$. §IV-C sets $r_{\text{captor}} > r_{\text{helper}}$. (§IV-C)
- §V lists $r_{\text{captor}}$, $r_{\text{helper}}$, $w_q$ and $w_d$ as 10, 100, 0.1 and
  0.002, in this order. (§V)
- The cost of a coll_pp: not published. §III says that the reward discourages it. (§III)

| symbol | value | source |
|---|---|---|
| $r_{\text{captor}}$ | 10 | §V |
| $r_{\text{helper}}$ | 100 | §V |
| $w_q$ | 0.1 | §V |
| $w_d$ | 0.002, with $d_{ie}$ in px | §V |

## Metrics

| name | when | definition | source |
|---|---|---|---|
| `caught_e` (his success rate, or capture rate) | at the end | the episode ended in a capture | §III, §VI |
| `time_to_catch` (his average timesteps to capture) | at the end | the number of steps to the capture. Only captures count | §VI-C |

## Training

TD3 (his Twin Delayed Deep Deterministic Policy Gradient): an actor-critic method. It
improves on DDPG, an older method of the same kind, and reduces the over-estimate of the
value. (§IV-A)

- One learner trains the policy of the pursuers. Every pursuer adds its steps to it (his
  shared experience). (§IV-A)
- Each pursuer sees the task as its own MDP (his Markov Decision Process: at each step a
  state, an action, a reward and the next state), with the discount $\gamma$. (§IV)
- Each $n$ trains its own policy. (§IV-A)
- The Repulsive pilot flies the evader. (§VI-A)

| setting | value | source |
|---|---|---|
| steps | 4e6 | §V-B, Fig. 8 |
| what one step counts: one step of the environment, or of one pursuer | not published | — |
| evader speed $v_T$ in training | 0 to 20 px per step. How it is drawn: not published | §V |
| actor and critic: layers, sizes and activations | not published | — |
| $\gamma$, learning rate, batch size and the other TD3 settings | not published | — |
| time and hardware | not published | — |

### Curriculum

His curriculum learning: training starts with an easier task and makes it harder, step
by step, until it reaches the real task. (§IV-D)

- Training starts with a large $d_{\text{cap}}$ and makes it smaller, down to the eval
  value. (§IV-D, §V)

| what | value | source |
|---|---|---|
| first $d_{\text{cap}}$ | not published | — |
| how and when $d_{\text{cap}}$ gets smaller | not published | — |

## Evaluation

- 100 episodes for each speed level of the evader. (§V-A)
- The speed levels are $v_T / v_p$ = 0.8 to 2.0, in steps of 0.2. (§V-A)
- $d_{\text{cap}}$ is 30 px. (§V)
- The policies trained against the Repulsive pilot fly against both evader pilots.
  (§VI-A)
- For the training curves, training stops at regular intervals, and the policy flies
  against the Repulsive pilot. Fig. 8 shows 3 training runs. (§VI-F)
- If the policy adds noise to its action in eval: not published.
- The seeds: not published.

| pilot | role | what flies | source |
|---|---|---|---|
| Repulsive | evader | a push away from the pursuers and the border | §V-A, eq. 3 |
| Fixed Paths | evader | one of three set paths | §V-A, Fig. 2 |
| Janosov | pursuer | the group chase of Janosov et al. 2017, turned into a turn rate | §V-B |
| Angelani | pursuer | the group chase of Angelani 2012, turned into a turn rate | §V-B |

- A learned baseline also flies the pursuers: the method of Hüttenrauch et al. 2019,
  trained in this simulation for 4e6 steps. Its training skips the curriculum. Its input
  holds the angle and the distance to the evader, and the heading of each pursuer (his
  communication set). (§V-B)

### Repulsive

The evader adds a push from each pursuer $j$ and a push from the border (his motion
vector). (§V-A, eq. 3)

$$
v = \sum_j \frac{p_j - p_e}{d_{je}^2} + \frac{R_{\text{arena}}\, \hat\gamma - p_e}{d_w^2}
$$

- $p_e$ (his $e$) is the position of the evader. $p_j$ (his $a_j$) is the position of
  pursuer $j$, and $d_{je}$ (his $d_j$) is the distance between them. (§V-A)
- $\gamma$ is the direction from the evader to the nearest point of the border.
  $\hat\gamma$ is the unit vector at the angle $\gamma$. $d_w$ is the distance from the
  evader to the border. (§V-A)
- Each push gets weaker with the square of the distance. (§V-A)
- The evader moves at the speed $v_T$. (§V-A)
- How the evader turns $v$ into the direction of its step: not published.

### Fixed Paths

- The evader follows one of three set paths, drawn in Fig. 2. It moves at the speed
  $v_T$. (§V-A)
- The equations of the paths: not published.

### Janosov and Angelani

- Each method gives a world velocity $(d_x, d_y)$. A P controller (a gain times the
  error) turns it into a turn rate. (§V-B, eq. 4, eq. 5)

$$
\psi^{\text{des}} = \arctan \frac{d_y}{d_x}, \qquad \omega = K \left( \psi - \psi^{\text{des}} \right)
$$

- The gain $K$ is tuned for the most captures in simulation. (§V-B)
- The value of $K$: not published.

## Presets

The main setup is `repulsive`. Each other row says what it changes. The preset names are
ours. (§V, §VI)

| preset | $n$ | what changes | source |
|---|---|---|---|
| `repulsive` | 1 to 8 | the main setup, as in the sections above | §V, §VI-B |
| `fixed-paths` | 1 to 8 | eval only: the Fixed Paths pilot flies the evader | §VI-A |
| `arena-size` | 3 in §VI-C, 8 in Fig. 5 | eval only: $R_{\text{arena}}$ times a factor, with the policy trained at 430 px. The factors are only in Fig. 5 | §VI-C, Fig. 5 |
| `nearest-4` | more than 4, in eval | eval only: the policy trained with $n = 4$. Each pursuer sees only its 4 nearest pursuers | §VI-E, Fig. 7 |
| `no-curriculum` | 3 | training skips the curriculum | §VI-F |
| `no-formation` | 3 | the per-step term is $-w_d\, d_{ie}$ | §VI-G |
| `speed-control` | 3 | the action is $(v_i, \omega_i)$, with $0 \le v_i \le v_p$. The network has two outputs | §VI-I |
| `real` | 3 | real drones, as below | §VII |

### Real drones

His proof-of-concept demonstration. (§VII)

- Three pursuer drones chase a drone that a person flies. (§VII)
- The policy trained in simulation flies the pursuers with no change (his direct
  sim-to-real transfer). (§VII)
- Each drone holds a fixed height. The pursuers share one height, and the evader flies at
  another. (§VII)
- The actor input holds the normalized relative positions of the evader and of the other
  pursuers. (§VII)
- The policy gives the turn rate. A low-level controller on each drone tracks the speed
  and the turn rate, holds the height and avoids collisions. (§VII)
- A motion capture system measures the pose of every drone. A local computer runs the
  policy of each pursuer and sends the command to the drone over a wireless link. (§VII)

| what | value | source |
|---|---|---|
| drone | Parrot AR Drone 2 | §VII |
| arena | round, radius 3 m | §VII |
| speed limit | 1.2 m/s | §VII |
| command rate | 20 Hz | §VII |
| the two heights | not published | — |
| $v_p$ in m/s, and $d_{\text{cap}}$ in m | not published | — |
