# Gavin, Lacroix & Bronz 2026, 1v1

One pursuer chases one evader. The pursuer carries a net and catches the evader with it.
Both sides learn with PPO at the same time, each against the newest policy of the other
side. Paper: [arXiv:2603.16279v1](https://arxiv.org/abs/2603.16279).

## Setup

- One drone has the role `pursuer`, and one drone has the role `evader`, so $N = 2$.
  Each role has its own policy. (§IV-B)
- $e$ is the evader, and $p$ is the pursuer.
- Both roles fly a drone with the same abilities. (§IV-B)
- The world is a box-shaped arena with no obstacles, above the ground. (§IV-B, §IV-B1)
- The pursuer tries to catch the evader as fast as it can. The evader tries to stay
  free. (§IV-B)
- Along each wall, a narrow buffer zone costs the evader. This keeps the evader in a
  smaller box in the centre of the arena. (§IV-B)

| what | value | source |
|---|---|---|
| arena box $L$ (his $L \times L \times H$) | set by the preset | §IV-B, §V-B |
| the evader's box | set by the preset | §V-B |
| centre $c$ and the height of the ground | not published | — |
| step limit $T$ | 10 s, 1000 steps | §IV-C |

### The net

- The pursuer carries a rigid round net of radius $R$. The net is fixed on the pursuer
  and aligned with its body frame. (§IV-B, Fig. 3)
- The pursuer holds the net taut, and lets it go at a capture. (§IV)
- $c_p$ (his $c_{\text{net}}$) is the centre of the net. (§IV-B1)

| what | value | source |
|---|---|---|
| net radius $R$ | not published | — |
| where the net sits on the drone, and which way it faces | not published | — |
| capture distance | not published | — |

## Start

- Each drone starts at a uniform random point of the arena. (§IV-B, §IV-C)

| what | value | source |
|---|---|---|
| clearance from the faces | not published | — |
| clearance between drones | not published | — |
| velocity, attitude and body rates | not published | — |

## Drone

- A high-fidelity quadrotor simulator, written in JAX. It models the air drag, the
  low-level control, the motor speeds and the transmission delays. (§III)
- The quadrotor model and its low-level control come from Heeg, Song & Scaramuzza 2024.
  (§III)
- The model is identified from real flight data. (§VII)
- Every episode flies the same drone parameters. (§IV-C)
- The simulator computes the collisions between the drones, with the arena and with the
  net. (§III)
- The action is CTBR: a mass-normalized collective thrust and three body rates (his
  $a^{\text{th}}_i$ and $a^{\omega}_i$). (§III, §IV-B1)
- An attitude autopilot turns the action into motor speeds. (Fig. 2)
- An SE(3) controller turns a position, velocity or acceleration command into thrust and
  body rates. It is the geometric tracking controller of Lee, Leok & McClamroch 2010, as
  built in RotorPy. The pilots use it. (§III, §V-B)

| what | value | source |
|---|---|---|
| mass, inertia, motors and drag | not published | — |
| physics step $\delta$ | not published | — |
| attitude autopilot | 1 kHz | Fig. 2 |
| policy step $\Delta t$ | 0.01 s (100 Hz) | §IV-C, Fig. 2 |
| limits of the thrust and the body rates | not published | — |
| transmission delay | not published | — |

## Observation

The actor input of drone $i$, with $j$ its opponent.

| block | content | size | source |
|---|---|---|---|
| own (his $o^{\text{self}}_i$) | $v_i$, $\operatorname{vec}(R_i)$ | 12 | §IV-B1 |
| others (his $o^{\text{opp}}_i$) | $p_j - p_i$, $v_j - v_i$, world frame | 6 | §IV-B1 |
| arena (his $o^{\text{env}}_i$) | the distance from $p_i$ to each bound of the arena, and to the ground | not published | §IV-B1 |

- Each drone sees its one opponent, so $K = 1$. (§IV-B1)
- The frame of $v_i$: not published.
- Training scales the inputs before a network reads them. It divides the relative
  positions by $k^p_i$ (his maximum range of view), and the velocities by $k^v_i$ (his
  maximum velocity). Each drone has its own $k^p_i$ and $k^v_i$. (§IV-B1)
- The values of $k^p_i$ and $k^v_i$, and the scale of the other inputs: not published.

The critic input of a role (his value network) holds the whole scene.

| block | content | size | source |
|---|---|---|---|
| scene | $p_j$, $v_j$ and $\operatorname{vec}(R_j)$ of both drones | 30 | §IV-B1 |
| opponents | this step's action $a_j$ of the opponent | 4 | §IV-B1 |

- The frame of the scene: not published.
- Training scales the critic input before the critic reads it. How: not published.
  (§IV-B1)

## Events

- **Capture:** the centre of the evader comes within the capture distance of the net.
  (§IV-B, Fig. 3)
- **Fail:** a drone hits the ground or leaves the arena bounds. (§IV-B1)
- **coll_pe** (his contact): the body of one drone touches the body of the other. The
  episode goes on. (§IV-B1)
- **End:** the episode ends at a capture, at a fail, or after $T$. (§IV-B1, §IV-C)

| what | value | source |
|---|---|---|
| contact distance | not published | — |
| which ends are terminal, and which are a truncation | not published | — |

## Reward

$$
\begin{aligned}
r_p ={}& \lambda_{\text{catch}}\, [\text{capture}] - \lambda_{\text{dist}} \lVert p_e - c_p \rVert - \lambda_{\text{PE}}\, [\text{coll\_pe}] - \lambda_{\text{fail}}\, [\text{fail}_p] - \lambda_{\text{cmd}} \lVert \omega^{\text{cmd}}_p \rVert \\
r_e ={}& -\lambda_{\text{catch}}\, [\text{capture}] + \lambda_{\text{dist}} \lVert p_e - c_p \rVert - \lambda_{\text{PE}}\, [\text{coll\_pe}] - \lambda_{\text{fail}}\, [\text{fail}_e] - \lambda_{\text{cmd}} \lVert \omega^{\text{cmd}}_e \rVert - \phi_{\text{bnd}}(d_{\text{bnd}})
\end{aligned}
$$

- Each term is paid at each step at which it holds. The paper writes no $\Delta t$ in the
  terms. (§IV-B1)
- The evader earns the distance term that the pursuer pays. (§IV-B1)
- $\omega^{\text{cmd}}$ is the commanded body rate (his $a^{\omega}$). (§IV-B1)
- A coll_pe costs the pursuer and the evader the same. (§IV-B1)
- A drone gets nothing when its opponent fails. (§IV-B1)
- $\phi_{\text{bnd}}$ costs the evader near the arena bounds. It starts when the distance
  $d_{\text{bnd}}$ to the bounds falls under a set value, and grows exponentially as
  $d_{\text{bnd}}$ shrinks. (§IV-B1)

| symbol | value | source |
|---|---|---|
| $\lambda_{\text{catch}}$ | 10.0 | Table I |
| $\lambda_{\text{dist}}$ | 0.001 | Table I |
| $\lambda_{\text{PE}}$ (his $\lambda_{\text{coll}}$) | 0.1 | Table I |
| $\lambda_{\text{fail}}$ | 30.0 | Table I |
| $\lambda_{\text{cmd}}$ | 2e−4 | Table I |
| $\lambda_{\text{bnd}}$ | 1.0 | Table I |
| where $\lambda_{\text{bnd}}$ enters $\phi_{\text{bnd}}$ | not published | — |
| where $\phi_{\text{bnd}}$ starts, and how fast it grows | not published | — |
| units of the weights | not published | — |

## Metrics

| name | when | definition | source |
|---|---|---|---|
| `caught_e` (his catch rate) | at the end | the pursuer catches the evader in 10 s | §V-B |
| evade rate | at the end | the evader stays free for 10 s, or the pursuer fails | §V-B |
| `time_limit` (his timeout) | at the end | the evader stays free for 10 s | Table III |
| pursuer crash | at the end | the pursuer fails, and the evader does not | §V-B |
| evader crash | at the end | the evader fails, and the pursuer does not | §V-B |
| `fail_pe` (his double crash) | at the end | both drones fail at the same step | §V-B |
| `time_to_catch` (his time to catch) | at the end | the time of the capture, in s. A fail or a timeout counts as 10 s | §V-B |

- `caught_e`, `time_limit`, pursuer crash, evader crash and `fail_pe` are the outcomes.
  They add up to 1. (Table III)
- Table III gives the mean and the std of `time_to_catch`. (Table III)

## Training

PPO, with one actor (his policy network) and one critic (his value network) for each
role. (§IV-B1)

| setting | value | source |
|---|---|---|
| actor | MLP 2 × 256, ReLU. The output layer gives the mean and the std of a Gaussian, then a tanh squash bounds the action | §IV-C |
| critic | MLP 2 × 256, ReLU, with a linear output | §IV-C |
| $\gamma$ | 0.99 | Table II |
| GAE $\lambda$ | 0.95 | Table II |
| PPO clip | 0.2 | Table II |
| value loss weight (his critic weight) | 0.5 | Table II |
| entropy weight | 0.01 | Table II |
| gradient norm clip | 0.5 | Table II |
| optimizer | not published | — |
| learning rate | 5e−4, constant | Table II |
| environments | 1024 (his per agent) | §IV-C, Table II |
| rollout | 128 steps | Table II |
| epochs × minibatches | 15 × 1 | Table II |
| environment steps | 2e9 | §IV-C |
| training steps (his total number of training steps) | 4e9 | Table II |
| time | about 1 h 35 min on one RTX 4090, at about 3.5e5 environment steps per second | §IV-C |
| reward scale and advantage scale | not published | — |

- Both roles learn at the same time (his co-evolution). (§I, §IV-B1)
- Each role plays the newest version of the other role. (§VII)

## Evaluation

- 10,000 episodes for each pair of a pursuer and an evader. (§V-B)
- Each preset flies the pursuer and the evader that trained in it. (§V-B)
- A pilot sees only the opponent's position and velocity, as a policy does. (§V-B)
- Each pilot gives a velocity or an acceleration command. The SE(3) controller turns it
  into CTBR. Which pilot gives which: not published. (§V-B)
- A network flies its mean action, or it draws an action: not published.

| pilot | role | what flies | source |
|---|---|---|---|
| PP | pursuer | pure pursuit: the pursuer flies in a straight line toward the position of the evader | §V-B |
| FRPN | pursuer | the Fast-Response Proportional Navigation of Pliska et al. 2024, a proportional navigation for multirotors | §V-B |
| hover | evader | the evader holds a fixed point in space | §V-B |
| APF | evader | artificial potential field: the pursuer and the arena bounds push the evader. The field of Zhang et al. 2023 | §V-B |

### Real flight

- A real quadrotor with a Betaflight flight controller flies the pursuer. A ground
  station computer simulates the evader. (§V-D)
- The arena is indoors, 8 × 8 × 5 m, the size of `Small`. (§V-D)
- A motion capture system (OptiTrack) measures the position and the attitude of the real
  drone at 200 Hz, and passes them to the simulation. (§V-D)
- The policy runs on the ground station. Its commands go to the drone over a radio link
  at 100 Hz. (§V-D)
- The policies fly with no fine-tuning after the simulation. (§V-D)

## Presets

The paper trains and evaluates two setups. They differ in the arena. (§V-B)

| preset | arena box $L$ | the evader's box, in the centre of the arena | source |
|---|---|---|---|
| `Large` | 40 × 40 × 14 m | 20 × 20 × 4 m | §V-B, Table III |
| `Small` | 8 × 8 × 5 m | 6 × 6 × 4 m | §V-B, Table III |
