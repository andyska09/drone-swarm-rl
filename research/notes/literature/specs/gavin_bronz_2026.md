# Gavin & Bronz 2026, Nv1

Pursuers chase one evader. Each pursuer carries a net below its body and catches the
evader with it. Both sides learn, each against a pool of the other side's past policies.
Paper: [arXiv:2607.05939v1](https://arxiv.org/abs/2607.05939).

## Setup

- $P$ drones have the role `pursuer`, and one drone has the role `evader`. Each role has
  one policy, and every drone of the role flies it. The experiments use $P = 3$.
  (§IV-A, §IV-B, §V-B)
- Both roles fly the same drone. (§IV-A)
- The world is an arena box with no obstacles, above a floor. (§IV-A, §IV-C)
- Each pursuer tries to catch the evader with its net. The evader tries to stay free
  until the time runs out. (§IV-A)
- Along each wall, a buffer zone costs the evader. This keeps the evader in a smaller box
  in the centre of the arena. (§IV-A)

| what | value | source |
|---|---|---|
| arena box $L$ | 32 × 32 × 16 m | §V-A |
| centre $c$ and floor height | not published | — |
| the evader's box | 12 × 12 × 6 m, in the centre of the arena | §V-A |
| step limit $T$ | 10 s | §V-A |

### The net

- Each pursuer carries a rigid round net of radius $R$, below its body and aligned with
  its body frame. (§IV-A)
- $c_p$ (his $c^{\text{net}}_p$) is the centre of the net of pursuer $p$. (§IV-C)

| what | value | source |
|---|---|---|
| net radius $R$ | not published | — |
| how far below the drone the net hangs | not published | — |
| capture distance | not published | — |

## Start

- Each drone starts at a uniform point of the arena. (§IV-A)

| what | value | source |
|---|---|---|
| clearance from the faces | not published | — |
| clearance between drones | not published | — |
| velocity, attitude and body rates | not published | — |

## Drone

- A high-fidelity quadrotor simulator, as in Gavin et al. 2026, 1v1. It simulates
  the collisions between drones, with the arena and with the nets. (§IV-E)
- The action is CTBR: a collective thrust and three body rates (his $a^{\text{th}}_i$ and
  $a^{\omega}_i$). (§IV-B)
- An attitude controller tracks the action. (Fig. 2)
- An INDI controller (incremental nonlinear dynamic inversion) turns a position, velocity
  or acceleration command into thrust and body rates. The pilots and the `PFSP-vel`
  preset use it. (§IV-E)

| what | value | source |
|---|---|---|
| mass, inertia and motors | not published | — |
| physics step $\delta$ | not published | — |
| attitude controller | 1000 Hz | Fig. 2 |
| policy step $\Delta t$ | 0.01 s (100 Hz) | Fig. 2 |
| limits of the thrust and the body rates | not published | — |

## Observation

The actor input of drone $i$. Each vector is in the body frame of drone $i$.

| block | content | size | source |
|---|---|---|---|
| own (his $o^{\text{self}}_i$) | $R_i^\top v_i$, $\operatorname{vec}(R_i)$ | 12 | §IV-B |
| others (his $o^{\text{others}}_i$) | for every other drone $j$, the evader included for a pursuer: $R_i^\top (p_j - p_i)$, $R_i^\top (v_j - v_i)$ | $6 (N - 1)$ | §IV-B |
| arena (his $o^{\text{env}}_i$) | $z_i$, and the distances $d_m$ to the arena bounds along $M$ rays, spread evenly in the horizontal plane | $1 + M$ | §IV-B |

- Each drone sees every other drone, so $K = N - 1$. (§IV-B)
- The number of rays $M$, and if the rays turn with the heading: not published.
- The order of the other drones: not published.
- Training scales the inputs before a network reads them. How: not published. (§IV-B)

The critic input of a role (his value network) holds the whole scene, in the world
frame. It is the same for every drone of the role.

| block | content | size | source |
|---|---|---|---|
| scene | $p_j$, $v_j$ and $\operatorname{vec}(R_j)$ of every drone | $15N$ | §IV-B |
| opponents | this step's action $a_j$ of each drone in the other role | 4 per drone | §IV-B |

- Training scales the critic input before the critic reads it. How: not published.
  (§IV-B)

## Events

- **Capture:** the centre of the evader comes within the capture distance of the net of
  a pursuer. (§IV-A)
- **Fail:** a drone hits the floor or leaves the arena bounds. (§IV-C)
- **coll_pp:** the body of a pursuer touches the body of another pursuer. (§IV-C)
- **coll_pe:** the body of a pursuer touches the body of the evader. The episode goes on.
  (§IV-C)
- **End:** the episode ends at a capture, a fail, a coll_pp, or after $T$. (§IV-C, eq. 1)

| what | value | source |
|---|---|---|
| contact distance | not published | — |
| which ends are terminal, and which are a truncation | not published | — |

## Reward

$$
\begin{aligned}
r_p ={}& \lambda_{\text{catch}}\, [\text{capture}] - \lambda_{\text{step}} - \lambda_{\text{PP}}\, [\text{coll\_pp}_p] - \lambda_{\text{PE}}\, [\text{coll\_pe}_p] - \lambda_{\text{fail}}\, [\text{fail}_p] - \lambda_{\text{cmd}} \lVert \omega^{\text{cmd}}_p \rVert - \lambda_{\text{dist}} \lVert p_e - c_p \rVert \\
r_e ={}& -\lambda_{\text{catch}}\, [\text{capture}] + \lambda_{\text{step}} - \lambda_{\text{PE}}\, [\text{coll\_pe}] - \lambda_{\text{fail}}\, [\text{fail}_e] - \lambda_{\text{cmd}} \lVert \omega^{\text{cmd}}_e \rVert - \phi_{\text{bnd}}(d_{\text{bnd}})
\end{aligned}
$$

- Each term is paid at each step at which it holds. The paper writes no $\Delta t$ in the
  terms. (§IV-C)
- One capture pays $\lambda_{\text{catch}}$ to every pursuer. (§IV-C)
- Each pursuer pays the distance from the evader to the centre of its own net. (§IV-C)
- $\omega^{\text{cmd}}$ is the commanded body rate (his $a^{\omega}$). (§IV-C)
- A coll_pe costs the pursuer and the evader the same. (§IV-C)
- A drone gets nothing when its opponent fails. (§IV-C)
- $\phi_{\text{bnd}}$ costs the evader near the arena bounds. It starts when the distance
  $d_{\text{bnd}}$ to the bounds falls under a set value, and grows exponentially as
  $d_{\text{bnd}}$ shrinks. (§IV-C)

| symbol | value | source |
|---|---|---|
| $\lambda_{\text{catch}}$ | 10.0 | Table I |
| $\lambda_{\text{step}}$ | 0.04 | Table I |
| $\lambda_{\text{PP}}$ (his $\lambda_{\text{collPP}}$) | 10.0 | Table I |
| $\lambda_{\text{PE}}$ (his $\lambda_{\text{collPE}}$) | 0.1 | Table I |
| $\lambda_{\text{fail}}$ | 30.0 | Table I |
| $\lambda_{\text{cmd}}$ | 2e−4 | Table I |
| $\lambda_{\text{dist}}$ | 0.001 | Table I |
| units of the weights | not published | — |
| where $\phi_{\text{bnd}}$ starts, and how fast it grows | not published | — |

## Metrics

| name | when | definition | source |
|---|---|---|---|
| `caught_e` (his catch rate) | at the end | the pursuers catch the evader within 10 s, with no crash | §V-A |
| `time_limit` (his evade rate) | at the end | the evader stays free for 10 s | §V-A |
| pursuer crash | at the end | a pursuer crashes, and the evader does not | §V-A |
| evader crash | at the end | the evader crashes, and no pursuer does | §V-A |
| `fail_pe` (his double crash) | at the end | a pursuer and the evader crash at the same step | §V-A |
| `time_to_catch` (his time to catch) | at the end | the time of the capture, in s. A crash or a timeout counts as 10 s | §V-A |

- The first five metrics are the outcomes. They add up to 1. (Table II)

## Training

PPO for many agents (his MAPPO), with one actor and one critic for each role. (§IV-B)

| setting | value | source |
|---|---|---|
| actor | MLP 2 × 256, ReLU. The network gives the mean and the std of a Gaussian, then a tanh squash | §IV-E |
| critic | MLP 2 × 256, ReLU, with a linear output | §IV-E |
| environments | 1024 | §IV-E |
| steps | 5e9 for each role, 1e10 in all | §IV-E |
| time | about 5.5 h on one RTX 4090, at about 3.5e5 steps per second | §IV-E |
| $\gamma$, GAE $\lambda$, PPO clip, learning rate, epochs, minibatches, entropy weight | not published | — |
| reward scale and advantage scale | not published | — |

### Pools

His PFSP: prioritized fictitious self-play.

- Each role keeps a pool of its own frozen past policies. A pool only grows. (§IV-D)
- The roles learn in turns. While one role learns, the other role's policies stay
  frozen. (§IV-D, Fig. 2)
- Each training episode plays one frozen opponent: the newest one in 50 % of the
  episodes, and one drawn by the weights below in the other 50 %. (§IV-D)
- When the learner's score against the whole opponent pool passes a set value, a copy of
  it joins its own pool, and the other role starts to learn. (§IV-D, Fig. 2)

The score of pursuer policy $i$ against evader policy $e$, in an episode that ends at
step $t$, with $T$ (his $\tau$) the step limit: (eq. 1, eq. 2)

$$
\rho_P(i, e) = \begin{cases} 1 - t / T & \text{if } i \text{ catches } e \\ 0 & \text{if } i \text{ crashes, or } e \text{ stays free} \\ 1 & \text{if } e \text{ crashes} \end{cases} \qquad \rho_E(e, i) = 1 - \rho_P(i, e)
$$

A pursuer that learns draws the evader policy $e$ of the pool $E$ with this weight. A
learning evader uses $\rho_E$ in place of $\rho_P$. (§IV-D)

$$
\frac{f\left(\rho_P(i, e)\right)}{\sum_{e' \in E} f\left(\rho_P(i, e')\right)}, \qquad f(x) = (1 - x)^2
$$

| what | value | source |
|---|---|---|
| the score at which a copy joins the pool | not published | — |
| the first member of each pool | not published | — |

## Evaluation

- 10,000 episodes for each pair of a pursuer and an evader. (§V-A)
- Eval flies the training arena, with the evader in its smaller box. (§V-A)
- A pilot sees only the opponent's position and velocity, as a policy does. (§V-B)
- A network flies its mean action, or it draws an action: not published.

| pilot | role | what flies | source |
|---|---|---|---|
| FRPN | pursuer | the proportional navigation of Pliska et al. 2024, for agile drones. It gives an acceleration | §V-B |
| APF | pursuer | the evader pulls, and the arena bounds and the other pursuers push. The sum, normalized, sets the direction of a velocity command of fixed speed | §V-B |
| FRPN + APF | pursuer | the FRPN acceleration plus the APF push | §V-B |
| APF | evader | the pursuers and the arena bounds push | §V-B |

## Presets

The main setup is `PFSP-CTBR`. Each other setup changes how a role trains, or its action.
(§V-B)

| preset | how the role trains | action | source |
|---|---|---|---|
| `PFSP-CTBR` | pools, as in Training | CTBR | §V-B |
| `PFSP-vel` | pools | a velocity command, which INDI turns into CTBR | §V-B |
| `SP` | against the newest opponent only | CTBR | §V-B |
| `noSP` | against a fixed APF opponent: the pursuers against the APF evader, the evader against the APF pursuers | CTBR | §V-B |
