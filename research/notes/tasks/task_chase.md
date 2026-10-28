# chase

Pursuers chase one evader. Each pursuer carries a net under its body and catches the
evader with it. Both sides learn. In the paths presets, the paths pilot flies the evader,
and only the pursuers learn.

Log: [chase_pools_log.md](../experiment_logs/chase_pools_log.md), and before the pools
[chase_duel_log.md](../experiment_logs/chase_duel_log.md). The paths presets:
[chase_modal_log.md](../experiment_logs/chase_modal_log.md)

## Setup

- One drone has the role `evader`, and $P$ drones have the role `pursuer`. Each role has
  its own policy. Each preset sets $P$ to 1 or 3.
- The world is the arena box of the preset.
- Each pursuer tries to catch the evader with its net. The evader tries to stay free
  until the time runs out.
- $e$ is the evader, and $p$ is a pursuer.

| what | value |
|---|---|
| step limit $T$ | 1000 steps (10 s) |

### The net

- Each pursuer carries a flat square net, 1.0 m on a side. The net hangs on a 0.4 m rope,
  straight down the body $z$ axis, and faces along the body $x$ axis.
- The centre of the net is $c_p = p_p - h R_p e_3$, with $h$ = 0.9 m.
- A drone $k$ touches the net of pursuer $p$ when its ball of radius 0.40 m meets the
  square. In the net frame, with $q = R_p^\top (p_k - p_p) + h e_3$, this is
  $\lvert q_x \rvert \le 0.40$ m, $\lvert q_y \rvert \le 0.5$ m and
  $\lvert q_z \rvert \le 0.5$ m.

## Start

- Each drone starts at a uniform point of the box, 2 m clear of every face:
  $p_i \sim U\left[c - (L/2 - 2\ \text{m}),\ c + (L/2 - 2\ \text{m})\right]$.
- Every two drones start at least 2 m apart: the first of 16 draws that meets this.
- If the paths pilot flies the evader, the evader starts on its path. Then the pursuers
  are drawn again by the same rules.
- The rest follows the default.

## Drone

Default.

## Observation

| input | content | size |
|---|---|---|
| actor | the default, with the sensed blocks of the role in each slot | $20 + (8 + s) K$ |
| critic | the default | $18N + 4$ per drone of the other role |

- Each drone sees every other drone, at any distance, so $K = N - 1$.
- Each slot of others also holds the sensed blocks of the role of drone $i$, after the
  role flag, in this order:

| block | content | size |
|---|---|---|
| attitude | $\operatorname{vec}(R_i^\top R_j)$, the attitude of drone $j$ in the body frame of drone $i$ | 9 |
| rpm | $n_{j,k} / n_{\max}$ for $k = 1, \dots, 4$, each motor speed of drone $j$ as a share of the top speed | 4 |

- $s$ is the size of the sensed blocks of the role: 0, 9, 4 or 13. The preset sets the
  blocks of each role.
- In the duel, the actor input is $28 + s$ wide, and the critic input is 40 wide.
- With open walls, a drone can fly outside the box. Then some of its wall distances are
  negative.

## Events

- **Capture:** the evader touches the net of a pursuer.
- **Fail:** with closed walls, a drone leaves the box. With open walls, a drone goes below
  the floor. The drone dies. A fail at the floor is a crash, and a fail at a wall or the
  ceiling is an OOB.
- **coll_pp:** a pursuer touches another pursuer, or the net of one pursuer touches the
  other. Both pursuers die.
- **coll_pe:** a pursuer touches the evader. The episode goes on.
- **Outside:** the evader is outside the box. The episode goes on.
- **End:** the episode ends at a capture, at a death, or after $T$ steps. A capture is
  terminal for every drone. A death is terminal for the drones that died, and a
  truncation for the others. The step limit is a truncation.
- A capture wins over a death at the same step.

## Reward

$$
\begin{aligned}
r_p ={}& \Delta t \left( -\lambda_{\text{step}} - \lambda_{\text{dist}} \lVert p_e - c_p \rVert - \lambda_{\text{PE}}\, [\text{coll\_pe}_p] - \lambda_{\text{cmd}} \lVert \omega^{\text{cmd}}_p \rVert \right) + \lambda_{\text{catch}}\, [\text{capture}] - \lambda_{\text{PP}}\, [\text{coll\_pp}_p] - \lambda_{\text{fail}}\, [\text{fail}_p] \\
r_e ={}& \Delta t \left( \lambda_{\text{step}} - \lambda_{\text{PE}}\, [\text{coll\_pe}_p \text{ for some } p] - \lambda_{\text{out}}\, [\text{outside}] - \lambda_{\text{cmd}} \lVert \omega^{\text{cmd}}_e \rVert \right) - \lambda_{\text{catch}}\, [\text{capture}] - \lambda_{\text{fail}}\, [\text{fail}_e]
\end{aligned}
$$

- One capture pays $\lambda_{\text{catch}}$ to every pursuer. The evader pays it once.
- Each pursuer pays the distance from the evader to the centre of its own net.

| symbol | value |
|---|---|
| $\lambda_{\text{step}}$ | 4.0 |
| $\lambda_{\text{dist}}$ | 0.1 per m |
| $\lambda_{\text{PE}}$ | 10.0 |
| $\lambda_{\text{cmd}}$ | 0.02 per rad/s |
| $\lambda_{\text{out}}$ | 0, or 57.0 (see Presets) |
| $\lambda_{\text{catch}}$ | 10.0 |
| $\lambda_{\text{PP}}$ | 10.0 |
| $\lambda_{\text{fail}}$ | 30.0 |

## Metrics

| name | when | definition |
|---|---|---|
| `caught_e` | at the end | the episode ended in a capture |
| `fail_pe` | at the end | the evader failed, and at the same step a pursuer failed or had a coll_pp |
| `crash_p` | at the end | a pursuer crashed |
| `OOB_p` | at the end | a pursuer went OOB |
| `coll_pp` | at the end | the episode ended in a coll_pp |
| `crash_e` | at the end | the evader crashed |
| `OOB_e` | at the end | the evader went OOB |
| `time_to_catch` | at the end | the time of the capture, in s. Only captures count |
| `catch_offset` | at the end | at a capture, the distance from the middle of the net to the evader, in the net plane: $\min_p \lVert (q_y,\ q_z) \rVert$, with $q$ of the evader in the net frame of $p$ |
| `distance` | mean over the steps | $\min_p \lVert p_e - c_p \rVert$ |
| `coll_pe` | mean over the steps | the share of steps with a coll_pe for some $p$ |

The first seven metrics are the outcomes. Each episode gets exactly one: the first that
holds, in the order of the table, or else `time_limit`. These eight add up to 1 for any
$N$.

## Presets

| preset | $L$ | $c$ | walls | $\lambda_{\text{out}}$ |
|---|---|---|---|---|
| `default` | 32 × 32 × 16 m | (0, 0, 8) m | closed | 0 |
| `small` | 24 × 24 × 16 m | (0, 0, 8) m | closed | 0 |
| `open` | 32 × 32 × 16 m | (0, 0, 8) m | open | 57.0 |
| `nemecek` | 20 × 20 × 10 m | (0, 0, 5) m | open | 57.0 |

- The floor is at $z = 0$ in every preset.
- In these four presets, $P = 1$, no role senses a block, and both sides learn.

Each other preset is one of these four, with these changes:

| preset | from | $P$ | the pursuer senses | the evader senses | the evader in training |
|---|---|---|---|---|---|
| `3v1` | `default` | 3 | none | none | learns |
| `paths_1v1` | `default` | 1 | none | none | paths |
| `paths_1v1_attitude` | `default` | 1 | attitude | none | paths |
| `paths_1v1_rpm` | `default` | 1 | rpm | none | paths |
| `paths_1v1_attitude_rpm` | `default` | 1 | attitude, rpm | none | paths |
| `paths_3v1` | `default` | 3 | none | none | paths |
| `paths_3v1_attitude` | `default` | 3 | attitude | none | paths |
| `paths_3v1_rpm` | `default` | 3 | rpm | none | paths |
| `paths_3v1_attitude_rpm` | `default` | 3 | attitude, rpm | none | paths |
| `default_p_attitude` | `default` | 1 | attitude | none | learns |
| `default_p_rpm` | `default` | 1 | rpm | none | learns |
| `default_p_attitude_rpm` | `default` | 1 | attitude, rpm | none | learns |
| `nemecek_p_attitude_e_attitude` | `nemecek` | 1 | attitude | attitude | learns |
| `nemecek_p_attitude_e_rpm` | `nemecek` | 1 | attitude | rpm | learns |
| `nemecek_p_attitude_e_attitude_rpm` | `nemecek` | 1 | attitude | attitude, rpm | learns |
| `nemecek_p_rpm_e_attitude` | `nemecek` | 1 | rpm | attitude | learns |
| `nemecek_p_rpm_e_rpm` | `nemecek` | 1 | rpm | rpm | learns |
| `nemecek_p_rpm_e_attitude_rpm` | `nemecek` | 1 | rpm | attitude, rpm | learns |
| `nemecek_p_attitude_rpm_e_attitude` | `nemecek` | 1 | attitude, rpm | attitude | learns |
| `nemecek_p_attitude_rpm_e_rpm` | `nemecek` | 1 | attitude, rpm | rpm | learns |
| `nemecek_p_attitude_rpm_e_attitude_rpm` | `nemecek` | 1 | attitude, rpm | attitude, rpm | learns |

## Training

- Runs set $\gamma$ = 0.99.
- In the paths presets, only the pursuers learn, against the paths evader, in one phase.
  The network of the evader keeps its first weights and never flies.
- In the other presets, with $G = 0$ generations, both sides learn at the same time,
  against each other. With $G > 0$, each side learns against a pool of frozen opponents,
  as below. Runs use $G$ = 13.

### Pools

- The run splits into $G + 1$ phases with the same number of updates, one for each
  generation $k = 0, \dots, G$.
- In each phase both sides learn at the same time. Each side trains against the other
  side's pool as it was at the end of generation $k - 1$.
- Each side has its own environments, its own blocks and its own opponent mix.
- The step budget of a run counts the steps of both sides.
- At the end of a phase, each learner joins its own pool. A pool only grows.

| pool | first member | then |
|---|---|---|
| pursuers | FRPN | `pursuer_00`, `pursuer_01`, … |
| evaders | paths | `evader_00`, `evader_01`, … |

- Generation 0 starts both sides from random weights.
- Each phase keeps the learner's weights, optimizer state and learning rate. It sets the
  learner's std back to 1.0.
- Each phase starts new episodes. Each side keeps its reward scale from phase to phase.
- A frozen network flies its mean action. The learner sees its opponent only through the
  observation.
- Only the learner's drones enter the loss and change the reward scale.
- The learner's critic sees the action that the opponent flies.

### The opponent mix

Before each phase, the learner plays each member $j$ of the opposing pool for 600
episodes, as in eval. $w_j$ is the learner's win rate:

| the pursuer wins | the evader wins | half a win to each side |
|---|---|---|
| `caught_e`, `crash_e`, `OOB_e` | `time_limit`, `crash_p`, `OOB_p`, `coll_pp` | `fail_pe` |

The weight of member $j$, in a pool of $M$ members:

$$
\mu_j = \frac{\gamma_h}{M} + (1 - \gamma_h) \frac{c_j^{\eta_h}}{\sum_l c_l^{\eta_h}}, \qquad c_j = 1 - w_j
$$

| symbol | value |
|---|---|
| $\gamma_h$ | 0.7 |
| $\eta_h$ | 2 |

- If every $c_j$ is 0, the second term is uniform.
- The environments of each side form 64 blocks of the same size. Member $j$ flies in $64\, \mu_j$
  blocks, rounded by the largest remainder. A block keeps its member for the whole phase.

## Evaluation

A role can be flown by its network, or by one of these pilots:

| pilot | role | what flies |
|---|---|---|
| cascade | any | the default cascade, to the reference point below |
| FRPN | pursuer | a guidance law that aims at where the evader will be |
| APF | evader | a potential field that pushes away from the pursuers and the walls |
| paths | evader | a hover, a circle or a figure-eight, drawn for each episode |

In the paths presets, the paths pilot flies the evader.

The reference points of the cascade:

- A pursuer aims at $p_e + h R_p e_3$, so that its net centre lands on the evader.
- The evader flies to a waypoint: a point up to 3 m from the evader in a random
  direction, kept 2 m inside the faces of the box. When the evader comes within 0.2 m of
  it, a new waypoint comes. Only the cascade evader uses the waypoint.

FRPN, APF and paths give a world acceleration, and the default acceleration chain flies
it. FRPN and APF first pass a speed limit: at most 20 m/s², and a speed gain up to
12 m/s.

### FRPN

$$
a_p = G \left( (1 - W) \frac{\Delta p + \Delta v\, t_{\text{go}}}{t_{\text{go}}^2} + W\, \Delta p \right), \qquad t_{\text{go}} = \frac{\lVert \Delta p \rVert}{\lVert \Delta v \rVert}
$$

$\Delta p = p_e + h R_p e_3 - p_p$ is the offset to the aim point, and
$\Delta v = v_e - v_p$.

| symbol | value |
|---|---|
| $G$ | 6.677 |
| $W$ | 0.337 |

### APF

$$
a_e = \sum_p \frac{k_{\text{pur}} + b_{\text{pur}}\, \sigma_p}{d_{ep}^{\,e_{\text{pur}}}}\, \hat r_p + \sum_{f:\ g_f < r_{\text{wall}}} \left( k_{\text{wall}} + b_{\text{wall}}\, \nu_f \right) \left( \frac{1}{g_f} - \frac{1}{r_{\text{wall}}} \right)^{e_{\text{wall}}} \hat n_f
$$

- $\hat r_p$ is the unit vector from pursuer $p$ to the evader, and
  $\sigma_p = \max\left(0,\ (v_p - v_e) \cdot \hat r_p\right)$ is its closing speed.
- $f$ runs over the six faces of the box. $g_f$ is the gap from the evader to face $f$,
  $\hat n_f$ is the unit normal of $f$ into the box, and
  $\nu_f = \max\left(0,\ -\hat n_f \cdot v_e\right)$ is the speed toward $f$.

| symbol | value |
|---|---|
| $k_{\text{pur}}$, $b_{\text{pur}}$, $e_{\text{pur}}$ | 2.135, 1.370, 1.032 |
| $k_{\text{wall}}$, $b_{\text{wall}}$, $e_{\text{wall}}$ | 57.89, 29.71, 0.5652 |
| $r_{\text{wall}}$ | 9.997 m |

### Paths

At the start of each episode, the evader draws a path:

| what | value |
|---|---|
| shape | hover, circle or figure-eight, each with a chance of 1/3 |
| size $r$ | hover 0. Circle $U[4.8, 8.0]$ m. Figure-eight $U[3.6, 7.2]$ m |
| radii | $(r_x, r_y) = (\kappa_x r,\ \kappa_y r)$, with $\kappa_x, \kappa_y \sim U[0.85, 1.0]$ |
| speed $V$ | $U[2.83, 5.66]$ m/s |
| turn rate | $\Omega = \pm V / \bar r$, with $\bar r = (r_x + r_y)/2$ and a random sign |
| start angle $\theta_0$ | $U[0, 2\pi)$ |
| centre $c_{\text{path}}$ | uniform, where the whole path is 2 m inside the faces of the box. The height is uniform too |

With $\theta = \theta_0 + \Omega t$, the path point $x(t)$ is:

$$
\text{circle: } c_{\text{path}} + (r_x \cos\theta,\ r_y \sin\theta,\ 0), \qquad \text{figure-eight: } c_{\text{path}} + \left(r_x \sin\theta,\ \tfrac{1}{2} r_y \sin 2\theta,\ 0\right), \qquad \text{hover: } c_{\text{path}}
$$

- The evader starts on its path, with the velocity of the path.
- It follows the path point with $a_e = \ddot x + 4\, (x - p_e) + 4\, (\dot x - v_e)$.

### The win-rate matrix

After a pool run, every pursuer plays every evader: FRPN, the cascade and the pursuer
pool, against paths, APF, the cascade and the evader pool. Each pair flies 1024 episodes,
as in eval. A cell holds the pursuer's win rate.
