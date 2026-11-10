# Huang et al. 2024

$N$ drones fly to their goals through a room full of columns, and avoid each other and
the columns. Each drone sees itself, its nearest drones and the columns near it, and sets
the thrust of each rotor. Paper: [arXiv:2309.13285v2](https://arxiv.org/abs/2309.13285).
Code: [quad-swarm-rl](https://github.com/Zhehui-Huang/quad-swarm-rl).

- A `source` that names a file gives a value from the code. The paths start at the root of
  quad-swarm-rl.
- A flag with only a line number is set in
  `swarm_rl/runs/obstacles/quad_obstacle_baseline.py`. The run of the base setting,
  `swarm_rl/runs/obstacles/quads_multi_obstacles.py`, adds to that file.

## Setup

- $N$ drones share one role, `drone`, and one policy. (§III-A, Fig. 1)
- Each drone $i$ flies to its goal $g_i$, and avoids the other drones and the obstacles.
  (§III-A)
- The scenario sets the goals, see Presets. Training uses the scenarios `random` and
  `same_goal`. (§III-B) Code: each episode draws one of the two, each with a chance of 1/2
  (`gym_art/quadrotor_multi/scenarios/utils.py:17`,
  `gym_art/quadrotor_multi/scenarios/mix.py:78-79`).
- The world is the arena box $B$ (his room), above a floor. The obstacles are columns from
  the floor to the ceiling. (§III-B, Fig. 1)
- The middle 8 × 8 m of the floor is a grid of 64 cells of 1 m². An obstacle stands at the
  centre of a cell. The density is the share of the cells that hold an obstacle. (§III-B)
- Each episode puts the obstacles in new random cells. (§III-B) Code:
  $\lfloor 64 \times \text{density} \rfloor$ different cells
  (`gym_art/quadrotor_multi/quadrotor_multi.py:304-325`).

| what | value | source |
|---|---|---|
| arena box $L$ | 10 × 10 × 10 m | §III-B |
| centre $c$ and floor height | $c$ = (0, 0, 5) m, floor at $z = 0$ | `gym_art/quadrotor_multi/quadrotor_single.py:146-147` |
| obstacle shape | a round column | Fig. 1 |
| obstacle size | 0.6 m, the diameter of a column | §IV, §IV-D. Code: radius = size / 2 (`gym_art/quadrotor_multi/obstacles/obstacles.py:10`) |
| obstacle density | 20 % | §IV |
| step limit $T$ | 1500 steps | §IV-A. Code: 15 s, `--quads_episode_duration=15.0` (line 12) |

## Start

- Each drone starts at the centre of a cell with no obstacle, at a random height. (§III-B)
  Code: each drone gets its own cell
  (`gym_art/quadrotor_multi/scenarios/obstacles/o_base.py:69-81`). Then the start moves by
  $U[-0.1, 0.1]$ m on each axis, and it stays at least 0.75 m high
  (`gym_art/quadrotor_multi/quadrotor_single.py:215-216, 394-402`).
- The attitude and the velocity are random. (§III-B)
- The scenario sets $g_i$, see Presets.

| what | value | source |
|---|---|---|
| height | $U[1, 3]$ m | §III-B |
| $R_i$, $v_i$ and $\omega_i$ | random | §III-B. Code: level, with a random heading whose nose points within 60° of the direction to the room centre. $v_i = 0$ and $\omega_i = 0$ (`gym_art/quadrotor_multi/quadrotor_single.py:424-434`, `swarm_rl/env_wrappers/quad_utils.py:62`) |
| motor thrust | 0 | `gym_art/quadrotor_multi/quadrotor_dynamics.py:385-387` |

## Drone

The Crazyflie 2.1, in the QuadSwarm simulator. The model follows Mellinger & Kumar [30]
and Lee et al. [31]. (§II, §III-A, §IV-F)

$$
\dot p = v, \qquad m \dot v = -m g\, e_3 + R \sum_{k=1}^{4} f_k\, e_3, \qquad \dot R = R\, [\omega]_\times, \qquad J \dot\omega = \tau - \omega \times J \omega
$$

- $f_k$ is the thrust of rotor $k$. $\tau$ is the torque of the four thrusts on their arms,
  plus a yaw torque of $0.006 f_k$ from each propeller, with the sign of its spin.
  (`gym_art/quadrotor_multi/quadrotor_dynamics.py:517-533`)
- Each motor follows its command $u_k \in [0, 1]$ with a lag. Over one physics step:
  (`gym_art/quadrotor_multi/quadrotor_dynamics.py:504-517`)

$$
s_k \leftarrow s_k + \min\left(1, \frac{4 \delta}{\tau_m}\right) \left(\sqrt{u_k} - s_k\right), \qquad f_k = f_{\max} \min\left(1,\ \max\left(0,\ s_k^2 + u_k \eta_k\right)\right)
$$

- $\eta_k$ is OU noise (Ornstein–Uhlenbeck: a random value that drifts back to 0), with
  $\theta$ = 0.15 and $\sigma$ = 0.01. It gets a new value once per policy step.
  (`gym_art/quadrotor_multi/quadrotor_dynamics.py:168-173, 208-214`,
  `gym_art/quadrotor_multi/numba_utils.py:81-105`)
- The integrator steps $p$, $v$ and $\omega$ by Euler, and turns $R$ by $\omega \delta$.
  Every 0.5 s, $R$ is made orthonormal again.
  (`gym_art/quadrotor_multi/quadrotor_dynamics.py:54, 535-563, 645`)
- $\omega$ stays in [−40, 40] rad/s on each axis.
  (`gym_art/quadrotor_multi/quadrotor_dynamics.py:49, 560`)
- Downwash is the air that a drone blows down. It pushes a drone that is up to 0.7 m
  below another drone, and within 0.1 m of its $z$ axis. The push is
  $\frac{6}{17}(7 - 10\, d_{ij}) + U[-0.1, 0.1]$ m/s² along the $-z$ axis of the upper
  drone, with a small random spin. (`gym_art/quadrotor_multi/aerodynamics/downwash.py:13-49`,
  `--quads_use_downwash=True`, line 21)

| what | value | source |
|---|---|---|
| body radius $\rho$ | 0.05 m | abstract, §IV-D. Code: the arm, 0.046 m (`gym_art/quadrotor_multi/quad_models.py:11`, `gym_art/quadrotor_multi/quadrotor_dynamics.py:158`) |
| $m$ | 0.028 kg, the sum of the parts | `gym_art/quadrotor_multi/quad_models.py:5-9` |
| $J$ | from the shapes and the masses of the parts | `gym_art/quadrotor_multi/inertia.py:295-303` |
| $g$ | 9.81 m/s² | `gym_art/quadrotor_multi/quadrotor_dynamics.py:12` |
| $f_{\max}$ | $1.9\, m g / 4$ per rotor | `gym_art/quadrotor_multi/quad_models.py:25`, `gym_art/quadrotor_multi/quadrotor_dynamics.py:134` |
| $\tau_m$ | 0.15 s | `gym_art/quadrotor_multi/quad_models.py:31-32` |
| drag | none | `gym_art/quadrotor_multi/quad_models.py:29-30`, `swarm_rl/env_wrappers/quad_utils.py:31` |
| physics step $\delta$ | 0.005 s (200 Hz) | `gym_art/quadrotor_multi/quadrotor_single.py:102, 157` |
| policy step $\Delta t$ | 0.01 s (100 Hz) | `gym_art/quadrotor_multi/quadrotor_single.py:102, 160` |

### Action and control

- The action is the thrust level of each rotor: $a_i \in [0, 1]^4$, and $u_k = a_{i,k}$.
  0 is no thrust, and 1 is $f_{\max}$. (§III-B) Code: the network gives
  $a_i \in [-1, 1]^4$, which is clipped, and $u_k = (a_{i,k} + 1)/2$
  (`gym_art/quadrotor_multi/quadrotor_control.py:43-57`,
  `swarm_rl/env_wrappers/quad_utils.py:24`).
- The policy sets the rotor thrusts directly, with no control loop between. (§II)
- Each action holds for 2 physics steps. (`gym_art/quadrotor_multi/quadrotor_single.py:102`,
  `gym_art/quadrotor_multi/quadrotor_dynamics.py:208-214`)

## Observation

The actor input of drone $i$ (his $o_i = (e_i, \eta_i, \zeta_i)$).

| block | content | size | source |
|---|---|---|---|
| own (his $e_i$) | $p_i - g_i$, $v_i$, $\operatorname{vec}(R_i)$, $\omega_i$, $z_i$ (his $h_i$). $v_i$ is in the world frame, and $\omega_i$ in the body frame | 19 | §III-B |
| others (his $\eta_i$) | for each of the $K$ nearest other drones $j$, nearest first: the position and the velocity of drone $i$ relative to drone $j$ (his $\tilde p_{ij}$, $\tilde v_{ij}$) | $6K$ | §III-B. Code: $p_j - p_i$ and $v_j - v_i$ in the world frame, drone by drone. The order is by $d_{ij} + \dot d_{ij} \cdot 1$ s (`gym_art/quadrotor_multi/quadrotor_multi.py:219-231, 259-270`) |
| obstacles (his $\zeta_i$) | an SDF on a 3 × 3 grid of points around the drone, 0.1 m apart | 9 | §III-B |

- $K = 2$, and $K \le N - 1$. (§IV, §III-B)
- An SDF (signed distance field) gives, at each point, the distance to the nearest
  obstacle. (§III-B) Code: the distance in the floor plane from the point to the surface of
  the nearest column. It is negative inside a column
  (`gym_art/quadrotor_multi/obstacles/utils.py:4-27`).
- The obstacle block holds 9 values for any number of obstacles, in any order. (§III-B)
- Code: each value of others is clipped, a position to ±10 m and a velocity to ±6 m/s.
  (`gym_art/quadrotor_multi/quadrotor_single.py:294-295`,
  `gym_art/quadrotor_multi/quadrotor_multi.py:241-243`)
- Code: the own block carries Gaussian sensor noise, with std 0.005 m on the position,
  0.01 m/s on $v_i$ and 1.75e−4 rad/s on $\omega_i$.
  (`gym_art/quadrotor_multi/sensor_noise.py:70-73, 235-261`,
  `swarm_rl/env_wrappers/quad_utils.py:30`)
- The networks read the inputs in raw SI units. (`--normalize_input=False`, line 7)

The critic input of drone $i$ is its actor input (his IPPO: PPO in which each drone learns
from its own view). (§III-B, §IV-B)

## Events

- **Contact** (his inter-robot collision): two drones touch, $d_{ij} \le d_c$. Both bounce
  off, and the episode goes on. (§II, §IV-A) Code: the velocities along the line between
  the centres swap, with noise. Then each speed drops to 20–80 %, and a random spin is
  added (`gym_art/quadrotor_multi/collisions/quadrotors.py:24-59`,
  `gym_art/quadrotor_multi/collisions/utils.py:8-33`).
- **Obstacle hit** (his robot-obstacle collision): the ball of radius $\rho$ of a drone
  meets an obstacle. The drone bounces off. (§III-B) Code: the test is in the floor plane,
  with the arm as $\rho$ (`gym_art/quadrotor_multi/obstacles/utils.py:31-43`). The bounce
  turns the velocity away from the column axis, with noise, and adds a random spin
  (`gym_art/quadrotor_multi/collisions/obstacles.py:23-50`).
- **Floor hit** (his crash with the floor): a drone comes down to the floor. (§III-B)
  Code: at $z_i \le$ 0.046 m, the drone stops. $v_i$ and $\omega_i$ become 0, the drone
  turns level, and its motors stop. It sits on the floor until its thrust lifts it again
  (`gym_art/quadrotor_multi/quadrotor_dynamics.py:378, 570-639`).
- **Wall hit:** a drone reaches a wall or the ceiling. Code: it stays in $B$, and it
  bounces back down with 20–80 % of its speed and a random spin
  (`gym_art/quadrotor_multi/quadrotor_dynamics.py:360-367`,
  `gym_art/quadrotor_multi/collisions/room.py:6-113`).
- **End:** the episode ends after $T$ steps, for all drones at once. (§IV-A) Code: this is
  the only end, and it is terminal (`gym_art/quadrotor_multi/quadrotor_single.py:353`,
  `gym_art/quadrotor_multi/quadrotor_multi.py:626-723`,
  `swarm_rl/env_wrappers/compatibility.py:42-50`).

| what | value | source |
|---|---|---|
| contact distance $d_c$ | 0.092 m, two arms | `--quads_collision_hitbox_radius=2.0` (line 16), `gym_art/quadrotor_multi/quadrotor_multi.py:154` |

## Reward

$$
\begin{aligned}
r_i ={}& -\lambda_{\text{dist}} \lVert p_i - g_i \rVert - \lambda_{\text{obst}}\, [\text{obstacle hit}_i] - \lambda_{\text{contact}}\, [\text{contact}_i] - \lambda_{\text{close}} \sum_{j=1}^{K} \max\left(1 - \frac{d_{ij}}{d_{\text{close}}},\ 0\right) \\
& - \lambda_{\text{floor}}\, [\text{floor}_i] - \lambda_{\text{spin}} \lVert \omega_i \rVert - \lambda_{\text{effort}} \lVert f_i \rVert + \lambda_{\text{up}} R_{i,33}
\end{aligned}
$$

- The first term is his $r_{i,\text{dist}}$. The next three are his $r_{i,\text{col}}$, and
  the last four are his $r_{i,\text{control}}$. (§III-B)
- A contact or an obstacle hit lasts more than one step, but it costs only once. (§III-B)
  Code: a drone pays at the first step of each touch
  (`gym_art/quadrotor_multi/quadrotor_multi.py:440, 467, 488, 501-504, 518`).
- The sum runs over the $K$ sensed drones. (§III-B) Code: over every other drone with
  $d_{ij} \le d_{\text{close}}$ (`gym_art/quadrotor_multi/quadrotor_multi.py:443-444, 507-511`,
  `gym_art/quadrotor_multi/collisions/quadrotors.py:95-103`).
- $[\text{floor}_i]$ is 1 when drone $i$ hits the floor. (§III-B) Code: it is 1 at each step
  that the drone sits on the floor. Then the up term is $-\lambda_{\text{up}}$ in place of
  $\lambda_{\text{up}} R_{i,33}$ (`gym_art/quadrotor_multi/quadrotor_single.py:45-48, 57`).
- $\lVert f_i \rVert$ is the norm of the four thrusts (his control effort). (§III-B) Code:
  the norm of the network output $a_i \in [-1, 1]^4$
  (`gym_art/quadrotor_multi/quadrotor_single.py:41`).
- Code: each step multiplies the terms of $\lambda_{\text{dist}}$, $\lambda_{\text{floor}}$,
  $\lambda_{\text{spin}}$, $\lambda_{\text{effort}}$ and $\lambda_{\text{up}}$ by
  $\delta$ = 0.005 s, and the term of $\lambda_{\text{close}}$ by $\Delta t$ = 0.01 s. The
  terms of $\lambda_{\text{contact}}$ and $\lambda_{\text{obst}}$ are one-off.
  (`gym_art/quadrotor_multi/quadrotor_single.py:60, 348-350`,
  `gym_art/quadrotor_multi/quadrotor_multi.py:510`)
- Training grows $\lambda_{\text{contact}}$, $\lambda_{\text{obst}}$ and
  $\lambda_{\text{close}}$, see Training.

| symbol | value | source |
|---|---|---|
| $\lambda_{\text{dist}}$ (his $\alpha_{\text{dist}}$) | 1.0 per m | `swarm_rl/env_wrappers/reward_shaping.py:9` |
| $\lambda_{\text{obst}}$ (his $\alpha_{\text{ocol}}$) | 5.0 | `--quads_obst_collision_reward=5.0` (line 21) |
| $\lambda_{\text{contact}}$ (his $\alpha_{\text{rcol}}$) | 5.0 | `--quads_collision_reward=5.0` (line 17) |
| $\lambda_{\text{close}}$ (his $\alpha_{\text{rclose}}$) | 4.0 | `--quads_collision_smooth_max_penalty=4.0` (line 17) |
| $d_{\text{close}}$ (his $d_{\text{rclose}}$) | 0.184 m, four arms | `--quads_collision_falloff_radius=4.0` (line 17), `gym_art/quadrotor_multi/quadrotor_multi.py:155` |
| $\lambda_{\text{floor}}$ (his $\alpha_{\text{floor}}$) | 1.0 | `swarm_rl/env_wrappers/reward_shaping.py:9` |
| $\lambda_{\text{spin}}$ (his $\alpha_{\omega}$) | 0.1 per rad/s | `swarm_rl/env_wrappers/reward_shaping.py:9` |
| $\lambda_{\text{effort}}$ (his $\alpha_{f}$) | 0.05 | `swarm_rl/env_wrappers/reward_shaping.py:9` |
| $\lambda_{\text{up}}$ (his $\alpha_{\text{orient}}$) | 1.0 | `swarm_rl/env_wrappers/reward_shaping.py:9` |

## Metrics

| name | when | definition | source |
|---|---|---|---|
| `success` (his success rate) | at the end | the share of the drones that reach their goal with no contact and no obstacle hit | §IV. Code: a drone reaches its goal when its distance, mean over 5 steps, falls below 0.5 m in `random` and 1.0 m in `same_goal`. Only the contacts and hits after the first 1.5 s count (`gym_art/quadrotor_multi/quadrotor_multi.py:451-454, 471-480, 542-546, 685-687`, `gym_art/quadrotor_multi/scenarios/obstacles/o_random.py:10`, `gym_art/quadrotor_multi/scenarios/obstacles/o_static_same_goal.py:13`) |
| `collision` (his collision rate) | at the end | the share of the drones with at least one contact or obstacle hit | §IV. Code: after the first 1.5 s (`gym_art/quadrotor_multi/quadrotor_multi.py:696`) |
| `distance` (his distance to goal) | at the end | $\lVert p_i - g_i \rVert$, mean over the drones and over the last 1 s | §IV. Code: `gym_art/quadrotor_multi/quadrotor_multi.py:649-650` |
| `flight_distance` (his flight distance) | at the end | the length of the path of a drone, mean over the drones | §IV |
| `inference_time` (his inference time) | at each step | the time from the start of the observation to the action | §IV |

- The paper gives `distance` for `random` and `same_goal` apart. (Fig. 3)

## Training

PPO for each drone on its own view (his IPPO), in its asynchronous form (APPO) from Sample
Factory. In APPO, the workers keep collecting steps while the learner updates. (§III-B)

| setting | value | source |
|---|---|---|
| actor | an MLP of 2 layers for each of the own, the others and the obstacle blocks. A multi-head attention (a layer that weights its inputs by learned scores) takes the others and the obstacle embeddings. The own embedding and the two attention outputs go into an MLP of 2 layers that gives the action | §III-C, Fig. 2 |
| hidden size and activation | 256, tanh | `--rnn_size=256` (line 5), `--nonlinearity=tanh` (line 4), `swarm_rl/models/quad_multi_model.py:148-175` |
| attention heads | 4 | `swarm_rl/models/quad_multi_model.py:170` |
| action noise | a Gaussian with one learned std per action, the same in every state | `--adaptive_stddev=False` (line 5) |
| critic | the network of the actor, with its own weights | `--actor_critic_share_weights=False` (line 4) |
| $\gamma$ | not published | — |
| GAE $\lambda$ | 1.0 | `--gae_lambda=1.00` (line 6) |
| PPO clip | not published | — |
| value clip | 5.0 | `--ppo_clip_value=5.0` (line 3) |
| entropy weight | 0 | `--exploration_loss_coeff=0.0` (line 6) |
| gradient norm clip | 5.0 | `--max_grad_norm=5.0` (line 6) |
| optimizer | not published | — |
| learning rate | 1e−4 | `--learning_rate=0.0001` (line 3) |
| rollout | 128 steps | `--rollout=128` (line 6) |
| minibatch | 1024 steps | `--batch_size=1024` (line 6) |
| epochs × minibatches | not published | — |
| environments | 144, each with $N$ drones: 36 workers × 4 | `--num_workers=36`, `--num_envs_per_worker=4` (`quads_multi_obstacles.py:12`) |
| steps | 1e9, and 2e9 in the size runs | the axes of Fig. 3–9 and of Fig. 10. Code: `--train_for_env_steps=1000000000` (line 2) |
| seeds | 4 | §IV. Code: 0, 1111, 2222, 3333 (`quads_multi_obstacles.py:6`) |
| weight init | Xavier uniform | `--policy_initialization=xavier_uniform` (line 4) |
| time and hardware | not published | — |

- Code: the reward is clipped to [−10, 10], with no scale. (`--reward_clip=10`,
  `--normalize_returns=False`, line 7)
- Code: the learner takes steps from a policy of any age. V-trace, a correction for steps
  from an older policy, is off. (`--max_policy_lag=100000000`, `--with_vtrace=False`,
  line 5)

### Replay

His replay buffer.

- When a contact or an obstacle hit happens, the state of the whole environment 1.5 s
  before it goes into a buffer. (§III-D)
- One episode can store several states, one for each collision. (§III-D) Code: at most one
  state in each 5 s of an episode, and a replayed episode stores none
  (`gym_art/quadrotor_multi/quad_experience_replay.py:28, 140-163`).
- At the start of an episode, a stored state comes back with the chance $\alpha_r$. Else a
  new episode starts. (§III-D)
- A replayed episode starts at the stored state and runs to the step limit. (§IV-A) Code:
  the stored state keeps its step count
  (`gym_art/quadrotor_multi/quad_experience_replay.py:99-104, 176-193`).
- After each episode, a state that came back more often than a set limit leaves the
  buffer. (§III-D)

| what | value | source |
|---|---|---|
| $\alpha_r$ (his replay rate) | 0.75 | `--replay_buffer_sample_prob=0.75` (line 10) |
| limit of returns | 10: a state leaves after its 10th return | `gym_art/quadrotor_multi/quad_experience_replay.py:48-54, 191` |
| buffer size | 20 states | `gym_art/quadrotor_multi/quad_experience_replay.py:17` |
| a state is saved | every 0.5 s | `gym_art/quadrotor_multi/quad_experience_replay.py:17-20, 140-142` |
| collisions that store a state | after the first 1.5 s | `gym_art/quadrotor_multi/quad_experience_replay.py:148-149` |
| the buffer starts | when the drones fly: over the last 100 episodes, and at least 10, the first drone sits on the floor for less than 2 s per episode, on the mean | `gym_art/quadrotor_multi/quadrotor_multi.py:174, 281-287, 355-359, 611-612` |

### Penalty ramp

- Code: $\lambda_{\text{contact}}$, $\lambda_{\text{obst}}$ and $\lambda_{\text{close}}$ grow
  in a straight line from 0 to their values over the first 3e8 environment steps. Each new
  value holds from the end of an episode. (`--anneal_collision_steps=300000000`, line 9,
  `swarm_rl/env_wrappers/quad_utils.py:79-88`,
  `swarm_rl/env_wrappers/reward_shaping.py:110-118`)

## Evaluation

- The curves of Fig. 3–10 give the metrics over the training steps. (Fig. 3–10) Code: from
  the training episodes, with the replayed episodes left out
  (`gym_art/quadrotor_multi/quadrotor_multi.py:682-718`).
- Table I sets the policy against two pilots, with $N$ = 8 and a density of 20 %. (§IV-D)
- Fig. 11 sets the policy against the same pilots with $N$ = 32: the densities 20–80 % at a
  size of 0.6 m, and the sizes 0.6–0.85 m at a density of 80 %. (§IV-D, Fig. 11)
- The scenarios `pursuit_evasion` and `swap_goals` test a policy that trained on `random`
  and `same_goal`. Each runs 20 episodes, with $N$ = 8 and a density of 20 %. (§IV-E)
- The paper draws maps of the critic value over positions in the floor plane, with every
  velocity at 0. (§IV-B, Fig. 6)
- The number of episodes for Table I and Fig. 11: not published.
- A network flies its mean action, or it draws an action: not published.

| pilot | what flies | source |
|---|---|---|
| SBC | a controller [30] gives an acceleration toward the goal. Safety barrier certificates (a rule that keeps the drones out of each other and out of the obstacles) change it to a safe acceleration, from the state of every drone and obstacle within a radius. The controller turns it into thrusts | §IV-D |
| GLAS | a learned policy (an imitation of a central planner, with a safety module) gives a velocity in the floor plane, from the drones and obstacles within a sensing radius. The controller of SBC turns it into thrusts. Each drone starts and ends at one height | §II, §IV-D |

- The radius of SBC and the sensing radius of GLAS: not published.

### Real flights

- The drones are Crazyflie 2.1. A Vicon system (cameras that track each drone) gives the
  positions at 100 Hz. The controller of each drone runs at 1000 Hz. (§IV-F)
- Each drone keeps a list of the obstacles, and builds its SDF from the list on board.
  (§IV-F)
- The drones fly the network of `small`. (§IV-F)
- 8 drones fly at a density of 20 %. (abstract) Fig. 12 shows 8 drones and 5 obstacles.
  (Fig. 12)
- Table II sets `small` against `distill` in simulation, over 20 episodes, with $N$ = 8 and
  a density of 20 %. (§IV-F, Table II)

## Presets

### Scenarios

| scenario | goals | used in | source |
|---|---|---|---|
| `random` (his random) | each drone has its own random goal | training | §III-B. Code: the centre of a free cell, a different cell for each drone, at a height of $U[1, 3]$ m (`gym_art/quadrotor_multi/scenarios/obstacles/o_random.py:42-43`, `gym_art/quadrotor_multi/scenarios/obstacles/o_base.py:69-81`) |
| `same_goal` (his static same goal) | one goal for all drones, at the place in the room farthest from any obstacle | training | §III-B. Code: the centre of the largest free square of cells, at a height of $U[1.5, 3]$ m (`gym_art/quadrotor_multi/scenarios/obstacles/o_static_same_goal.py:41, 48`, `gym_art/quadrotor_multi/scenarios/obstacles/o_base.py:124-153`) |
| `pursuit_evasion` (his pursuit evasion) | one goal for all drones. It moves along a Bézier curve (a smooth curve set by a few control points) | eval only | §IV-E. Code: a new curve of degree 2 every 6 s (`gym_art/quadrotor_multi/scenarios/obstacles/o_ep_rand_bezier.py:20-49`) |
| `swap_goals` (his swap goals) | the drones swap their goals after a random time | eval only | §IV-E. Code: the goals form a shape around the centre of the largest free square, and a shuffle swaps them every 4–6 s (`gym_art/quadrotor_multi/scenarios/obstacles/o_swap_goals.py:22-23, 37-52`) |

### Setups

`base` is the base setting of §IV. Each other row changes the values that its source
names, and keeps the rest of `base`. (§IV)

| preset | $N$ | $K$ | density | size | source |
|---|---|---|---|---|---|
| `base` | 8 | 2 | 20 % | 0.6 m | §IV |
| `n16`, `n32` | 16, 32 | 2 | 20 % | 0.6 m | §IV-C.1, Fig. 7 |
| `k1`, `k2`, `k6`, `k16`, `k31` | 32 | 1, 2, 6, 16, 31 | 20 % | 0.6 m | §IV-C.2, Fig. 8 |
| `d40`, `d60`, `d80` | 8 | 2 | 40, 60, 80 % | 0.6 m | §IV-C.3, Fig. 9 |
| `s07`, `s08`, `s085` | 8 | 2 | 80 % | 0.7, 0.8, 0.85 m | §IV-C.4, Fig. 10 |

### Variants

Each variant changes `base` as its row says.

| preset | change | source |
|---|---|---|
| `no_replay` | no replay | §IV-A, Fig. 3 |
| `no_attention` | no replay, and no attention | §IV-A, Fig. 3 |
| `no_sdf` (his base policy) | no replay, no attention, and the $L$ nearest obstacles in place of the SDF. This is the policy of Batra et al. [16] | §IV-A, Fig. 3 |
| `plr` | PLR (prioritized level replay: it replays start states by a score, here the L1 value loss) in place of the replay | §IV-A, Fig. 4 |
| `range` | others: every drone within 4 m, read by the attention model of [16], in place of the $K$ nearest drones and their MLP | §IV-A, Fig. 5 |
| `small` | hidden size 10 in the three encoders and the attention, and one attention head: 1820 weights in all | §IV-F |
| `distill` | the network of `small`, trained by policy distillation (it learns to copy a large trained policy) | §IV-F, Table II |

| what | value | source |
|---|---|---|
| $L$ in `no_sdf` | not published | — |
| how `no_attention` joins the embeddings | not published | — |
| the large policy that `distill` copies | not published | — |
