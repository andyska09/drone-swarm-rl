# tunnel

Archived: `swarm/envs/tunnel.py` was deleted on 2026-10-27 (commit `b31304e`). This
note keeps the task as it was.

Two drones fly opposite ways through a bent tube, each along its own line, and pass each
other. The bends hide the drones from each other until late. The task measures what a
drone must sense to pass the other drone.

Log: [tunnel_log.md](../experiment_logs/tunnel_log.md)

## Setup

- Two drones share one role, `drone`, and one policy.
- The world is a closed tube with a square cross-section, bent twice. Seen from above, it
  has a left leg, a middle piece and a right leg, like `/‾\`. Its middle is at the arena
  centre $c$, and each episode turns it to a random yaw.
- Each drone flies along its own line, from the end of one leg to the end of the other.
  The two drones fly opposite ways.
- A line is the centre line of the tube, moved sideways and up or down by an offset. It
  keeps the same distance to the walls through the bends.

| what | value |
|---|---|
| tube width and height | 2.4 m |
| inner corner | 111°, through the solid, so each bend turns 69° |
| middle piece | 2.4 m of inner wall |
| leg | 10 m along a line, from the bend to the end of the line |
| closed end | 1 m past each end of a line |
| centre line | 24.05 m |
| line offset $o_i$ | uniform in $[-0.5, 0.5]$ m sideways and up or down, new for each drone in each episode |
| step limit $T$ | 1500 steps (15 s) |

| symbol | meaning |
|---|---|
| $s_i$ | how far drone $i$ is along its line: the arc length of the nearest point of the line, seen from above |
| $q_i(s)$ | the point of the line of drone $i$ at the arc length $s$. Past the end of the line, it is the end point |

## Start

| what | training | eval |
|---|---|---|
| start line | the centre line, moved by a new offset $\sim U[-0.5, 0.5]^2$ m, in the direction of drone $i$ | the same |
| start point | at an arc length $\sim U[0,\ \ell - 2\ \text{m})$ on the start line, with $\ell$ the length of that line | at the start of the start line |
| spacing | the first of 16 draws in which the drones start at least 2 m apart | the same |
| the rest | the default | the default |

## Drone

Default.

## Observation

Drone $i$ sees drone $j$ when, seen from above, the segment between their centres stays
inside the tube, and $d_{ij} < d$. An unseen drone gets mask 0 and a row of zeros.

| input | content | size |
|---|---|---|
| actor, own | the default, with $z_i$ measured from the floor of the tube and $w_i$ measured to the tunnel walls | 20 |
| actor, others and mask | the default, with the sight rule above | 8 |
| actor, goal | $R_i^\top (q_i(s_i) - p_i)$ and $R_i^\top (q_i(s_i + 3\ \text{m}) - p_i)$ | 6 |
| critic | for each drone: $\tilde p_j$, $\tilde v_j$, $\operatorname{vec}(\tilde R_j)$, $\omega_j$, $s_j$, $o_j$, and $[\text{drone } j \text{ sees the other drone}]$ | 44 |

A tilde marks the tunnel frame: its origin is at $c$, $x$ runs along the middle piece, $y$
points toward the outer wall of the middle piece, and $z$ points up.

## Events

- **Wall hit:** a drone touches a wall, the floor or the ceiling. The drone dies.
- **Contact:** the drones touch. Both drones die.
- **Finish:** a drone reaches the end of its line.
- **End:** the episode ends at a death or a finish, which is terminal, or after $T$ steps,
  which is a truncation.

## Reward

$$
r_i = \lambda_{\text{prog}} \left( s_i(t+1) - s_i(t) \right) - \Delta t\, \lambda_{\text{line}} \lVert p_i - q_i(s_i) \rVert - \lambda_{\text{crash}}\, [\text{drone } i \text{ dies}]
$$

| symbol | value |
|---|---|
| $\lambda_{\text{prog}}$ | 1.0 per m |
| $\lambda_{\text{line}}$ | 1.0 per m |
| $\lambda_{\text{crash}}$ | 10.0 |

## Metrics

| name | when | definition |
|---|---|---|
| `success` | at the end | a drone finished, and both drones are alive |
| `contact` | at the end | the episode ended in a contact |
| `wall` | at the end | the episode ended in a wall hit |
| `timeout` | at the end | the episode reached $T$ |
| `closest` | at the end | the smallest $d_{01}$ in the episode |
| `line_distance_unseen` | at the end | per drone, the mean of $\lVert p_i - q_i(s_i) \rVert$ over the steps where the other drone is out of sight |
| `line_distance` | mean over the steps | per drone, $\lVert p_i - q_i(s_i) \rVert$ |
| `speed` | mean over the steps | per drone, $\left(s_i(t+1) - s_i(t)\right) / \Delta t$ |

## Presets

| preset | $d$ |
|---|---|
| `d_inf` | $\infty$ |
| `d6`, `d5`, `d4`, `d3` | 6, 5, 4, 3 m |

## Training

Default.

## Evaluation

- Each drone starts at the start of its start line (see Start).
- The cascade flies drone $i$ to $q_i(s_i + 3\ \text{m})$.
- The line follower is a baseline for one seat or for both. It flies along its own line
  at 2 m/s. It gives the acceleration
  $4\, (q_i(s_i) - p_i) + 4\, (2\ \text{m/s} \cdot \hat t_i - v_i)$, where $\hat t_i$
  points from $q_i(s_i)$ to $q_i(s_i + 1\ \text{m})$. The default acceleration chain flies
  it. By default, it flies drone 1, and the policy flies drone 0.
