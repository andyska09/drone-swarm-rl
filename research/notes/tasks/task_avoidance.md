# avoidance

Two drones fly head-on through a closed tube and pass each other. The task measures what
a drone must sense to do this. The input is the sight range $d$, and the reading is the
success rate.

Log: [avoidance_log.md](../experiment_logs/avoidance_log.md)

## Setup

- Two drones share one role, `drone`, and one policy.
- The world is a closed tube with a square cross-section. It stands on the ground, and
  each episode turns it to a random yaw.
- Each drone $i$ flies to its goal $g_i$, the start point of the other drone.

| what | value |
|---|---|
| tube width and height | 3 m |
| tube length $\ell$ | 5 m to 20 m, see Training |
| step limit $T$ | 1100 steps |

## Start

A tilde marks the tube frame: its origin is in the middle of the tube, $x$ runs along the
tube, and $z$ points up.

- Drone 0 starts 1 m from the end wall at $-x$, and drone 1 starts 1 m from the end wall
  at $+x$: $\tilde p_{i,x} = \mp (\ell/2 - 1\ \text{m})$.
- Across the tube, each start is uniform in the middle 1.6 m × 1.6 m:
  $\tilde p_{i,y},\ \tilde p_{i,z} \sim U[-0.8, 0.8]$ m.
- The rest follows the default.

## Drone

Default.

## Observation

Drone $i$ sees drone $j$ when both drones fly and $d_{ij} < d$. An unseen drone gets
mask 0 and a row of zeros.

| input | content | size |
|---|---|---|
| actor, own | the default, with $w_i$ measured to the tube walls | 20 |
| actor, others and mask | the default, with the sight rule above | 8 |
| actor, goal | $R_i^\top (g_i - p_i)$, $-R_i^\top v_i$ | 6 |
| critic | for each drone: $\tilde p_j$, $\tilde v_j$, $\operatorname{vec}(\tilde R_j)$, $\omega_j$, $\tilde g_j$ and $[\text{drone } j \text{ arrived}]$. Then $\ell$, and the time left, $10\ \text{s} - t$ | 46 |

## Events

- **Wall hit:** a drone touches a wall, the floor or the ceiling. The drone dies.
- **Contact:** the drones touch. Both drones die.
- **Arrival:** a drone comes within 1 m of its goal. It freezes there, and sight and
  contact skip it from then on.
- **End:** the episode ends at a death, when both drones arrived, or after 10 s. Each of
  these ends is terminal.

## Reward

$$
r_i = \lambda_{\text{arrive}}\, [\text{drone } i \text{ arrives}] - \lambda_{\text{crash}}\, [\text{drone } i \text{ dies}]
$$

| symbol | value |
|---|---|
| $\lambda_{\text{arrive}}$ | 1 |
| $\lambda_{\text{crash}}$ | 1 |

## Metrics

| name | when | definition |
|---|---|---|
| `success` | at the end | both drones arrived |
| `wall` | at the end | a drone hit a wall |
| `contact` | at the end | the drones touched |
| `timeout` | at the end | the episode ended at 10 s |
| `closest` | at the end | the smallest $d_{01}$ while both drones fly |
| `time_to_arrive` | at the end | per drone, the time from the start to its arrival. The mean counts only the episodes where both drones arrived |
| `length` | at the end | $\ell$ of the episode |

## Presets

| preset | $d$ |
|---|---|
| `d_inf` | $\infty$ |
| `d6`, `d5`, `d4`, `d3` | 6, 5, 4, 3 m |
| `d0` | 0 m |

## Training

- Runs set the entropy weight to 0.
- $\ell$ starts at 5 m. After an update, $\ell$ grows by 5 m, up to 20 m, if both of these
  are true:
  - every episode that ended in the update flew the current $\ell$,
  - more than 80% of these episodes were a success.
- A new $\ell$ holds for the episodes that start after the change.

## Evaluation

- Eval flies the 20 m tube.
- The cascade flies drone $i$ to $g_i$.
