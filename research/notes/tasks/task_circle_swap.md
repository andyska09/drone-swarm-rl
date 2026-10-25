# circle_swap

$N$ drones start on a circle, and each drone flies to the point opposite its start. All
paths cross the middle, so the drones must avoid each other.

Log: [circle_swap_log.md](../experiment_logs/circle_swap_log.md)

## Setup

- $N$ drones share one role, `drone`, and one policy.
- The world is the default arena.
- The goal $g_i$ of drone $i$ is the point opposite its start.

| what | value |
|---|---|
| circle radius $r$ | 3 m, around the arena centre $c$ |
| step limit $T$ | 700 steps (7 s) |

## Start

- Drone $i$ starts at $p_i = c + r\, u_i$, with $u_i = (\cos\theta_i,\ \sin\theta_i,\ 0)$
  and $\theta_i = 2\pi i / N$.
- Its goal is $g_i = c - r\, u_i$.
- The rest follows the default.

## Drone

Default.

## Observation

| input | content | size |
|---|---|---|
| actor | the default, and the goal block $R_i^\top (g_i - p_i)$, $-R_i^\top v_i$ | $26 + 8K$ |
| critic | the default, and the goal $g_j$ of every drone | $21N$ |

## Events

Default.

## Reward

$$
r_i = -\Delta t \left( \lambda_{\text{dist}} \lVert g_i - p_i \rVert + \lambda_{\text{spin}} \lVert \omega_i \rVert + \lambda_{\text{close}} \sum_{j \ne i} \max\left(1 - \frac{d_{ij}}{d_{\text{close}}},\ 0\right) - \lambda_{\text{up}} R_{i,33} \right) - \lambda_{\text{crash}} N\, [\text{drone } i \text{ dies}]
$$

| symbol | value |
|---|---|
| $\lambda_{\text{dist}}$ | 1.0 per m |
| $\lambda_{\text{spin}}$ | 0.1 per rad/s |
| $\lambda_{\text{close}}$ | 10.0 |
| $d_{\text{close}}$ | 1.5 m |
| $\lambda_{\text{up}}$ | 1.0 |
| $\lambda_{\text{crash}}$ | 25.0 |

## Metrics

| name | when | definition |
|---|---|---|
| `distance` | at the end | $\lVert g_i - p_i \rVert$, per drone |
| `worst_distance` | at the end | $\max_i \lVert g_i - p_i \rVert$ |
| `closest` | at the end | the smallest $d_{ij}$ in the episode |

## Presets

| preset | $N$ | $K$ |
|---|---|---|
| `pair` | 2 | 1 |
| `circle8` | 8 | 7 |

## Training

Default.

## Evaluation

The cascade flies drone $i$ to $g_i$.
