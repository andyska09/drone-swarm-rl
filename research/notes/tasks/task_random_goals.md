# random_goals

$N$ drones fly to random goals, one goal after another, and each drone knows only its own
goal. The task measures which information about the other drones helps a policy in this
traffic. The setup is the reliability test of
[Krátký et al. 2026](https://arxiv.org/abs/2512.08574), Section V-C.

## Setup

- $N$ drones share one role, `drone`, and one policy.
- The world is the default arena.
- The goals lie in the goal box $G$, in the middle of the arena.
- Each drone has one goal $g_i$ at a time.

| what | value |
|---|---|
| goal box $G$ | 20 × 20 × 1 m around $c$ |
| goal radius $\epsilon$ | 0.5 m |
| step limit $T$ | 1000 steps (10 s) |

## Start

- The drones start at random places in $G$, at least 2 m apart.
- Each drone gets a goal $g_i \sim U(G)$.
- The rest follows the default.

## Drone

Default.

## Observation

Drone $i$ sees every other drone that flies, nearest first. In `blind`, it sees no other
drone, so every slot has mask 0.

| input | content | size |
|---|---|---|
| actor | the default with $K = N - 1$, the sensed blocks in each slot, and the goal block $R_i^\top (g_i - p_i)$, $-R_i^\top v_i$ | $26 + (8 + s) K$ |
| critic | the default, and for each drone $j$: $g_j$, $n_{j,k} / n_{\max}$ for $k = 1, \dots, 4$, and $[\text{drone } j \text{ flies}]$ | $26 N$, plus the actor input |

Each slot of others holds the sensed blocks of the preset after its default content:

| block | content | size |
|---|---|---|
| attitude | $\operatorname{vec}(R_i^\top R_j)$, the attitude of drone $j$ in the body frame of drone $i$ | 9 |
| rpm | $n_{j,k} / n_{\max}$ for $k = 1, \dots, 4$, each motor speed of drone $j$ as a share of the top speed | 4 |

$s$ is the size of the sensed blocks: 0, 9, 4 or 13.

## Events

- **Goal:** a drone reaches its goal when it comes within $\epsilon$ of it, at any speed.
  Its next goal $g_i \sim U(G)$ comes in the same step.
- **Death:** the default. A death is terminal for that drone. The dead drone stays where
  it was at the start of that step and leaves the scene, so sight and contact skip it, and
  the other drones fly on.
- **End:** the episode ends when no drone flies, or after $T$ steps, which is a truncation
  for every flying drone.

## Reward

$$
r_i = \lambda_{\text{goal}} \left( \lVert g_i - p_i \rVert - \lVert g_i - p_i' \rVert \right) - \lambda_{\text{crash}}\, [\text{drone } i \text{ dies}]
$$

$p_i$ and $p_i'$ are the positions of drone $i$ at the start and at the end of the step.
$g_i$ is its goal at the start of the step.

| symbol | value |
|---|---|
| $\lambda_{\text{goal}}$ | 1 per m |
| $\lambda_{\text{crash}}$ | 30 |

## Metrics

| name | when | definition |
|---|---|---|
| `goals` | at the end | the number of goals that a drone reached, mean over the drones |
| `contact` | at the end | the share of the drones that died by touching another drone |
| `wall` | at the end | the share of the drones that died in any other way |
| `closest` | at the end | the smallest $d_{ij}$ between two flying drones in the episode |

## Presets

| preset | $N$ | a drone senses of the others |
|---|---|---|
| `alone` | 1 | no other drone exists |
| `blind` | 10 | nothing |
| `base` | 10 | the default slot |
| `attitude` | 10 | the default slot, attitude |
| `rpm` | 10 | the default slot, rpm |
| `attitude_rpm` | 10 | the default slot, attitude, rpm |

## Training

Default.

## Evaluation

The cascade flies drone $i$ to $g_i$.
