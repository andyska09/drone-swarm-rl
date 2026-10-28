# chase_modal

Pursuers chase an evader that flies a fixed path, and only the pursuers learn. The task
measures what a pursuer must sense of the other drones to catch the evader and to keep
clear of its teammates. The input is what a pursuer senses of each other drone, and the
reading is the outcome of the episodes.

Log: [chase_modal_log.md](../experiment_logs/chase_modal_log.md)

Each section lists only what differs from [chase](task_chase.md). `As in chase.` means
that the section of chase holds.

## Setup

- $P$ drones have the role `pursuer`, and one drone has the role `evader`. Each preset
  sets $P$ to 1 or 3.
- The paths pilot flies the evader, in training and in eval.
- The world is the arena box of the chase preset `default`.

## Start

The evader starts on its path, with the velocity of the path. Then the pursuers are drawn
by the rules of chase.

## Drone

Default.

## Observation

Each slot of others also holds the sensed blocks of the preset, after the role flag, in
this order:

| block | content | size |
|---|---|---|
| attitude | $\operatorname{vec}(R_i^\top R_j)$, the attitude of drone $j$ in the body frame of drone $i$ | 9 |
| rpm | $n_{j,k} / n_{\max}$ for $k = 1, \dots, 4$, each motor speed of drone $j$ as a share of the top speed | 4 |

| input | content | size |
|---|---|---|
| actor | the default, with the sensed blocks in each slot | $20 + (8 + s) K$ |
| critic | as in chase | $18N + 4$ per drone of the other role |

- $s$ is the size of the sensed blocks of the preset: 0, 9, 4 or 13.
- Each drone sees every other drone, at any distance. $K = N - 1$, so the evader and
  each teammate fill one slot each.

## Events

As in chase.

## Reward

As in chase.

## Metrics

As in chase.

## Presets

Every preset has the box, the walls and the reward weights of the chase preset `default`.

| preset | $P$ | sensed blocks | pursuer actor input |
|---|---|---|---|
| `1v1` | 1 | none | 28 |
| `1v1_attitude` | 1 | attitude | 37 |
| `1v1_rpm` | 1 | rpm | 32 |
| `1v1_attitude_rpm` | 1 | attitude, rpm | 41 |
| `3v1` | 3 | none | 44 |
| `3v1_attitude` | 3 | attitude | 71 |
| `3v1_rpm` | 3 | rpm | 56 |
| `3v1_attitude_rpm` | 3 | attitude, rpm | 83 |

## Training

- Only the pursuers learn, against the paths evader, in one phase.
- The network of the evader keeps its first weights and never flies.

## Evaluation

As in chase, with the paths pilot in the seat of the evader.
