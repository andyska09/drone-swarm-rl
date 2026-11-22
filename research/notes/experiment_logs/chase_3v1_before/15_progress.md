# 15_progress

**Question:** Do three pursuers catch more when each one earns for the metres that its net closes on the evader?
**Differs from** `14_cost10`: the progress reward. Each pursuer earns 0.25 for each metre that its own net closes on the evader, in place of the clock and the distance cost. A catch pays 30, not 10. This is the reward of today.
**Runs:** MetaCentrum, 2 runs, 2026-11-10, commit `18a4c40`: `c6de8e` (seed 0), `398c2a` (seed 1).

- Why: with the clock, the rest of an episode was worth about −5 to a pursuer that lives on, so a death also ended its costs.

## Results

Eval of the last checkpoint, 1024 episodes. "Training" is the mean of the last 100 updates.

| run | reward | seed | `caught_e` eval | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | time to catch | `distance` | `caught_e` training | std | explained variance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `a573b5` | cost 10 | 0 | 0.250 | 0.176 | 0.017 | 0.034 | 0.52 | 6.1 s | 5.0 m | 0.238 | 0.02–0.03 | 0.68 |
| `c6de8e` | progress | 0 | 0.194 | 0.561 | 0.024 | 0.014 | 0.21 | 4.6 s | 5.3 m | 0.202 | 0.018–0.022 | 0.55 |
| `052a62` | cost 10 | 1 | 0.166 | 0.468 | 0.011 | 0.012 | 0.34 | 5.5 s | 5.6 m | 0.180 | 0.03–0.04 | 0.58 |
| `398c2a` | progress | 1 | 0.255 | 0.325 | 0.026 | 0.037 | 0.36 | 5.4 s | 4.8 m | 0.258 | 0.016–0.019 | 0.47 |

Training over the updates, the mean of each range:

| updates | 1–100 | 101–200 | 201–500 | 501–1000 | 1001–1907 |
|---|---|---|---|---|---|
| std, `c6de8e` | 0.92 | 0.75 | 0.36 | 0.086 | 0.030 |
| std, `398c2a` | 0.92 | 0.71 | 0.33 | 0.080 | 0.027 |
| `caught_e`, `c6de8e` | 0.09 | 0.16 | 0.20 | 0.21 | 0.21 |
| `caught_e`, `398c2a` | 0.09 | 0.18 | 0.22 | 0.25 | 0.26 |
| `coll_pp`, `c6de8e` | 0.30 | 0.40 | 0.39 | 0.51 | 0.53 |
| `coll_pp`, `398c2a` | 0.34 | 0.53 | 0.34 | 0.36 | 0.34 |

- The std does not blow up: it falls from 1.0 to about 0.02.
- The training catch is flat over updates 1001–1907.

The tables below come from 4096 eval episodes of each run, with mean actions and every step saved. The cost-10 runs are those of `14_cost10`, and the cost-30 runs those of `09_critic_input`. FRPN flew the pursuers in the same episodes.

The outcomes, with the 95 % Wilson interval of the catch:

| run | reward | `caught_e` | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | median time to catch |
|---|---|---|---|---|---|---|---|
| `c6de8e` | progress | 0.212 [0.200, 0.225] | 0.537 | 0.031 | 0.013 | 0.207 | 4.2 s |
| `398c2a` | progress | 0.253 [0.240, 0.267] | 0.324 | 0.030 | 0.034 | 0.358 | 5.2 s |
| `a573b5` | cost 10 | 0.245 [0.232, 0.258] | 0.155 | 0.013 | 0.031 | 0.556 | 6.5 s |
| `052a62` | cost 10 | 0.177 [0.166, 0.189] | 0.459 | 0.010 | 0.010 | 0.345 | 5.5 s |
| `2d0b98` | cost 30 | 0.175 [0.164, 0.187] | 0.093 | 0.009 | 0.010 | 0.712 | 6.4 s |
| `6fce56` | cost 30 | 0.243 [0.230, 0.256] | 0.044 | 0.013 | 0.005 | 0.695 | 6.0 s |
| FRPN | — | 0.524 [0.509, 0.539] | 0.322 | 0.012 | 0.142 | 0.000 | 1.5 s |

- The catch does not change with the reward: 0.212 and 0.253, against 0.175–0.245 in the four older runs. A one-way ANOVA over the three rewards gives p = 0.83.
- The catch comes sooner: a median of 4.2 and 5.2 s, against 5.5–6.5 s.
- `coll_pp` is 0.32–0.54, against 0.16–0.46 at a cost of 10 and 0.04–0.09 at 30.

The catch by the path of the evader. Each path has 1344–1386 episodes:

| path | `c6de8e` | `398c2a` | `a573b5` | `052a62` | `2d0b98` | `6fce56` | FRPN |
|---|---|---|---|---|---|---|---|
| hover | 0.210 | 0.260 | 0.224 | 0.170 | 0.156 | 0.219 | 0.603 |
| circle | 0.211 | 0.225 | 0.228 | 0.184 | 0.174 | 0.225 | 0.559 |
| figure-eight | 0.216 | 0.275 | 0.282 | 0.177 | 0.195 | 0.284 | 0.413 |

- The learned catch does not depend on the path. FRPN catches 0.603 of the hovers and 0.413 of the figure-eights.

The distance from the evader to each net, nearest / second / third, the median over the episodes still running (m):

| time | 1 s | 2 s | 4 s | 8 s |
|---|---|---|---|---|
| `c6de8e` | 6.5 / 10.1 / 14.1 | 3.4 / 5.1 / 7.1 | 3.3 / 4.9 / 6.7 | 3.4 / 4.9 / 6.6 |
| `398c2a` | 7.0 / 10.6 / 14.7 | 4.0 / 6.0 / 8.0 | 3.0 / 4.6 / 6.2 | 2.8 / 4.3 / 5.7 |
| the four older runs | 7.5–7.8 / 11.5–11.9 / 15.9–16.4 | 4.9–5.3 / 7.1–8.5 / 9.4–11.7 | 3.3–4.1 / 5.2–6.4 / 7.0–8.6 | 2.8–3.7 / 4.5–5.8 / 6.2–7.6 |
| FRPN | 4.7 / 9.3 / 14.6 | 2.2 / 4.3 / 7.0 | — | — |

The median closing speed of a net on the evader, by its distance (m/s):

| distance | 0–1 m | 1–2 m | 2–3 m | 3–5 m | 5–8 m | 8–12 m | 12–40 m |
|---|---|---|---|---|---|---|---|
| `c6de8e` | 0.88 | 0.34 | 0.28 | 0.42 | 0.85 | 2.77 | 5.96 |
| `398c2a` | 0.79 | 0.26 | 0.22 | 0.32 | 0.75 | 2.82 | 5.61 |
| the four older runs | 0.56–0.89 | 0.17–0.27 | 0.15–0.23 | 0.21–0.33 | 0.43–0.65 | 1.20–1.89 | 3.29–4.54 |
| FRPN | 4.15 | 4.79 | 5.82 | 6.63 | 7.70 | 8.19 | 6.21 |

- All six policies hold their nets on a ring around the evader: at 4 s and 8 s the nets are 2.8–4.1, 4.3–6.4 and 5.7–8.6 m away.
- At 1–5 m all six close at 0.15–0.42 m/s. FRPN closes at 4.79–6.63 m/s there.
- Far from the evader the two new runs close faster: 5.61–5.96 m/s beyond 12 m, against 3.29–4.54.

The motion of a pursuer relative to the evader while its net is 2–5 m away. Medians, from every 10th step:

| run | toward the evader | around it | turn rate around it | around it, hover path | pursuer speed, median / top 10 % |
|---|---|---|---|---|---|
| `c6de8e` | 0.45 m/s | 3.72 m/s | 1.08 rad/s | 3.45 m/s | 5.43 / 9.98 m/s |
| `398c2a` | 0.34 | 2.97 | 0.86 | 2.59 | 4.32 / 8.05 |
| `a573b5` | 0.37 | 2.61 | 0.73 | 2.27 | 3.82 / 6.64 |
| `052a62` | 0.34 | 3.75 | 1.08 | 3.40 | 5.20 / 8.70 |
| `2d0b98` | 0.26 | 3.24 | 0.87 | 2.84 | 4.34 / 7.12 |
| `6fce56` | 0.29 | 2.07 | 0.62 | 1.67 | 3.42 / 6.20 |
| FRPN | 6.60 | 2.78 | 0.90 | 2.01 | 9.34 / 12.70 |

- A learned pursuer moves at 2.07–3.75 m/s around the evader and at 0.26–0.45 m/s toward it. FRPN moves at 6.60 m/s toward it.

The `coll_pp`, at the step of the hit. Rank 1 is the pursuer whose net is nearest to the evader. "Same way" is the share of pairs that turned around the evader in the same direction 0.5 s before the hit, and "base" is that share for any two pursuers with their nets 2–5 m away:

| run | hits | body on body | ranks 1+2 / 2+3 / 1+3 | nearest net to the evader, median | under 2 m | pair within 2 m before, median | pair speed, median | same way | base |
|---|---|---|---|---|---|---|---|---|---|
| `c6de8e` | 2200 | 0.47 | 0.57 / 0.35 / 0.08 | 3.60 m | 0.14 | 0.21 s | 5.22 m/s | 0.62 | 0.57 |
| `398c2a` | 1328 | 0.44 | 0.52 / 0.40 / 0.08 | 3.26 m | 0.17 | 0.27 s | 4.17 m/s | 0.65 | 0.57 |
| `a573b5` | 633 | 0.39 | 0.57 / 0.37 / 0.06 | 3.22 m | 0.20 | 0.29 s | 3.99 m/s | 0.68 | 0.53 |
| `052a62` | 1879 | 0.53 | 0.58 / 0.36 / 0.06 | 3.90 m | 0.10 | 0.17 s | 6.07 m/s | 0.60 | 0.53 |
| `2d0b98` | 382 | 0.30 | 0.50 / 0.44 / 0.06 | 4.05 m | 0.08 | 0.20 s | 4.57 m/s | 0.54 | 0.49 |
| `6fce56` | 179 | 0.30 | 0.64 / 0.32 / 0.04 | 3.09 m | 0.22 | 0.38 s | 3.40 m/s | 0.72 | 0.54 |
| FRPN | 1320 | 0.66 | 0.85 / 0.11 / 0.04 | 1.32 m | 0.74 | 0.12 s | 11.92 m/s | 0.62 | 0.61 |

- The learned hits come on the ring: the nearest net is a median 3.09–4.05 m from the evader. For FRPN it is 1.32 m.
- In 0.30–0.53 of the learned hits, the two bodies touch. In the others a net touches a pursuer.

The episodes with no catch. "Closest" is the median of the closest net of each episode. The off-net columns are shares of the episodes with a net within 1 m, at its closest step:

| run | closest | a net within 1 m | closing at the closest step | off across / sideways / up-down | height q10 / q50 / q90 | a body touch |
|---|---|---|---|---|---|---|
| `c6de8e` | 1.44 m | 0.25 | 0.03 m/s | 0.52 / 0.30 / 0.45 | −1.36 / 0.19 / 1.89 m | 0.187 |
| `398c2a` | 1.24 m | 0.33 | 0.15 m/s | 0.43 / 0.35 / 0.48 | −1.13 / 0.37 / 1.99 m | 0.247 |
| `a573b5` | 1.31 m | 0.30 | 0.09 m/s | 0.45 / 0.37 / 0.47 | −1.03 / 0.49 / 2.01 m | 0.160 |
| `052a62` | 1.47 m | 0.25 | −0.02 m/s | 0.54 / 0.23 / 0.47 | −1.17 / 0.33 / 2.01 m | 0.200 |
| `2d0b98` | 1.49 m | 0.23 | 0.00 m/s | 0.44 / 0.33 / 0.49 | −1.29 / 0.35 / 2.14 m | 0.149 |
| `6fce56` | 1.42 m | 0.26 | 0.05 m/s | 0.41 / 0.36 / 0.48 | −1.28 / 0.12 / 1.78 m | 0.232 |
| FRPN | 0.90 m | 0.62 | 1.21 m/s | 0.37 / 0.30 / 0.55 | −0.21 / 0.60 / 1.39 m | 0.592 |

- In 0.23–0.33 of the learned episodes with no catch, a net comes within 1 m. At its closest step it closes at −0.02 to 0.15 m/s.
