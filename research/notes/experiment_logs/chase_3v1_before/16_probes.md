# 16_probes

**Question:** What holds the trained pursuers on the ring: the net kill, the teammates in the input, or the payoff of closing in?
**Differs from** `15_progress`: no training. Three probes fly the last checkpoints, with 4096 episodes, the eval seeds and mean actions.
**Runs:** flights only, 2026-11-10, on commit `79e1f0e` with the presets `paths_3v1_body` and `paths_3v1_netpass` added.

## Results

The net kill. Each policy flies in `paths_3v1` and in `paths_3v1_netpass`, where a net that touches a pursuer is no `coll_pp`. The keys are the same, so each episode is the same in both up to its first net touch. "Saved" are the `coll_pp` episodes of `paths_3v1` that end in another way in `paths_3v1_netpass`:

| policy | catch | `coll_pp` | `time_limit` | episodes to a catch / away from it | McNemar p | saved | saved, then catch / `time_limit` / `crash_p` / `OOB_p` | still `coll_pp` |
|---|---|---|---|---|---|---|---|---|
| `c6de8e` | 0.213 → 0.248 | 0.538 → 0.385 | 0.206 → 0.321 | 140 / 0 | 2.7e-32 | 627 | 0.223 / 0.753 / 0.019 / 0.005 | 1575 |
| `398c2a` | 0.252 → 0.280 | 0.326 → 0.213 | 0.357 → 0.441 | 113 / 0 | 2.2e-26 | 464 | 0.244 / 0.737 / 0.006 / 0.013 | 873 |
| FRPN | 0.524 → 0.562 | 0.322 → 0.276 | 0.000 → 0.001 | 154 / 0 | 2.3e-35 | 191 | 0.806 / 0.010 / 0.031 / 0.152 | 1129 |

- Without the net kill the same policies catch more, by 0.034 and 0.028 (140 and 113 episodes), and FRPN by 0.038. Every changed episode moves toward a catch.
- Most saved episodes of the learned policies run to the time limit: 0.753 and 0.737 of them, against 0.010 for FRPN.

Ghost teammates. In the actor input of each pursuer, the offsets of the two teammate slots move to 15 m, along the same direction. "All" changes the input of all three pursuers, and "nearest" only of the pursuer nearest to the evader. The real teammates still fly:

| policy | input | catch | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | nets at 4 s | closing at 1–2 / 2–3 / 3–5 m | closest net, no catch |
|---|---|---|---|---|---|---|---|---|---|
| `c6de8e` | real | 0.212 | 0.537 | 0.031 | 0.013 | 0.207 | 3.3 / 4.9 / 6.7 m | 0.34 / 0.28 / 0.42 m/s | 1.44 m |
| `c6de8e` | all | 0.188 | 0.533 | 0.100 | 0.030 | 0.150 | 3.5 / 5.2 / 7.1 m | 0.38 / 0.40 / 0.56 m/s | 1.50 m |
| `c6de8e` | nearest | 0.197 | 0.552 | 0.045 | 0.017 | 0.190 | 3.5 / 5.2 / 7.0 m | 0.39 / 0.34 / 0.50 m/s | 1.49 m |
| `398c2a` | real | 0.253 | 0.324 | 0.030 | 0.034 | 0.358 | 3.0 / 4.6 / 6.2 m | 0.26 / 0.22 / 0.32 m/s | 1.24 m |
| `398c2a` | all | 0.197 | 0.311 | 0.129 | 0.082 | 0.282 | 3.4 / 4.9 / 6.7 m | 0.34 / 0.30 / 0.45 m/s | 1.47 m |
| `398c2a` | nearest | 0.240 | 0.342 | 0.057 | 0.047 | 0.314 | 3.1 / 4.7 / 6.3 m | 0.37 / 0.37 / 0.47 m/s | 1.29 m |

- The ring stays: the nearest net at 4 s is 3.1–3.5 m away, against 3.0 and 3.3 m with the real input.
- The catch is 0.188–0.240, against 0.212 and 0.253. With "all", `crash_p` rises to 0.100 and 0.129.

The payoff of closing in, from the 4096-episode records of `15_progress`. A sample is a pursuer at every 10th step up to 6 s, alive, with its net 2–5 m from the evader. G is its own discounted return from that step, at γ = 0.99, with the reward of its run. The columns are its closing speed on the evader, in m/s:

| run | < 0 | 0–0.5 | 0.5–1 | 1–2 | 2–4 | > 4 |
|---|---|---|---|---|---|---|
| `c6de8e` | 0.04 | 0.31 | 0.36 | 0.47 | 0.67 | 1.34 |
| `398c2a` | 0.37 | 0.63 | 0.67 | 0.69 | 1.01 | 1.71 |
| `a573b5` | −4.19 | −4.13 | −4.11 | −4.07 | −3.95 | −3.65 |
| `052a62` | −4.30 | −4.31 | −4.24 | −4.18 | −4.20 | −3.95 |
| `2d0b98` | −4.47 | −4.42 | −4.40 | −4.34 | −4.25 | −3.94 |
| `6fce56` | −4.09 | −4.10 | −4.12 | −4.07 | −3.94 | −3.54 |

This pursuer catches within 1 s:

| run | < 0 | 0–0.5 | 0.5–1 | 1–2 | 2–4 | > 4 |
|---|---|---|---|---|---|---|
| `c6de8e` | 0.004 | 0.009 | 0.011 | 0.017 | 0.027 | 0.056 |
| `398c2a` | 0.003 | 0.006 | 0.008 | 0.015 | 0.031 | 0.064 |
| `a573b5` | 0.004 | 0.007 | 0.011 | 0.017 | 0.035 | 0.063 |
| `052a62` | 0.002 | 0.005 | 0.009 | 0.014 | 0.024 | 0.050 |
| `2d0b98` | 0.003 | 0.006 | 0.008 | 0.012 | 0.022 | 0.056 |
| `6fce56` | 0.003 | 0.006 | 0.008 | 0.014 | 0.029 | 0.072 |

This pursuer is in a `coll_pp` within 1 s:

| run | < 0 | 0–0.5 | 0.5–1 | 1–2 | 2–4 | > 4 |
|---|---|---|---|---|---|---|
| `c6de8e` | 0.067 | 0.069 | 0.072 | 0.076 | 0.081 | 0.082 |
| `398c2a` | 0.039 | 0.036 | 0.036 | 0.036 | 0.037 | 0.034 |
| `a573b5` | 0.016 | 0.015 | 0.015 | 0.017 | 0.018 | 0.019 |
| `052a62` | 0.050 | 0.055 | 0.057 | 0.056 | 0.062 | 0.068 |
| `2d0b98` | 0.006 | 0.007 | 0.005 | 0.006 | 0.006 | 0.007 |
| `6fce56` | 0.004 | 0.005 | 0.005 | 0.005 | 0.006 | 0.005 |

- From the ring, a pursuer that closes at more than 4 m/s has a higher G than one that moves away, in all six runs, by 0.35–1.34.
- Its own catch within 1 s is 14–25 times higher at more than 4 m/s than below 0. Its own `coll_pp` within 1 s differs by −0.005 to +0.018.
- These tables compare states inside the flights of each policy, with no change of the action.
