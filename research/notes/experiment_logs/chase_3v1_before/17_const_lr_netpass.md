# 17_const_lr_netpass

**Question:** Do three pursuers catch more with a constant lr of 5e-4, and when they train with no net kill?
**Differs from** `15_progress`, run `c6de8e`: a constant lr (`--set adaptive_lr=false --set lr=5e-4`), the preset `paths_3v1_netpass`, or both. Seed 0.
**Runs:** MetaCentrum, 3 runs, 2026-11-10, commit `fba6aec`. No run of `paths_3v1_body` was submitted.

- Why a constant lr: Gavin & Bronz 2026 train with a constant lr of 5e-4 (Table II). The adaptive lr fell to 3.9–4.4e-5 in `15_progress`.

| run | preset | lr |
|---|---|---|
| `c6de8e` (before) | `paths_3v1` | adaptive |
| `be848f` | `paths_3v1` | 5e-4, constant (`--set adaptive_lr=false --set lr=5e-4`) |
| `4f3c3b` | `paths_3v1_netpass` | adaptive |
| `d11920` | `paths_3v1_netpass` | 5e-4, constant |

## Results

Eval on the preset of the run, 1024 episodes, mean actions, the last checkpoint unless a table says otherwise. The tables with FRPN come from 4096 episodes, as in `15_progress`.

The training catch and the std, the mean of each 200 updates:

| updates | 1–200 | 201–400 | 401–600 | 601–800 | 801–1000 | 1001–1200 | 1201–1400 | 1401–1600 | 1601–1800 | 1801–1907 |
|---|---|---|---|---|---|---|---|---|---|---|
| catch, `c6de8e` | 0.13 | 0.19 | 0.21 | 0.22 | 0.21 | 0.21 | 0.21 | 0.20 | 0.21 | 0.20 |
| catch, `be848f` | 0.24 | 0.47 | 0.47 | 0.45 | 0.47 | 0.55 | 0.58 | 0.56 | 0.51 | 0.50 |
| catch, `4f3c3b` | 0.15 | 0.29 | 0.35 | 0.35 | 0.36 | 0.36 | 0.37 | 0.37 | 0.37 | 0.37 |
| catch, `d11920` | 0.30 | 0.57 | 0.61 | 0.62 | 0.62 | 0.60 | 0.57 | 0.57 | 0.56 | 0.53 |
| std, `c6de8e` | 0.835 | 0.434 | 0.174 | 0.089 | 0.055 | 0.041 | 0.033 | 0.027 | 0.023 | 0.020 |
| std, `be848f` | 0.669 | 0.256 | 0.136 | 0.094 | 0.078 | 0.066 | 0.059 | 0.054 | 0.051 | 0.050 |
| std, `4f3c3b` | 0.816 | 0.370 | 0.154 | 0.081 | 0.050 | 0.035 | 0.027 | 0.022 | 0.019 | 0.017 |
| std, `d11920` | 0.677 | 0.276 | 0.143 | 0.093 | 0.073 | 0.063 | 0.056 | 0.052 | 0.051 | 0.050 |

Over updates 1001–1907, the mean of:

| run | `lr_p` | `approx_kl_p` | `clipfrac_p` | explained variance |
|---|---|---|---|---|
| `c6de8e` | 7.0e-5 | 0.020 | 0.26 | 0.54 |
| `be848f` | 5e-4 | 0.022 | 0.28 | 0.77 |
| `4f3c3b` | 7.1e-5 | 0.021 | 0.26 | 0.59 |
| `d11920` | 5e-4 | 0.023 | 0.29 | 0.75 |

- With the constant lr the std stays at about 0.05. With the adaptive lr it falls to 0.017–0.020.

Eval at four checkpoints, 1024 episodes:

| checkpoint | 400 | 900 | 1300 | 1907 |
|---|---|---|---|---|
| `be848f` catch | 0.512 | 0.492 | 0.562 | 0.487 |
| `be848f` `coll_pp` | 0.283 | 0.397 | 0.292 | 0.280 |
| `d11920` catch | 0.592 | 0.597 | 0.573 | 0.499 |
| `d11920` `coll_pp` | 0.334 | 0.349 | 0.369 | 0.387 |

- The best checkpoint catches more than the last: 0.562 at update 1300 against 0.487 (`be848f`), and 0.597 at update 900 against 0.499 (`d11920`).

The outcomes, 4096 episodes, with the 95 % Wilson interval of the catch:

| run | catch | `coll_pp` | `crash_p` | `OOB_p` | `time_limit` | median time to catch |
|---|---|---|---|---|---|---|
| `c6de8e` | 0.212 [0.200, 0.225] | 0.537 | 0.031 | 0.013 | 0.207 | 4.2 s |
| `be848f` | 0.498 [0.482, 0.513] | 0.278 | 0.062 | 0.093 | 0.069 | 4.1 s |
| `4f3c3b` | 0.388 [0.373, 0.403] | 0.348 | 0.013 | 0.052 | 0.199 | 5.0 s |
| `d11920` | 0.509 [0.493, 0.524] | 0.392 | 0.022 | 0.064 | 0.013 | 3.2 s |
| FRPN | 0.524 [0.509, 0.539] | 0.322 | 0.012 | 0.142 | 0.000 | 1.5 s |

- The constant lr more than doubles the catch in `paths_3v1`: 0.498, against 0.212.
- With the adaptive lr, netpass catches 0.388, against 0.212. With the constant lr it changes the catch little: 0.509, against 0.498.
- Flown in `paths_3v1`, 1024 episodes, the netpass runs catch 0.335 (`4f3c3b`, `coll_pp` 0.481) and 0.396 (`d11920`, `coll_pp` 0.513).

The catch by the path of the evader:

| path | `c6de8e` | `be848f` | `4f3c3b` | `d11920` | FRPN |
|---|---|---|---|---|---|
| hover | 0.210 | 0.511 | 0.379 | 0.528 | 0.603 |
| circle | 0.211 | 0.501 | 0.364 | 0.506 | 0.559 |
| figure-eight | 0.216 | 0.481 | 0.421 | 0.493 | 0.413 |

The behaviour:

| | `c6de8e` | `be848f` | `4f3c3b` | `d11920` | FRPN |
|---|---|---|---|---|---|
| nets at 4 s, median (m) | 3.3 / 4.9 / 6.7 | 2.1 / 3.6 / 5.6 | 2.7 / 4.0 / 5.5 | 1.9 / 3.0 / 4.4 | 2.4 / 3.9 / 6.0 |
| nets at 8 s, median (m) | 3.4 / 4.9 / 6.6 | 1.9 / 3.5 / 5.7 | 2.5 / 3.8 / 5.1 | 1.8 / 2.7 / 4.0 | — |
| closing at 1–2 / 2–3 / 3–5 m (m/s) | 0.34 / 0.28 / 0.42 | 0.44 / 0.45 / 0.69 | 0.38 / 0.40 / 0.60 | 0.46 / 0.59 / 1.14 | 4.79 / 5.82 / 6.63 |
| closing at 5–8 / 8–12 / 12–40 m (m/s) | 0.85 / 2.77 / 5.96 | 1.40 / 4.20 / 5.66 | 1.26 / 4.00 / 5.82 | 3.67 / 7.69 / 7.04 | 7.70 / 8.19 / 6.21 |
| net 2–5 m away: toward / around the evader (m/s) | 0.45 / 3.72 | 0.71 / 2.81 | 0.54 / 3.47 | 0.94 / 3.53 | 6.60 / 2.78 |
| pursuer speed, median / top 10 % (m/s) | 5.43 / 9.98 | 5.26 / 10.70 | 5.32 / 8.96 | 6.52 / 12.69 | 9.34 / 12.70 |
| height over the evader, net within 3 m, q10 / q50 / q90 (m) | −1.36 / 0.19 / 1.89 | −0.67 / 0.60 / 1.69 | −0.81 / 0.77 / 2.25 | −0.66 / 0.57 / 1.66 | −0.21 / 0.60 / 1.39 |
| no catch: closest net, median; share within 1 m | 1.44 m; 0.25 | 1.12 m; 0.41 | 1.06 m; 0.44 | 1.08 m; 0.44 | 0.90 m; 0.62 |
| `coll_pp`: nearest net to the evader, median; share under 2 m | 3.60 m; 0.14 | 2.10 m; 0.46 | 2.70 m; 0.26 | 1.90 m; 0.55 | 1.32 m; 0.74 |
| `coll_pp`: ranks 1+2 / 2+3 / 1+3 | 0.57 / 0.35 / 0.08 | 0.68 / 0.26 / 0.06 | 0.52 / 0.36 / 0.12 | 0.63 / 0.28 / 0.10 | 0.85 / 0.11 / 0.04 |
| angle of the two nearest at the evader, median | 74° | 89° | 75° | 80° | 78° |

- The ring moves in: at 4 s the nearest net is 2.1 m (`be848f`) and 1.9 m (`d11920`) from the evader, against 3.3 m.
- The hits move toward the evader: at a `coll_pp` the nearest net is a median 1.90–2.10 m away, against 3.60 m.

The payoff of closing in from the ring, as in `16_probes`:

| run | G, < 0 | 0–0.5 | 0.5–1 | 1–2 | 2–4 | > 4 m/s | own catch within 1 s, < 0 / > 4 | own `coll_pp` within 1 s, < 0 / > 4 |
|---|---|---|---|---|---|---|---|---|
| `be848f` | 2.21 | 2.92 | 3.01 | 3.02 | 3.10 | 3.05 | 0.011 / 0.089 | 0.045 / 0.067 |
| `4f3c3b` | 0.89 | 1.17 | 1.13 | 1.33 | 1.57 | 2.22 | 0.007 / 0.069 | 0.043 / 0.045 |
| `d11920` | 3.03 | 3.71 | 3.75 | 3.76 | 3.78 | 3.67 | 0.019 / 0.091 | 0.073 / 0.112 |

- With the constant lr, G no longer rises above 0.5 m/s of closing: 2.92–3.10 and 3.67–3.78.

The episodes with no catch, last checkpoint, 4096 episodes. A death is a `coll_pp`, an OOB or a crash:

| run | no catch | `coll_pp` | `OOB_p` | `crash_p` | `time_limit` | deaths | `coll_pp` share of the deaths |
|---|---|---|---|---|---|---|---|
| `c6de8e` | 0.788 | 0.537 | 0.013 | 0.031 | 0.207 | 0.581 | 0.92 |
| `be848f` | 0.502 | 0.278 | 0.093 | 0.062 | 0.069 | 0.433 | 0.64 |
| `4f3c3b` | 0.612 | 0.348 | 0.052 | 0.013 | 0.199 | 0.413 | 0.84 |
| `d11920` | 0.491 | 0.392 | 0.064 | 0.022 | 0.013 | 0.478 | 0.82 |

- `coll_pp` is the largest failure in all four runs: 0.64–0.92 of the deaths.

The std, the KL and the clip fraction over training, the mean of each range:

| updates | 1–100 | 101–200 | 201–500 | 501–1000 | 1001–1907 |
|---|---|---|---|---|---|
| std, `be848f` | 0.840 | 0.499 | 0.221 | 0.093 | 0.057 |
| std, `d11920` | 0.838 | 0.516 | 0.238 | 0.091 | 0.055 |
| `approx_kl_p`, `be848f` | 0.0065 | 0.0088 | 0.0132 | 0.0187 | 0.0224 |
| `approx_kl_p`, `d11920` | 0.0064 | 0.0082 | 0.0120 | 0.0189 | 0.0232 |
| `approx_kl_p`, `c6de8e` | 0.0143 | 0.0174 | 0.0179 | 0.0197 | 0.0201 |
| `clipfrac_p`, `be848f` | 0.080 | 0.119 | 0.183 | 0.248 | 0.283 |
| `clipfrac_p`, `d11920` | 0.079 | 0.109 | 0.166 | 0.249 | 0.290 |
| `clipfrac_p`, `c6de8e` | 0.197 | 0.232 | 0.235 | 0.253 | 0.256 |

- With the constant lr, the KL rises from 0.006 to 0.022–0.023 over the run, and the clip fraction from 0.08 to 0.28–0.29. With the adaptive lr the KL stays at 0.014–0.020.
