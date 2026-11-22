# 05_frpn_noise

**Question:** How do the catch and the return of FRPN pursuers change with noise on their actions, with one pursuer and with three?
**Differs from** `03_cost30_slot0`: no training. FRPN flies the pursuers against the paths evader. Each action gets Gaussian noise of std σ, and then the clip.
**Runs:** eval only, 2026-11-02, 512 episodes for each σ.

- "Return" is the mean episode return of one pursuer, as eval reports it: no discount and no bootstrap.

## Results

| σ | 1v1 caught | 1v1 return | 3v1 caught | 3v1 `coll_pp` | 3v1 `OOB_p` | 3v1 return |
|---|---|---|---|---|---|---|
| 0 | 0.76 | −16 | 0.49 | 0.32 | 0.17 | −13 |
| 0.3 | 0.71 | −19 | 0.47 | 0.33 | 0.17 | −14 |
| 1 | 0.25 | −41 | 0.19 | 0.35 | 0.29 | −25 |
| 3 | 0.04 | −47 | 0.04 | 0.05 | 0.82 | −23 |
| 10 | 0.00 | −43 | 0.00 | 0.01 | 0.98 | −19 |

- With one pursuer, noise only hurts: the catch falls from 0.76 to 0.25 at σ = 1, and the return from −16 to −41.
- With three pursuers, σ = 1 has the lowest return (−25). The return is −13 at σ = 0 and −19 at σ = 10.
- With more noise the three pursuers stop hitting each other and leave the box: at σ = 10, `coll_pp` is 0.01 and `OOB_p` 0.98. A `coll_pp` kills two pursuers, and an exit kills one.

A new policy starts at σ = 1, the value of `init_std`.
