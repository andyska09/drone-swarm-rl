# 01_pool

**Question:** Do three pursuers learn to catch the evader in a pool run?
**Differs from** the code of today: the reward is the one before the progress reward. Each pursuer pays a clock of 4 per s and a distance cost of 0.1 per m per s, a catch pays 10, and a `coll_pp` costs 10. The critic has one input for all three pursuers. The other drones are in the actor input nearest first, so the evader has no fixed slot. Settings: preset `3v1`, `gamma=0.99`, `generations=13`, `ent_coef=0.001`, 1.4e9 steps, seed 0.
**Runs:** 1 run, 2026-10-27, commit `f6f5ef5`, `chase_3v1_s0_20261027-211843_000913`.

- In generation 0 the three pursuers learn against the paths evader only, for 50M steps.

## Results

The ends of the training episodes of generation 0, at its last update:

| `caught_e` | `crash_p` | `OOB_p` | `coll_pp` | `time_limit` |
|---|---|---|---|---|
| 0.03 | 0.59 | 0.23 | 0.11 | 0.05 |

- A pursuer dies in 0.93 of the episodes: 0.59 by a crash, 0.23 out of the box and 0.11 by a `coll_pp`.

Other numbers of the run, and FRPN in the same task:

| what | value |
|---|---|
| the generation-0 pursuers against the paths evader, mean actions, 600 episodes | win 0.01 |
| the best pursuers of any generation against the paths evader | win 0.07 |
| `coll_pp` in the pursuers' training episodes, in the later generations | up to 0.64 |
| three FRPN pursuers against the paths evader, 256 episodes | catch 0.44, `coll_pp` 0.36 |
| one FRPN pursuer against the paths evader | catch 0.75 |

- No pursuers of the run win more than 0.07 against the paths evader. Three FRPN pursuers catch 0.44.
