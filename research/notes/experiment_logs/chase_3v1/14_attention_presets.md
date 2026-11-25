# 14_attention_presets

**Question:** Does the attention net lift the catch (`caught_e`) of three pursuers above the MLP, in the presets where three pursuers catch less than one?
**Differs from** the MLP rows with `ent_coef` 0.001: `net=attention`. `apf_3v1` of `11_ent_easy`, `ent_coef0.001` of `09_apf_x8_free_ent`, and `apf_3v1_weak` and `apf_3v1_weak_v4_a7` of `12_weak_pursuer`, one config each, with 5e8 steps. A fifth config is `apf_3v1_weak_v4_a7` with `net=attention` and 2e9 steps. Seeds 0 and 1. The MLP rows do not run again.
**Runs:** RCI, `tools/rci/sweeps/14_attention_presets.sh`, 10 jobs.

- `05_attention` ran the attention net in `paths_3v1` with `ent_coef` 0.003. No run before this one tested it with 0.001, or against another evader.
- The rows that the attention net must beat:

| config | steps | MLP `caught_e` | MLP `coll_pp` | one pursuer `caught_e` | the MLP row is in |
|---|---|---|---|---|---|
| `apf_3v1` | 5e8 | 0.953 | 0.045 | 1.000 | `11_ent_easy` |
| `apf_x8_free_3v1` | 5e8 | 0.706 | 0.136 | 0.977 | `09_apf_x8_free_ent` |
| `apf_3v1_weak` | 5e8 | 0.748 | 0.144 | 0.999 | `12_weak_pursuer` |
| `apf_3v1_weak_v4_a7` | 5e8 | 0.148 | 0.173 | 0.702 | `12_weak_pursuer` |

- `apf_3v1_weak_v4_a7` runs with 5e8 steps and with 2e9 steps. The MLP has no row with 2e9 steps for it. So the row with 2e9 steps compares to the attention row with 5e8 steps.
- Beside the catch, the result is `coll_pp`. In `05_attention`, the attention net has a `coll_pp` of 0.009, and the MLP of `04_new_base` has 0.022.
- An attention job with 5e8 steps took 46 minutes in `05_attention`. So a job with 5e8 steps has a time limit of 2 hours, and a job with 2e9 steps has 3:59:00.
