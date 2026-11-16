- lr ,ent coeff na konci snizovat (to znamena konst a pak linear decay or something)
- nahodny restarty, delka epizody
- koeff pro penalizaci cmd, postupne zvysovat
- penazilace velikosti vah v rewardu primo

## Sweep plan — draft by TARS, 2026-11-15

Base: `paths_3v1`, `gamma=0.99`, `normalize_obs=false`, `adaptive_lr=false`, `lr=5e-4`,
`ent_coef=0.003`, 5e8 steps. Score: eval `caught_e` at the last checkpoint, 4096 episodes.

Order: 1. 8 seeds of the base. 2. The `lr` × `ent_coef` grid. 3. The breadth pass: each
other row alone, from the best cell of the grid.

### Number knobs

The base value is bold. "Tried" means in 3v1.

| knob | tried before | values to try | new code |
|---|---|---|---|
| `lr`, constant | 5e-4 | 2.5e-4, **5e-4**, 1e-3, 2e-3 | no |
| `ent_coef` | 0, 0.001, 0.003 | 0.001, **0.003**, 0.01, 0.03 | no |
| `gamma` | 0.995, with the adaptive lr only | 0.98, **0.99**, 0.995 | no |
| `gae_lambda` | never | 0.9, **0.95**, 0.98 | no |
| `clip_eps` | never | 0.1, **0.2**, 0.3 | no |
| `update_epochs` | never | 2, **4**, 8 | no |
| `num_minibatches` | never | 16, **32**, 64 | no |
| `num_steps` | never | 32, **64**, 128 | no |
| `num_envs` | never | 2048, **4096**, 8192 | no |
| `vf_coef` | never | 0.25, **0.5**, 1.0 | no |
| `max_grad_norm` | never | 0.25, **0.5**, 1.0 | no |
| `init_std` | never | 0.3, 0.5, **1.0** | no |
| `hidden` | never | 128,128 / **256,256** / 512,512 / 256,256,256 | no |
| `activation` | relu, with the adaptive lr only | **tanh**, relu | no |
| `normalize_reward` | never | **true**, false | no |
| `max_steps`, the episode length | never | 500, **1000**, 2000 | `--set-env` |
| `reward.coll_pp` | 30 and 10, with the adaptive lr only | 0, **10** | `--set-env` |

### Other approaches

| approach | from | new code |
|---|---|---|
| lr and `ent_coef` constant, then a linear fall at the end | line 1 above | yes |
| each env starts its first episode at a random time | line 2 above | yes, the trainer has none now |
| a `cmd` cost that grows over the training | line 3 above | yes |
| a cost on the size of the network weights, in the loss | line 4 above | yes, the optimizer has none now |
| a floor for the std, for example 0.3 | the log, Open | yes |
| train in `paths_3v1_netpass`, eval in `paths_3v1` | the log | no |
| `paths_3v1_body`, the body catch | the log | no |
| the best MLP setup with `net=attention` | you | no |