# 03_breadth

**Question:** Which one change to a training setting or a task value gives a higher catch (`caught_e`) in `paths_3v1` than the best cell of `02_lr_ent`?
**Differs from** `02_lr_ent`, cell `lr1.25e-4_ent0.003`: each config changes one value. `gamma` 0.98, 0.995. `gae_lambda` 0.9, 0.98. `clip_eps` 0.1, 0.3. `update_epochs` 2, 8. `num_minibatches` 16, 64. `num_steps` 32, 128. `num_envs` 2048, 8192. `init_std` 0.3, 0.5. `max_grad_norm` 1.0. Task values: `max_steps` 2000, `reward.coll_pp` 0. Seeds 0 and 1. The base does not run again: its row is `lr1.25e-4_ent0.003` in `02_lr_ent`.
**Runs:** RCI, `tools/rci/sweeps/03_breadth.sh`, 38 jobs.

## Results

| config | seeds | caught_e | min | max | coll_pp | death_rate_p |
|---|---|---|---|---|---|---|
| clip_eps0.1 | 2/2 | 0.945 | 0.944 | 0.946 | 0.048 | 0.035 |
| num_steps128 | 2/2 | 0.938 | 0.934 | 0.943 | 0.058 | 0.040 |
| num_envs8192 | 2/2 | 0.928 | 0.918 | 0.938 | 0.068 | 0.047 |
| gamma0.995 | 2/2 | 0.920 | 0.905 | 0.935 | 0.076 | 0.052 |
| update_epochs2 | 2/2 | 0.917 | 0.906 | 0.928 | 0.077 | 0.053 |
| max_steps2000 | 2/2 | 0.886 | 0.880 | 0.892 | 0.107 | 0.074 |
| max_grad_norm1.0 | 2/2 | 0.883 | 0.867 | 0.898 | 0.108 | 0.075 |
| num_minibatches64 | 2/2 | 0.871 | 0.863 | 0.879 | 0.110 | 0.080 |
| gae_lambda0.9 | 2/2 | 0.848 | 0.831 | 0.866 | 0.142 | 0.098 |
| num_minibatches16 | 2/2 | 0.825 | 0.780 | 0.870 | 0.163 | 0.113 |
| clip_eps0.3 | 2/2 | 0.803 | 0.788 | 0.818 | 0.185 | 0.128 |
| gae_lambda0.98 | 2/2 | 0.803 | 0.737 | 0.868 | 0.183 | 0.127 |
| coll_pp0 | 2/2 | 0.640 | 0.559 | 0.721 | 0.346 | 0.236 |
| init_std0.5 | 2/2 | 0.637 | 0.597 | 0.677 | 0.323 | 0.229 |
| init_std0.3 | 2/2 | 0.604 | 0.479 | 0.728 | 0.342 | 0.244 |
| update_epochs8 | 2/2 | 0.593 | 0.542 | 0.643 | 0.390 | 0.265 |
| num_envs2048 | 2/2 | 0.572 | 0.527 | 0.617 | 0.394 | 0.271 |
| gamma0.98 | 2/2 | 0.553 | 0.527 | 0.579 | 0.372 | 0.273 |
| num_steps32 | 2/2 | 0.532 | 0.406 | 0.658 | 0.428 | 0.293 |

- The best config is `clip_eps` 0.1: 0.945, with 0.944 and 0.946 in the two seeds. The base row in `02_lr_ent` is 0.832.
- Five configs catch more than 0.91: `clip_eps` 0.1, `num_steps` 128, `num_envs` 8192, `gamma` 0.995 and `update_epochs` 2.
- Seven configs catch less than 0.65: `reward.coll_pp` 0, `init_std` 0.5 and 0.3, `update_epochs` 8, `num_envs` 2048, `gamma` 0.98 and `num_steps` 32.
