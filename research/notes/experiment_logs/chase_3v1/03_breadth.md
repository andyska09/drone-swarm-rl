# 03_breadth

**Question:** Which one change to a training setting or a task value gives a higher catch (`caught_e`) in `paths_3v1` than the best cell of `02_lr_ent`?
**Differs from** `02_lr_ent`, cell `lr1.25e-4_ent0.003`: each config changes one value. `gamma` 0.98, 0.995. `gae_lambda` 0.9, 0.98. `clip_eps` 0.1, 0.3. `update_epochs` 2, 8. `num_minibatches` 16, 64. `num_steps` 32, 128. `num_envs` 2048, 8192. `init_std` 0.3, 0.5. `max_grad_norm` 1.0. Task values: `max_steps` 2000, `reward.coll_pp` 0. Seeds 0 and 1. The base does not run again: its row is `lr1.25e-4_ent0.003` in `02_lr_ent`.
**Runs:** RCI, `tools/rci/sweeps/03_breadth.sh`, 38 jobs.
