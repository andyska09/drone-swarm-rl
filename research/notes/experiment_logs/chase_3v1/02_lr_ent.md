# 02_lr_ent

**Question:** Which `lr` and `ent_coef` give the highest catch (`caught_e`) in `paths_3v1`?
**Differs from** `01_base_seeds`: `lr` 1.25e-4, 2.5e-4, 5e-4, 1e-3 × `ent_coef` 0.001, 0.003, 0.01, 0.03. Seeds 0 and 1.
**Runs:** RCI, `tools/rci/sweeps/02_lr_ent.sh`, 32 jobs.
