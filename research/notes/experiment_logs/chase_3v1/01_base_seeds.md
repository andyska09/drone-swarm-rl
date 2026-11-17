# 01_base_seeds

**Question:** How much does the catch (`caught_e`) of the base setup in `paths_3v1` change from one seed to the next?
**Differs from** the defaults in `swarm/learn/config.py`: task `chase`, preset `paths_3v1`, 5e8 steps, `gamma=0.99`, `normalize_obs=false`, `adaptive_lr=false`, `lr=5e-4`, `ent_coef=0.003`. Seeds 0 to 7.
**Runs:** RCI, `tools/rci/sweeps/01_base_seeds.sh`, 8 jobs.
