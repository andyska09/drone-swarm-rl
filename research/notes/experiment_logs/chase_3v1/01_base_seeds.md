# 01_base_seeds

**Question:** How much does the catch (`caught_e`) of the base setup in `paths_3v1` change from one seed to the next?
**Differs from** the defaults in `swarm/learn/config.py`: task `chase`, preset `paths_3v1`, 5e8 steps, `gamma=0.99`, `normalize_obs=false`, `adaptive_lr=false`, `lr=5e-4`, `ent_coef=0.003`. Seeds 0 to 7.
**Runs:** RCI, `tools/rci/sweeps/01_base_seeds.sh`, 8 jobs.

## Results

| config | seeds | caught_e | min | max | coll_pp | death_rate_p |
|---|---|---|---|---|---|---|
| base | 8/8 | 0.753 | 0.647 | 0.884 | 0.198 | 0.147 |

- The mean catch of the 8 seeds is 0.753.
- The lowest seed catches 0.647 and the highest 0.884, a gap of 0.237.

Each seed, from the same eval of the last checkpoint:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `caught_e` | 0.763 | 0.647 | 0.732 | 0.884 | 0.763 | 0.811 | 0.718 | 0.705 |
| `coll_pp` | 0.175 | 0.266 | 0.198 | 0.087 | 0.201 | 0.164 | 0.245 | 0.245 |
| `death_rate_p` | 0.135 | 0.204 | 0.152 | 0.067 | 0.144 | 0.117 | 0.176 | 0.177 |
