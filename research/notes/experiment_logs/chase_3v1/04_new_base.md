# 04_new_base

**Question:** What catch (`caught_e`) does `paths_3v1` give when `gamma` 0.995, `clip_eps` 0.1 and `num_steps` 128 run together?
**Differs from** the base of `03_breadth`: `gamma` 0.995, `clip_eps` 0.1 and `num_steps` 128, all three in one config. Seeds 0 and 1.
**Runs:** RCI, 2 jobs, with this command:

```bash
python3 tools/rci/submit.py --walltime 0:30:00 --seeds 0 1 -- \
    --task chase --preset paths_3v1 --steps 5e8 \
    --set normalize_obs=false --set adaptive_lr=false \
    --set lr=1.25e-4 --set ent_coef=0.003 \
    --set gamma=0.995 --set clip_eps=0.1 --set num_steps=128 \
    --out runs/04_new_base/gamma0.995_clip_eps0.1_num_steps128
```

## Results

| config | seeds | caught_e | min | max | coll_pp | death_rate_p |
|---|---|---|---|---|---|---|
| gamma0.995_clip_eps0.1_num_steps128 | 2/2 | 0.975 | 0.971 | 0.979 | 0.022 | 0.016 |

- The three changes together catch 0.975: 0.979 and 0.971 in the two seeds.
- The best row of `03_breadth`, `clip_eps` 0.1 alone, is 0.945.
