# 05_attention

**Question:** Does the attention net catch (`caught_e`) as much as the MLP in `paths_3v1`, with the settings of `04_new_base`?
**Differs from** `04_new_base`: `net=attention`. Seeds 0 and 1.
**Runs:** RCI, 2 jobs, with this command:

```bash
python3 tools/rci/submit.py --walltime 2:00:00 --seeds 0 1 -- \
    --task chase --preset paths_3v1 --steps 5e8 \
    --set normalize_obs=false --set adaptive_lr=false \
    --set lr=1.25e-4 --set ent_coef=0.003 \
    --set gamma=0.995 --set clip_eps=0.1 --set num_steps=128 \
    --set net=attention \
    --out runs/05_attention/net_attention
```

## Results

| config | seeds | caught_e | min | max | coll_pp | death_rate_p |
|---|---|---|---|---|---|---|
| net_attention | 2/2 | 0.988 | 0.987 | 0.989 | 0.009 | 0.007 |

- The attention net catches 0.988: 0.989 and 0.987 in the two seeds.
- The MLP of `04_new_base` catches 0.975: 0.979 and 0.971.
