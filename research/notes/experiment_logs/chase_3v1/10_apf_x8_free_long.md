# 10_apf_x8_free_long

**Question:** Does twice the training time lift the catch (`caught_e`) of the three pursuers in `apf_x8_free_3v1` with `ent_coef` 0.001?
**Differs from** `09_apf_x8_free_ent`, config `ent_coef0.001`: 1e9 steps. Seeds 0 and 1.
**Runs:** RCI, 2 jobs, with this command:

```bash
python3 tools/rci/submit.py --walltime 1:00:00 --seeds 0 1 -- \
    --task chase --preset apf_x8_free_3v1 --steps 1e9 \
    --set normalize_obs=false --set adaptive_lr=false \
    --set lr=1.25e-4 --set ent_coef=0.001 \
    --set gamma=0.995 --set clip_eps=0.1 --set num_steps=128 \
    --out runs/10_apf_x8_free_long/steps1e9
```

- In `09_apf_x8_free_ent`, with 5e8 steps, the three pursuers catch 0.706. The catch in training still rises at the end: 0.555, 0.580 and 0.648 in the last three tenths of seed 0.
- Beside the catch, the result is the std of each action at the end of the run.
