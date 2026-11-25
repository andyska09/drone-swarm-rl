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

- In `09_apf_x8_free_ent`, with 5e8 steps, the catch in training still rises at the end: 0.555, 0.580 and 0.648 in the last three tenths of seed 0.
- Beside the catch, the result is the std of each action at the end of the run.

## Results

| config | seeds | caught_e | min | max | coll_pp | death_rate_p |
|---|---|---|---|---|---|---|
| steps1e9 | 2/2 | 0.805 | 0.733 | 0.876 | 0.064 | 0.045 |

The spread of the pursuers' actions in training, as the mean of the two seeds. The std of each action starts at 1.0, and the table gives its value at the end of the run. The row of 5e8 steps is `ent_coef0.001` of `09_apf_x8_free_ent`. A tenth of the run with 1e9 steps is twice as long.

| steps | caught_e in training, last tenth | entropy_p, first tenth | entropy_p, last tenth | std thrust | std roll | std pitch | std yaw |
|---|---|---|---|---|---|---|---|
| 5e8 | 0.634 | 5.55 | 2.53 | 0.44 | 0.32 | 0.33 | 0.79 |
| 1e9 | 0.774 | 5.38 | −0.07 | 0.20 | 0.12 | 0.13 | 0.83 |
