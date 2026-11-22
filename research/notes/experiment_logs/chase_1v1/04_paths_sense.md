# 04_paths_sense

**Question:** Does one pursuer catch the paths evader more often, or sooner, when it senses the attitude or the motor speeds of the evader?
**Differs from** the code of today: the task was `chase_modal`, with the presets `1v1`, `1v1_attitude`, `1v1_rpm` and `1v1_attitude_rpm`. They are `paths_1v1` to `paths_1v1_attitude_rpm` of `chase` now, and the table uses these names. The reward is the one before the progress reward. Settings: `gamma=0.99`, 2e8 steps (762 updates), the adaptive lr, `ent_coef=0.001`, seeds 0 to 2.
**Runs:** MetaCentrum, 12 runs, 2026-10-28, commit `23b65a8`. The code of today cannot load these runs. The 12 runs of the same batch with three pursuers are in [chase_modal_log.md](../../archive/chase_modal_log.md#runs-2026-10-28).

## Results

Eval of the last checkpoint with `--seat evader paths`, 1024 episodes.

| config | seeds | caught_e | seed 0 | seed 1 | seed 2 | time_to_catch | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|---|---|---|---|---|
| paths_1v1 | 3/3 | 0.96 | 0.95 | 0.94 | 0.98 | 3.7 s | 3.9 s | 3.8 s | 3.3 s |
| paths_1v1_attitude | 3/3 | 0.99 | 0.96 | 1.00 | 1.00 | 2.7 s | 3.9 s | 2.2 s | 2.0 s |
| paths_1v1_rpm | 3/3 | 0.98 | 1.00 | 0.97 | 0.97 | 3.0 s | 2.6 s | 3.1 s | 3.2 s |
| paths_1v1_attitude_rpm | 3/3 | 0.97 | 0.91 | 1.00 | 1.00 | 3.0 s | 4.3 s | 2.1 s | 2.5 s |

- Every preset catches 0.96–0.99 in the mean of its seeds.
- 7 of the 9 runs with a sensed block catch in 2.0–3.2 s. The fastest run with no block needs 3.3 s.
