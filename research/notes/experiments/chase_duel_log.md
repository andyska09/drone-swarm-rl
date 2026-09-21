# Task 4 — duel, run log

What each duel run was and what it scored. Design and gates are in
[task_chase_duel.md](../task_chase_duel.md).

All four: `chase`, preset `duel`, `gamma=0.99`, `seed=0`, 4e8 steps, 1525 updates.
Only the named fields differ.

| run | setting | updates |
|---|---|---|
| `chase_duel_s0_20260920-153504_737629` | `swap_min=10` | 1525 |
| `chase_duel_s0_20260920-212916_a45f5b` | `swap_min=0` | killed at 941 |
| `chase_duel_s0_20260920-230541_d7464a` | `swap_min=0`, `ent_coef=0.01` | 1525 |
| `chase_duel_s0_20260921-113249_a45f5b` | `swap_min=0` | 1525 |

## Scores

Every number is `_final` from `evals/`, 1024 episodes, the last checkpoint.

| metric | swap ON | swap OFF (941) | swap OFF, ent 0.01 | swap OFF (1525) |
|---|---|---|---|---|
| `evader_died` vs cascade pursuer — the gate | 0.2793 | **0.1650** | 0.2588 | **0.5898** |
| `caught` vs cascade evader | **0.9609** | 0.6240 | 0.6211 | 0.6514 |
| `caught` ours vs ours | 0.7041 | 0.0225 | 0.4795 | 0.0195 |
| `crashed` vs cascade pursuer | 0.4678 | 0.2715 | 0.5186 | 0.6514 |
| `distance` vs cascade pursuer | 8.57 m | 11.38 m | 7.71 m | 14.12 m |

## What it points to

No run gives a good pursuer and a good evader together.

- The swap builds the pursuer (0.96) and starves the evader (0.28).
- No swap builds the evader for a while (0.165 at update 941), then breaks it
  (0.590 at 1525).

**The evader dies on the wall.** `distance` climbs run over run to 14.12 m. The
usable edge of a 32 m box is 14 m. `crashed` climbs with it to 0.65. Nothing
costs the evader for running straight. So it learns to run straight, and the wall
ends the episode.

**The evader knows one pursuer.** The same policy dies 0.8% of the time against
our pursuer and 59% against the cascade pursuer. A new opponent pushes it where
it has never been.

**Ours-vs-ours `caught` hides both.** It read 0.48 in the `ent_coef=0.01` run.
The pursuer scored 0.62 against the cascade in every run. That rise was the
evader falling, not the pursuer rising. Read the seat evals, not the self-play
number.

## Ranked against Gavin

Every difference we could find, sorted by how likely it is to explain the table
above. From 1v1 §IV and Tables I-II, Nv1 §IV-V, and our own code.

| # | difference | Gavin | ours | matters? |
|---|---|---|---|---|
| 1 | critic input | value net sees the opponent's position, speed, rotation and its action this step (CTDE) | critic sees the same obs as the actor | very high |
| 2 | action bound | net outputs mean and spread, then `tanh` squash | free `log_std`, sample, then clip to [-1, 1] | very high |
| 3 | evader wall rule | 1v1: buffer zone at each wall, evader only, `λbnd = 1.0`. Nv1: evader boxed in 12 × 12 × 6 | none, whole arena | very high |
| 4 | training steps | 2e9 to 4e9 | 4e8 | high |
| 5 | action spread | net outputs it per state | one fixed vector for every state | high |
| 6 | opponent pool | Nv1: PFSP, each episode plays a frozen past opponent | only the newest opponent | high |
| 7 | update shape | 15 epochs × 1 minibatch = 15 big steps | 4 epochs × 32 minibatches = 128 small steps | medium |
| 8 | obs scaling | position ÷ view range, speed ÷ max speed | raw metres and m/s | medium |
| 9 | learning rate | 5e-4, no decay | 3e-4, decay on | medium |
| 10 | reward scaling | not published | reward normalization on | medium |
| 11 | activation | ReLU | tanh | low |
| 12 | opponent frame | world frame | body frame | low |
| 13 | angular speed in obs | not present | present | low |
| 14 | envs × rollout | 1024 × 128 | 4096 × 64 | low |
| 15 | wall obs | distance to each bound and the ground | 4 rays in the level body frame, plus height | low |

Rows 3 and 6 are the two findings above. Row 2 is the section below.

## Mechanism, from our code

`ent_coef=0.01` is Gavin's value and it does not transfer. He squashes the action
with `tanh`, so a wider spread stops paying. We sample a Gaussian and clip it in
`core.fly`, so the entropy of the policy has no ceiling — it ran 6.70 → 35.28 and
never turned. The evader traded not-crashing for noise: `crashed` 0.5186, the
worst of the three finished runs.
