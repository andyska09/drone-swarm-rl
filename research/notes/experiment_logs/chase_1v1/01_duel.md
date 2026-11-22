# 01_duel

**Question:** Does the swap train a good pursuer and a good evader in one run of `chase` with one pursuer?
**Differs from** the code of today: the two sides learn in turn. A turn ends when the pursuer's score crosses a limit, and never before `swap_min` updates. With `swap_min=0` the swap is off, and both sides learn in every update. The reward is the one before the progress reward: the pursuer pays a clock and a distance cost, and a catch pays 10. Settings: preset `duel` (the box of `default`), `gamma=0.99`, `lr=3e-4` with a linear decay, 4e8 steps (1525 updates), seed 0.
**Runs:** 6 runs, 2026-09-20 to 2026-09-22, commits `47273bd` and `ac29531`. The code has no swap now. More tables: [chase_duel_log.md](../../archive/chase_duel_log.md).

- Two runs can have one hash, because the hash covers only the config. So the table names a run by its start time.
- The critic reads the actor's input, or from `ac29531` the scene: the world state of both drones and the action of the opponent.

## Results

Eval of the last checkpoint, 1024 episodes. The time to catch counts an episode with no catch as 10 s. The two crash columns come from the eval against the cascade pursuer.

| run | commit | swap | critic input | `ent_coef` | updates in which the evader learns | caught, cascade evader | time to catch, cascade evader | evader crashes | cascade pursuer crashes | caught, ours against ours | time to catch, ours against ours |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0920-153504_737629` | `47273bd` | on, `swap_min=10` | the actor's | 0 | 68 | 0.9609 | 3.55 s | 0.2793 | 0.1885 | 0.7041 | 5.68 s |
| `0920-212916_a45f5b` | `47273bd` | off | the actor's | 0 | 941, then killed | 0.6240 | 6.77 s | 0.1650 | 0.1064 | 0.0225 | 9.89 s |
| `0920-230541_d7464a` | `47273bd` | off | the actor's | 0.01 | 1525 | 0.6211 | 6.72 s | 0.2588 | 0.2598 | 0.4795 | 7.73 s |
| `0921-113249_a45f5b` | `47273bd` | off | the actor's | 0 | 1525 | 0.6514 | 6.88 s | 0.5898 | 0.0615 | 0.0195 | 9.89 s |
| `0921-175411_a45f5b` | `ac29531` | off | the scene | 0 | 1525 | 0.3447 | 8.33 s | 0.0977 | 0.3174 | 0.0078 | 9.95 s |
| `0922-094921_737629` | `ac29531` | on, `swap_min=10` | the scene | 0 | 103 | 0.7227 | 6.28 s | 0.3965 | 0.0527 | 0.5010 | 7.51 s |

- No run has a good pursuer and a good evader. The highest catch against the cascade evader is 0.9609 (swap on, the actor's input), and the fewest evader crashes are 0.0977 (swap off, the scene).
- With the scene as the critic input, the catch against the cascade evader falls in both settings: 0.9609 → 0.7227 with the swap, and 0.6514 → 0.3447 without it.
- With the swap, the evader learns in 68 and in 103 of the 1525 updates.

The pursuers and the evaders of the two runs with the swap, each pursuer against each evader (`caught`):

| pursuer of | evader of `0920-153504_737629` | evader of `0922-094921_737629` | cascade evader |
|---|---|---|---|
| `0920-153504_737629`, the actor's input | 0.7041 | 0.5557 | 0.9609 |
| `0922-094921_737629`, the scene | 0.4111 | 0.5010 | 0.7227 |

- The pursuer with the actor's input catches more against all three evaders.

The std of the actions at the end of each of the 14 `chase` runs up to 2026-09-23, by `ent_coef`:

| `ent_coef` | action bound | runs | std at the end |
|---|---|---|---|
| 0.01 | clip (`0920-230541_d7464a`) | 1 | 112 to 60784 |
| 0.01 | tanh squash | 1 | 371 to 942 |
| lower | clip or tanh squash | 12 | 0.02 to 0.51 |

- Only the two runs with `ent_coef` 0.01 end with a std above 1.
- The run with the squash and 0.01 ended at a `distance` of 11.40 m, where an untrained policy starts. Every other trained run ended at 0.5 to 4.0 m.
