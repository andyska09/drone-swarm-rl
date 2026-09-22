# Task 4 — duel, run log

What each duel run was and what it scored. Design and gates are in
[task_chase_duel.md](../task_chase_duel.md).

All six: `chase`, preset `duel`, `gamma=0.99`, `seed=0`, 4e8 steps, 1525 updates.
Only the named fields differ.

| run | critic | setting | updates |
|---|---|---|---|
| `chase_duel_s0_20260920-153504_737629` | shared obs | `swap_min=10` | 1525 |
| `chase_duel_s0_20260920-212916_a45f5b` | shared obs | `swap_min=0` | killed at 941 |
| `chase_duel_s0_20260920-230541_d7464a` | shared obs | `swap_min=0`, `ent_coef=0.01` | 1525 |
| `chase_duel_s0_20260921-113249_a45f5b` | shared obs | `swap_min=0` | 1525 |
| `chase_duel_s0_20260921-175411_a45f5b` | CTDE, `ac29531` | `swap_min=0` | 1525 |
| `chase_duel_s0_20260922-094921_737629` | CTDE, `ac29531` | `swap_min=10` | 1525 |

## Scores

Every number is `_final` from `evals/`, 1024 episodes, the last checkpoint.

**All six runs were measured with the old metric names**, before `info` was split
into `end` and `step`. Reading them today: `_final` is gone, `evader_died` is now
`evader_crashed`, and the `distance` row below is the gap at the **last step**, while
`distance` now means the gap **averaged over the chase**. Do not compare the last row
against a new run.

The last two columns are the CTDE critic. Each one matches the shared-obs run at the
same `swap_min` on every other field, so the pairs to read are 1 with 6, and 4 with 5.

| metric | swap ON | swap OFF (941) | swap OFF, ent 0.01 | swap OFF (1525) | swap OFF, CTDE | swap ON, CTDE |
|---|---|---|---|---|---|---|
| `caught` vs cascade evader — the gate | **0.9609** | 0.6240 | 0.6211 | 0.6514 | **0.3447** | 0.7227 |
| time to catch vs cascade evader | **3.55 s** | 6.77 s | 6.72 s | 6.88 s | **8.33 s** | 6.28 s |
| the evader crashes, `evader_died` — the gate | 0.2793 | 0.1650 | 0.2588 | **0.5898** | **0.0977** | 0.3965 |
| the cascade pursuer crashes, the remainder | 0.1885 | 0.1064 | 0.2598 | 0.0615 | **0.3174** | 0.0527 |
| `caught` ours vs ours | 0.7041 | 0.0225 | 0.4795 | 0.0195 | 0.0078 | 0.5010 |
| time to catch, ours vs ours | 5.68 s | 9.89 s | 7.73 s | 9.89 s | 9.95 s | 7.51 s |
| `distance` vs cascade pursuer | 8.57 m | 11.38 m | 7.71 m | 14.12 m | 6.76 m | 12.13 m |

**Time to catch is right-censored at 10 s**, as both papers do: an episode that ends
in a crash or a timeout is written down as the full 10 s, not as the step it stopped.
Our episode is 1000 steps at `policy_dt = 0.01 s`, so the horizon matches theirs
exactly. Computed from `episodes.npz`, not from the `length` summary, which averages
the real end step and is a different quantity.

**Only the `ours vs ours` time is comparable to a published table.** Their number is
a learned pursuer against a learned evader. The cascade evader is a waypoint
follower, so the row above it is our own benchmark, not theirs.

Rows 3 and 4 are one number split in two, both from the `pursuer_cascade` eval.
`crashed` in `info` is `jnp.any` over the drones, so it **contains** `evader_died` and
the two overlap. Row 3 is `crashed − evader_died`, which is the cascade pursuer's
share. It slightly undercounts: if both drones crash on the same step, `jnp.any`
records one episode, not two.

Best evader so far is `swap OFF, CTDE` at 0.0977. Best pursuer is still
`swap ON` with the shared-obs critic at 0.9609. No column holds both.

## What it points to

No run gives a good pursuer and a good evader together.

- The swap builds the pursuer (0.96) and starves the evader (0.28).
- No swap builds the evader for a while (0.165 at update 941), then breaks it
  (0.590 at 1525).

**The evader dies on the wall.** `crashed` climbs run over run to 0.65 and
`evader_died` to 0.59. Nothing costs the evader for running straight. So it learns
to run straight, and the wall ends the episode.

The first version of this line read `distance` as the evader's distance from the
arena centre and matched 14.12 m to the 14 m usable edge of a 32 m box. That is
wrong: `distance` is the gap from the nearest pursuer's net to the evader. The two
14s are different quantities. The crash rates above carry the claim; `distance`
does not.

**The evader knows one pursuer.** The same policy dies 0.8% of the time against
our pursuer and 59% against the cascade pursuer. A new opponent pushes it where
it has never been.

**Ours-vs-ours `caught` hides both.** It read 0.48 in the `ent_coef=0.01` run.
The pursuer scored 0.62 against the cascade in every run. That rise was the
evader falling, not the pursuer rising. Read the seat evals, not the self-play
number.

## The CTDE critic — two runs

`ac29531` moved the critic to Gavin's input: from the actor's 28 numbers to 40 —
every drone's exact `x, v, R, omega` in world frame, plus the opponent's action this
step. The actor did not change. Scores are the last two columns of the table above.

Read as two pairs: the evader got much better without the swap (0.5898 → 0.0977) and
the pursuer got worse in **both** settings (0.9609 → 0.7227, 0.6514 → 0.3447).

Gate: the old `swap_min=10` run passed rows 1 and 2. Both CTDE runs pass row 1 only.

### The crashes changed owner, they did not go away

Rows 2 and 3 of the score table, swap OFF: the evader's crashes fall 0.5898 → 0.0977
while the cascade pursuer's rise 0.0615 → 0.3174. Total barely moves, 0.6514 → 0.4150.
Episodes are 15% longer, which does not explain a five-fold rise.

### Where they actually fly

Distance to the nearest arena bound, over the live steps of `pursuer_cascade`,
straight from `trajectory.npz`:

| | evader mean | evader under 3 m | cascade pursuer under 3 m |
|---|---|---|---|
| swap OFF, shared obs | 4.40 m | 25.2% | 20.7% |
| swap OFF, CTDE | **3.73 m** | **40.6%** | **45.1%** |

**The evader did not learn to survive in open air. It learned to fly along the wall
and take the pursuer down with it.** That is the exploit 1v1 §IV names in one
sentence, and `λbnd` is its published answer. Row 3 stays at the top of the list.

Nv1 runs the same 32 × 32 × 16 arena and reports its evader settling into 12 × 12 × 6
in the centre — at least 10 m off the side walls and 5 m off the floor and ceiling.
That is the calibration target for `ϕbnd`, whose threshold and curve neither paper
publishes.

### Cross-eval: is the pursuer worse, or just specialised?

Every pursuer against every evader, `caught`. Needs a loader patch for the pre-split
checkpoints — their actor is `Dense_0..2` and `log_std`, the same names and shapes the
split actor uses, one level short of nesting under `actor`.

| pursuer | vs shared-obs evader | vs CTDE evader | vs cascade |
|---|---|---|---|
| shared obs | **0.7041** | **0.5557** | **0.9609** |
| CTDE | 0.4111 | 0.5010 | 0.7227 |

**The old pursuer wins all three columns.** A pursuer that had merely specialised
against a harder opponent would win the middle one. So this is a real regression, not
a change of opponent. It also crashes more: 0.146 against the shared-obs evader, where
the old pursuer crashes 0.070 against the CTDE evader.

### But the critic itself got better

`explained_var` and `value_loss`, pursuer turns only, the swap-ON pair:

| updates | old `ev` | CTDE `ev` | old `vl` | CTDE `vl` |
|---|---|---|---|---|
| 0-200 | 0.912 | 0.881 | 0.400 | 0.436 |
| 600-1000 | 0.909 | **0.938** | 0.435 | **0.260** |
| whole run | 0.913 | **0.922** | 0.431 | **0.329** |

**Better value, worse policy.** That is published, not a bug.
[Lyu, Xiao, Daley and Amato, arXiv:2102.04402](https://arxiv.org/abs/2102.04402): the
policy-gradient variance with a centralized critic is at least as large as with a
decentralized one, because the decentralized critic has already averaged out the other
agents' randomness.

It is visible in one line of `learn/ppo.py`:
`delta = tr.reward + gamma * next_value * tr.cont - tr.value`. With the old critic
nothing in `delta_t` depended on the opponent's next move. With CTDE, `next_value` is
`V(s_{t+1}, a_opp,t+1)` — so this drone's advantage now swings with a choice it did
not cause and its actor cannot see.

**Read this as half a recipe, not a wrong one.** Gavin pays for that variance with a
bounded action, a state-dependent spread, and 2e9 to 4e9 steps. At `ac29531` we had
none of the three.

### The swap starved the evader again

From the `training` column: the evader trained **68** of 1525 updates with the old
critic and **103** of 1525 with CTDE. Both swap-ON evaders are close to untrained, so
the evader column of that pair is not worth reading.

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

**Closed since:** row 1 in `ac29531`, rows 2 and 5 in `0be3492`. Nothing else has
moved. Row 3 is now the best-evidenced of the open ones — see the wall measurement
above.

## Mechanism, from our code

`ent_coef=0.01` is Gavin's value and it did not transfer. He squashes the action
with `tanh`, so a wider spread stops paying. We sampled a Gaussian and clipped it in
`core.fly`, so the entropy of the policy had no ceiling — it ran 3.70 → 35.28 and
never turned. The evader traded not-crashing for noise: `crashed` 0.5186, the
worst of the three finished runs.

`0be3492` fixed it. The actor's spread now comes from the net per state, the draw is
squashed with `tanh`, and entropy carries the squash's log-determinant. Measured on
the new code:

| `std` | old formula | now |
|---|---|---|
| 0.61 | 3.68 | 2.40 |
| 1.00 | 5.68 | **2.68** ← the peak |
| 7.39 | 13.68 | −28.3 |
| 1636, where that run ended | 35.28 | −10398 |

Uniform on `(-1, 1)⁴` has entropy `log(2⁴) = 2.77`, so the peak sits just under the
bound, as it must. A wider spread now costs entropy instead of buying it.
`ent_coef=0.01` has not been run under this code yet.
