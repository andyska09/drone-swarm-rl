# AMSPBH — pool training for chase

How `chase` trains both sides against pools of frozen opponents. The method is AMSPBH
from [Sánchez Roncero 2026](../papers/2506.02849v3.md): §III-C for the training and the
Hedge sampler (eq. 12), §IV-A for the settings. We keep to the paper. Where it is silent,
we follow Němeček's code,
`research/code_sources/rl_tracking_eagle-cleaned-up-logic/examples/deep_eagle/examples/train_pipeline.py`.

The env, the observation, the reward and the eval seats are the duel's: read
[task_chase.md](task_chase.md), then [task_chase_duel.md](task_chase_duel.md). This note
replaces the duel's swap. The seed opponents are in [baselines.md](baselines.md).

## The run

```bash
python run/train.py --task chase --preset default --seed 0 \
    --set gamma=0.99 --set generations=13 --steps 1.4e9
```

`generations` is a `TrainConfig` field. At 0, the default, every task trains as before,
and in `chase` both sides learn at once. Above 0, `chase` trains with pools. Any other
task raises an error.

`--steps` is the whole run. It splits evenly into `2 × (generations + 1)` phases of
`num_updates // phases` updates. The few updates left over are not run. The run above has
28 phases of 190 updates, 49.8M steps each. At the duel's 36k steps/s it takes about 11 h,
plus the Hedge passes.

## Generations

| generation | phase 1 | phase 2 |
|---|---|---|
| 0 | the evader learns against FRPN | the pursuer learns against `paths` |
| k ≥ 1 | the evader learns against the pursuer pool of generation k − 1 | the pursuer learns against the evader pool of generation k − 1 |

- One side learns at a time, and the evader goes first.
- Each side trains against the other side's pool **as it was at the end of generation
  k − 1**. So the pursuer never meets the evader of its own generation while it learns.
  After both phases, both new policies join their pools.
- Generation 0 starts both sides from random weights.
- Each phase continues the learner's TrainState: weights, Adam moments and the lr. The
  paper calls this full-agent loading.
- Except the std: each phase sets the learner's std back to `init_std`. Ours. The std
  shrinks within a phase, and the learner must explore again against the new pool.
- Each phase starts fresh episodes. Each side keeps its own reward scale across phases.
- Pools only grow. The opponent's identity never enters the observation.

## Pools

| pool | who trains against it | seed | then |
|---|---|---|---|
| pursuer pool | the evader | `frpn` | `pursuer_00`, `pursuer_01`, … |
| evader pool | the pursuer | `paths` | `evader_00`, `evader_01`, … |

- `paths` is one member. Each episode it draws hover, circle or figure-eight with equal
  chance.
- APF and the cascade are test-only. They never enter training.

## Hedge

Before each phase, the learner as it stands plays every member of the opposing pool, 600
episodes each. It runs as eval does: both sides fly their mean action, on the fixed eval
seeds (`EVAL_SEED`), with no auto-reset.

A win comes from the seven outcomes in `info["end"]`:

| the pursuer wins | the evader wins | half a win to each |
|---|---|---|
| `caught_evader`, `crash_evader`, `OOB_evader` | `timeout`, `crash_pursuer`, `OOB_pursuer` | `fail_both` |

`w_j` is the learner's win rate against member `j` of a pool of `N`. Eq. 12:

```
c_j = 1 − w_j
p_j = γh / N + (1 − γh) · c_j^ηh / Σ c_l^ηh        γh = 0.7, ηh = 2
```

If every `c_j` is 0, the second term is uniform. In generation 0 the pool has one member,
so its weight is 1.

## Opponents in training

- The envs form 64 blocks: 4096 envs make 64 blocks of 64. A block keeps one pool member
  for the whole phase.
- A member of weight `p_j` gets `p_j × 64` blocks, rounded by largest remainder so that
  the counts add up to 64. With `γh = 0.7` and at most 14 members, each member gets at
  least 3 blocks.
- A frozen network flies its mean action. Němeček does the same (`deterministic=True`,
  `train_pipeline.py:228`).
- A seed starts with each of its episodes. `paths` draws a path and puts the evader on it,
  and both seeds reset their PIDs. A path counts time from its own episode start.
- Only the learner's rows enter the loss. Only the learner's side updates its reward
  scale.
- The learner's critic sees the action the opponent really flies, also at the states it
  bootstraps from.

## PPO

| | value | from |
|---|---|---|
| lr | 2e-4 at the start, then adaptive | the paper |
| adaptive lr | target KL 0.016. KL > 0.032: lr ÷ 1.5. 0 < KL < 0.008: lr × 1.5. The lr stays in [1e-5, 1e-2] | the paper's target, rsl_rl's rule |
| entropy coefficient | 0.001 | the paper |
| action std | one learned number per action, 1.0 at the start; the action is clipped to [−1, 1] | the paper is silent; the default of rsl_rl, rl_games and skrl, the PPO libraries of Isaac Lab |

- After the forward pass of each minibatch, read the learner's mean `approx_kl`. It is a
  sample estimate of KL(old ‖ new), the KL that rsl_rl computes in closed form.
- The rule then sets the lr, and the step of that same minibatch uses it, as in rsl_rl.
- Each role has its own lr in its own TrainState, so a frozen role keeps its lr.
- Every task uses the adaptive lr. `anneal_lr` is gone.

Everything else stays ours: `[256, 256]`, tanh, 4 epochs × 32 minibatches, `vf_coef`
0.5, 4096 envs, rollout 64, clip 0.2, GAE 0.95, grad clip 0.5, reward normalization, raw
obs, and `gamma` 0.99 for chase.
The reward, the observation, the CTDE critic, the 100 Hz policy, the arena, the net, the
spawn, no domain randomization, and the final evals are ours too.

## After a failure

This changes the value target only. The reward does not change.

| the episode ends with | the drone that failed | every other drone |
|---|---|---|
| a catch | — | no bootstrap |
| a failure: `crash`, `OOB` or `collPP` | no bootstrap | bootstraps |
| the step limit | — | bootstraps |

A failure used to be terminal for every drone. So the pursuer gained when the evader
failed, because it stopped paying the clock. At γ = 0.99 that is worth about 4, against a
catch of 10. The paper also bootstraps past an out-of-bounds (§III-B).

`chase.step` sets `info["truncated"]` whenever an episode ends without a catch, and
`vecenv` bootstraps `alive × truncated`, so only the survivors bootstrap. The `timeout`
outcome stays on the step limit.

## The run folder

| file | what it holds |
|---|---|
| `metrics.csv` | one row per update. Adds `generation`, `training` (the side that learns) and an `lr` per role. The frozen side's learner stats stay empty: losses, entropy, KL, clip fraction, explained variance |
| `hedge.csv` | one row per generation, side and pool member: the win rate and the weight |
| `pool/<role>_<generation>.pkl` | one policy per side per generation. Loads as a `--seat` source |
| `latest.pkl` | the training state at the end of the last finished phase |
| `checkpoints/` | numbered policy snapshots, as before |

`python run/train.py --resume runs/<run>` restarts the unfinished phase from its start,
with the same pool.

`python run/matrix.py runs/<run>` plays every pursuer against every evader after
training: the pool members, plus APF and the cascade, 1024 episodes per cell. It writes
the pursuer's win rates to `matrix.csv` and draws them in `matrix.png`, as in the paper's
Fig. 3.

## Where it lives

| file | what it holds |
|---|---|
| `swarm/learn/pool.py` | the generations, the phases, Hedge, the pool files, resume, and the paper's constants |
| `swarm/learn/seats.py` | who flies a seat: a network, the cascade, FRPN, APF or a path. Eval and training share it |
| `swarm/learn/ppo.py` | the adaptive lr, the opponent's rows in the rollout, the learner-only loss |
| `swarm/learn/vecenv.py` | a seed's start on reset, the learner-only reward scale |
| `swarm/envs/chase.py` | `truncated` after a failure |
| `run/matrix.py` | the win-rate matrix |
