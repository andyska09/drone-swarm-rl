# Task chase — training with pools

How `chase` trains both sides against pools of frozen opponents. The method is AMSPBH
from [Sánchez Roncero 2026](../papers/2506.02849v3.md): §III-C for the training and the
Hedge sampler (eq. 12), §IV-A for the settings. We keep to the paper. Where it is silent,
we follow Němeček's code,
`research/code_sources/rl_tracking_eagle-cleaned-up-logic/examples/deep_eagle/examples/train_pipeline.py`.

The env, the observation, the reward and the eval seats are in
[task_chase.md](task_chase.md). The seed opponents are in
[task_chase_baselines.md](task_chase_baselines.md). The PPO settings that every task
shares, the adaptive lr among them, are in [choices.md](choices.md), Learner. The runs
and the history of this design are in
[experiments/chase_pools_log.md](experiments/chase_pools_log.md).

## The run

```bash
python run/train.py --task chase --preset default --seed 0 \
    --set gamma=0.99 --set generations=13 --steps 1.4e9
```

`generations` is a `TrainConfig` field. At 0, the default, no task uses pools, and in
`chase` both sides learn at once. Above 0, `chase` trains with pools. Any other
task raises an error.

`--steps` is the whole run. It splits evenly into `2 × (generations + 1)` phases of
`num_updates // phases` updates. The few updates left over are not run. The run above has
28 phases of 190 updates, 49.8M steps each.

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
  `reset_std=false` turns it off.
- Each phase starts fresh episodes. Each side keeps its own reward scale across phases.
  The normalizer counts per drone column, so a frozen column stays exact.
- Pools only grow. The opponent's identity never enters the observation.

## Pools

| pool | who trains against it | seed | then |
|---|---|---|---|
| pursuer pool | the evader | `frpn` | `pursuer_00`, `pursuer_01`, … |
| evader pool | the pursuer | `paths` | `evader_00`, `evader_01`, … |

- `paths` is one member. Each episode it draws hover, circle or figure-eight with equal
  chance.
- APF and the cascade are test-only. They never enter training.
- The seeds stay in the pools in every generation, as in the paper: Fig. 6 and §IV-C,
  "heuristic opponents (FRPN, trajectories) retain non-trivial weight even in later
  generations". Fig. 2's `Π0P = {πP0}` reads otherwise, but the results do not.

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

## Where we differ from the paper

Every difference from Roncero, and from Němeček, is in one table:
[literature/comparisons.md](literature/comparisons.md), Chase.

## The run folder

| file | what it holds |
|---|---|
| `metrics.csv` | one row per update. Adds `generation`, `training` (the side that learns) and an `lr` per role. The frozen side's learner stats stay empty: losses, entropy, KL, clip fraction, explained variance |
| `hedge.csv` | one row per generation, side and pool member: the win rate and the weight |
| `pool/<role>_<generation>.pkl` | one policy per side per generation. Loads as a `--seat` source |
| `latest.pkl` | the training state at the end of the last finished phase |
| `checkpoints/` | numbered policy snapshots, as in every run |

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
| `swarm/learn/ppo.py` | the opponent's rows in the rollout, the learner-only loss |
| `swarm/learn/vecenv.py` | a seed's start on reset, the learner-only reward scale |
| `run/matrix.py` | the win-rate matrix |
