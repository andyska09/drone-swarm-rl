# Handoff — action bound, action std, and the entropy bonus

Repo: `/Users/ondrej/workspace/drone-swarm`, branch `main`, working tree clean.
Read `CLAUDE.md` and `research/notes/goals.local.md` first. Start every reply with
`TARS:` and use the simple-English output style.

## What the session did

The task was: make the action bound and the action std match Gavin's papers, and
keep it clean. That is done and committed.

- `0be3492 action std now depends on state` — the code change.
- `fb33384 slop` — the `choices.md` note, rewritten after the run failed.

The change, in `swarm/learn/ppo.py`: two linear heads on the shared trunk. The
mean goes out as is. A second raw number per action goes through `softplus` plus
`MIN_STD` and becomes the std, so the std is per state. The draw is a Gaussian
sample, the action is `tanh(draw)`. `log_prob` and `entropy` charge for the tanh
stretch. `Transition` keeps the pre-squash draw, because `atanh` of an action is
unstable at ±1. `opp_actions` no longer clips. `evaluate.py` flies `tanh(mean)`.
`config.py` swapped `init_log_std = -0.5` for `init_std = 0.6`.

The reasoning, the paper quotes with section numbers, and the gradient maths all
live in `research/notes/choices.md`, in the two bullets after **Actions**. Do not
restate them; read them there.

Two breakages, on purpose: old checkpoints do not load (new actor shape), and old
`runs/*/config.json` do not load (`init_log_std` is gone).

Checks: `pytest tests/` is 50 passed. `scratch/check_squash.py` verifies the
density integrates to 1 over `[-1,1]^4` and that the entropy estimate matches a
sampled one.

## The finding that matters

The first run with the new code, `runs/chase_duel_s0_20260922-220845_82fcd7`
(`ent_coef=0.01`, `swap_min=0`, `gamma=0.99`, 4e8 steps), failed. Its policy went
limp: `distance` 11.40 m, `caught` 0.031, `rho` 0.079, episodes running to
timeout, almost no crashes.

A survey of all 14 `chase` runs settled the cause. `ent_coef` is the only variable
that ever breaks the std. Twelve runs at `0` all end with std between 0.02 and
0.51. The two runs at `0.01` are the only broken ones — 112 to 60784 under the old
clip, 371 to 942 under the new squash. The two runs that solve `chase default`
(`caught` 0.996 and 1.000) both end at std 0.08 to 0.13.

Why the bonus breaks each design differently is written out in `choices.md`. The
short version: under the clip the entropy pushes the std up for ever; under the
squash that push reverses, but a new term pulls the mean to zero and wins once the
std passes about 15, so `m/s → 0` and the rate loop averages the command to
nothing.

Three notes for reading results:

- `distance` is the honest tell. Trained runs sit at 0.5 to 4.0 m. 11 m is an
  untrained or limp policy.
- Do not read the duel self-play `caught`. It says who is winning, not who is
  good. Use the two seat evals.
- `metrics.csv` does not log the std. Reading it needs a checkpoint. The user
  declined adding it; offer again only if asked.

## What to do next

The user chose **direction 1**: keep the squash and the per-state std, and leave
`ent_coef` at its default `0.0`. No code change is needed. The run was not started
in this session.

```bash
conda run -n drone-swarm python run/train.py \
    --task chase --preset duel --seed 0 --steps 4e8 \
    --set gamma=0.99 --wandb
```

Its twin is `runs/chase_duel_s0_20260921-113249_a45f5b` — same cell, old action
bound. Numbers to beat: `distance` 3.61, `caught` 0.034, `rho` 0.031, final std
0.06 to 0.19, and in the seat evals `evader_died` 0.5898 and `caught` 0.6514.

Then the three evals, because the self-play number hides both sides:

```bash
R=runs/<new run>
conda run -n drone-swarm python run/eval.py $R --seat pursuer cascade  # evader gate
conda run -n drone-swarm python run/eval.py $R --seat evader cascade   # pursuer gate
conda run -n drone-swarm python run/eval.py $R
```

Early signs to watch: `distance` must fall to 1 to 4 m; the two `*_entropy`
columns should drift down slowly from about 2.4 and stay in single digits; a dive
into the hundreds means the std is running again. If `distance` parks near 11 m
with `ent_coef = 0`, the squash itself is at fault, and the fix is a smooth bound
on the std, `s = MIN_STD + (1 - MIN_STD) * sigmoid(raw)` — direction 2 from the
session, with the ceiling of 1 argued in `choices.md`.

After that the user plans to implement full PFSP from the Nv1 paper
(`research/papers/2607.05939v1.md`, §IV-D) for the 1v1 case. They will not use the
`swap_min` alternation. Design notes for the duel are in
`research/notes/task_chase_duel.md`, and the run log in
`research/notes/experiments/chase_duel_log.md` — note that log still uses the old
metric names (`evader_died`, `caught_final`, `distance_mean`).

## Open questions the user has not decided

- Learning-rate schedule. I recommended `anneal_lr = False` with a constant lr,
  because Gavin publishes `Decay learning rates: False` and lr `5e-4`, which makes
  any per-phase lr reset unnecessary. No decision was made.
- Whether to switch the trunk to ReLU. It needs no code, only
  `--set activation=relu`, and it matches both papers. Not decided.

## Suggested skills

- `code-review` — before the next commit that touches `swarm/learn/`, review
  against the repo's own standards.
- `simplify` — the user pushed back hard on comment and naming style during this
  session. Use it on any new diff in `swarm/learn/ppo.py`.
- `grilling` — useful when the PFSP design is drafted, to stress it before code.
- `research` — if the next agent has to pin down PFSP details the Nv1 paper leaves
  out.

No document-format skill is needed. Notes in this repo are plain Markdown under
`research/notes/`, and the papers are already transcribed as sibling `.md` files,
so grep those instead of opening a PDF.

## House rules that bit during this session

- Do only what is asked. Do not add knobs, tests, or metrics unasked.
- Comments: as few as possible, plain, and never a story or a claim about what the
  code does not do.
- Answer the question that was asked. A question is not a signal of an error.
- Everything stays under the repo root; scratch goes in `scratch/`, which is
  gitignored. This handoff is the one exception, because `/handoff` asks for the
  OS temp directory.
- `research/papers/master_thesis.*` is unpublished and local only. Never name it,
  quote it, or add it to any tracked file.
