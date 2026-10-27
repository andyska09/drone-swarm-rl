# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules — every AI agent follows these, in every reply and every change

These rules are not optional. They come before your own defaults and before any skill's
instructions. If a rule seems to clash with what I ask, ask me before you act.

### Talking to me

- Start every message with `TARS:`.
- Write every reply in the [simple-english](.claude/output-styles/simple-english.md)
  output style.
- **Answer the question I asked, and nothing more.** Be short and direct.
- A question is only a question: answer it, and keep your plan and conclusions until I
  say they change. If something I said changes an earlier conclusion, say so in one line
  at the end, then stop and wait.
- When I give you new information, take it in and answer. Redo the analysis only if I
  ask for it.
- I am new to this field. Explain every term in plain words. When I ask what a term
  means, explain the idea, not the literature.

### Doing the work

- **Do only what I ask.** Do not add features, knobs, tests or metrics that I did not ask
  for. Suggest them in the chat instead.
- A **decision** that changes the plan is mine: ask it as its own question, with the
  options and your pick, and act after I answer.
- Keep it simple. Simple means logic that is easy to follow, not few lines.
- Keep diffs small. I want to read a change and know what it does.
- **Task specs** in `research/notes/tasks/` say what each task is now: the model, not
  the code. Write or change one by the [spec rules](research/notes/tasks/README.md).
- The reasons, the history of the decisions, the run results and the open questions go
  to `research/notes/experiment_logs/<name>_log.md`. Comparisons with papers go to
  `research/notes/literature/comparisons.md`.
- Everything stays under the repo root, `handoff.md` included. Scripts, probes and build
  artifacts go in `scratch/` (gitignored), never in `/tmp` or a session scratchpad.
- Unpublished papers under `research/papers/` stay local, and `CLAUDE.local.md` names
  them. Never commit one, never `git add -f` one, and never quote, summarise or name one
  in a tracked file, this one included.

### Code and tests

Write them by [CODING_STANDARDS.md](CODING_STANDARDS.md).

### For Codex

Edit only the named file. Do not run tests. Stop after the edit, summarize it, and wait
for approval before you touch another file. The only exception: I tell you to run wild.

## What this repository is

A research workspace for multi-robot / swarm RL. Our own code lives in `swarm/`;
`research/` holds notes, papers, and the reference codebases we learn from but do
not edit.

**The deliverable is a JAX multirotor simulator** fast enough to train swarm
policies on thousands of parallel environments — written from scratch. What
it is for: **decentralized swarm interception** — N drones catching one agile
evader, each seeing only itself, its neighbours and the target, with no
communication between drones.

The work cycle for a task: write it in `swarm/envs/`, fly it with the cascade to
prove it is possible at all, train a policy with `run/train.py`, score both in the
same seat with `run/eval.py`, then watch it in `run/replay.py`.

Physics and control-stack reference: [ctu-mrs/mrs_multirotor_simulator](https://github.com/ctu-mrs/mrs_multirotor_simulator)
— `multirotor_model.hpp` (state + ODE) and `uav_system.hpp` (the six-controller
cascade). We reimplement it in JAX; we do not wrap the C++.


## swarm/ — our simulator

Every command below needs the `conda run -n drone-swarm` prefix. Pins in
`requirements.txt` (jax 0.11.1, flax 0.12.9, numpy 2.5.2, matplotlib 3.11.2,
pytest 9.1.1).

Run `pytest tests/`, never bare `pytest`, because other folders hold tests too.

```bash
pytest tests/test_control.py                           # one file
pytest tests/test_dynamics.py -k tumble                # one test
python run/train.py --task circle_swap --preset pair --steps 5e6  # PPO -> runs/<name>/
python run/train.py --task circle_swap --preset pair --set gamma=0.99  # any TrainConfig field
python run/train.py --task chase --preset default --set gamma=0.99 --set generations=13 --steps 1.4e9  # pools
python run/train.py --resume runs/<name>               # what it resumes: --help
python run/matrix.py runs/<name>                       # every pursuer against every evader
python run/eval.py runs/<name>                         # -> runs/<name>/evals/latest/
python run/eval.py runs/<name> --seat drone cascade   # the cascade in one role's seat
python run/replay.py                                   # serve + open the 3D viewer
python run/plot.py runs/<name>/evals/latest            # time plots of one episode
bash tools/mrs_golden/build.sh      # regenerate tests/golden/ from the C++
python3 tools/metacentrum/submit.py --walltime 1:00:00 -- --task chase --preset default --set generations=13 --steps 1.4e9  # on MetaCentrum: tools/metacentrum/metacentrum.md. 1 h is enough for a 1.4e9 pool run: always give --walltime 1:00:00
```

Each task file lists its presets in `PRESETS`. `train.py` stops unless it gets both
`--task` and `--preset`. A preset picks env values only — hyperparameters live in
`learn/config.py` and are overridden with `--set`.

- `sim/dynamics.py` — `Params`/`State` as `flax.struct.dataclass`,
  `step(state, throttle, params, dt)` with `throttle ∈ [0,1]⁴`. RK4 over the 18
  rigid-body states, then re-orthonormalize, then the exponential motor lag.
- `sim/control.py` — the cascade, read top to bottom in the order it runs:
  `position -> velocity -> acceleration -> attitude -> rate -> mixer`, plus
  `cascade_step` which chains all six. PID state is carried explicitly in
  `PIDState`; only the heading branch exists, not MRS's heading-rate branch.
- `sim/guidance.py` — the hand-written opponents for chase: FRPN, APF and the flown
  paths. Each one gives a world acceleration.
- `envs/` — one file per task, plain functions, no gymnax; auto-reset belongs to the
  trainer. **The env contract** every task file meets is in the docstrings of
  `envs/__init__.py`: what a task file holds, the obs, the end flags and the metrics.
  Read it before writing a task. Every task carries a drone axis, `N = 1` included, and
  each has a spec in `research/notes/tasks/task_<name>.md`.
- **`envs/core.py` holds everything the tasks share**: `Common` (the `EnvParams`
  fields every task has, which each task's `EnvParams` inherits), `fly` (the rate
  loop and the plant for one policy step), `own_obs`/`target_obs`/`others`,
  `spawn`, `hits`, `is_dead`, and the CTBR action conversion. A task file holds only
  what makes that task different. Do not copy a helper into a task file.
- **No task names a drone by number.** `roles` is static, so a task reads its own
  index arrays from it — see `chase._sides`. That is what lets a longer `roles`
  add more pursuers with no other change.
- The reward is a per-step cost, not a bonus: `-policy_dt * (distance + spin + …)`
  with a one-off crash penalty. `avoidance` is the exception: +1 when a drone arrives,
  −1 for a crash, 0 at every other step. `circle_swap` adds a proximity cost and an
  upright term. **`chase` keeps three words apart and never mixes them**: `fail` is leaving
  the box through the ground, a wall or the ceiling (in a preset without walls only the
  ground, and the evader pays `outside` per second instead), `collPP` is a pursuer hitting a
  pursuer (both end the episode), and `collPE` is a pursuer hitting the evader, which
  costs per step and does not end it. `collPP` and `collPE` are Gavin's names, used
  for the `RewardConfig` field, the function and the `info` key alike. The outcomes
  split a `fail` into `crash` (the ground, or a broken state) and `OOB` (a wall or
  the ceiling). **This reward parks the drone 0.2-0.6 m short of the goal** — the open
  problem, with the fixes other papers use, is in
  [research/notes/literature/reward_terms.md](research/notes/literature/reward_terms.md).
- **`EnvParams.roles` is the field everything hangs off.** It is
  `pytree_node=False`; its length is the drone count, and `role_slices` in
  `learn/ppo.py` turns each distinct name in it into its own set of weights.
- `learn/` — our PPO and the training loop. Each module's docstring says what it holds.
  `ppo.py` and `vecenv.py` know nothing about drones: they take an env *module* and one
  network per role.
- **Every role is its own learner.** One `TrainState` per role, so gradient
  clipping, Adam state, the adaptive lr, advantage scaling and reward scaling never
  mix two sides of a game. `cfg.generations` above 0 trains chase with pools; the
  design is in [research/notes/tasks/task_chase.md](research/notes/tasks/task_chase.md),
  Training.
- **Eval writes into the run it measures**, at `runs/<run>/evals/<name>/`:
  `eval.json` (summary + checkpoint + seeds + commit), `episodes.npz` (per-episode
  arrays), `trajectory.npz` (the first 8 episodes, full state + action), `header.json`
  (task, preset, arena, roles — the viewer never imports the env). Eval holds the
  scene frozen once an episode ends; the env does not, because with `N = 1` the
  episode ends the same step and the trainer resets it.
- `tools/viewer/index.html` — one file, three.js from a CDN, no build step. Parses
  the `.npz` in JS (`np.savez` writes a *stored* zip, so no inflate library).
  `run/replay.py` serves the repo and lists every eval at `/evals.json`, so the
  viewer opens with a dropdown; dragging the files onto the page still works.
- `sim/` is written for **one drone**. Both the drone axis and the parallel-env axis
  are `jax.vmap` at the env boundary. No Python branches on array values, no
  `.item()`, no value-dependent shapes — `test_vmap_matches_python_loop` enforces it.
- Every design decision for the simulator and for what all tasks share, every
  deviation, and every "revisit later" lives in
  [research/notes/design_choices.md](research/notes/design_choices.md). The shared
  model is in `research/notes/tasks/defaults.md`, and a task's own model, pool training
  included, is in its task spec. **Read it before changing the physics, and update it
  when a decision changes.**
- `conftest.py` turns on `jax_enable_x64` because the C++
is double precision.
- One deliberate deviation from the C++, patched by `tools/mrs_golden/transpose.diff`:
MRS re-orthonormalizes as `R·L⁻¹`, which is not orthonormal and drifts ~3e-3 off
SO(3); we use `R·L⁻ᵀ`. Hover is not a gate — it is an equilibrium, so a
transposed allocation matrix or a missing `ω × Jω` sails through it. Only
`tumble` and `spin_down` exercise the torque path.

## Tests

Each file in `tests/` opens with a docstring that says what it checks.

## research/ and the reference code

- `notes/` — working notes, Czech is fine. `design_choices.md` holds the shared design
  decisions and their reasons. `tasks/` holds the task specs: `defaults.md` for what
  every task shares, and `task_<name>.md` for one task. `experiment_logs/` holds the run
  logs and the history of the decisions. `literature/` holds the literature reviews
  and `comparisons.md`, which puts our setup beside the reference works. `learning/`
  holds explainers of the theory, and `archive/` the notes about removed code.
- `code_sources/` (GITIGNORED) — reference code, **not ours**:
  `mrs_multirotor_simulator/` (the C++ we reimplement), `PPO_example/` (the
  supervisor's JAX PPO), `quad-swarm-rl/` (PyTorch swarm, upstream clone),
  `rl_tracking_eagle-cleaned-up-logic/` (Němeček's pursuit-evasion code). **To run**
  `PPO_example` or `quad-swarm-rl`, read
  [reference_code.md](research/notes/reference_code.md).
- **Papers: grep the transcript, don't open the PDF.** Every PDF has a
  `pdftotext -layout` transcript as a sibling `.md`; open the PDF only for
  figures. [bibliography_index.md](research/papers/bibliography_index.md) is the
  entry point — a tracked table of paper → files → one-line summary. Add a row
  whenever a PDF is added, and generate its transcript with
  `pdftotext -layout <file>.pdf <file>.md`.
- Only the PDFs are gitignored. Transcripts and `bibliography_index.md` are
  tracked, so a fresh clone can grep every paper without the binaries.
- The unpublished papers (see Rules) are excluded through `.git/info/exclude`, which is
  itself untracked, so no tracked file names them. Add an index row only for a paper
  that is publicly published.
- `quad-swarm-rl` is a **live clone with its own `.git` and an `origin` pointing at
  the upstream GitHub repo**. Never commit or push from inside it. Treat every
  `code_sources/` tree as a read-only reference unless the user says otherwise.

## Shell traps

- **Bare `python` is not on PATH.** Every command goes through
  `conda run -n drone-swarm`.
- **`conda run` swallows stdout.** If you need to read the output, use
  `conda run --no-capture-output -n drone-swarm python ...`. Without the flag a
  script that prints fine in a terminal returns nothing here.

## Agent skills

### Issue tracker

None: the task specs in `research/notes/tasks/` take the place of issues. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `research/notes/tasks/defaults.md` is the glossary, and `research/notes/design_choices.md` holds the ADRs. See `docs/agents/domain.md`.
