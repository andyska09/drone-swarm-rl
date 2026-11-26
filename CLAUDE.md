# CLAUDE.md

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
- The design and the results of each experiment go to one file in
  `research/notes/experiment_logs/`, in the shape that its
  [README](research/notes/experiment_logs/README.md) gives. A paper's setup goes to a
  paper spec in `research/notes/literature/specs/`, by its
  [README](research/notes/literature/specs/README.md).
- Everything stays under the repo root, `handoff.md` included. Scripts, probes and build
  artifacts go in `scratch/` (gitignored), never in `/tmp` or a session scratchpad.
- Unpublished papers under `research/papers/` stay local, and `CLAUDE.local.md` names
  them. Never commit one, never `git add -f` one, and never quote, summarise or name one
  in a tracked file, this one included.
- **I stage, commit and push.** Use git only to read.
- After a change, run only the tests of the changed file. Run the full suite when I ask.

### Code and tests

Write them by [CODING_STANDARDS.md](CODING_STANDARDS.md).

### For Codex

Edit only the named file. Do not run tests. Stop after the edit, summarize it, and wait
for approval before you touch another file. The only exception: I tell you to run wild.

## What this repository is

A research workspace for multi-robot / swarm RL. Our own code lives in `swarm/`;
`research/` holds notes, papers, and the reference codebases we learn from but do
not edit.

What it is for: **decentralized swarm interception** — N drones catching one agile
evader, each seeing only itself, its neighbours and the target, with no
communication between drones.

The work cycle for a task: write it in `swarm/envs/`, train a policy with `run/train.py`, score in the
same seat with `run/eval.py`, then watch it in `run/replay.py`. Usually this also encompasses running different kinds of experiments and modifications to the envs or train parameters and setup. 

Physics and control-stack reference: [ctu-mrs/mrs_multirotor_simulator](https://github.com/ctu-mrs/mrs_multirotor_simulator)
— `multirotor_model.hpp` (state + ODE) and `uav_system.hpp` (the six-controller
cascade). We reimplement it in JAX; we do not wrap the C++.


## The map of each part

Read the map of a part before you read or change a file of that part.

- Our code, in `swarm/`, `run/`, `tests/` and `tools/viewer/`:
  [swarm/CLAUDE.md](swarm/CLAUDE.md).
- The notes, the papers and the reference code, in `research/`:
  [research/CLAUDE.md](research/CLAUDE.md).

## Commands

Every command needs the `conda run -n drone-swarm` prefix, because bare `python` is not
on PATH. `conda run` swallows stdout: to read the output, use
`conda run --no-capture-output -n drone-swarm python ...`. The pinned versions are in
[requirements.txt](requirements.txt).

Run `pytest tests/`, never bare `pytest`, because other folders hold tests too.

```bash
pytest tests/test_control.py                           # one file
pytest tests/test_dynamics.py -k tumble                # one test
python run/train.py --task circle_swap --preset pair --steps 5e6  # PPO -> runs/<name>/
python run/train.py --task circle_swap --preset pair --set gamma=0.99  # any TrainConfig field
python run/train.py --task chase --preset paths_3v1 --set-env reward.coll_pp=0  # any EnvParams field of the preset
python run/train.py --task chase --preset default --set gamma=0.99 --set generations=13 --steps 1.4e9  # pools
python run/train.py --resume runs/<name>               # what it resumes: --help
python run/matrix.py runs/<name>                       # every pursuer against every evader
python run/eval.py runs/<name>                         # -> runs/<name>/evals/latest/, or evals/<role>_<who>/ with a --seat or a preset's pilot
python run/eval.py runs/<name> --seat drone cascade   # the cascade in one role's seat
python run/replay.py                                   # serve + open the 3D viewer, the spec page and the training curves
python run/plot.py runs/<name>/evals/<eval>            # time plots of one episode
python run/compare.py runs/<a> runs/<b>                # evals and training curves side by side
python run/sweep.py runs/<name>                        # the table of an experiment, one row per config
python run/info.py runs/<name> --show std approx_kl    # what the files of each run hold; no --show prints all: --help
bash tools/mrs_golden/build.sh      # regenerate tests/golden/ from the C++
python3 tools/metacentrum/submit.py --walltime 1:00:00 -- --task chase --preset default --set generations=13 --steps 1.4e9  # on MetaCentrum: tools/metacentrum/metacentrum.md. 1 h is enough for a 1.4e9 pool run: always give --walltime 1:00:00
python3 tools/rci/submit.py --walltime 1:00:00 -- ...  # the same on RCI: tools/rci/rci.md
```

Each task file lists its presets in `PRESETS`. `train.py` stops unless it gets both
`--task` and `--preset`. A preset picks env values only — hyperparameters live in
`swarm/learn/config.py` and are overridden with `--set`.

## Agent skills

### Issue tracker

None: the task specs in `research/notes/tasks/` take the place of issues. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `research/notes/tasks/defaults.md` is the glossary, and the repo keeps no ADRs. See `docs/agents/domain.md`.
