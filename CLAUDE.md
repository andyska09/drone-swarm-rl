# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT NOTE TO ALL AI CHATBOTS

When talking to me start the message with "TARS:"

### Style

see output-styles

- Blunt, minimal code. No base classes, protocols, registries. One file per
  concern, readable top to bottom.
- **Comments: as few as possible.** Only where the code cannot say it itself —
  a constraint, a non-obvious reason, a trap. Never restate what the line does.
  One clear sentence beats three. No banner blocks, no section dividers.
- Same for docstrings: a module gets a short one, a function only if its
  contract is not obvious from its name and signature.

**Answer the question I actually asked. Nothing more.** If I ask what a term means, explain
the term - do not re-analyse the plan, do not revise your recommendations, do not rewrite
your conclusions. A question is not a signal that you were wrong. When I hand you new
information, absorb it and answer; only redo the analysis if I explicitly tell you to. If
something I said genuinely changes an earlier conclusion, say so in one line at the end and
stop there - wait for me to ask before expanding it.

I am new to this field. When I ask about a term, assume I want the concept explained plainly,
not a literature review.

When I shout and curse at you DO NOT apologize, it wastes tokens, just follow orders. 
Next when writing code DO NOT write stupid comments and docstrings. Keep the code clean and high quality. When implementing stuff KEEP IT SIMPLE. This important. I do not want to read 1000 lines of diffs, I want to look at the change and know what it does. 

### Scratch files

**Everything stays under the repo root.** No scripts, probes, or build artifacts in
`/tmp` or any session scratchpad directory — put them in `scratch/` (gitignored).

#### FOR CODEX:

Edit only the named file. Do not run tests. Stop after the edit, summarize it, and wait for approval before touching another file. Unless I say to you to run wild.

## What this repository is

A research workspace for multi-robot / swarm RL. Our own code lives in `swarm/`;
`research/` holds notes, papers, and two vendored reference codebases we learn
from but do not edit.

**The deliverable is a JAX multirotor simulator** fast enough to train swarm
policies on thousands of parallel environments — written from scratch, because
understanding the dynamics is the point of the exercise, not a side effect. What
it is for: **decentralized swarm interception** — N drones catching one agile
evader, each seeing only itself, its neighbours and the target, with no
communication between drones.

The task ladder, the platform backlog, and every pass/fail gate live in
[research/notes/goals.local.md](research/notes/goals.local.md) (gitignored, so a
fresh clone does not have it). **Read it before planning work.** The short version:

1. **M1** rigid body — 6-DOF state `(x, v, R, ω, rpm)`, RK4, motor lag, mixer.
   *Done — `swarm/sim/dynamics.py`, gated on the golden trajectories.*
2. **M2** control cascade — hover and A→B with hand-tuned gains, *no learning*.
   *Done — `swarm/sim/control.py`, all six rungs, gated on the cascade goldens.*
3. **M3** RL env — CTBR actions (collective thrust + body rates), hover.
   *Env done — `swarm/envs/hover.py`, gated on the cascade flying it. The gate on
   our PPO actually learning it is still open.*
4. **Learner** — ours, in `swarm/learn/`, not vendored. One update compiles to one
   XLA program and a Python loop calls it, so metrics and checkpoints exist while
   a run is going. *Done, and it learns `a_to_b`/`hover` — 0.085 m against the
   cascade's 0.021 m. Eval and the replay viewer are in.*
5. **Then** waypoint / recover, N drones on a circle swapping sides, 1v1
   interception, Nv1 interception — plus the platform each needs (runs and
   configs, wandb, evaluation against the cascade, replay rendering, attention
   neighbour encoder, RCI cluster).

Physics and control-stack reference: [ctu-mrs/mrs_multirotor_simulator](https://github.com/ctu-mrs/mrs_multirotor_simulator)
— `multirotor_model.hpp` (state + ODE) and `uav_system.hpp` (the six-controller
cascade). We reimplement it in JAX; we do not wrap the C++.


## swarm/ — our simulator

`conda run -n drone-swarm`. Pins in `requirements.txt` (jax 0.11.1, flax 0.12.9).

```bash
pytest                              # THE GATE — golden replay + analytic checks
python run/sim.py tumble --steps 500 --every 50        # open loop, you give throttles
python run/fly.py --target 3 -2 5                      # closed loop, the cascade flies
python run/train.py --task a_to_b --preset default --steps 5e6  # PPO -> runs/<name>/
python run/eval.py runs/<name>                         # -> runs/<name>/evals/latest/
python run/eval.py runs/<name> --policy cascade        # the cascade in the same seat
python run/replay.py                                   # serve + open the 3D viewer
bash tools/mrs_golden/build.sh      # regenerate tests/golden/ from the C++
```

- `sim/dynamics.py` — `Params`/`State` as `flax.struct.dataclass`,
  `step(state, throttle, params, dt)` with `throttle ∈ [0,1]⁴`. RK4 over the 18
  rigid-body states, then re-orthonormalize, then the exponential motor lag.
- `sim/control.py` — the cascade, read top to bottom in the order it runs:
  `position -> velocity -> acceleration -> attitude -> rate -> mixer`, plus
  `cascade_step` which chains all six. PID state is carried explicitly in
  `PIDState`; only the heading branch exists, not MRS's heading-rate branch.
- `envs/` — one file per task (`a_to_b.py` so far), each with `reset`, `step`,
  `get_obs` and `PRESETS`. Plain functions, no gymnax; auto-reset belongs to the
  trainer. `__init__.py` holds the shared `Obs` layout and `make(task, preset)`,
  which imports the task module by name. Every task carries a drone axis, `N = 1`
  included — the interface is spelled out in
  [research/notes/plan_t1t2.md](research/notes/plan_t1t2.md).
- `learn/` — our PPO. `ppo.py` (ActorCritic + `make`, one compiled update),
  `vecenv.py` (vmap, auto-reset, normalization), `runner.py` (the update loop, run
  directory, checkpoints, wandb), `config.py` (hyperparameters), `evaluate.py`
  (fixed eval seeds, mean actions, no auto-reset). Knows nothing about drones: it
  takes an env *module* and one network per role.
- **Eval writes into the run it measures**, at `runs/<run>/evals/<name>/`:
  `eval.json` (summary + checkpoint + seeds + commit), `episodes.npz` (per-episode
  arrays), `trajectory.npz` (the first 8 episodes, full state), `header.json`
  (task, preset, arena, roles — the viewer never imports the env). Eval holds the
  scene frozen once an episode ends; the env does not, because with `N = 1` the
  episode ends the same step and the trainer resets it.
- `tools/viewer/index.html` — one file, three.js from a CDN, no build step. Parses
  the `.npz` in JS (`np.savez` writes a *stored* zip, so no inflate library).
  `run/replay.py` serves the repo and lists every eval at `/evals.json`, so the
  viewer opens with a dropdown; dragging the files onto the page still works.
- Written for **one drone**; batching is `jax.vmap` at the env boundary. No Python
  branches on array values, no `.item()`, no value-dependent shapes —
  `test_vmap_matches_python_loop` enforces it.
- Every design decision, deviation, and "revisit later" lives in
  [research/notes/choices.md](research/notes/choices.md). **Read it before
  changing the physics, and update it when a decision changes.**

**The golden trajectories are the gate.** `tools/mrs_golden/` runs the patched C++
once per scenario and dumps the full state at every step to
`tests/golden/*.csv`; the test replays the same input in JAX and demands 1e-9
(worst measured: 1.1e-13). Five open-loop scenarios drive the plant directly;
`rate_step`, `attitude_step`, `velocity_step` and `position_step` come from
`UavSystem` and check the controllers too, so their command columns hold the
reference, not motor throttles. `research/code_sources/` is gitignored, so those CSVs
are the only copy of the reference in a fresh clone — never regenerate them to
make a failing test pass. `conftest.py` turns on `jax_enable_x64` because the C++
is double precision.

One deliberate deviation from the C++, patched by `tools/mrs_golden/transpose.diff`:
MRS re-orthonormalizes as `R·L⁻¹`, which is not orthonormal and drifts ~3e-3 off
SO(3); we use `R·L⁻ᵀ`. Hover is not a gate — it is an equilibrium, so a
transposed allocation matrix or a missing `ω × Jω` sails through it. Only
`tumble` and `spin_down` exercise the torque path.

## Layout and conventions

```
swarm/             our implementation
├── sim/
│   ├── dynamics.py   the plant: Params, State, derivative, RK4 step
│   └── control.py    the six-rung cascade + cascade_step
├── envs/
│   ├── __init__.py   the Obs layout + make(task, preset)
│   └── a_to_b.py     task 1: reset, step, get_obs, PRESETS
└── learn/            our PPO: ppo.py, vecenv.py, runner.py, config.py, evaluate.py
run/
├── sim.py          open-loop rollout CLI
├── fly.py          closed-loop cascade CLI
├── train.py        PPO training CLI
├── eval.py         measure a checkpoint (or the cascade) -> runs/<run>/evals/<name>/
└── replay.py       serve the repo, list every eval, open the viewer
tests/
├── test_dynamics.py  the M1 gate
├── test_control.py   the M2 gate
├── test_env.py       the env gate: the cascade flies the task
├── test_learn.py     the trainer gate: it learns, and it is reproducible
├── test_runner.py    the run gate: a run says what produced it and resumes
├── test_eval.py      the eval gate: the cascade in the seat scores the M2 number
└── golden/           C++ reference trajectories (CSV) + params.txt
tools/
├── mrs_golden/     C++ harness that generated tests/golden/
└── viewer/         index.html — three.js replay, reads trajectory.npz directly
research/
├── notes/          working notes (Czech is fine here)
│   ├── goals.local.md  task ladder, platform backlog, gates (GITIGNORED)
│   ├── choices.md    every design decision and deviation from the C++
│   ├── lit_review.md the papers, sorted by milestone
│   ├── learning/     explainers: ODE/RK4, plant, cascade, sim loop
│   └── experiments/  one tracked <exp>.md per experiment
├── papers/         PDFs (gitignored) + .md transcripts and the index (tracked)
└── code_sources/   vendored reference code, NOT our implementation — GITIGNORED
    ├── mrs_multirotor_simulator/  the C++ we reimplement
    ├── PPO_example/     JAX/gymnax single-agent PPO (supervisor's teaching repo)
    └── quad-swarm-rl/   PyTorch/Sample-Factory quadrotor swarm (upstream clone)
```

- **Papers: grep the transcript, don't open the PDF.** Every PDF has a
  `pdftotext -layout` transcript as a sibling `.md`; open the PDF only for
  figures. [bibliography_index.md](research/papers/bibliography_index.md) is the
  entry point — a tracked table of paper → files → one-line summary. Add a row
  whenever a PDF is added, and generate its transcript with
  `pdftotext -layout <file>.pdf <file>.md`.
- Only the PDFs are gitignored. Transcripts and `bibliography_index.md` are
  tracked, so a fresh clone can grep every paper without the binaries.
- **Some material under `research/papers/` is unpublished and must stay local.**
  Those files are excluded through `.git/info/exclude`, which is itself untracked
  — so the exclusion names them and no tracked file ever does. Do not add them to
  the index table, do not quote, summarise, or name them in any tracked file (this
  one included), and never `git add -f` one. Only add an index row for a paper
  that is publicly published. See `CLAUDE.local.md` (untracked) for specifics.
- `quad-swarm-rl` is a **live clone with its own `.git` and an `origin` pointing at
  the upstream GitHub repo**. Never commit or push from inside it. Treat every
  `code_sources/` tree as a read-only reference unless the user says otherwise.

## PPO_example — JAX learner (the baseline we build on)

Has its own [CLAUDE.md](research/code_sources/PPO_example/CLAUDE.md), which is
authoritative when working inside that directory. Read it plus its `README.md`,
`ppo/train.py`, `envs/interceptor2d.py` — the whole thing is ~1,000 lines.

PureJaxRL-style PPO where the entire training loop is a single XLA program.
`ppo/train.py` (ActorCritic + `make_train`), `ppo/wrappers.py`, `ppo/config.py`
(every hyperparameter), `envs/interceptor2d.py` (env + reward + `PRESETS`),
`run/{train,eval,plot}.py` (CLI → `runs/<preset>_s<seed>/`), `cluster/` (RCI SLURM).

```bash
# PPO_example/CLAUDE.md names a conda env `agiflight`; it does NOT exist here.
# Use `conda run -n drone-swarm`.
python tests/test_smoke.py     # ~15 s, THE GATE: env checks + a learning gate
python run/train.py --steps 3e6 --num-envs 256 --num-steps 64 --num-minibatches 8   # CPU
python run/train.py            # GPU defaults: 4096 envs, 50M steps
python run/eval.py runs/straight_s0
python run/plot.py runs/straight_s0
```

Smoke tests must be green before any change is called done and before any GPU
job. Writing a new env: follow the contract in
[envs/README.md](research/code_sources/PPO_example/envs/README.md) — gymnax-style
`flax.struct` state with identical dtypes from `reset_env`/`step_env`, static
params as `pytree_node=False`, `info` carrying `terminated`/`truncated`/
`terminal_obs`, and all values chosen in `PRESETS`, never in `run/`.

The traps that repo already paid for: `env.step` auto-resets on done (log
terminal data *before* it, or use `step_env`); obs-normalization stats are part
of the policy and live in `params.pkl`; timeout is truncation (bootstrap V),
capture/miss are terminals; small batch + `ent_coef=0.01` makes entropy blow up
(keep ≥2048 envs on GPU). Cluster gotchas are in `cluster/README.md` — deploy
source by git only, verify `git log -1` on the cluster after every pull, and
`pip install --user` without `--no-deps` shadows the container's CUDA jax.

## quad-swarm-rl — PyTorch swarm reference (attention)

Separate stack, separate env: Python ≥3.11, `pip install -e .` (numpy 1.26,
torch 2.5, sample-factory, numba, gym/gymnasium). Two halves:

- `gym_art/quadrotor_multi/` — the simulator: `quadrotor_dynamics.py`,
  `quadrotor_multi.py` (the multi-agent env), `collisions/`, `obstacles/`,
  `aerodynamics/` (downwash), and `scenarios/` — one file per task
  (`static_same_goal`, `swarm_vs_swarm`, `mix`, …), each subclassing
  `scenarios/base.py`. Numba-accelerated hot paths (`--quads_use_numba`).
- `swarm_rl/` — the training glue on top of Sample Factory APPO:
  `models/quad_multi_model.py` + `models/attention_layer.py` (the neighbor
  encoder — this is the "swarming with attention" piece goal 3 points at),
  `env_wrappers/reward_shaping.py`, `runs/` (Sample Factory launcher scripts
  = experiment configs), `sim2real/` (PyTorch → C for Crazyflie firmware).

```bash
bash train.sh                                    # baseline multi-drone APPO run
python -m sample_factory.launcher.run --run=swarm_rl.runs.single_quad.single_quad \
    --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --quads_render=True \
    --train_dir=... --experiment=... --quads_view_mode topdown
./run_tests.sh                                   # = python -m unittest (all tests)
python -m unittest gym_art.quadrotor_multi.tests.test_multi_env   # single test module
tensorboard --logdir=./                          # from the experiment folder
```

Key training flags to recognise: `--quads_mode` (scenario), `--quads_obs_repr`,
`--quads_neighbor_encoder_type=attention`, `--quads_neighbor_visible_num`,
`--quads_use_obstacles`, `--quads_use_downwash`, `--replay_buffer_sample_prob`.
