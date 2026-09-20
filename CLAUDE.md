# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT NOTE TO ALL AI CHATBOTS

When talking to me start the message with "TARS:"

## Design Philosophy

When building always do only what I ask for. Do not add unwanted features. You can ask about them in the chat or raise it with me when chatting with me, but do not put them into code right away. 

When designing code, keep it simple. By this I mean the logic should be easy to follow and clean - lines of code is wrong metric for this. This requires you to think about it rather then just bust out code. If there two approaches possible with no clear better one - ask me for input. Always.

For all your responses always use [.claude/output-styles/simple-english.md](.claude/output-styles/simple-english.md) output style. 

### Style

For code:
- Blunt, minimal code. No base classes, protocols, registries. One file per
  concern, readable top to bottom.
- When implementing stuff KEEP IT SIMPLE. This important. I do not want to read 1000 lines of diffs, I want to look at the change and know what it does. 
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
policies on thousands of parallel environments — written from scratch. What
it is for: **decentralized swarm interception** — N drones catching one agile
evader, each seeing only itself, its neighbours and the target, with no
communication between drones.

Two tasks run today: `a_to_b` (one drone flies to a point and holds it) and
`circle_swap` (N drones on a circle swap to the opposite side without touching).
Chase, then interception, come next.

The work cycle for a task: write it in `swarm/envs/`, fly it with the cascade to
prove it is possible at all, train a policy with `run/train.py`, score both in the
same seat with `run/eval.py`, then watch it in `run/replay.py`.

The task ladder, the platform backlog, what is done, and every pass/fail gate
live in [research/notes/goals.local.md](research/notes/goals.local.md)
(gitignored, so a fresh clone does not have it). **Read it before planning work.**
Do not copy it here — this file goes stale, that one does not.

Physics and control-stack reference: [ctu-mrs/mrs_multirotor_simulator](https://github.com/ctu-mrs/mrs_multirotor_simulator)
— `multirotor_model.hpp` (state + ODE) and `uav_system.hpp` (the six-controller
cascade). We reimplement it in JAX; we do not wrap the C++.


## swarm/ — our simulator

Every command below needs the `conda run -n drone-swarm` prefix. Pins in
`requirements.txt` (jax 0.11.1, flax 0.12.9, numpy 2.5.2, matplotlib 3.11.2,
pytest 9.1.1).

No pytest alone, since there is also code in other folders. Always need to specify pytest tests/ 

```bash
pytest tests/test_control.py                           # one file
pytest tests/test_dynamics.py -k tumble                # one test
python run/sim.py tumble --steps 500 --every 50        # open loop, you give throttles
python run/fly.py --target 3 -2 5                      # closed loop, the cascade flies
python run/train.py --task a_to_b --preset default --steps 5e6  # PPO -> runs/<name>/
python run/train.py --set gamma=0.99 --set ent_coef=0.01        # any TrainConfig field
python run/eval.py runs/<name>                         # -> runs/<name>/evals/latest/
python run/eval.py runs/<name> --policy cascade        # the cascade in the same seat
python run/replay.py                                   # serve + open the 3D viewer
python run/plot.py runs/<name>/evals/latest            # time plots of one episode
bash tools/mrs_golden/build.sh      # regenerate tests/golden/ from the C++
```

Presets: `a_to_b` has `default` and `hover`; `circle_swap` has `pair` and
`circle8`; `chase` has `default` (1 pursuer) and `three`. A preset picks env
values only — hyperparameters live in `learn/config.py` and are overridden with
`--set`.

- `sim/dynamics.py` — `Params`/`State` as `flax.struct.dataclass`,
  `step(state, throttle, params, dt)` with `throttle ∈ [0,1]⁴`. RK4 over the 18
  rigid-body states, then re-orthonormalize, then the exponential motor lag.
- `sim/control.py` — the cascade, read top to bottom in the order it runs:
  `position -> velocity -> acceleration -> attitude -> rate -> mixer`, plus
  `cascade_step` which chains all six. PID state is carried explicitly in
  `PIDState`; only the heading branch exists, not MRS's heading-rate branch.
- `envs/` — one file per task (`a_to_b.py`, `circle_swap.py`, `chase.py`), each with
  `reset`, `step`, `get_obs`, `reference`, `compute_reward`, `RewardConfig`,
  `EnvParams` and `PRESETS`. Plain functions, no gymnax; auto-reset belongs to the
  trainer. `__init__.py` holds the shared `Obs` layout, `flat`, `take`,
  `make(task, preset)` and **the env contract every task file must meet** — read it
  before writing a task. Every task carries a drone axis, `N = 1` included, and each
  has a note beside it in `research/notes/task_<name>.md`.
- **`envs/core.py` holds everything the tasks share**: `Common` (the 15 `EnvParams`
  fields every task has, which each task's `EnvParams` inherits), `fly` (the rate
  loop and the plant for one policy step), `own_obs`/`target_obs`/`neighbors`,
  `spawn`, `hits`, `is_dead`, and the CTBR action conversion. A task file holds only
  what makes that task different. Do not copy a helper into a task file.
- **No task names a drone by number.** `roles` is static, so a task reads its own
  index arrays from it — see `chase._sides`. That is what lets `PRESETS["three"]`
  add two more pursuers with no other change.
- **Every task uses the same observation shape**: `Obs(own, others, others_mask,
  target)`. `own` is body-frame velocity, `R`, `omega` and height. `others` holds
  the K nearest drones, whatever their role — body-frame offset, body-frame
  velocity, and a 0/1 flag saying whether that drone shares my role, so a teammate
  and an opponent are told apart by a number, not by sitting in different blocks.
  `target` is the body-frame offset to a goal point, and a task with no goal point
  gives it width 0 — `chase` does. `others_mask` is a 0/1 column, and `flat` zeroes
  the masked rows before it concatenates, so an empty seat becomes zeros and never
  changes the vector width. Add a task, not a new obs format.
- The reward is a per-step cost, not a bonus: `-policy_dt * (distance + spin + …)`
  with a one-off crash penalty. `circle_swap` adds a proximity cost and an upright
  term. **`chase` keeps three words apart and never mixes them**: `crash` is a wall
  or the ground, `collPP` is a pursuer hitting a pursuer (both end the episode), and
  `collPE` is a pursuer hitting the evader, which costs per step and does not end it
  — Gavin's names, used for the `RewardConfig` field, the function and the `info`
  key alike. **This reward parks the drone 0.2-0.6 m short of the goal** — the open
  problem, with the fixes other papers use, is in
  [research/notes/parking_error.md](research/notes/parking_error.md).
- **`EnvParams.roles` is the field everything hangs off.** It is
  `pytree_node=False`; its length is the drone count, and `role_slices` in
  `learn/ppo.py` turns each distinct name in it into its own set of weights.
- `learn/` — our PPO. `ppo.py` (ActorCritic + `make`, one compiled update),
  `vecenv.py` (vmap, auto-reset, normalization), `runner.py` (the update loop, run
  directory, checkpoints, wandb), `config.py` (hyperparameters), `evaluate.py`
  (fixed eval seeds, mean actions, no auto-reset). Knows nothing about drones: it
  takes an env *module* and one network per role.
- **Every role is its own learner.** One `TrainState` per role, so gradient
  clipping, Adam state, advantage scaling and reward scaling never mix two sides of
  a game. `cfg.train_roles` names the roles that take a step; empty means all. That
  is how task 4 alternates — freeze the pursuer, train the evader, switch.
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
- Every design decision, deviation, and "revisit later" lives in
  [research/notes/choices.md](research/notes/choices.md). **Read it before
  changing the physics, and update it when a decision changes.**
- `conftest.py` turns on `jax_enable_x64` because the C++
is double precision.
- One deliberate deviation from the C++, patched by `tools/mrs_golden/transpose.diff`:
MRS re-orthonormalizes as `R·L⁻¹`, which is not orthonormal and drifts ~3e-3 off
SO(3); we use `R·L⁻ᵀ`. Hover is not a gate — it is an equilibrium, so a
transposed allocation matrix or a missing `ω × Jω` sails through it. Only
`tumble` and `spin_down` exercise the torque path.

## Tests — what each one gates

| file | gate |
|---|---|
| `test_dynamics.py` | the plant against the goldens |
| `test_control.py` | the cascade against the goldens |
| `test_env.py` | the cascade flies `a_to_b`; `chase` scales with `roles`, one catch pays every pursuer, a net on another pursuer kills both, and `scripted` decides who flies the evader |
| `test_learn.py` | it learns, the same seed gives the same numbers, and a frozen role does not move |
| `test_runner.py` | a run says what produced it and resumes |
| `test_eval.py` | the cascade in the policy's seat scores the cascade's number |

## research/ and the reference code

- `notes/` — working notes, Czech is fine. `goals.local.md` (GITIGNORED) holds
  the task ladder and the gates. `choices.md` holds every design decision.
  `task_<name>.md` is one note per task. `parking_error.md` is the open reward
  question, with the reward and observation terms every reference paper uses.
  `comparisons.md` puts our observation layout beside both reference repos.
  `lit_review.md` sorts the papers by milestone. `learning/` are explainers,
  `experiments/` is one file per run.
- `code_sources/` (GITIGNORED) — reference code, **not ours**:
  `mrs_multirotor_simulator/` (the C++ we reimplement), `PPO_example/` (the
  supervisor's JAX PPO), `quad-swarm-rl/` (PyTorch swarm, upstream clone).
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
  encoder — this is the "swarming with attention" piece we want),
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
