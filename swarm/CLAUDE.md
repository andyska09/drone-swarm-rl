# swarm/ — our simulator

## Read before you change

- **The physics, or a decision that all tasks share:**
  [design_choices.md](../research/notes/design_choices.md). It holds every design
  decision for the simulator and for what all tasks share, every deviation, and every
  "revisit later". Update it when a decision changes.
- **A new task:** [env_contract.md](../research/notes/tasks/env_contract.md), the contract
  that every task file meets: what a task file holds, the obs, the end flags and the
  metrics.
- **A reward, an event or a metric:** the task spec,
  `research/notes/tasks/task_<name>.md`. It holds the task's own model: its events, reward
  and metrics. The shared model and its words, the name rule `<name>_<r>` included, are
  in [defaults.md](../research/notes/tasks/defaults.md). Why a distance cost leaves a
  drone 0.2-0.6 m short of its goal is in
  [reward_terms.md](../research/notes/literature/reward_terms.md).

## The code

- `sim/dynamics.py` — `Params`/`State` as `flax.struct.dataclass`,
  `step(state, throttle, params, dt)` with `throttle ∈ [0,1]⁴`. RK4 over the 18
  rigid-body states, then re-orthonormalize, then the exponential motor lag.
- `sim/control.py` — the cascade, read top to bottom in the order it runs:
  `position -> velocity -> acceleration -> attitude -> rate -> mixer`, plus
  `cascade_step` which chains all six. PID state is carried explicitly in
  `PIDState`; only the heading branch exists, not MRS's heading-rate branch.
- `sim/guidance.py` — the guidance laws behind chase's pilots: FRPN, APF and the flown
  paths. Each one gives a world acceleration.
- `envs/` — one file per task, plain functions, no gymnax; auto-reset belongs to the
  trainer. Every task carries a drone axis, `N = 1` included.
- **`envs/core.py` holds everything the tasks share**: `Common` (the `EnvParams`
  fields every task has, which each task's `EnvParams` inherits), and the helpers for
  the flight of one policy step, the obs, the start, the contact rules and the pilots.
  Read its function list before you write a helper. A task file holds only what makes
  that task different. Do not copy a helper into a task file.
- **No task names a drone by number.** `roles` is static, so a task reads its own
  index arrays from it — see `chase._sides`. That is what lets a longer `roles`
  add more pursuers with no other change.
- **`EnvParams.roles` is the field everything hangs off.** It is
  `pytree_node=False`; its length is the drone count, and `role_slices` in
  `envs/__init__.py` gives each distinct name in it its drones.
- `learn/` — our PPO and the training loop. `ppo.py` and `vecenv.py` know nothing about
  drones: they take an env *module* and one network per role.
- **Every role is its own learner.** One `TrainState` per role, so gradient
  clipping, Adam state, the adaptive lr, advantage scaling and reward scaling never
  mix two sides of a game. `cfg.generations` above 0 trains chase with pools; the
  design is in [task_chase.md](../research/notes/tasks/task_chase.md), Training.
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
- `conftest.py` turns on `jax_enable_x64` because the C++ is double precision.
- One deliberate deviation from the C++, patched by `tools/mrs_golden/transpose.diff`:
  MRS re-orthonormalizes as `R·L⁻¹`, which is not orthonormal and drifts ~3e-3 off
  SO(3); we use `R·L⁻ᵀ`. Hover is not a gate — it is an equilibrium, so a
  transposed allocation matrix or a missing `ω × Jω` sails through it. Only
  `tumble` and `spin_down` exercise the torque path.
