# Literature review — RL control for multirotors → pursuit-evasion → multi-agent PE

Three layers, narrowing. Each entry: what it actually does, and why it matters
here. Papers we hold PDFs for are also in
[bibliography_index.md](../papers/bibliography_index.md).

---

## 0. Existing surveys — read these instead of re-deriving the map

- **Mu et al. 2023, *A survey of the pursuit–evasion problem in swarm intelligence*** —
  [10.1631/FITEE.2200590](https://doi.org/10.1631/FITEE.2200590) (open access)
  The one survey that covers our exact target. Splits PE three ways: game
  theory (Isaacs), control theory + AI, bio-inspired. Start here.
- **2025, *A Survey on UAV Control with
  Multi-Agent Reinforcement Learning*** — Drones 9(7):484,
  [mdpi.com/2504-446X/9/7/484](https://www.mdpi.com/2504-446X/9/7/484)
  MARL-for-UAV specifically: CTDE, credit assignment, scalability.
- **Bhattacharya et al. 2024, *A Survey of Offline and Online Learning-Based
  Algorithms for Multirotor UAVs*** — [arXiv:2402.04418](https://arxiv.org/abs/2402.04418)
  Single-vehicle learning-based control, broad and shallow.
- **2026, *A Review of RL for Multirotor UAVs from a Hierarchical Control
  Perspective*** — [10.3390/drones10060448](https://doi.org/10.3390/drones10060448)
  Organises the field by *where in the cascade the policy is inserted* —
  stabilisation / perception-action / task planning. Directly the axis our
  "policy insertion point is a config value" commitment sits on.

---

## 1. RL control for a single drone

### 1.1 The lineage

- **Hwangbo et al. 2017, *Control of a Quadrotor with RL*** —
  [arXiv:1707.05110](https://arxiv.org/abs/1707.05110)
  First convincing motor-level neural controller: recovers from being thrown by
  hand. Established that RL can own the innermost loop.
- **Molchanov et al. 2019, *Sim-to-(Multi)-Real*** —
  [arXiv:1903.04628](https://arxiv.org/abs/1903.04628)
  One policy, several physical quadrotors. Domain randomisation over mass,
  inertia, motor lag, delay is what makes the transfer survive. This is the
  ancestor of `quad-swarm-rl`.
- **Song et al. 2023, *Reaching the Limit in Autonomous Racing: Optimal Control
  vs RL*** — Science Robotics,
  [arXiv:2310.10943](https://arxiv.org/abs/2310.10943)
  Why RL beats MPC at the limit: not better control, but the freedom to optimise
  a task-level objective directly instead of a tracking error on a precomputed
  trajectory.
- **Kaufmann et al. 2023, *Champion-level drone racing using deep RL* (Swift)** —
  Nature 620:982, [10.1038/s41586-023-06419-4](https://www.nature.com/articles/s41586-023-06419-4)
  Beats human world champions. Sim-trained, plus a residual observation/dynamics
  model fit on real flights to close the gap. The field's existence proof.

### 1.2 The design decisions, settled empirically

- **Kaufmann et al. 2022, *A Benchmark Comparison of Learned Control Policies
  for Agile Quadrotor Flight*** — [arXiv:2202.10796](https://arxiv.org/abs/2202.10796)
  Head-to-head on identical observations: linear velocity vs **CTBR** vs
  single-rotor thrusts. CTBR wins on the sim-to-real tradeoff. **This is why
  M3's action space is collective thrust + body rates** — it justifies inserting
  the policy at the rate controller rather than at the motors.
- **SimpleFlight 2024, *What Matters in Learning a Zero-Shot Sim-to-Real RL
  Policy for Quadrotor Control?*** — [arXiv:2412.11764](https://arxiv.org/abs/2412.11764)
  Ablation over exactly our choices. Rotation matrix ≫ quaternion (~64%);
  action-smoothness term `‖u_t − u_{t−1}‖₂` at weight 0.4 with both terms
  normalised; 100 Hz control beats 50 Hz.
- **Eschmann et al. 2023, *Learning to Fly in Seconds*** —
  [arXiv:2311.13081](https://arxiv.org/abs/2311.13081)
  Asymmetric actor-critic + a very fast custom simulator: 18 s of training on a
  laptop to a flying RPM-level policy, deployable on a microcontroller. The
  argument that simulator throughput *is* the research variable — which is the
  whole premise of this repo.
- **Yu et al. 2025, *Equivariant RL Frameworks for Quadrotor Low-Level Control*** —
  [arXiv:2502.20500](https://arxiv.org/abs/2502.20500)
  Encodes the rotational and reflectional symmetries of quadrotor dynamics into
  the network, removing redundancy the policy would otherwise have to learn.
  Sample efficiency only — reach for it if training turns out slow.
- **Towards Task-Oriented Flying 2025** — [arXiv:2504.15129](https://arxiv.org/abs/2504.15129)
  Design guidelines connecting task definition ↔ training setup ↔ deployment.
  Useful checklist when writing `PRESETS`.

### 1.3 Simulators — the competition

The GPU tier — what anyone training a swarm today actually runs on.

| what | backend | diff | multi-drone | reported speed | note |
|---|---|---|---|---|---|
| [Crazyflow](https://arxiv.org/abs/2606.01478) ([code](https://github.com/learnsyslab/crazyflow)) | **JAX/XLA**; MuJoCo only for render, raycast, contact | yes | yes, `n_worlds × n_drones` | 914M steps/s (RTX 4090, 262K worlds); ~700M at 1M worlds; 4.2M drones | **Closest thing to what we are building.** Schoellig's lab; successor to `gym-pybullet-drones`. Layered control (motor → thrust+torque → thrust+attitude → position). Steal: one monolithic PyTree for drone + controller state, `lax.scan` over steps. Their motor model is asymmetric up/down; ours is a single symmetric τ. Crazyflie model fit from real flights — not the MRS plant, and no six-stage cascade. |
| [OmniDrones](https://arxiv.org/abs/2309.12825) | Isaac Sim / Omniverse | no | yes | high, not comparable | The biggest **task suite**: hover, track, fly-through, transport. The reference for "what an env suite looks like". Tied to NVIDIA drivers. `thu-uav`'s PE work (§3.2) is built on it. |
| [Aerial Gym](https://arxiv.org/abs/2305.16510) (RA-L version [arXiv:2503.01471](https://arxiv.org/abs/2503.01471), [site](https://ntnu-arl.github.io/aerial_gym_simulator/)) | Isaac Gym (PyTorch) | no | yes | ~3.8M steps/s | NTNU. Heavy on depth/LiDAR rendering. |
| [DiffAero](https://arxiv.org/abs/2509.10247) ([code](https://github.com/flyingbitac/diffaero)) | PyTorch, GPU-native | yes | limited | "orders of magnitude" over CPU | Physics **and** rendering on GPU. Built for analytic policy gradients (see §2). |
| [MuJoCo-Drones-Gym](https://arxiv.org/abs/2606.08039) | MJX | yes | yes | GPU/TPU batched | 2026. Vision-based quadrotor RL. Comparison point for env API only. |
| [RLtools / l2f](https://github.com/rl-tools/learning-to-fly) ([paper](https://arxiv.org/abs/2311.13081)) | C++ templates, CPU or GPU | no | single | ~5 months of flight/s on a laptop GPU | 18 s to a flying policy on an M1; deploys to a microcontroller. The throughput-is-the-research-variable argument. |
| [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world) | Taichi → CUDA | partly | yes | tens of M frames/s | General robotics, not drone-specific. Watch, do not adopt. |

**The speed column is not a ranking.** Every row uses a different GPU, drone count,
control rate and definition of "step". It only shows the size class.

The CPU tier is still cited and nobody trains swarms on it: `gym-pybullet-drones`,
PyFlyt, RotorPy, Flightmare, AirSim, RotorS, Gazebo, and the MRS C++ simulator we
reimplement. AirSim and Flightmare are unmaintained. Mapped in
**Dimmig et al. 2024, *Survey of Simulators for Aerial Robots*** —
[arXiv:2311.02296](https://arxiv.org/abs/2311.02296), RA-M,
[10.1109/MRA.2024.3433171](https://doi.org/10.1109/MRA.2024.3433171).

---

## 2. Pursuit-evasion, 1v1 — quadrotors, not point masses

The dividing line in this literature: **point-mass/Dubins kinematics** (most of
it, decades deep) vs **full 6-DOF quadrotor with a real control stack** (the
last ~2 years, small and directly relevant).

- **Isaacs 1965, *Differential Games*** — the origin. Homicidal chauffeur,
  Hamilton-Jacobi-Isaacs equation, the notion of a barrier surface separating
  capture from escape. Everything geometric downstream is this.
- **RL in pursuit-evasion differential games: safety, stability, robustness 2025** —
  [arXiv:2507.19516](https://arxiv.org/abs/2507.19516)
  The bridge between HJI theory and learned policies; what guarantees survive.
- **Xiao & Feroskhan 2023, *Learning Multi-Pursuit Evasion for Safe Targeted
  Navigation of Drones* (AMS-DRL)** —
  [arXiv:2304.03443](https://arxiv.org/abs/2304.03443)
  From the *evader's* side: reach a goal while N pursuers attack. Asynchronous
  multi-stage training over a bipartite pursuer/evader graph, with a Nash
  equilibrium convergence argument.
- **Learned Controllers for Agile Quadrotors in Pursuit-Evasion Games 2025** —
  [arXiv:2506.02849](https://arxiv.org/abs/2506.02849)
  **The closest paper to our end goal.** 1v1, both sides learned, CTBR actions,
  full quadrotor dynamics. Names the three failure modes of competitive
  self-play — non-stationarity, catastrophic forgetting, strategy cycling —
  and answers with AMSPBH: PSRO-style population, new policies trained as best
  responses to a *mixture* of past opponents rather than the current one.
- **Agile Interception of a Flying Target using Competitive RL 2026** —
  [arXiv:2603.16279](https://arxiv.org/abs/2603.16279)
  Net-carrying interceptor vs agile target, both PPO, both CTBR. Notable for us:
  **their simulator is quadrotor dynamics + low-level cascade implemented in
  JAX for GPU-parallel rollouts** — i.e. exactly this repo's deliverable, used
  for exactly this task. Read the sim section closely.
- **Learning Agile Intruder Interception using Differentiable Quadrotor
  Dynamics 2026** — [arXiv:2607.02472](https://arxiv.org/abs/2607.02472)
  Drops RL for **analytic policy gradients** through differentiable dynamics.
  Observation is only the 3D bearing unit vector (what a monocular camera
  actually gives you), interception up to 10 m/s, +30% over point-mass
  baselines. JAX gives us this gradient for free — worth remembering.
- **Pliska, Vrba, Báča, Saska 2024, *Towards Safe Mid-Air Drone Interception*** —
  RA-L, [arXiv:2405.13542](https://arxiv.org/abs/2405.13542),
  [project page](https://mrs.fel.cvut.cz/towards-interception)
  Classical (non-RL) counterpart from the MRS group whose C++ simulator we are
  reimplementing: net-equipped interceptor, EPN guidance, onboard estimation.
  The baseline any learned policy here gets compared against.

---

## 3. Multi-agent pursuit-evasion

### 3.1 MARL machinery you will need

- **Yu et al. 2021, *The Surprising Effectiveness of PPO in Cooperative
  Multi-Agent Games* (MAPPO)** — [arXiv:2103.01955](https://arxiv.org/abs/2103.01955)
  Shared-parameter PPO with a centralised critic beats off-policy MARL on the
  standard benchmarks with almost no tuning. Practical consequence: **the PPO in
  `PPO_example` is already 90% of a MARL algorithm** — vmap over agents, feed
  the critic global state.
- **Baker et al. 2019, *Emergent Tool Use From Multi-Agent Autocurricula*** —
  [arXiv:1909.07528](https://arxiv.org/abs/1909.07528)
  Hide-and-seek. Six rounds of strategy/counter-strategy emerge from
  competition alone. The canonical demonstration that the opponent *is* the
  curriculum.
- **Huang et al. 2024, *Collision Avoidance and Navigation for a Quadrotor Swarm
  Using End-to-end DRL*** — [arXiv:2309.13285](https://arxiv.org/abs/2309.13285),
  PDF in `research/papers/`
  The paper behind `quad-swarm-rl`. **Attention over the K nearest neighbours**
  — the mechanism M6 points at — plus SDF obstacles and collision-episode
  replay. Also the source of the cost-style reward shaping we copy.

### 3.2 Cooperative capture

- **Chen et al. 2023, *A Dual Curriculum Learning Framework for Multi-UAV
  Pursuit-Evasion in Diverse Environments*** —
  [arXiv:2312.12255](https://arxiv.org/abs/2312.12255)
  N pursuers, one **faster** evader, obstacles, drone dynamics. Curriculum on
  both axes at once — task difficulty and environment complexity — because a
  fast evader makes the naive reward unlearnable from scratch.
- **Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments 2024/25** —
  RA-L, [arXiv:2409.15866](https://arxiv.org/abs/2409.15866)
  The strongest sim-to-real result in multi-UAV PE: real dynamics and physical
  constraints (not fixed-altitude 2D), an evader-prediction network for partial
  observability, an adaptive environment generator for generalisation.
- **AMBUSH 2026, *Collaborative Capture in Complex Environments with Neural
  Acceleration*** — [arXiv:2607.01029](https://arxiv.org/abs/2607.01029)
  Argues a single parameterised primitive — ambush — lets slower pursuers catch
  a faster evader in cluttered maps, where end-to-end RL fails. A useful
  antagonist to the pure-RL framing.
- **Intercepting an Agile Target with Net-Carrying Drones using Competitive
  MARL 2026** — [arXiv:2607.05939](https://arxiv.org/abs/2607.05939)
  Team of pursuers vs learned evader, MAPPO + **Prioritized Fictitious
  Self-Play**, CTBR. The multi-agent sequel to arXiv:2603.16279 and the most
  complete statement of the end target: our M5/M6 plus a learned opponent.
- **Decentralized Consensus Inference-based Hierarchical RL for
  Multi-Constrained UAV Pursuit-Evasion 2025** —
  [arXiv:2506.18126](https://arxiv.org/abs/2506.18126)
  The swarm is the *evader* side: hold formation coverage over target zones
  while collectively escaping predators, under a communication limit.
  Hierarchical — a high level picks the intent, a low level flies it — which is
  what the configurable insertion point buys us.
- **TERL 2025, *Large-Scale Multi-Target Encirclement Using Transformer-Enhanced
  RL*** — [arXiv:2503.12395](https://arxiv.org/abs/2503.12395)
  Transformer neighbour encoding at swarm scale — the scaling story for the
  attention encoder.

### 3.3 The geometric baselines you must beat

Not RL, but this is what reviewers compare against, and what gives the reward
shaping its shape:

- **Apollonius circle** methods — the locus of points a pursuer and a faster
  evader reach simultaneously; capture requires the pursuers' Apollonius circles
  to tile a closed encirclement around the evader. See
  [Apollonius partitions based PE strategies via MARL](https://www.sciencedirect.com/science/article/abs/pii/S0925231225003157)
  (Neurocomputing 2025) for the hybrid, and
  [Collaborative PE of multi-UAVs based on Apollonius circle with obstacles](https://www.tandfonline.com/doi/full/10.1080/09540091.2023.2168253).
- **Voronoi partition / area-minimisation** pursuit — pursuers shrink the
  evader's Voronoi cell monotonically.
- **EPN / proportional-navigation guidance** — the 1v1 classical baseline
  (arXiv:2405.13542 above).

---

## 4. What the field agrees on

1. **CTBR is the action space** for anything agile. Motor-level buys marginal
   agility and costs sim-to-real robustness; velocity-level throws away the
   dynamics you need. (2202.10796, confirmed by every 2026 interception paper.)
2. **Throughput is the bottleneck, so simulators went GPU-native and often
   differentiable.** JAX / Isaac / MJX, 10⁶–10⁸ steps/s. Nobody trains these on
   a CPU physics engine any more.
3. **Naive self-play does not converge.** Every competitive PE paper since 2025
   carries an opponent-sampling scheme — PSRO population + hedge sampling
   (2506.02849), prioritized fictitious self-play (2607.05939), asynchronous
   multi-stage (2304.03443). Pick one before writing the training loop, not
   after it diverges.
4. **A faster evader breaks the naive reward.** The answer is curriculum
   (2312.12255), structural priors (AMBUSH, Apollonius), or both.
5. **Attention over K nearest neighbours** is the settled neighbour encoder:
   permutation-invariant and agent-count-agnostic. (2309.13285, TERL.)

## 5. Gaps

- Almost no multi-agent PE work uses **full 6-DOF dynamics for both sides at
  once**; teams are usually point masses and only the interceptor is a real
  quadrotor. arXiv:2607.05939 is the exception.
- Aerodynamic coupling (downwash) is modelled in swarm navigation
  (`quad-swarm-rl`) but ignored in every PE paper found.
- Bearing-only observation (arXiv:2607.02472) is realistic and rare; everyone
  else assumes relative position.
- No open, fast, **JAX** PE benchmark with real dynamics. arXiv:2603.16279 built
  one and did not release it.

## 6. Reading order for this repo

| milestone | read |
|---|---|
| M3 obs/action/reward | 2202.10796, 2412.11764 |
| M3/M4 training setup | `PPO_example`, 2311.13081, 2504.15129 |
| M5 vmap over agents | 2103.01955, Crazyflow (2606.01478) |
| M6 attention encoder | 2309.13285, TERL 2503.12395 |
| PE task design | 2506.02849, 2603.16279, 2405.13542 |
| multi-agent PE | 2607.05939, 2409.15866, 2312.12255 |
