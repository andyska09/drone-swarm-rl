# Bibliography Index

Search index for `research/papers/`. Every PDF has a `pdftotext -layout` transcript
(`.md`) next to it — grep the `.md`, open the `.pdf` only for figures.

Regenerate a transcript with:
`pdftotext -layout <file>.pdf <file>.md`

| paper | files | what it is |
|---|---|---|
| Huang et al. 2024, *Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end DRL* ([arXiv:2309.13285](https://arxiv.org/abs/2309.13285)) | `2309.13285v2.pdf`, `.md` | The paper behind `quad-swarm-rl`. Rotor-thrust actions, attention over K nearest neighbours, SDF obstacles, collision-episode replay. Source of the cost-style reward we copy. |
| Gavin, Lacroix & Bronz 2026, *Agile Interception of a Flying Target using Competitive Reinforcement Learning* ([arXiv:2603.16279](https://arxiv.org/abs/2603.16279)) | `2603.16279v1.pdf`, `.md` | The 1v1 predecessor of the Nv1 paper below. Pursuer and evader both learn with PPO in a co-evolution loop, CTBR actions on a high-fidelity quadrotor model written in JAX — the same stack we are building. Adds real indoor flights, which the Nv1 paper does not have. |
| Chen, Yu et al. 2024, *What Matters in Learning a Zero-Shot Sim-to-Real RL Policy for Quadrotor Control?* ([arXiv:2412.11764](https://arxiv.org/abs/2412.11764)) | `2412.11764v2.pdf`, `.md` | SimpleFlight: a PPO recipe for single-drone trajectory tracking, ablated factor by factor and flown on a Crazyflie. Rotation matrix + velocity in the actor, time vector in the critic only, `‖u_t − u_{t−1}‖₂` smoothness reward at λ = 0.4, SysID instead of blanket domain randomization, large batches. CTBR actions at 100 Hz — our action space. Source of the action-difference penalty in `parking_error.md`. |
| Gavin & Bronz 2026, *Intercepting an Agile Target with Net-Carrying Drones using Competitive MARL* ([arXiv:2607.05939](https://arxiv.org/abs/2607.05939)) | `2607.05939v1.pdf`, `.md` | Nv1 interception, both sides learned. MAPPO with CTDE, CTBR actions at 100 Hz over a 1 kHz attitude controller, Prioritized Fictitious Self-Play against a frozen opponent pool. Closest prior art to our Nv1 rung — same obs split (self / others / arena), and it terminates on pursuer-pursuer collision but keeps a soft penalty for pursuer-evader contact. |
| Pliska, Vrba, Báča & Saska 2024, *Towards Safe Mid-Air Drone Interception: Strategies for Tracking & Capture* ([arXiv:2405.13542](https://arxiv.org/abs/2405.13542), RA-L 9(10) 8810-8817) | `2405.13542v1.pdf`, `.md` | The non-learning interception baseline, and the source of the FRPN guidance law both Gavin papers compare against. Runs on `ctu-mrs/mrs_multirotor_simulator` — the same C++ plant we reimplement. Net is a circular plane of **2 m radius** hung below the interceptor; capture is the target's centre **passing through** that plane. Evaluated on 100 target trajectories up to 10 m/s and 8 m/s², 5 random starts within 20 m, 100 s each. Baselines: pure pursuit, LPN, GPN, MPC. |
