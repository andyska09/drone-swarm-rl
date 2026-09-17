# Why the learned policy will not park

Our policy drifts around a point 0.2-0.6 m short of the goal at ~0.2 m/s. The
cascade in the same seat parks at 0.014 m. This note tests five hypotheses
against primary sources and lists what those sources actually do about it.

Every claim is marked **[P]** (primary source, with path/line or paper section) or
**[I]** (my inference). Measurements of our own runs are **[M]**.

## Sources

| short name | reference | where | what it flies | position error it reports |
|---|---|---|---|---|
| Hwangbo 2017 | Hwangbo et al., *Control of a Quadrotor with Reinforcement Learning*, arXiv:1707.05110 | web | 1 drone, rotor thrusts | **1.3 cm** |
| Molchanov 2019 | Molchanov et al., *Sim-to-(Multi)-Real*, arXiv:1903.04628v2 | web | 1 drone, rotor thrusts, real hardware | 0.09 m sim / 0.21 m real |
| Eschmann 2023 | Eschmann et al., *Learning to Fly in Seconds*, arXiv:2311.13081 + code `arplaboratory/learning-to-fly` | web | 1 drone, real hardware | 0.26 m mean / 0.24 m median |
| Swift 2023 | Kaufmann et al., *Champion-level drone racing*, Nature 2023 | web | 1 drone, CTBR, racing | racing, no hover number |
| Yu & Lee 2023 | Yu & Lee, arXiv:2311.06144 | web | 1 drone, integral term in the obs | not published |
| Huang 2024 | Huang et al., arXiv:2309.13285 — the paper behind `quad-swarm-rl` | `research/papers/2309.13285v2.md` | 8 drones, rotor thrusts, attention | 0.09 - 0.43 m |
| quad-swarm-rl | the code of Huang 2024 | `research/code_sources/quad-swarm-rl/` | same | "arrived" is 0.5 m |
| Wang 2024 | Wang, Zheng & Lin, arXiv:2402.09075 | web | ACC cruise plant, DDPG — not a quadrotor | 2.9e-3 to 2.7e-1 m |
| SimpleFlight 2024 | *What Matters in Learning a Zero-Shot Sim-to-Real RL Policy for Quadrotor Control?*, arXiv:2412.11764v2 + code `thu-uav/SimpleFlight` | web | 1 drone, **CTBR at 100 Hz — our action space** | 0.016 m on a figure eight |
| Gavin 1v1 2026 | Gavin, Lacroix & Bronz, arXiv:2603.16279 | `research/papers/2603.16279v1.md` | 1v1 interception, CTBR | interception, no hover number |
| Gavin Nv1 2026 | Gavin & Bronz, arXiv:2607.05939 | `research/papers/2607.05939v1.md` | Nv1 interception, CTBR | interception, no hover number |

Ours, for comparison: 0.21 m (`a_to_b default`), 0.34 m (`circle_swap pair`),
0.58 m (`circle_swap circle8`). The M2 cascade in the same seat: 0.014 m.

## Our reward and observation, for the record

- **[P]** `swarm/envs/a_to_b.py:14-18` — `distance=1.0, crash=10.0, spin=0.1`.
- **[P]** `swarm/envs/a_to_b.py:118-121` — `-policy_dt * (1.0·‖g-x‖ + 0.1·‖ω‖) - crash·died`.
- **[P]** `swarm/envs/circle_swap.py:15-21, 148-153` — adds `close=10.0` and `- 1.0·R₃₃`.
- **[P]** `swarm/envs/a_to_b.py:93-106` — obs: body velocity, R (9), ω, z, body-frame
  goal offset, body-frame `-v`. **Velocity is observed.** No action history, no integral.
- **[P]** `swarm/learn/config.py:19,26,32` — `gamma=0.998`, `ent_coef=0.0`, `init_log_std=-0.5`.
- **[I]** There is **no action term of any kind** in either reward — no effort, no
  smoothness. Every source below has at least one.

## First: our number is inside the published band

**[P]** Molchanov et al. 2019 (arXiv:1903.04628v2, Results) — 0.09 m position error in
sim, 0.21 m on hardware, on a cost that is nearly identical to ours.
**[P]** Huang et al. 2024 (`research/papers/2309.13285v2.md:277-278, 294`) — distance to
goal 0.09 m (random goals), 0.24 m (same goal), 0.43 m (train-from-scratch, Table II).
**[P]** Eschmann et al. 2023 (arXiv:2311.13081, Table II) — 0.26 m mean / 0.24 m median
position error, real-hardware position hold, best ablation configuration.
**[P]** `quad-swarm-rl/gym_art/quadrotor_multi/scenarios/base.py:31` — their "reached
goal" test is `approch_goal_metric = 0.5`, i.e. mean distance over the last 5 steps
below **0.5 m** (unwound at `quadrotor_multi.py:542-546`). Success is 0.5 m, not 1 cm.

**[I]** So 0.2-0.6 m is the normal outcome of this reward family, not a bug in our
plant or trainer. Nobody in this lineage parks; the ones that do changed the reward.

## The five hypotheses

**(a) No integral action — SUPPORTED.**
**[P]** Hwangbo et al. 2017 (arXiv:1707.05110, Conclusion): "We had a small steady state
error (1.3 cm) which can be easily diminished with a constant state offset." A constant
offset is by definition a bias a memoryless policy cannot see.
**[P]** Yu & Lee (arXiv:2311.06144, §IV-D): "RL-based quadrotor low-level control often
suffers from steady-state errors. To address this ... an integral term is formulated as
`ė_Ix = -α·e_Ix + e_x` with α > 0 chosen to mitigate the integral windup." `e_Ix ∈ R³`
is in the **observation**, `o = (e_x, e_Ix, e_v, R, e_Ω)` (§IV-A), and also in the reward
as `-k_Ix‖e_Ix‖²`.
**[I]** Our policy has neither. A constant thrust/tilt bias is unobservable to it.

**(b) Linear distance cost — REFUTED as stated, but the tie-break part is real.**
**[P]** Wang, Zheng & Lin (arXiv:2402.09075, Table ACC DATE) measure exactly this on an
ACC plant with DDPG: quadratic reward → steady-state error **2.7e-1 m**; absolute-value
reward → **2.9e-3 m**. Linear is the *better* of the two, because the quadratic's
gradient vanishes at zero. Their fix for the quadratic case is an integral penalty
`C_I = c_I^T Z c_I` (→ 7.3e-3 m / 9.2e-3 m). Not a quadrotor.
**[I]** What is true and unfixable by weight tuning: a cost that depends only on
‖x-g‖ is *exactly indifferent* between orbiting at radius r and hovering at radius r.
Nothing in our reward breaks that tie. Every source below breaks it with a velocity
term, an action term, or a bonus cliff.
**[I]** I found no primary source that names the orbiting / limit-cycle behaviour in a
learned quadrotor controller. The indifference argument is mine.

**(c) Discount horizon too short — REFUTED.**
**[P]** SimpleFlight (`thu-uav/SimpleFlight`, `cfg/algo/mappo.yaml`) uses `gamma: 0.995`
at 100 Hz — a 200-step / 2 s horizon — and parks inside a 0.02 m bonus radius.
**[P]** Gavin et al. 2026 (`research/papers/2603.16279v1.md:232`) use `0.99` at 100 Hz.
**[I]** Ours is 0.998 → 500 steps ≈ 5 s, *longer* than either. Not the cause.

**(d) Policy action noise — no primary source found.**
**[M]** Eval uses mean actions, so exploration noise is not in the eval trajectory.
**[I]** It still shapes training: `init_log_std=-0.5` is σ≈0.61 on a `[-1,1]` action, and
with `ent_coef=0.0` nothing forces it down but nothing measures it either. The mean
action learned is the one that is optimal *under that noise*. Cheap check: read
`log_std` out of the checkpoint. No source attributes residual error to this.

**(e) No velocity term — SUPPORTED, and it is the cleanest split in the literature.**
**[P]** Hwangbo (arXiv:1707.05110 §IV-B): `r_t = 4e-3‖p_t‖ + 2e-4‖a_t‖ + 3e-4‖ω_t‖ +
5e-4‖v_t‖`. Velocity weight is 0.125× the position weight. Result: **1.3 cm**.
**[P]** Molchanov (arXiv:1903.04628v2, eq. 14): `c_t = (‖e_p‖ + α_v‖e_v‖ + α_ω‖e_ω‖ +
α_a‖a‖ + α_R·cos⁻¹((Tr(R)-1)/2))·dt`, baseline **α_v = 0**, α_ω = 0.1, α_a = 0.05,
α_R = 0. Result: 0.09 m sim / 0.21 m real.
**[P]** `quad-swarm-rl/swarm_rl/env_wrappers/reward_shaping.py:9` — `vel=0.0`. And it is
a **dead key**: `grep -rn 'rew_coeff\["vel"\]\|cost_vel\|rew_vel' --include=*.py` over
the whole tree returns nothing. `compute_reward_weighted`
(`gym_art/quadrotor_multi/quadrotor_single.py:34-66`) sums only pos, effort, crash,
orient, spin. **No run script under `swarm_rl/runs/` sets `vel` at all.**
**[P]** Yu & Lee (arXiv:2311.06144 §IV-B): `r1 = -k_x‖e_x‖² - k_Ix‖e_Ix‖² - k_v‖e_v‖²
- k_b3‖e_b3‖ - k_ω12‖e_ω12‖² - r_crash`.
**[I]** Hwangbo (velocity term, 1.3 cm) vs Molchanov/Huang (α_v = 0, 0.09-0.43 m) is a
suggestive pair, not a controlled experiment — different algorithms, tasks and hardware.
Treat it as a hypothesis to test in our env, not a proven cause.

## What the sources put in the reward to fix it

1. **Success bonus inside a radius.**
   **[P]** SimpleFlight `omni_drones/envs/single/hover.py`:
   `reward_pos = -pos_error * self.reward_distance_scale`,
   `reward_pos_bonus = ((pos_error <= 0.02) * 10).float()`,
   `reward_head = -head_error * (reward_pos_bonus > 0)`,
   `reward_head_bonus = ((head_error <= 0.02) * 10 * (reward_pos_bonus > 0)).float()`.
   With `reward_distance_scale: 10.0` (`cfg/task/Hover.yaml`) the linear term at 1 m is
   -10/step and the bonus is +10/step — a cliff worth 1 m of shaping, at 2 cm.
   Note the heading reward is **gated**: it pays nothing until you are inside 2 cm.
   **[P]** Gavin et al. 2026 (`2603.16279v1.md:240-242`): `λcatch = 10.0` against
   `λdist = 0.001` — a 10⁴ ratio between the terminal bonus and the distance shaping.

2. **Exponential shaping instead of linear.**
   **[P]** SimpleFlight `omni_drones/envs/single/track.py`:
   `reward_pos = self.reward_distance_scale * torch.exp(-distance)`, with
   `reward_distance_scale: 5.0` (`cfg/task/Track.yaml`).
   **[P]** Eschmann, shipped code `arplaboratory/learning-to-fly`,
   `include/learning_to_fly/simulator/parameters/reward_functions/abs_exp.h`:
   `r = params.scale * exp(-params.scale_inner * weighted_abs_cost)` where
   `weighted_abs_cost = position·‖p‖ + orientation·(1-q_w²) + linear_velocity·‖v‖ +
   angular_velocity·‖ω‖ + linear_acceleration·… + angular_acceleration·… + action·‖a-a_baseline‖`.
   Note this is the *absolute* cost inside the exponential, not the squared form printed
   in the paper. The numeric weights are "supplied in the supplementary material"
   (arXiv:2311.13081 §Reward) and are not in the paper body; I could not open the config
   that sets them.
   **[I]** `exp(-k·cost)` is also a built-in survival bonus: the reward is positive and
   bounded, so dying forfeits income. Ours is strictly negative, so dying is a payout.

3. **Multiplicative gating of the attitude terms on proximity.**
   **[P]** SimpleFlight `track.py`:
   `reward = reward_pos + reward_pos * (reward_up + reward_spin) + reward_action_norm +
   reward_action_smoothness + reward_acc + reward_jerk + reward_snap`, with
   `reward_up = w * 0.5/(1+tiltage²)` and `reward_spin = w * 0.5/(1+spin²)`,
   both weights 1.0 (`cfg/task/Track.yaml`).

4. **Action-difference penalty, not action magnitude.** The single best-measured fix.
   **[P]** SimpleFlight (arXiv:2412.11764v2 §IV-C, Tab. I): `r = r_task + λ·r_smooth`,
   `r_smooth = e^{-A}`, both normalized to [0,1], `λ = 0.4`. Real-world MED on
   figure-eight (slow / normal / fast), `A = ‖u_t‖₂` vs `A = ‖u_t - u_{t-1}‖₂`:
   **0.044 / 0.066 / 0.110 m** vs **0.016 / 0.028 / 0.051 m**. Same task, same rig,
   2.4-2.8× worse for the magnitude penalty. Their `u_t` is CTBR — our action exactly.
   **[P]** Swift (Kaufmann et al., Nature 2023, Methods): `r_cmd = λ₄‖a_t^ω‖ +
   λ₅‖a_t - a_{t-1}‖²`. Coefficients are in Extended Data Table 1a, not in the body text.

5. **Integral of position error in the observation.** Yu & Lee, quoted above.

6. **Previous action in the observation — contested.**
   **[P]** Swift: 31-dim obs = platform state (15) + relative gate pose (12) +
   **previous action (4)**.
   **[P]** SimpleFlight (arXiv:2412.11764v2 §V-B-1): "Including the previous action
   u_{t-1} ... results in a slight performance drop ... including the previous action in
   the input introduces non-stationary into the environment."

7. **Time vector in the critic only.**
   **[P]** SimpleFlight §V-B-1: time in both actor and critic causes OOD on long
   flights; critic-only is their choice. Relevant to us only if we truncate.

8. **Curriculum on reward weights.**
   **[P]** Eschmann (arXiv:2311.13081): "Every 100,000 steps the weights are adjusted by
   multiplying them by the C_p* factors ... until they reach the C_target,* values",
   moving toward "punishing position errors and particularly control actions more harshly".

**[P]** Termination: SimpleFlight Hover has **none** — `done = progress_buf >=
max_episode_length` only, 500 steps at 100 Hz. No arrival terminal anywhere in these
sources; arrival is rewarded, never ended.

## Term-by-term

| term | ours | Huang / quad-swarm-rl | Molchanov | Hwangbo | Eschmann | SimpleFlight Hover | SimpleFlight Track | Gavin 1v1 | Swift |
|---|---|---|---|---|---|---|---|---|---|
| distance shape | linear ‖·‖ | linear ‖·‖ | linear ‖·‖ | linear ‖·‖ | `exp(-k·Σ)` | linear + cliff | `exp(-d)` | linear ‖·‖ | progress Δd |
| distance weight | 1.0 | `pos=1.0` | 1.0 | 4e-3 | n/p | 10.0 | 5.0 | λdist=0.001 | λ₁ n/p |
| success bonus | none | none | none | none | none | **+10 @ 0.02 m** | none | λcatch=10.0 | none |
| velocity | **none** | `vel=0.0`, dead key | α_v=0 | **5e-4** (0.125×) | `linear_velocity·‖v‖` | none | none | none | none |
| body rate | 0.1·‖ω‖ | `spin=0.1` | α_ω=0.1 | 3e-4 (0.075×) | `angular_velocity·‖ω‖` | none | `0.5/(1+spin²)`, w=1.0, gated | λcmd=2e-4·‖aω‖ | λ₄‖a^ω‖ |
| action magnitude | **none** | `effort=0.05` | α_a=0.05 | 2e-4 (0.05×) | `action·‖a-ā‖` | none | `reward_action_norm`, init 0.0 | — | — |
| action difference | **none** | none | none | none | via accel terms | `w=0.0` | `w·exp(-Δu)`, init **2.0**, max 2.0 | none | λ₅‖Δa‖² |
| orientation | -1.0·R₃₃ (swap only) | `orient=1.0`·R₃₃ | α_R=0 | none | `orientation·(1-q_w²)` | `((up_z+1)/2)²` | `0.5/(1+tilt²)`, gated | none | perception term |
| crash | -10 / -25 one-off | `crash=1.0` per step on floor | n/p | none | termination | none | z<0.1 → done | λfail=30.0 | 5.0 |
| survival | none | none | none | none | implicit (`r>0`) | implicit (`r>0`) | implicit (`r>0`) | λstep=0.04 (Nv1) | none |
| integral in obs | no | no | no | no | no | no | no | no | no |
| prev action in obs | no | no | no | no | action history | `action_history_step: 5` | `action_history_step: 5` | no | yes (4 of 31) |
| gamma | 0.998 | n/p | n/p | n/p | n/p | 0.995 | 0.995 | 0.99 | n/p |

`n/p` = the source does not publish it. Gavin coefficients: `2603.16279v1.md:240-242`.
Nv1 `λstep=0.04`: `2607.05939v1.md:319`.

## What to try, cheapest first

1. **Success bonus inside a radius.** One line, no new state. Copy SimpleFlight Hover:
   `+B` when `‖g-x‖ ≤ r`. Scale it against our shaping — at `distance=1.0` and
   `policy_dt=0.01` a 1 m error costs 0.01/step, so a bonus of 0.1/step is a 10 m
   equivalent cliff. **Why:** it is the only mechanism in any source that makes the last
   centimetres worth more than the first metre. Source: SimpleFlight `hover.py`.
2. **Add a velocity term.** `+ k_v·‖v‖` at Hwangbo's ratio, `k_v = 0.125 × distance`.
   **Why:** it is the only term that distinguishes orbiting at radius r from hovering at
   radius r, and it is the one thing Hwangbo (1.3 cm) has that Molchanov and
   quad-swarm-rl (0.09-0.43 m) do not. Sources: arXiv:1707.05110 §IV-B; 1903.04628 eq. 14.
3. **Action-difference penalty on the CTBR command.** Needs `prev_action` in `EnvState`
   (not in the obs). **Why:** the only controlled A/B in the whole set — 2.4-2.8× lower
   error than an action-magnitude penalty, same rig, CTBR actions, 100 Hz.
   Source: arXiv:2412.11764v2 Tab. I.
4. **Swap the distance term for `exp(-k·d)` or wrap the whole cost in `exp(-k·cost)`.**
   Also flips the reward positive, which removes the "suicide is a payout" exit
   `task_circle_swap.md` already documents. **Why:** sharpens the gradient near zero and
   adds a survival income. Sources: SimpleFlight `track.py`; `abs_exp.h`.
5. **Integral of position error in the observation**, `ė_I = -α·e_I + e`, body frame.
   **Why:** the only fix that removes a *constant bias* rather than trading it off, and
   Hwangbo says his residual is exactly such a bias. Costs 3 obs columns and one state
   field. Source: arXiv:2311.06144 §IV-A/§IV-D.
6. **Read `log_std` out of the checkpoint** before doing anything about it. Free.
   **Why:** rules (d) in or out. No source; our own diagnostic.

Do not bother shortening gamma — 0.998 is already longer than both published horizons.
