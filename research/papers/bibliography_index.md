# Bibliography Index

Search index for `research/papers/`. Every PDF has a `pdftotext -layout` transcript
(`.md`) next to it — grep the `.md`, open the `.pdf` only for figures.

Regenerate a transcript with:
`pdftotext -layout <file>.pdf <file>.md`

| paper | files | what it is |
|---|---|---|
| Huang et al. 2024, *Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end DRL* ([arXiv:2309.13285](https://arxiv.org/abs/2309.13285)) | `2309.13285v2.pdf`, `.md` | The paper behind `quad-swarm-rl`. Rotor-thrust actions, attention over K nearest neighbours, SDF obstacles, collision-episode replay. Source of the cost-style reward we copy. |
| Gavin & Bronz 2026, *Intercepting an Agile Target with Net-Carrying Drones using Competitive MARL* ([arXiv:2607.05939](https://arxiv.org/abs/2607.05939)) | `2607.05939v1.pdf`, `.md` | Nv1 interception, both sides learned. MAPPO with CTDE, CTBR actions at 100 Hz over a 1 kHz attitude controller, Prioritized Fictitious Self-Play against a frozen opponent pool. Closest prior art to our Nv1 rung — same obs split (self / others / arena), and it terminates on pursuer-pursuer collision but keeps a soft penalty for pursuer-evader contact. |
