# Task tunnel — run log

The runs of the tunnel task and their results. The spec is in
[task_tunnel.md](../task_tunnel.md).

## The first CPU run

`d_inf`, 3e7 steps, 1024 envs, the drones start in the legs only. The drones flew the
legs at 4.8 m/s and hit a wall at the first bend in 96% of the episodes, so they never
reached a meeting.

## Three runs of 1e9 steps

`d_inf`, 1e9 steps, seeds 0–2 (`7c1d6b`, `e0b72d`, `c027a4`).

- Training stopped improving after about 4e8 steps. It reached 35% success, 56% wall
  and 9% contact.
- In eval from the far ends of the legs, success was 0–0.5% and wall 93–100%. The drones
  sped up to 4.5–6 m/s and hit the first bend.
- In seed 1, 61% of the wall deaths came less than 1.5 m from a bend.

So the score mixed two skills: flying the tube and dodging. This is why the avoidance
task exists ([avoidance_log.md](avoidance_log.md)).
