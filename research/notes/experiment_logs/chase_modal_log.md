# chase_modal — run log

What each chase_modal run was and what it scored, and why the setup is as it is. The task
is in [task_chase_modal.md](../tasks/task_chase_modal.md).

## The question

Three pursuers that share one policy hit each other often. chase_modal asks if a pursuer
that senses more of each other drone hits its teammates less and catches more. The evader
flies a fixed path, so only the sensing changes between runs.

## Design decisions, 2026-10-28

- **The steps are base, +attitude, +rpm and +attitude+rpm.** Base is the chase obs: the
  offset and the velocity of every other drone, at any distance. The user picked the
  steps. The rotation rate of the other drones is not a step: the user meant the tilt.
- **The spec says attitude, not tilt.** `defaults.md` uses tilt for the part of $R$ without
  the heading. The block is the whole $R_i^\top R_j$. So it also gives the heading of
  drone $j$, the side its net faces, and the place of each of its motors.
- **Why the attitude can matter.** From the offset and the velocity, a policy without
  memory can only extend a straight line. The attitude gives the direction of drone $j$'s
  thrust, so the direction it accelerates in now. The motor speeds give the size of that
  thrust and the torques, 0.03 s after drone $j$ commanded them (the motor lag). So they
  are close to its action. But a thrust and a torque in the body of drone $j$ say little
  without its attitude.
- **rpm is a share of the top speed, $n / n_{\max}$.** A raw motor speed is about 4255 rpm
  at hover, and it saturates the first tanh layer. So rpm is the one input that is not in
  raw SI units. Hover is 0.55, and the range is 0.15 to 1.
- **The sensed blocks go to every other drone, the evader too.** In 1v1 the only other
  drone is the evader, so the blocks are the pursuer's advantage over it. So 1v1 shows if
  a pursuer uses the blocks at all. It is also the control for 3v1: a gain that 3v1 shows
  and 1v1 does not comes from the teammates.
- **The task fixes the evader to paths.** The user picked this over a train setting that
  names a pilot for any task. So `--task chase_modal` always trains only the pursuers. Eval
  still needs `--seat evader paths`, because eval does not read the task's pilot yet.
- **The box rules stay those of chase `default`.** The rules for leaving the box change the
  results a lot, but the user keeps the defaults for now.

## Before the first run

Generation 0 of the 3v1 pool run `chase_3v1_s0_20261027-211843_000913` is the run `3v1`:
three pursuers learn against paths only, for 50M steps. At its last update, the training
episodes ended so:

| `caught_e` | `crash_p` | `OOB_p` | `coll_pp` | `time_limit` |
|---|---|---|---|---|
| 0.03 | 0.59 | 0.23 | 0.11 | 0.05 |

- With its mean action, this pursuer then won 0.01 of 600 episodes against paths.
- In the later generations of that run, `coll_pp` grew to 0.64 of the pursuer's training
  episodes. Against paths, no pursuer of the run won more than 0.07.
- FRPN, which does not avoid its teammates, catches paths in 0.75 of the episodes in 1v1.
  In 3v1 it catches in 0.44, and two pursuers hit each other in 0.36 (256 episodes).
- In three older 1v1 pool runs in `default`, with older code, the generation-0 pursuer won
  0.59, 0.50 and 0.12 against paths (`797201`, `967c5e`, `9e4242`).

## Open

| question | state |
|---|---|
| the step budget of 3v1 | after 50M steps, the 3v1 pursuers still crash in 0.59 of the episodes. A run needs more steps, or every step of sensing scores near 0 |
| seeds | the older 1v1 runs spread from 0.12 to 0.59. Plan 2 to 3 seeds for each preset |
| `coll_pp` is cheap | a `coll_pp` costs a pursuer 10, and a crash costs 30. If every step of sensing gives the same `coll_pp`, the reward is a likely reason |
