# Env contract

What every task file in `swarm/envs/` holds, in code. `tests/test_contract.py` checks a
part of it on every task and preset.

## A task file

A task module is a set of plain functions:

```
NUM_ACTIONS             width of one drone's action
EnvParams(core.Common)  the task's values; PRESETS names the ones we use
EnvState                reset and step carry it, nothing else touches it
reset(key, params)      -> obs, state
step(key, state, action, params)
                        -> obs, state, reward (n,), done (), info
get_obs(state, params)  -> Obs
reference(state, params)
                        -> (n, 3), each drone's goal point, or NaN for a drone
                           without one. The viewer draws it as a marker
PILOTS                  {name: pilot}, the hand-written pilots. pilot(params,
                        rows) -> act, start, as in swarm/learn/seats.py, flying
                        the drones of `rows`. A preset's `piloted` can name one
                        to fly a role in training
PROGRESS                the metrics the trainer prints in its progress line
SWARM                   only for the attention networks: the role with a free
                        number of drones. Its drones go through the attention
scenery(state, params)  only for what the viewer must draw: walls,
                        {"walls": (k, 3), the corners of a closed outline on
                        the floor in the world, "wall_height": ()}, or nets,
                        {"net": (n, 2), each drone's net side and rope, 0 for
                        no net}
eval_params(params)     only if eval flies other params than training
                        -> params
curriculum(params, metrics)
                        only if training changes the params as it goes.
                        The trainer calls it after every update with that
                        update's metrics -> the next params, or None to
                        keep these
```

## The end flags and the metrics

The trainer owns the step limit: it ends an episode after `max_steps` steps. A deadline
that is a rule of the task, like avoidance's 10 s `timeout`, belongs in `step`.
`info` carries three flags per drone and two metric dicts:

```
alive      (n,) the drone has not died
flying     (n,) the drone still takes part: alive, and not at its goal in a task
           where a drone leaves at its goal. Only these drones train
bootstrap  (n,) when `done`, the critic estimates what the drone would still get.
           False makes the end terminal for the drone
end        read at the step the episode ends — a catch, a crash, a final distance.
           A NaN means this episode has no value for it, and it is left out of
           the average instead of counted as a zero.
step       averaged over the steps — a gap, a per-step collision rate.
```

At the step limit every flying drone bootstraps, unless the task ended the episode at
the same step. The trainer adds `time_limit` to `end`.

A metric with a drone axis is reported once per role, named `<metric>_<r>` with `r`
the role's first letter, for example `return_p`; a task with one role keeps the bare
name. A metric without one is a scene fact and keeps its name. Put a number in the
group that matches how you want to read it — nothing else decides it.

## The obs

`Obs` in `swarm/envs/__init__.py` is what the networks see. Batching adds leading
dimensions to these shapes. The learner never looks inside a row: the task picks the
columns.

```
actor   {role: (n, f)}, one row for each drone of the role, in drone order
critic  (c,), the whole scene
own     {role: (n, 20)}, the own block of each drone of the role
pairs   {role: (n, N, p)}: row i, column j is drone j, in drone order, as drone i
        senses it
world   (N, 18), each drone's state in the world frame
```

The attention networks read `own`, `pairs` and `world`. A task without them leaves None.
They read each other drone of the `SWARM` role as one row of the attention, and each
other drone through a fixed slot.
