"""Shared observation layout and task loading.

A task module in this package is a set of plain functions:

    NUM_ACTIONS             width of one drone's action
    EnvParams(core.Common)  the task's values; PRESETS names the ones we use
    EnvState                reset and step carry it, nothing else touches it
    reset(key, params)      -> obs, state
    step(key, state, action, params)
                            -> obs, state, reward (n,), done (), info
    get_obs(state, params)  -> Obs, one row per drone, plus `scene` for the critic
    reference(state, params)
                            -> (n, 3), where each drone is trying to fly. The
                               cascade baseline flies to it and the viewer draws
                               it as the marker sphere.
    is_dead                 only if core.is_dead is the wrong rule for the task
    action_signs(state, params)
                            only if a drone sees the world in a mirror
                            -> (n, NUM_ACTIONS) of +1 and -1. step multiplies
                               the action by it, so a baseline that reads the
                               state, not the obs, multiplies its action by it
                               first
    scenery(state, params)  only for what the viewer must draw: walls,
                            {"walls": (k, 3), the corners of a closed outline on
                            the floor in the world, "wall_height": ()}, or lines,
                            {"lines": (n, 2, 3), the start and the end of each
                            drone's line}
    eval_params(params)     only if eval flies other params than training
                            -> params
    curriculum(params, metrics)
                            only if training changes the params as it goes.
                            The trainer calls it after every update with that
                            update's metrics -> the next params, or None to
                            keep these

`info` must carry `alive`, `died_this_step`, `truncated`, and two metric dicts:

    end   read at the step the episode ends — a catch, a crash, a final distance.
          A NaN means this episode has no value for it, and it is left out of the
          average instead of counted as a zero.
    step  averaged over the steps — a gap, a per-step collision rate.

A task where a drone leaves the episode at its goal also gives `arrived` and
`arrived_this_step`. The trainer stops training an arrived drone, and the drone stays
`alive`, so it is not a death.

A metric with a drone axis is reported once per role, named `<role>_<metric>`; a
task with one role keeps the bare name. A metric without one is a scene fact and
keeps its name. Put a number in the group that matches how you want to read it —
nothing else decides it.
"""

import importlib

import flax.struct
import jax.numpy as jnp


@flax.struct.dataclass
class Obs:
    """Per-drone observations; batching adds leading dimensions to these shapes.

    `others` holds drones, whatever their role — a teammate and an opponent are
    told apart by the last feature, not by sitting in different blocks. `target`
    holds a place to fly to, and a task that has none gives it zero width.

    `scene` is the critic's input: the whole game.
    """

    own: jnp.ndarray  # (N, f)
    others: jnp.ndarray  # (N, K, 7), the K nearest drones, body frame
    others_mask: jnp.ndarray  # (N, K)
    target: jnp.ndarray  # (N, t)
    scene: jnp.ndarray  # (18N,), every drone's exact state, world frame


DRONE_AXIS = {"own": -2, "others": -3, "others_mask": -2, "target": -2}


def flat(obs):
    """Flatten each drone's observations, zero the empty seats, and add the mask."""

    lead = obs.own.shape[:-1]
    return jnp.concatenate(
        [
            obs.own,
            (obs.others * obs.others_mask[..., None]).reshape(lead + (-1,)),
            obs.others_mask,
            obs.target,
        ],
        axis=-1,
    )


def flat_critic(obs, opp_action):
    """The critic's input: the whole scene and the opponents' actions this step.

    Both blocks are per scene, so they broadcast over the rows.
    """

    lead = obs.own.shape[:-1]
    grow = lambda x: jnp.broadcast_to(x[..., None, :], lead + x.shape[-1:])
    return jnp.concatenate([grow(obs.scene), grow(opp_action)], axis=-1)


def take(obs, drones):
    """Select drone indices while preserving leading batch dimensions."""

    return obs.replace(
        **{f: jnp.take(getattr(obs, f), drones, axis=a) for f, a in DRONE_AXIS.items()}
    )


def make(task, preset):
    env = importlib.import_module(f"swarm.envs.{task}")
    return env, env.PRESETS[preset]
