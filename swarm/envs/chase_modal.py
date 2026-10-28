"""chase against the paths evader. The presets differ only in what a pursuer senses of the
other drones."""

from swarm.envs.chase import (  # pylint: disable=unused-import
    NUM_ACTIONS,
    PILOTS,
    PROGRESS,
    EnvParams,
    EnvState,
    get_obs,
    reference,
    reset,
    scenery,
    step,
)

PILOTED = ("evader", "paths")

SENSES = ((), ("attitude",), ("rpm",), ("attitude", "rpm"))

PRESETS = {
    "_".join((f"{p}v1",) + sense): EnvParams(roles=("pursuer",) * p + ("evader",), sense=sense)
    for p in (1, 3)
    for sense in SENSES
}
