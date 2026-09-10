"""Open-loop rollout of the plant. No controller, no policy — you hand it
throttles and watch what the drone does."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from swarm.sim import dynamics

SCENARIOS = ("free_fall", "hover", "tumble", "spin_down")


def rollout(state, cmds, params, dt):
    def body(s, u):
        s = dynamics.step(s, u, params, dt)
        return s, s

    return jax.lax.scan(body, state, cmds)[1]


def scenario(name, params, steps):
    """Initial state and one throttle vector per step. Mirrors tests/golden/."""

    hover_rpm = jnp.sqrt(params.mass * params.g / (params.n_motors * params.kf))
    hover_u = (hover_rpm - params.min_rpm) / (params.max_rpm - params.min_rpm)

    if name == "free_fall":
        return dynamics.rest_state(rpm=0.0), jnp.zeros((steps, 4))

    if name == "hover":
        return dynamics.rest_state(rpm=hover_rpm), jnp.full((steps, 4), hover_u)

    if name == "tumble":
        u = jnp.array([0.55, 0.40, 0.52, 0.44])
        return dynamics.rest_state(rpm=0.0), jnp.tile(u, (steps, 1))

    if name == "spin_down":
        state = dynamics.rest_state(rpm=hover_rpm).replace(omega=jnp.array([2.0, -1.5, 3.0]))
        return state, jnp.full((steps, 4), hover_u)

    raise ValueError(f"unknown scenario {name!r}, pick from {SCENARIOS}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenario", choices=SCENARIOS)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--every", type=int, default=10, help="print every Nth step")
    args = p.parse_args()

    params = dynamics.default_params()
    state, cmds = scenario(args.scenario, params, args.steps)
    traj = rollout(state, cmds, params, args.dt)

    print(f"{'t':>6}  {'x':>28}  {'v':>28}  {'omega':>28}  {'orth':>9}")
    for k in range(0, args.steps, args.every):
        R = traj.R[k]
        orth = jnp.abs(R.T @ R - jnp.eye(3)).max()
        print(
            f"{(k + 1) * args.dt:6.2f}  {fmt(traj.x[k])}  {fmt(traj.v[k])}"
            f"  {fmt(traj.omega[k])}  {orth:9.2e}"
        )


def fmt(v):
    return " ".join(f"{a:8.3f}" for a in v)


if __name__ == "__main__":
    main()
