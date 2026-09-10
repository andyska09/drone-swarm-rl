"""Fly to a point with the full MRS cascade. Hand-tuned gains, no learning."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from swarm.sim import control, dynamics


def rollout(state, target, heading, params, dt, steps):
    gains = control.cascade_gains(params)

    def body(carry, _):
        st, pids = carry
        throttles, pids = control.cascade_step(st, target, heading, pids, gains, params, dt)
        st = dynamics.step(st, throttles, params, dt)
        return (st, pids), st

    return jax.lax.scan(body, (state, control.cascade_init()), None, length=steps)[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=float, nargs=3, default=[3.0, -2.0, 5.0])
    p.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--heading", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--every", type=int, default=100)
    args = p.parse_args()

    params = dynamics.default_params()
    target = jnp.array(args.target)
    start = dynamics.rest_state(0.0).replace(x=jnp.array(args.start))
    traj = rollout(start, target, args.heading, params, args.dt, args.steps)

    print(f"{'t':>6}  {'x':>28}  {'v':>28}  {'error':>8}")
    for k in range(0, args.steps, args.every):
        error = float(jnp.linalg.norm(traj.x[k] - target))
        print(f"{(k + 1) * args.dt:6.2f}  {fmt(traj.x[k])}  {fmt(traj.v[k])}  {error:8.4f}")

    print(f"\nsettled at {fmt(traj.x[-1])}  error {float(jnp.linalg.norm(traj.x[-1] - target)):.4f} m")


def fmt(v):
    return " ".join(f"{a:8.3f}" for a in v)


if __name__ == "__main__":
    main()
