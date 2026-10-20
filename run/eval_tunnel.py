"""Fly a tunnel run against drones that only follow their lines and never dodge:
python run/eval_tunnel.py runs/<name> [--line-drone 1] [--episodes 1024]

`--line-drone 0 1` puts both drones on their lines: the baseline where nobody dodges."""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from swarm import envs
from swarm.envs import core, tunnel
from swarm.learn import evaluate, ppo, runner, seats
from swarm.sim import control, guidance

SPEED = 2.0
# The follower flies along the chord to the point this far ahead, so it starts each
# turn before the corner.
LEAD = 1.0


def line_follower(params, rows):
    """Track the nearest point on the own line and fly along the line at SPEED."""

    def act(state, obs, pids):
        del obs
        p = tunnel.to_tunnel(state.drone.x, state.heading, params)
        s, _, line = tunnel.follow(p, state.offset, params)
        here = tunnel.on_line(s, line, state.offset)
        ahead = tunnel.on_line(s + LEAD, line, state.offset)
        along = (ahead - here) / core.norm(ahead - here)[:, None]
        point = tunnel.to_world(here, state.heading, params)
        velocity = SPEED * along @ tunnel.yaw(state.heading).T
        own = jax.tree.map(lambda q: q[rows], state.drone)
        a = jax.vmap(guidance.follow)(
            own.x, own.v, (point[rows], velocity[rows], jnp.zeros((len(rows), 3)))
        )
        return seats._acceleration_to_action(params, own, a, pids)

    return act, lambda state: (state, seats._per_row(control.pid_init(), len(rows)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument(
        "--checkpoint", default="latest", help="'latest', an update number, or a path"
    )
    p.add_argument(
        "--line-drone",
        type=int,
        nargs="+",
        default=[1],
        help="the drones that only follow their lines",
    )
    args = p.parse_args()

    run = pathlib.Path(args.run)
    cfg = evaluate.load_config(run)
    env, env_params = envs.make(cfg.task, cfg.preset)
    rows = jnp.array(args.line_drone)
    follow, start = line_follower(env_params, rows)
    saved = runner.load(run, evaluate.checkpoint_name(args.checkpoint))
    net = seats.net_policy(cfg, env, saved["params"], ppo.role_slices(env_params.roles))

    def act(state, obs, pids):
        action, _ = net(state, obs, None)
        flown, pids = follow(state, obs, pids)
        return action.at[rows].set(flown), pids

    # One column per seat, so the policy and the follower are not averaged together.
    policy = [i for i in range(env_params.n_drones) if i not in args.line_drone]
    slices = {"follower": rows} | ({"policy": jnp.array(policy)} if policy else {})

    def measure(key):
        _, _, _, reward, info, live = evaluate.rollout(key, env, env_params, act, start)
        return evaluate.summarize(reward, info, live, slices)

    keys = jax.random.split(jax.random.PRNGKey(evaluate.EVAL_SEED), args.episodes)
    per_episode = jax.jit(jax.vmap(measure))(keys)
    summary = {k: evaluate._mean_defined(v) for k, v in per_episode.items()}

    out = run / "evals" / ("line_drone" + "_".join(map(str, args.line_drone)))
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(
        json.dumps(
            {
                "task": cfg.task,
                "preset": cfg.preset,
                "line_drones": args.line_drone,
                "follower_speed": SPEED,
                "checkpoint": args.checkpoint,
                "update": saved["update"],
                "episodes": args.episodes,
                "eval_seed": evaluate.EVAL_SEED,
                "git": runner.git_state(),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    np.savez(out / "episodes.npz", **{k: np.asarray(v) for k, v in per_episode.items()})
    evaluate.save_replay(
        out, cfg, env, env_params, act, start, keys[: evaluate.TRAJECTORIES], {}
    )
    for key in sorted(summary):
        print(f"{key:>30}  {summary[key]:10.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
