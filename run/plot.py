"""Time plots for one episode of an eval: python run/plot.py runs/<run>/evals/<name>"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from swarm import envs
from swarm.envs import core

AXES = ("x", "y", "z")


def load(path, episode, drone):
    path = pathlib.Path(path)
    traj = np.load(path / "trajectory.npz")
    header = json.loads((path / "header.json").read_text())
    _, params = envs.make(header["task"], header["preset"])

    k = max(int(traj["live"][episode].sum()), 1)
    cut = lambda name: np.asarray(traj[name])[episode, :k, drone]
    throttle, rate_ref = core.action_to_command(cut("action"), params)

    return header, params, k, {
        "x": cut("x"),
        "omega": cut("omega"),
        "rpm": cut("rpm"),
        "throttle": np.asarray(throttle),
        "rate_ref": np.asarray(rate_ref),
        "goal": cut("goal"),
    }


def draw(header, params, t, d):
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(9, 10))

    ax[0].plot(t, np.linalg.norm(d["goal"] - d["x"], axis=-1), color="k")
    ax[0].set_ylabel("distance [m]")

    for i, name in enumerate(AXES):
        color = f"C{i}"
        ax[1].plot(t, d["rate_ref"][:, i], color=color, ls="--", lw=1, alpha=0.7)
        ax[1].plot(t, d["omega"][:, i], color=color, label=name)
    ax[1].set_ylabel("body rate [rad/s]")
    ax[1].legend(loc="upper right", ncol=3, fontsize=8, title="dashed = command")

    ax[2].plot(t, d["throttle"], color="k")
    ax[2].set_ylim(-0.05, 1.05)
    ax[2].set_ylabel("collective throttle")

    model = params.model
    hover = np.sqrt(model.mass * model.g / (model.n_motors * model.kf))
    for i in range(d["rpm"].shape[-1]):
        ax[3].plot(t, d["rpm"][:, i], lw=1, label=f"m{i}")
    for y, name in ((model.min_rpm, "min"), (model.max_rpm, "max"), (hover, "hover")):
        ax[3].axhline(y, color="k", ls=":", lw=1)
        ax[3].annotate(name, (t[-1], y), fontsize=7, va="bottom", ha="right")
    ax[3].set_ylabel("motor rpm")
    ax[3].set_xlabel("time [s]")
    ax[3].legend(loc="upper right", ncol=4, fontsize=8)

    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle(f"{header['task']} · {header['preset']}")
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("eval", help="runs/<run>/evals/<name>")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--drone", type=int, default=0)
    p.add_argument("--out", help="default: <eval>/plot_e<episode>.png")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    header, params, k, d = load(args.eval, args.episode, args.drone)
    fig = draw(header, params, np.arange(k) * header["policy_dt"], d)

    out = args.out or pathlib.Path(args.eval) / f"plot_e{args.episode}.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}  ({k} steps)")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
