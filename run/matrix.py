"""Every pursuer against every evader of a pool run: python run/matrix.py runs/<name>

Writes the pursuer's win rate to runs/<name>/matrix.csv and draws it in matrix.png."""

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from swarm.learn import evaluate, pool, rundir

# The test-only baselines, played after each pool's seed.
BASELINES = {"pursuer": ["cascade"], "evader": ["apf", "cascade"]}
# A blue ramp from light to dark, and the figure's surface and ink.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def lineup(run, cfg, side):
    """The seed, the baselines, then each generation the run has finished."""

    trained = pool.members(side, cfg.generations + 1)[1:]
    return [pool.SEEDS[side], *BASELINES[side]] + [
        name for name in trained if (run / "pool" / f"{name}.pkl").exists()
    ]


def draw(rates, pursuers, evaders, out):
    fig, ax = plt.subplots(
        figsize=(0.6 * len(evaders) + 2.5, 0.45 * len(pursuers) + 1.6), facecolor=SURFACE
    )
    ax.set_facecolor(SURFACE)
    mesh = ax.pcolormesh(
        rates,
        cmap=LinearSegmentedColormap.from_list("blue", RAMP),
        vmin=0.0,
        vmax=1.0,
        edgecolors=SURFACE,
        linewidth=2,
    )
    for i, j in np.ndindex(rates.shape):
        # White on the dark half of the ramp, dark ink on the light half.
        ax.text(j + 0.5, i + 0.5, f"{rates[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="#ffffff" if rates[i, j] > 0.6 else INK)
    # A wider gap sets the seed and the baselines apart from the generations.
    ax.axhline(1 + len(BASELINES["pursuer"]), color=SURFACE, linewidth=6)
    ax.axvline(1 + len(BASELINES["evader"]), color=SURFACE, linewidth=6)

    ax.set_xticks(np.arange(len(evaders)) + 0.5, evaders, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pursuers)) + 0.5, pursuers)
    ax.tick_params(length=0, labelcolor=MUTED)
    ax.invert_yaxis()
    ax.set_xlabel("evader", color=MUTED)
    ax.set_ylabel("pursuer", color=MUTED)
    ax.set_title("Pursuer win rate", loc="left", color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--episodes", type=int, default=1024)
    args = p.parse_args()

    run = pathlib.Path(args.run)
    cfg = rundir.load_config(run)
    env, env_params = evaluate.eval_env(run)
    pursuers, evaders = lineup(run, cfg, "pursuer"), lineup(run, cfg, "evader")
    nets = {
        (side, name): pool.load_member(run, side, name)
        for side, names in (("pursuer", pursuers), ("evader", evaders))
        for name in names
        if name not in env.PILOTS
    }

    matches = {}

    def win_rate(pursuer, evader):
        sources = {
            side: name if name in env.PILOTS else None
            for side, name in (("pursuer", pursuer), ("evader", evader))
        }
        named = tuple(sorted(sources.items()))
        if named not in matches:
            matches[named] = pool.match(cfg, env, env_params, sources, "pursuer", args.episodes)
        seated = (("pursuer", pursuer), ("evader", evader))
        weights = {side: nets[side, name] for side, name in seated if (side, name) in nets}
        return float(matches[named](weights))

    rates = []
    for pursuer in pursuers:
        rates.append([win_rate(pursuer, evader) for evader in evaders])
        print(f"{pursuer:>12}  " + " ".join(f"{r:.2f}" for r in rates[-1]), flush=True)
    rates = np.array(rates)

    with (run / "matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pursuer", *evaders])
        writer.writerows([name, *row] for name, row in zip(pursuers, rates))
    draw(rates, pursuers, evaders, run / "matrix.png")
    print(f"wrote {run / 'matrix.csv'} and {run / 'matrix.png'}")


if __name__ == "__main__":
    main()
