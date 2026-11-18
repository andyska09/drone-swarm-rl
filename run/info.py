"""Print what the files of a run hold.

python run/info.py <folder> [<folder> ...] [--show <word> ...]

A folder is a run, or a folder with runs below it. Each run prints four parts:
  config   every value of config.json
  eval     every value of each evals/*/eval.json
  metrics  every column of metrics.csv, as the mean of each tenth of the run
  std      the std of each action of each role, from latest.pkl. A role that did not
           learn keeps its first std.

With no --show, every row prints. --show keeps a part by its name, and each row whose
name holds another word:
  python run/info.py runs/04_new_base --show std approx_kl caught_e
"""

import argparse
import csv
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from swarm.learn import rundir

TENTHS = 10


def flat(tree, prefix=""):
    if isinstance(tree, dict):
        return [row for key, value in tree.items() for row in flat(value, f"{prefix}{key}.")]
    return [(prefix[:-1], json.dumps(tree))]


def config(run):
    return flat(json.loads((run / "config.json").read_text()))


def evals(run):
    files = sorted(run.glob("evals/*/eval.json"))
    return [row for f in files for row in flat(json.loads(f.read_text()), f"{f.parent.name}.")]


def cell(values):
    kept = [v for v in values if v not in ("", "nan")]
    if not kept:
        return "-"
    try:
        return f"{sum(float(v) for v in kept) / len(kept):.4g}"
    except ValueError:
        return "/".join(sorted(set(kept)))


def metrics(run):
    file = run / "metrics.csv"
    if not file.exists():
        return []
    with file.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    n = len(rows)
    tenths = [rows[n * i // TENTHS : n * (i + 1) // TENTHS] for i in range(TENTHS)]
    return [(name, " ".join(cell([r[name] for r in tenth]) for tenth in tenths)) for name in rows[0]]


def std(run):
    if not (run / "latest.pkl").exists():
        return []
    saved = rundir.load(run)["params"]
    return [
        (role, " ".join(f"{math.exp(v):.4g}" for v in weights["actor"]["params"]["log_std"]))
        for role, weights in saved.items()
    ]


PARTS = {"config": config, "eval": evals, "metrics": metrics, "std": std}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+", type=pathlib.Path)
    p.add_argument("--show", nargs="+", default=[])
    args = p.parse_args()

    runs = [f.parent for folder in args.folders for f in sorted(folder.rglob("config.json"))]
    if not runs:
        p.error("these folders hold no run")
    for run in runs:
        print(f"== {run}")
        for part, rows in PARTS.items():
            for name, value in rows(run):
                if not args.show or part in args.show or any(word in name for word in args.show):
                    print(f"{part:8s}{name:36s}{value}")


if __name__ == "__main__":
    main()
