"""Put runs side by side: python run/compare.py runs/<a> runs/<b> ...

First every eval that the runs hold, one column each. Then each run's training curve: for
each role that learns, the policy's std and its lr / std, then the task's PROGRESS metrics."""

import argparse
import csv
import importlib
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from swarm.learn import runner

UPDATES = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
WINDOW = 10  # each point of the curve averages the updates this close to it
ONE_ACTION = 0.5 * (math.log(2 * math.pi) + 1)  # the entropy of one action at std 1


def number(text):
    return float(text) if text else math.nan


def mean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def print_evals(runs):
    found = [
        (f"{run.name[-6:]}/{path.parent.name}", json.loads(path.read_text()))
        for run in runs
        for path in sorted((run / "evals").glob("*/eval.json"))
    ]
    if not found:
        return
    width = max(len(label) for label, _ in found) + 2
    print(" " * 16 + "".join(label.rjust(width) for label, _ in found))
    print("update".rjust(16) + "".join(str(e["update"]).rjust(width) for _, e in found))
    for key in sorted({k for _, e in found for k in e["summary"]}):
        values = (e["summary"].get(key, math.nan) for _, e in found)
        print(key.rjust(16) + "".join(f"{v:.3f}".rjust(width) for v in values))


def print_curve(run):
    spec = json.loads((run / "config.json").read_text())
    env = importlib.import_module(f"swarm.envs.{spec['train']['task']}")
    with (run / "metrics.csv").open() as f:
        rows = [{k: number(v) for k, v in row.items()} for row in csv.DictReader(f)]

    # A role that a pilot flies leaves its entropy column empty.
    learners = [
        k[len("entropy"):] for k in runner.progress_columns(rows[0], ("entropy",))
        if any(not math.isnan(row[k]) for row in rows)
    ]
    columns = ["update", "steps"]
    columns += [f"{name}{r}" for r in learners for name in ("std", "lr", "lr/std")]
    columns += runner.progress_columns(rows[0], env.PROGRESS)

    last = int(rows[-1]["update"])
    print(f"\n{run}  commit {spec['git']['commit'][:7]}  {last} updates")
    print("".join(c.rjust(13) for c in columns))
    for u in [u for u in UPDATES if u < last] + [last]:
        near = [row for row in rows if abs(row["update"] - u) <= WINDOW]
        point = {k: mean(row[k] for row in near) for k in near[0]}
        for r in learners:
            point[f"std{r}"] = math.exp(point[f"entropy{r}"] / env.NUM_ACTIONS - ONE_ACTION)
            point[f"lr/std{r}"] = point.get(f"lr{r}", math.nan) / point[f"std{r}"]
        print(f"{u:13d}" + "".join(f"{point.get(c, math.nan):13.4g}" for c in columns[1:]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=pathlib.Path)
    args = p.parse_args()

    print_evals(args.runs)
    for run in args.runs:
        print_curve(run)


if __name__ == "__main__":
    main()
