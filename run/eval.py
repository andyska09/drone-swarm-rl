"""Measure a run: python run/eval.py runs/<name> --episodes 1024 [--seat evader apf]"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from swarm.learn import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument("--checkpoint", default="latest", help="'latest', an update number, or a path")
    p.add_argument("--seat", action="append", nargs=2, default=[], metavar=("ROLE", "SOURCE"),
                   help="who flies one role: a name from the task's PILOTS, another run "
                        "directory, or a .pkl")
    p.add_argument("--name", help="subdirectory under evals/")
    p.add_argument("--preset", help="fly another preset of the run's task, for example a bigger arena")
    args = p.parse_args()

    path, summary = evaluate.evaluate(
        args.run, args.episodes, args.checkpoint, args.seat, args.name, args.preset
    )
    for key in sorted(summary):
        print(f"{key:>24}  {summary[key]:10.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
