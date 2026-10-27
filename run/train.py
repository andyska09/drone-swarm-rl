"""Train a policy: python run/train.py --task circle_swap --preset pair --steps 5e6"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from swarm.learn import config, runner


def cast(current, text):
    if isinstance(current, bool):
        return text.lower() in ("1", "true", "yes")
    if isinstance(current, tuple):
        return tuple(int(v) for v in text.split(","))
    return type(current)(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task")
    p.add_argument("--preset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", "--steps", type=lambda value: int(float(value)))
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="any other TrainConfig field")
    p.add_argument("--out", default="runs")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--resume", metavar="RUN",
                   help="run a run on from its last checkpoint; every other flag is ignored")
    args = p.parse_args()

    if args.resume:
        path, _ = runner.resume(args.resume, use_wandb=args.wandb)
        print(f"wrote {path}")
        return
    if args.task is None or args.preset is None:
        p.error("--task and --preset are required")

    cfg = config.TrainConfig(task=args.task, preset=args.preset, seed=args.seed)
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    for override in args.set:
        name, _, text = override.partition("=")
        setattr(cfg, name, cast(getattr(cfg, name), text))

    path, _ = runner.train(cfg, root=args.out, use_wandb=args.wandb)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
