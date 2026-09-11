"""Train a policy: python run/train.py --task a_to_b --preset default --steps 5e6"""

import argparse
import dataclasses
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
    p.add_argument("--task", default="a_to_b")
    p.add_argument("--preset", default="default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", "--steps", type=lambda value: int(float(value)))
    p.add_argument("--num-envs", type=int)
    p.add_argument("--num-steps", type=int)
    p.add_argument("--num-minibatches", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="any other TrainConfig field")
    p.add_argument("--out", default="runs")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    cfg = config.TrainConfig()
    for field in dataclasses.fields(cfg):
        value = getattr(args, field.name, None)
        if value is not None:
            setattr(cfg, field.name, value)
    for override in args.set:
        name, _, text = override.partition("=")
        setattr(cfg, name, cast(getattr(cfg, name), text))

    path, _ = runner.train(cfg, root=args.out, use_wandb=args.wandb)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
