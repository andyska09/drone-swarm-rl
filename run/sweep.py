import argparse
import json
import pathlib

SCORE = "caught_e"
CONTEXT = ("coll_pp", "death_rate_p", "time_to_catch")


def mean(values):
    return sum(values) / len(values)


def own_eval(run):
    config = json.loads((run / "config.json").read_text())
    seats = {role: "checkpoint" for role in config["env"]["roles"]}
    seats.update([config["env"]["piloted"]] if config["env"]["piloted"] else [])
    own = (seats, config["train"]["preset"], "latest")
    for file in sorted(run.glob("evals/*/eval.json")):
        found = json.loads(file.read_text())
        if (found["seats"], found["preset"], found["checkpoint"]) == own:
            return found["summary"]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep", type=pathlib.Path)
    sweep = p.parse_args().sweep

    rows = []
    for config in sorted(d for d in sweep.iterdir() if d.is_dir()):
        runs = [d for d in config.iterdir() if d.is_dir()]
        evals = [found for found in map(own_eval, runs) if found]
        if evals:
            score = [e[SCORE] for e in evals]
            context = [mean([e[k] for e in evals]) for k in CONTEXT]
            rows.append((mean(score), min(score), max(score), *context, config.name, len(evals), len(runs)))

    print(f"| config | seeds | {SCORE} | min | max | " + " | ".join(CONTEXT) + " |")
    print("|---" * (5 + len(CONTEXT)) + "|")
    for *numbers, name, done, total in sorted(rows, reverse=True):
        print(f"| {name} | {done}/{total} | " + " | ".join(f"{v:.3f}" for v in numbers) + " |")


if __name__ == "__main__":
    main()
