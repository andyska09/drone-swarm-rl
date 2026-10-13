#!/usr/bin/env python3
"""Submit run/train.py jobs to MetaCentrum PBS. See metacentrum.md.

python3 tools/metacentrum/submit.py [--seeds 0 1 2] [--dry-run] -- <train.py arguments>
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOB = Path(__file__).resolve().parent / "job.sh"


def job_name(train):
    """The start of the run folder name, so qstat and logs/ match runs/."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--task")
    p.add_argument("--preset")
    p.add_argument("--seed", default="0")
    p.add_argument("--resume")
    a, _ = p.parse_known_args(train)
    if a.resume:
        return f"resume_{Path(a.resume).name}"
    return f"{a.task}_{a.preset}_s{a.seed}"


def submit(a, train):
    cmd = ["qsub", "-N", job_name(train),
           "-l", f"select=1:ncpus={a.ncpus}:ngpus=1:mem={a.mem}:gpu_cap={a.gpu_cap}",
           "-l", f"walltime={a.walltime}",
           "-j", "oe", "-o", f"{ROOT}/logs/",
           # -v splits on commas; a value in quotes keeps them (--set hidden=128,128).
           "-v", f"ROOT={ROOT},ARGS='{' '.join(train)}'",
           str(JOB)]
    if a.dry_run:
        print("DRY RUN:", " ".join(cmd))
        return
    (ROOT / "logs").mkdir(exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"  ERROR: {r.stderr.strip()}", file=sys.stderr)
    else:
        print(f"  Submitted: {r.stdout.strip()}")


def main():
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, help="one job per seed")
    p.add_argument("--ncpus", type=int, default=2)
    p.add_argument("--mem", default="32gb")
    p.add_argument("--walltime", default="12:00:00")
    p.add_argument("--gpu-cap", default="compute_80", dest="gpu_cap")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    a = p.parse_args(argv[:cut])
    train = argv[cut + 1:]
    if not train:
        p.error("give the train.py arguments after --")

    jobs = [train + ["--seed", str(s)] for s in a.seeds] if a.seeds else [train]
    print(f"Submitting {len(jobs)} job(s)...")
    for args in jobs:
        print(f"\n[{job_name(args)}]")
        submit(a, args)


if __name__ == "__main__":
    main()
