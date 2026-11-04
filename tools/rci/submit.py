#!/usr/bin/env python3
"""Submit run/train.py jobs to the RCI cluster (SLURM). See rci.md.

python3 tools/rci/submit.py [--seeds 0 1 2] [--dry-run] -- <train.py arguments>
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOB = Path(__file__).resolve().parent / "job.sh"
# The A100 partitions and their time limits in hours, shortest first.
PARTITIONS = {"amdgpufast": 4, "amdgpu": 24, "amdgpulong": 72, "amdgpuextralong": 504}


def job_name(train):
    """The start of the run folder name, so squeue and runs/ match."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--task")
    p.add_argument("--preset")
    p.add_argument("--seed", default="0")
    p.add_argument("--resume")
    a, _ = p.parse_known_args(train)
    if a.resume:
        return f"resume_{Path(a.resume).name}"
    return f"{a.task}_{a.preset}_s{a.seed}"


def partition(walltime):
    """The shortest partition that takes this walltime, given as H:MM:SS."""
    hours = int(walltime.split(":")[0])
    return next(name for name, limit in PARTITIONS.items() if hours < limit)


def submit(a, train):
    cmd = ["sbatch", f"--job-name={job_name(train)}",
           f"--partition={partition(a.walltime)}",
           "--gres=gpu:a100:1", f"--cpus-per-task={a.ncpus}", f"--mem={a.mem}",
           f"--time={a.walltime}",
           f"--chdir={ROOT}", f"--output={ROOT}/logs/%j.out",
           str(JOB)] + train
    if a.dry_run:
        print("DRY RUN:", " ".join(cmd))
        return
    (ROOT / "logs").mkdir(exist_ok=True)
    subprocess.run(cmd, check=False)


def main():
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, help="one job per seed")
    p.add_argument("--ncpus", type=int, default=2)
    p.add_argument("--mem", default="32G")
    p.add_argument("--walltime", default="12:00:00")
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
