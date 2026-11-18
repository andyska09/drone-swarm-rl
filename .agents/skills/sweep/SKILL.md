---
name: sweep
description: Run one designed experiment on the RCI cluster and fill in its results.
argument-hint: "The name of the experiment, for example 02_lr_ent"
disable-model-invocation: true
---

Run one designed **experiment** on RCI and fill in its results. The user gives its name. Its file, `research/notes/experiment_logs/*/<name>.md`, holds the design. Its **submit command** is `bash tools/rci/sweeps/<name>.sh`, or the command on the **Runs** line of the file when it has no batch script. The user pushed the design, and that push is the approval.

- The shape of the file is in `research/notes/experiment_logs/README.md`.
- The cluster is in `tools/rci/rci.md`. Read its step 3 and its Traps before step 1 below.
- The user owns git. Read it only.

## Steps

1. **Submit.** Run `tools/rci/local_submit.sh '<submit command>'`. A line that starts with `STOP` means that it sent no job: tell the user. Its last line, `jobs:`, holds the job numbers. Done when its `squeue` table lists every job of the batch.
2. **Watch.** Run `tools/rci/local_wait.sh <job numbers>` in the background. Done when it prints an end state for every job.
3. **Report each failed job.** A failed job stays failed. Copy its log to the Mac: `rsync -a rci:/mnt/personal/<user>/drone-swarm-rl/logs/<job id>.out scratch/`, where `<user>` is the `user` line of `ssh -G rci`. Tell the user the cause and the fix, and wait for the user's word.
4. **Download.** Run `tools/rci/local_copy.sh <name>`. Done when its count of `latest.pkl` files is the count of finished runs.
5. **Fill in the results.** Write the `## Results` part of the file, by the README. Done when the table has one row for each config of the submit command.
6. **Stop.** Send the user the link to the file.

## On the cluster

`git pull` and the submit command go only through `local_submit.sh`, which holds the checks. Beside the three scripts, run only these there: `squeue`, `sacct`, and `scancel` on your own jobs.
