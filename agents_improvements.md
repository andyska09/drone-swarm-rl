
- nuance leads to slop
- stop agents from writing markdown/comments
- no code comments
- better experiments logs
    - no stories. just result tables or something. no prose.
    - fixed format.

- clean up of scratch - something that is used a lot should be moved to tools/

- better validation of code - needs exploring what it means in the context of a research repo
    - good code/api design
    - every breaking of a rule instead of adding a line to claude.md, generalize and make it a linter rule or test
    - better tests
    - verification process?
    

## RULES FROM CHAT 

TARS: Here is the list. Tell me if a line is wrong or if one is missing.

## Access

- I reach the cluster with `ssh rci`, with your key `~/.ssh/id_rsa`.
- On the cluster I do only this: `git pull`, submit, `squeue`, `sacct`, and `scancel` on my own jobs. I copy results with `rsync` from the Mac.
- I never compute on the login node.
- I never `git pull` on the cluster while a job waits in the queue.
- We follow the RCI rules. If I am not sure, I ask you.
- The setup goes step by step, with your word for each step.

## Approval

- You own git: you review, commit and push.
- A batch is a tracked script, `tools/rci/sweeps/<sweep>.sh`. Your push of it is your approval.
- A batch has 40 jobs at most. We change the limit as we go.
- **Changed today:** when a job fails, I do not submit it again. I find the cause, I tell you the cause and my fix, and I wait for your word.
- After a batch I stop. You pick the next batch.

## A job

- It trains, then it runs the eval of the last checkpoint only, with 4096 episodes.
- The time limit is 1 h, on `amdgpufast`.

## The score

- We rank by eval `caught_e` at the last checkpoint.
- The other numbers are context only: `coll_pp`, `death_rate_p`, and the peak of the training curve.

## The experiments

- The base: `paths_3v1`, `gamma=0.99`, `normalize_obs=false`, `adaptive_lr=false`, `lr=5e-4`, `ent_coef=0.003`, 5e8 steps.
- The MLP comes first. Then the best setup runs with the attention network.
- The order:
  1. 8 seeds of the base.
  2. The `lr` × `ent_coef` grid. It can get more knobs after we talk.
  3. The breadth pass. Each config changes one thing: one knob, or one new approach.
- The plan lives in [notes.md](notes.md). I only append there. I never change your lines.
- Env values in a grid get a new flag, `--set-env`. It goes through your code review first.
- Line 4 of your notes means a cost on the size of the network weights, in the loss.
- Line 2 means a random start time for each env, and other values of `max_steps`.

## Names and folders

- One name marks a sweep, a number and some words, for example `01_base_seeds`.
- The runs: `runs/<sweep>/<config label>/<the usual run folder>`. The label holds the values that differ from the base.
- The old runs stay where they are.
- I download each run without `checkpoints/`. `latest.pkl` comes with it.

## Logs

- The folder is `research/notes/experiment_logs/chase_sweeps/`, with one file per sweep.
- Its `README.md` is the index: the best config now, and one row per sweep. It takes the place of the `## Facts` list.
- One entry holds four things:
  1. One line: the sweep folder, the commit, the seeds, the steps.
  2. One table.
  3. Notes, 5 bullets at most.
  4. The Decision, after we talk.
- A script builds the table. I type no number by hand.
- Keep it short. No Setup text.

## My rhythm

- While a batch runs, I check the queue about every 10 minutes.

## Not decided yet

- When I write `run/sweep.py` and the README rule for a sweep entry.
- The seeds per config, and the values of the grid. They wait for the first batch.
- Where these rules go, so that a later session has them.
- If `job.sh` loads `ml` on its own. For now I submit from a login shell.
- If I cancel test job `12047911` or let it run.
- The changes in `run/replay.py` and the dashboard for the deeper folders.