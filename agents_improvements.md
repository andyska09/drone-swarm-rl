
- nuance leads to slop
- stop agents from writing markdown/comments
- no code comments
- better experiments logs
    - no stories. just result tables or something. no prose.
    - fixed format.

- better validation of code - needs exploring what it means in the context of a research repo
    - good code/api design
    - every breaking of a rule instead of adding a line to claude.md, generalize and make it a linter rule or test
    - better tests
    - better verification process/loop?
    

## Not decided yet

- Env values in a grid get a new flag, `--set-env`. It goes through your code review first.
- The seeds per config, and the values of the grid. They wait for the first batch.
- If `job.sh` loads `ml` on its own. For now I submit from a login shell.
- The changes in `run/replay.py` and the dashboard for the deeper folders.
- How the code handles a drone whose state turns NaN. Now each task has its own NaN check, and `ppo.py` hides a NaN value in the bootstrap with `isfinite`. A fix in `fly` with a flag for each task was too complicated, and I took it back. It needs a better design first.
