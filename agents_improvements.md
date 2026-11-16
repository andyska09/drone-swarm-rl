
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
    

## Not decided yet

- Env values in a grid get a new flag, `--set-env`. It goes through your code review first.
- The seeds per config, and the values of the grid. They wait for the first batch.
- If `job.sh` loads `ml` on its own. For now I submit from a login shell.
- The changes in `run/replay.py` and the dashboard for the deeper folders.
