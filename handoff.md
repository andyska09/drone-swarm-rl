# Handoff — chase eval baselines

Repo: `/Users/ondrej/workspace/drone-swarm`, branch `main`, last commit `7d54585`.
Read `CLAUDE.md` and `research/notes/goals.local.md` first. Start every reply with
`TARS:` and use the simple-English output style.

## State

The eval baselines for `chase` are done and tested. **Nothing is committed.**
`pytest tests/` gives 51 passed.

- **The design, the values and the results are in
  [research/notes/baselines.md](research/notes/baselines.md).** Do not restate them here.
- New files: `swarm/sim/guidance.py`, `research/notes/baselines.md`.
- Changed files: `swarm/learn/evaluate.py`, `swarm/envs/chase.py`, `swarm/learn/runner.py`,
  `run/eval.py`, `tests/test_env.py`, `tests/test_eval.py`, `CLAUDE.md`, `handoff.md`.
- `runs/chase_baselines/` (gitignored) holds all 10 baseline matchups.
  `scratch/baseline_evals.py` rebuilds it in about a minute. Rebuild it after any change
  to `guidance.py`.

One interface changed. `make_policy` returns `act, start, step`. `start(state)` returns the
state and the first carry, and a path seat puts the evader on its path there. `rollout`
takes `start` in place of a carry.

## Open

1. **Notes that are out of date.** The user will rework `choices.md` in another chat,
   because its sentences are too long. These are still open:
   - `research/notes/choices.md` has no section on the baselines.
   - `research/notes/task_chase.md` still says `crash` and `crashed`. The code now says
     `fail`, `failed()` and the seven outcomes. Its "Baseline" section and the FRPN row in
     "Open" are old.
   - `CLAUDE.md` has no line for `sim/guidance.py`, no eval example with
     `--seat evader apf`, and no row for the new `test_eval.py` checks. Its chase paragraph
     still says `crash`.
2. **One comment with history**, in `swarm/learn/evaluate.py:93-94` ("as in Němeček's copy
   of MRS"). The user has not answered yet.
3. **Six scratch probes do not run.** Five broke before this work: `analyze_circle8`,
   `analyze_pair`, `eval_plumbing`, `probe_offset`, `probe_settle`. `net_flight.py` broke
   when `rollout` started to take `start`, and `task_chase.md` cites it. The user has not
   decided between deleting all six and fixing `net_flight.py`.
4. **Code review of the diff.** The user postponed it, and rates the code quality as low.
   Check mainly `guidance.py` and `guided_policy` against `CLAUDE.md`.
5. Commit only when the user asks.

The uncommitted edit to `CLAUDE.md` from an earlier chat stays as it is. The user decided.

The next big step is the opponent pool (AMSPBH, see `goals.local.md`). These baselines are
its generation 0.

## House rules that bit

- Be short and direct. Answer only the question asked. Explain every term in plain words.
- Ask before you add anything that was not asked for. If two designs are equally good, ask.
- Comments and docstrings say what the code does now. No history and no sources in code:
  git keeps the history, and the notes keep the sources.
- Everything stays under the repo root. Scratch goes in `scratch/`.
