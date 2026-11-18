# Experiment logs

One file for each experiment, in a folder for its line of work: `<line>/<name>.md`. An
experiment is one question and the runs that answer it. `run/sweep.py` and
`tools/rci/sweeps/` call the same thing a sweep.

The name is a number and some words, `02_lr_ent`. The same name marks the runs,
`runs/<name>/`, and the batch script, `tools/rci/sweeps/<name>.sh`.

A file holds the design, the tables, and short lines that state what a table shows. What
we do next goes into the question of the next experiment.

Write every file in the
[simple-english](../../../.claude/output-styles/simple-english.md) output style.

## The design

The top of the file. Before the run, the agent writes a draft and lists its picks. The
user edits it:

```markdown
# 02_lr_ent

**Question:** Which `lr` and `ent_coef` give the highest catch in `paths_3v1`?
**Differs from** `01_base_seeds`: `lr` 2.5e-4, 5e-4, 1e-3, 2e-3 × `ent_coef` 0.001, 0.003, 0.01, 0.03. Seeds 0 and 1.
**Runs:** RCI, `tools/rci/sweeps/02_lr_ent.sh`, 32 jobs.
```

- The exact configs live in the batch script, or in the command on the **Runs** line for a
  single run. The file points to them.
- Each config trains into `--out runs/<name>/<label>`, one run per seed. The label names
  the values that differ from the base.
- A batch script on RCI holds 40 jobs at most.

The design is done when the user pushes the file, and the batch script if it has one. The
push is the approval.

## The results

Under `## Results`, after the run:

1. The table that `python run/sweep.py runs/<name>` prints, as it is.
2. At most 3 lines. Each line states only what the table shows.

An analysis that the user asks for adds one table, under one line that says what it
measures. Every table in a file is one that the design or the user asked for.

## The old logs

The `<name>_log.md` files in this folder are from before 2026-11-16. They stay as they
are. Their `## Facts` lists still hold what those runs showed.
