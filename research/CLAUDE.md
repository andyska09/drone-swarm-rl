# research/ — notes, papers and the reference code

- `notes/` — working notes `design_choices.md` holds the shared design
  decisions and their reasons. `tasks/` holds the task specs: `defaults.md` for what
  every task shares, and `task_<name>.md` for one task. `experiment_logs/` holds the
  experiments, one file each. `literature/` holds the literature reviews,
  and `literature/specs/` the paper specs, which `tools/specs/index.html` shows beside
  our task specs. `learning/` holds explainers of the theory, and `archive/` the notes
  that no longer hold.
- `code_sources/` (GITIGNORED) — reference code, **not ours**:
  `mrs_multirotor_simulator/` (the C++ we reimplement), `PPO_example/` (the
  supervisor's JAX PPO), `quad-swarm-rl/` (PyTorch swarm, upstream clone),
  `rl_tracking_eagle-cleaned-up-logic/` (Němeček's pursuit-evasion code). **To run**
  `PPO_example` or `quad-swarm-rl`, read [reference_code.md](notes/reference_code.md).
- `quad-swarm-rl` is a **live clone with its own `.git` and an `origin` pointing at
  the upstream GitHub repo**. Never commit or push from inside it. Treat every
  `code_sources/` tree as a read-only reference unless the user says otherwise.
- **Papers: grep the transcript, don't open the PDF.** Every PDF in `papers/` has a
  `pdftotext -layout` transcript as a sibling `.md`; open the PDF only for
  figures. [bibliography_index.md](papers/bibliography_index.md) is the
  entry point — a tracked table of paper → files → one-line summary. Add a row
  whenever a PDF of a published paper is added, and generate its transcript with
  `pdftotext -layout <file>.pdf <file>.md`.
- The PDFs and their transcripts are gitignored. Of `papers/`, only
  `bibliography_index.md` is tracked. `.git/info/exclude`, which is itself untracked,
  excludes the unpublished papers, so no tracked file names them.
