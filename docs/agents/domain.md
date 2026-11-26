# Domain docs

This repo is single-context. Its domain docs live under `research/notes/`, by these names:

| A skill says | Read and write |
|---|---|
| `GLOSSARY.md` | `research/notes/tasks/defaults.md` for the words every task shares, and `research/notes/tasks/task_<name>.md` for one task |
| ADR, `docs/adr/` | none: the repo keeps no ADRs |

Before you explore an area, read the spec of each task you touch.

Write into these files in their own format, by the spec rules in `research/notes/tasks/README.md`.

## Use the glossary's words

When your output names a domain concept (in a proposal, a hypothesis or a test name), use the word exactly as `defaults.md` defines it. If the word you need is missing, tell the user: either the project does not use that idea, or the glossary has a gap.
