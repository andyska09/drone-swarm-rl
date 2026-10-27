# Coding standards

The rules for code and tests in this repo. `/code-review` checks a diff against them.

## Code style

- Blunt, minimal code. Ideally one file readable top to bottom. KISS and DRY.
- Comments: as few as possible. Write one only where the code cannot say it itself: a
  constraint, a non-obvious reason, a trap. Never restate what the line does. One clear
  sentence beats three. No banner blocks, no section dividers.
- **Comments and docstrings say what the code does now.** No story and no claim about
  what the code does not do. No history and no sources in code: git keeps the history,
  and the notes keep the sources.
- Docstrings: a module gets a short one. A function gets one only if its contract is not
  obvious from its name and signature.

## Tests

- A test checks behaviour that matters: a physics result, a rule of a task, or a promise
  that other code relies on.
- Keep each test fast: small batches and short rollouts, so `pytest tests/` stays quick.
