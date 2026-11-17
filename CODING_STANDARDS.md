# Coding standards

The rules for code and tests in this repo. `/code-review` checks a diff against them.

## Code style

- Blunt, minimal code. Ideally one file readable top to bottom. KISS and DRY.
- **No comments and no docstrings.** `tests/test_conventions.py` fails on one. Just design and write better code. If a rule must hold, write a test for it.
  Two things stay: a `#!` line, and the docstring of a script that prints it as its
  `--help` text.

## Tests

- A test checks behaviour that matters: a physics result, a rule of a task, or a promise
  that other code relies on.
- Keep each test fast: small batches and short rollouts, so `pytest tests/` stays quick.
