# Task specs

A task spec, `task_<name>.md`, says what the task is now: the model, not the code.
`defaults.md` holds what every task shares, and it is the template.
`env_contract.md` says what a task file holds in code. The spec rules do not cover it.

`chase` has two specs: `task_chase_1v1.md` for its presets with one pursuer, and
`task_chase_3v1.md` for its presets with three. So each one stands beside the papers of
its kind on the spec page. A change to `chase` goes into both.

## Spec rules

- Describe the task at the level of a paper's method section. Name no function, field,
  file or array.
- Say what the task does. Do not say what it does not do, and do not set "X is"
  against "Y is not".
- Write `Setup` and `Events` in plain words, with a table of values. Keep the symbols,
  formulas and tables of every other section.
- Describe only what is built now. No reasons, no history, no plans, no open questions.
- List only what differs from `defaults.md`, in its sections and in its order. A
  section with no change holds one word: `Default.`
- `Events` says when an event happens and how the episode ends. `Reward` says who
  pays for it. Neither repeats the other.
- Use tables, formulas and short lists. Use a paragraph only for the purpose at the top.
- Give each value once, with its unit.
- A table of values can end with a `source` column. A cell links to the paper spec that
  the value comes from, or says `ours`. A cell with no known source stays empty.
- One word, one meaning. `defaults.md` holds the words and the shared symbols.
- Write it in the [simple-english](../../../.claude/output-styles/simple-english.md) style.
- Write formulas in LaTeX: `$...$` inline, and a `$$` line before and after a block.
  Do not put `|` in math, because it breaks a table row. Write `\lVert x \rVert` and
  `\lvert x \rvert`.
- Check each line against the code when you write or change a spec.
