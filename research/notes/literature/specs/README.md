# Paper specs

A paper spec, `<authors>_<year>.md`, says what the task and the training of one paper are,
in the form of our task specs. The spec page, `tools/specs/index.html` served by
`run/replay.py`, shows any two specs side by side, one `##` section at a time.

## Rules

Follow the [spec rules](../../tasks/README.md), with these changes:

- Describe what the paper describes. Check each line against the paper itself, not
  against a note about it.
- Use the `##` headings of our task specs, in this order: Setup, Start, Drone,
  Observation, Events, Reward, Metrics, Training, Evaluation, Presets. A `###` part sits
  inside its `##` section.
- The spec is complete. Each section says what the paper does, and never `Default.`
- A value that the paper does not give reads `not published`.
- Use our words and symbols from [defaults.md](../../tasks/defaults.md). Give the paper's
  own name in brackets the first time: $\lambda_{\text{PP}}$ (his
  $\lambda_{\text{collPP}}$). A word that `defaults.md` does not have gets a short
  meaning where it first appears.
- Each value names its place in the paper. A table of values ends with a `source`
  column: `§III-B`, `eq. 7`, `Table 2`. A list item ends with its source in brackets.
- The setups of one paper go in `## Presets`, as a table of differences, like the
  presets of a task spec.
- The `#` title is the paper's short name, for example `Gavin & Bronz 2026, Nv1`. The
  page menu shows it.
- Only a paper that is publicly published gets a spec.
