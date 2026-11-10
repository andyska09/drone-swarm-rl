# Experiment logs

One log for each line of work, `<name>_log.md`: why the setup is as it is, what each run
scored, and the open questions. What a task is now lives in its spec in
[tasks/](../tasks/).

## Facts

Under its title and first paragraph, each log holds a `## Facts` list: every lesson that
still holds, one line each, with its date and a link to the section that shows it. Read
it before you pick the next run. When a result changes a fact, change its line or delete
it.

## An entry

One `## <what>, <date>` section for each batch of runs or each check, in four parts:

1. **Setup**: what ran, and how it differs from the entry before: the commit, the preset,
   the `--set` values, the steps and the seeds.
2. **Results**: a table.
3. **What it shows**: only what the numbers in the table show.
This 4. step is only done after a discussion with the user. He is the one that has the final word.
4. **Decision**: what we do next because of it, and the questions it leaves open.

Each claim in an entry rests on a number in its table. A guess stays in the chat, and the
plan goes to `handoff.md`.
