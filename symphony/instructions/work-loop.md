# Work Loop

Inspect repository state and task context before editing.

Make a concise plan and identify validation for the specific change.

If the issue description includes `Branch/ref: <name>`, fetch and check out
that branch or ref before editing. Confirm the checked-out commit with:

```bash
git status
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

Create a working branch named `symphony/<issue-identifier>-<short-slug>` from
the checked-out base branch. Do not commit directly to the base branch.

Keep edits narrowly scoped to the issue. Prefer the existing configuration,
request-building, runner, and summarization paths.

Do not use `/tmp` or `/home` for meaningful scratch, caches, generated outputs,
or downloaded artifacts. Use repo-local ignored paths or store paths already
documented in `dataset-and-models.md`.

During nontrivial work, periodically post concise Linear progress comments for
meaningful implementation progress, design decisions, run launches, blockers,
or validation changes.

If a result is partial or still running, label it as a snapshot rather than a
final result.
