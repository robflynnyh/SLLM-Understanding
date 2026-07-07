# Validation And Handoff

Run the most targeted command or test that demonstrates the task is complete.

For code changes, run syntax checks and the smallest relevant smoke test. Common
checks:

```bash
python -m py_compile scripts/*.py
python scripts/build_emonet_requests.py --help
python scripts/summarize_emonet_predictions.py --help
```

For dataset/request changes, build a small request file and inspect one row.

For model-runner changes, run a small limit such as `--limit 5` before any
larger run. Preserve raw outputs and parse errors.

For metric changes, validate against an existing prediction file, for example:

```bash
python scripts/summarize_emonet_predictions.py \
  --predictions runs/moss_train100_target_paper_0_10_raw.jsonl
```

For documentation-only changes, run:

```bash
git diff --check
```

If validation cannot run, document the exact blocker and the command that
should be run later.

Commit completed changes on the issue branch, push the branch to `origin`, and
open a GitHub pull request using:

```bash
/exp/exp4/acp21rjf/scripts/github-create-pr.sh \
  --base <base-branch> \
  --head <pushed-branch> \
  --title "<PR title>" \
  --body-file <pr-body.md>
```

Use the issue `Branch/ref` as the PR base when provided; otherwise use the
repository default branch.

Post one Linear completion comment summarizing files changed, validation,
output paths if any, GitHub PR URL, and residual risk.

Move the issue to `In Review` only when the requested work is complete and the
GitHub handoff has succeeded. Do not move implementation work directly to
`Done`; leave final acceptance to a human reviewer.

Do not move the issue to `In Review` if the work is incomplete, blocked, not
pushed, or missing a PR. Post a blocker comment with the exact missing step or
failure instead.
