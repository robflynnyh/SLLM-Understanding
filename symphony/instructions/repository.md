# Repository

This repository prepares and evaluates speech-language model experiments on
EmoNet-Voice Bench.

Important paths:

- `configs/` contains dataset, prompt, and model configuration.
- `scripts/` contains dataset preparation, request building, model runners, and
  metric summarization.
- `runs/`, `scratch/`, `.cache/`, `.uv-cache/`, and `.venv-*` are local or
  generated paths and are ignored by Git.
- `symphony/` contains Symphony workflow configuration and future-agent
  instructions. `symphony/.env` is local-only and must not be committed.

Do not commit raw audio, Hugging Face cache files, model checkpoints, local
virtualenvs, credentials, or bulky generated run artifacts.

Commit small durable scripts, configs, docs, and lightweight summaries when
they are part of the requested deliverable. For large generated outputs, record
the external path, generation command, and validation command instead.

Keep edits scoped to the issue. Prefer existing request builders, model runners,
and summary scripts over introducing new ad hoc evaluation paths.

Use `rg`/`rg --files` for searching. Keep broad scans bounded and avoid dumping
large generated JSONL outputs into comments or final responses.
