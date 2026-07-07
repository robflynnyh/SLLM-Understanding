# Symphony

This directory stores the Symphony workflow and repo-specific instructions for
SLLM-understanding.

- `WORKFLOW.md` is the workflow consumed by the local Symphony service.
- `instructions/` contains binding instructions loaded by `WORKFLOW.md`.
- `symphony/.env` is local-only and must not be committed.
- Keep temporary Symphony artifacts under `symphony/.scratch/`,
  `symphony/tmp/`, or `symphony/logs/`; these paths are ignored by Git.

Do not commit raw audio, model checkpoints, credentials, local caches, or bulky
generated run outputs.
