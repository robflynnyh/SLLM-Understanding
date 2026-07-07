---
tracker:
  kind: linear
  project_slug: "sllm-understanding-0a901958c5f5"
  api_key: $LINEAR_API_KEY
  active_states:
    - Todo
    - In Progress
  terminal_states:
    - Closed
    - Cancelled
    - Canceled
    - Duplicate
    - Done
polling:
  interval_ms: 30000
workspace:
  root: /exp/exp4/acp21rjf/symphony-workspaces-SLLM-understanding
hooks:
  timeout_ms: 120000
  after_create: |
    if [ -f /exp/exp4/acp21rjf/SLLM-understanding/symphony/.env ]; then
      set -a
      . /exp/exp4/acp21rjf/SLLM-understanding/symphony/.env
      set +a
    fi
    git clone --depth 1 "$SOURCE_REPO_URL" .
    if [ -n "${SOURCE_REF:-}" ]; then
      git fetch --depth 1 origin "$SOURCE_REF"
      git checkout -B symphony-source FETCH_HEAD
    fi
agent:
  max_concurrent_agents: 1
  max_turns: 30
codex:
  command: codex --config shell_environment_policy.inherit=all --config model_reasoning_effort=medium app-server
  approval_policy: never
  thread_sandbox: danger-full-access
  issue_label_overrides: true
  turn_sandbox_policy:
    type: dangerFullAccess
---

You are working on Linear issue {{ issue.identifier }} for the
SLLM-understanding repository.

Title: {{ issue.title }}
Current status: {{ issue.state }}
URL: {{ issue.url }}

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

Required instruction index:
- Before planning or editing, read every file in `symphony/instructions/` in
  the order below.
- Treat those files as binding project instructions, not optional background.
- In your plan, state that you read them and note any instruction that directly
  affects the issue.

Read order:
1. `symphony/instructions/linear-context.md`
2. `symphony/instructions/repository.md`
3. `symphony/instructions/dataset-and-models.md`
4. `symphony/instructions/work-loop.md`
5. `symphony/instructions/experiment-execution.md`
6. `symphony/instructions/validation-and-handoff.md`

Instruction map:
- Linear issue context, comment rereads, and clarification handling:
  `symphony/instructions/linear-context.md`
- Repository layout, artifact policy, and commit boundaries:
  `symphony/instructions/repository.md`
- EmoNet-Voice dataset roots, manifest policy, model paths, and default
  evaluation mode: `symphony/instructions/dataset-and-models.md`
- Branch setup, planning, implementation style, and progress comments:
  `symphony/instructions/work-loop.md`
- Local/Mimas execution, GPU queueing, callbacks, and long-run discipline:
  `symphony/instructions/experiment-execution.md`
- Validation, GitHub PR handoff, Linear review handoff, and final state rules:
  `symphony/instructions/validation-and-handoff.md`
