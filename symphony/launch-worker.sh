#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/exp/exp4/acp21rjf}"
REPO_ROOT="${REPO_ROOT:-$ROOT/SLLM-understanding}"
SESSION_NAME="${SESSION_NAME:-symphony-sllm-understanding}"
LOG_DIR="${LOG_DIR:-$ROOT/symphony-logs-SLLM-understanding}"
SHARED_ENV="${SHARED_ENV:-$ROOT/symphony-config/.env}"
REPO_ENV="${REPO_ENV:-$REPO_ROOT/symphony/.env}"
SYMPHONY_DIR="${SYMPHONY_DIR:-$ROOT/symphony/elixir}"
SYMPHONY_BIN="${SYMPHONY_BIN:-$SYMPHONY_DIR/bin/symphony}"
WORKFLOW="${WORKFLOW:-$REPO_ROOT/symphony/WORKFLOW.md}"

screen -wipe >/dev/null 2>&1 || true
mkdir -p "$LOG_DIR"

if screen -ls | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
  echo "already running: ${SESSION_NAME}"
  exit 0
fi

screen \
  -L \
  -Logfile "$LOG_DIR/symphony-service.log" \
  -dmS "$SESSION_NAME" \
  bash -lc "set -a && . '$SHARED_ENV' && . '$REPO_ENV' && set +a && cd '$SYMPHONY_DIR' && '$SYMPHONY_BIN' --i-understand-that-this-will-be-running-without-the-usual-guardrails --logs-root '$LOG_DIR' '$WORKFLOW'"

echo "started: ${SESSION_NAME}"
echo "log: ${LOG_DIR}/symphony-service.log"
