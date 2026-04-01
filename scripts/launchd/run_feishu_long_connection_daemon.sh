#!/bin/zsh
set -euo pipefail

ROOT_DIR="${FEISHU_WORKER_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
VENV_PYTHON="${ROOT_DIR}/.codex_local/venvs/feishu_long_connection/bin/python"
LOG_DIR="${ROOT_DIR}/.codex_local/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "[feishu-worker] missing python runtime: ${VENV_PYTHON}" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}"
export PYTHONUNBUFFERED=1

# Keep the worker alive through lock screen and idle periods.
# On macOS laptops, closing the lid may still sleep unless AC/clamshell conditions are met.
exec /usr/bin/caffeinate -dims \
  "${VENV_PYTHON}" \
  "${ROOT_DIR}/scripts/run_feishu_long_connection.py" \
  --project-root "${ROOT_DIR}" \
  --log-level INFO
