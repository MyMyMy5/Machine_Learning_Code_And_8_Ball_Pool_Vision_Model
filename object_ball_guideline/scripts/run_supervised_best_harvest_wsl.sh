#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.wsl_envs/guideline_train/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  bash "$REPO_ROOT/tools/setup_supervised_train_wsl.sh"
fi

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "$REPO_ROOT/tools/run_supervised_best_harvest.py" "$@"
