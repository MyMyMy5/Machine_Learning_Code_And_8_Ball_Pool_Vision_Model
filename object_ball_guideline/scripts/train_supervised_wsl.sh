#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.wsl_envs/guideline_train/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  bash "$REPO_ROOT/tools/setup_supervised_train_wsl.sh"
fi

DATA_ROOT="/mnt/c/My_Project/SAM_3/guideline_line/data_zoomprobe"
OUTPUT_ROOT="$REPO_ROOT/runs/supervised_train"
NEGATIVE_ARGS=()

REJECTED_ROOT="$DATA_ROOT/Rejected"
if [[ -d "$REJECTED_ROOT" ]]; then
  NEGATIVE_ARGS+=(--negative-root "$REJECTED_ROOT")
fi

SELECTED_NEGATIVE_ROOT="/mnt/c/My_Project/8_BALL_POOL/data/images/negative_selected"
if [[ -d "$SELECTED_NEGATIVE_ROOT" ]]; then
  NEGATIVE_ARGS+=(--negative-root "$SELECTED_NEGATIVE_ROOT")
fi

"$PYTHON_BIN" -m src.supervised.train \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  "${NEGATIVE_ARGS[@]}" \
  "$@"
