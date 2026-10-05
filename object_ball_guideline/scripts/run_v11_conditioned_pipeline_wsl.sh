#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.wsl_envs/guideline_train/bin/python"
GOLD_ROOT="${GOLD_ROOT:-$REPO_ROOT/runs/gold_board_v1}"
DIAG_ROOT="${DIAG_ROOT:-$GOLD_ROOT/current_best_diagnostics}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  bash "$REPO_ROOT/tools/setup_supervised_train_wsl.sh"
fi

"$PYTHON_BIN" "$REPO_ROOT/tools/build_gold_board.py" \
  --index-json "$REPO_ROOT/runs/supervised_train_v2/index.json" \
  --data-root "/mnt/c/My_Project/SAM_3/guideline_line/data_zoomprobe" \
  --rejected-root "/mnt/c/My_Project/SAM_3/guideline_line/data_zoomprobe/Rejected" \
  --negative-root "/mnt/c/My_Project/8_BALL_POOL/data/images/negative_selected" \
  --output-root "$GOLD_ROOT" \
  --holdout-fraction 0.2 \
  --frame-bucket-size 25

"$PYTHON_BIN" "$REPO_ROOT/tools/diagnose_guideline_failures.py" \
  --gold-board "$GOLD_ROOT/gold_board.json" \
  --manifest "$REPO_ROOT/runs/supervised_best_manifest.json" \
  --output-root "$DIAG_ROOT" \
  --split old_val \
  --split old_hard_val \
  --split harvest_holdout \
  --split rejected_negatives \
  --split flat_negatives \
  --max-candidates 24 \
  --crop-batch-size 32 \
  --save-outputs

"$PYTHON_BIN" "$REPO_ROOT/tools/make_guideline_contact_sheet.py" \
  --diagnostics-json "$DIAG_ROOT/diagnostics.json" \
  --output "$DIAG_ROOT/worst50_contact_sheet.png" \
  --max-items 50

bash "$REPO_ROOT/scripts/train_conditioned_supervised_wsl.sh"
