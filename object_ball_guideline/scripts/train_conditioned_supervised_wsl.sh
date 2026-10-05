#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.wsl_envs/guideline_train/bin/python"
DATA_ROOT="${DATA_ROOT:-/mnt/c/My_Project/SAM_3/guideline_line/data_zoomprobe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs/supervised_train_v11_object_ball_imgneg_conditioned_unet}"
GOLD_ROOT="${GOLD_ROOT:-$REPO_ROOT/runs/gold_board_v1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$REPO_ROOT/runs/supervised_train_v2/guideline_unet_best.pt}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EPOCHS="${EPOCHS:-35}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
POS_WEIGHT="${POS_WEIGHT:-18.0}"
CANDIDATE_ONLY="${CANDIDATE_ONLY:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  bash "$REPO_ROOT/tools/setup_supervised_train_wsl.sh"
fi

"$PYTHON_BIN" -m src.supervised.train_conditioned \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --learning-rate "$LEARNING_RATE" \
  --pos-weight "$POS_WEIGHT" \
  --negative-root "$DATA_ROOT/Rejected" \
  --negative-root "/mnt/c/My_Project/8_BALL_POOL/data/images/negative_selected" \
  --negative-crops-per-image 4 \
  --exclude-ids-json "$GOLD_ROOT/train_exclude_ids.json" \
  --resume "$RESUME_CHECKPOINT" \
  $([[ "$CANDIDATE_ONLY" == "1" ]] && printf '%s' "--candidate-only")
