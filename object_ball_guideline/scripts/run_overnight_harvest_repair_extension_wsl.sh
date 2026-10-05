#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_RUN="$REPO_ROOT/runs/supervised_train_v10_harvest_repair"
RESUME_CHECKPOINT="$REPO_ROOT/runs/supervised_train_v2/guideline_unet_best.pt"
LOG_ROOT="$REPO_ROOT/runs/overnight_harvest_repair_logs"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
mkdir -p "$LOG_ROOT"

has_process() {
  local pattern="$1"
  ps -eo cmd | grep -F "$pattern" | grep -v grep >/dev/null
}

wait_for_active_queue() {
  while has_process "scripts/run_overnight_harvest_repair_experiments_wsl.sh" || has_process "src.supervised.train"; do
    echo "[$(date -Is)] extension waiting for active training queue to finish"
    sleep 60
  done
}

wait_for_index_cache() {
  while [[ ! -f "$CACHE_RUN/index.json" || ! -f "$CACHE_RUN/index_meta.json" ]]; do
    echo "[$(date -Is)] extension waiting for index cache at $CACHE_RUN"
    sleep 60
  done
}

prepare_output_root() {
  local output_root="$1"
  mkdir -p "$output_root"
  cp "$CACHE_RUN/index.json" "$output_root/index.json"
  cp "$CACHE_RUN/index_meta.json" "$output_root/index_meta.json"
}

run_experiment() {
  local name="$1"
  local epochs="$2"
  local learning_rate="$3"
  local pos_weight="$4"
  local output_root="$REPO_ROOT/runs/$name"
  local log_path="$LOG_ROOT/${name}.log"

  echo "[$(date -Is)] extension start $name epochs=$epochs lr=$learning_rate pos_weight=$pos_weight batch_size=$BATCH_SIZE workers=$NUM_WORKERS prefetch=$PREFETCH_FACTOR"
  prepare_output_root "$output_root"
  bash "$REPO_ROOT/scripts/train_supervised_wsl.sh" \
    --output-root "$output_root" \
    --resume "$RESUME_CHECKPOINT" \
    --epochs "$epochs" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --learning-rate "$learning_rate" \
    --pos-weight "$pos_weight" \
    2>&1 | tee "$log_path"
  echo "[$(date -Is)] extension finished $name"
}

wait_for_active_queue
wait_for_index_cache

run_experiment "supervised_train_v10_repair_lr2e-4_pw24" 35 "0.0002" "24.0"
run_experiment "supervised_train_v10_repair_lr1e-4_pw18" 35 "0.0001" "18.0"
run_experiment "supervised_train_v10_repair_lr5e-5_pw18" 35 "0.00005" "18.0"

echo "[$(date -Is)] all extension harvest repair experiments finished"
