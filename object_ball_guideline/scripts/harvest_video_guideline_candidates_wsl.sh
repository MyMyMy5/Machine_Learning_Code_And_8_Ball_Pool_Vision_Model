#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.wsl_envs/guideline_train/bin/python"
SAM3_ROOT='/mnt/c/My_Project/SAM_3'
EXTRACT_SCRIPT="$SAM3_ROOT/guideline_line/extract_video_frames.py"

VIDEO_DIR='/mnt/c/My_Project/SAM_3/guideline_line/video_drop_model_harvest'
FRAMES_DIR='/mnt/c/My_Project/SAM_3/guideline_line/video_frames_model_harvest'
INFERENCE_DIR='/mnt/c/My_Project/SAM_3/guideline_line/video_infer_model_harvest'
HARVEST_DIR='/mnt/c/My_Project/SAM_3/guideline_line/video_infer_harvest_2'

SAMPLE_FPS='1.0'
EXTRACT_WORKERS='4'
EXTRACT_BACKEND='ffmpeg'
FFMPEG_HWACCEL='cuda'
DEDUP='0'
DETECTION_MIN_SCORE='-999.0'
DETECTION_MIN_MASK_PIXELS='1'
CROP_BATCH_SIZE='32'
NUM_WORKERS='4'
RECURSE_VIDEOS='0'
SKIP_EXISTING_INFERENCE='0'

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-dir)
      VIDEO_DIR="$2"
      shift 2
      ;;
    --frames-dir)
      FRAMES_DIR="$2"
      shift 2
      ;;
    --inference-dir)
      INFERENCE_DIR="$2"
      shift 2
      ;;
    --harvest-dir)
      HARVEST_DIR="$2"
      shift 2
      ;;
    --sample-fps)
      SAMPLE_FPS="$2"
      shift 2
      ;;
    --extract-workers)
      EXTRACT_WORKERS="$2"
      shift 2
      ;;
    --extract-backend)
      EXTRACT_BACKEND="$2"
      shift 2
      ;;
    --ffmpeg-hwaccel)
      FFMPEG_HWACCEL="$2"
      shift 2
      ;;
    --dedup)
      DEDUP='1'
      shift
      ;;
    --no-dedup)
      DEDUP='0'
      shift
      ;;
    --detection-min-score)
      DETECTION_MIN_SCORE="$2"
      shift 2
      ;;
    --detection-min-mask-pixels)
      DETECTION_MIN_MASK_PIXELS="$2"
      shift 2
      ;;
    --crop-batch-size)
      CROP_BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --recurse-videos)
      RECURSE_VIDEOS='1'
      shift
      ;;
    --skip-existing-inference)
      SKIP_EXISTING_INFERENCE='1'
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  bash "$REPO_ROOT/tools/setup_supervised_train_wsl.sh"
fi

if [[ ! -f "$EXTRACT_SCRIPT" ]]; then
  echo "Missing extractor: $EXTRACT_SCRIPT" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 && [[ "$EXTRACT_BACKEND" == "ffmpeg" ]]; then
  echo "ffmpeg is not installed in WSL; install it or pass --extract-backend opencv" >&2
  exit 1
fi

mkdir -p "$VIDEO_DIR" "$FRAMES_DIR" "$HARVEST_DIR"

echo "Video drop folder: $VIDEO_DIR"
echo "Extracted frames:  $FRAMES_DIR"
echo "Legacy inference:  $INFERENCE_DIR"
echo "Harvest output:    $HARVEST_DIR"
echo "Extraction backend: $EXTRACT_BACKEND"
echo "ffmpeg hwaccel:     $FFMPEG_HWACCEL"
echo "Dedup enabled:      $DEDUP"

echo "Extracting frames..."
extract_args=(
  "$EXTRACT_SCRIPT"
  "--video-dir" "$VIDEO_DIR"
  "--output-root" "$FRAMES_DIR"
  "--backend" "$EXTRACT_BACKEND"
  "--mode" "sample_fps"
  "--sample-fps" "$SAMPLE_FPS"
  "--image-ext" ".jpg"
  "--jpeg-quality" "95"
  "--workers" "$EXTRACT_WORKERS"
)
if [[ "$EXTRACT_BACKEND" == "ffmpeg" ]]; then
  extract_args+=("--ffmpeg-hwaccel" "$FFMPEG_HWACCEL")
fi
if [[ "$DEDUP" == "1" ]]; then
  extract_args+=(
    "--dedup"
    "--dedup-threshold" "0.014"
    "--dedup-center-crop-ratio" "0.82"
    "--dedup-resize-width" "224"
    "--dedup-blur-kernel" "5"
    "--min-gap-seconds" "0.0"
  )
else
  extract_args+=(
    "--no-dedup"
    "--dedup-threshold" "0.014"
    "--dedup-center-crop-ratio" "0.82"
    "--dedup-resize-width" "224"
    "--dedup-blur-kernel" "5"
    "--min-gap-seconds" "0.0"
  )
fi
if [[ "$RECURSE_VIDEOS" == "1" ]]; then
  extract_args+=("--recursive")
fi
"$PYTHON_BIN" "${extract_args[@]}"

if ! find "$FRAMES_DIR" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' \) -print -quit | grep -q .; then
  echo "No extracted frames were found under: $FRAMES_DIR" >&2
  exit 1
fi

echo "Backfilling any legacy per-frame inference outputs..."
filter_args=(
  "$REPO_ROOT/tools/filter_guideline_harvest.py"
  "--frames-dir" "$FRAMES_DIR"
  "--inference-dir" "$INFERENCE_DIR"
  "--output-root" "$HARVEST_DIR"
  "--detection-min-score" "$DETECTION_MIN_SCORE"
  "--detection-min-mask-pixels" "$DETECTION_MIN_MASK_PIXELS"
  "--recurse-frames"
)
"$PYTHON_BIN" "${filter_args[@]}"

echo "Running primary-model harvest over extracted frames..."
harvest_args=(
  "--frames-dir" "$FRAMES_DIR"
  "--output-root" "$HARVEST_DIR"
  "--recurse"
  "--detection-min-score" "$DETECTION_MIN_SCORE"
  "--detection-min-mask-pixels" "$DETECTION_MIN_MASK_PIXELS"
  "--crop-batch-size" "$CROP_BATCH_SIZE"
  "--num-workers" "$NUM_WORKERS"
)
if [[ "$SKIP_EXISTING_INFERENCE" == "1" ]]; then
  harvest_args+=("--skip-existing")
fi
"$PYTHON_BIN" "$REPO_ROOT/tools/run_supervised_best_harvest.py" "${harvest_args[@]}"

echo "Done."
echo "Predicted overlays:  $HARVEST_DIR/Predicted"
echo "Predicted masks:     $HARVEST_DIR/Masks_Predicted"
echo "No-prediction frames:$HARVEST_DIR/No_Prediction"
