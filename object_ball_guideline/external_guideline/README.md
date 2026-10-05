# SAM 3 Guideline Line Workflow

This folder contains an isolated workflow to teach SAM 3 to detect only the thin white aiming line.

## Files

- `guideline_line/live_label_overlay.py`
  Live overlay with click-to-propose segmentation, brush refine, and dataset export.
- `guideline_line/train_lora_guideline.py`
  LoRA training for line-only masks (BCE + Dice, tiny-line oversampling, prompt randomization).
- `guideline_line/eval_lora_guideline.py`
  Base-vs-adapter evaluation and promotion gate checks.
- `guideline_line/image_click_labeler.py`
  Image-folder annotation mode (click line, save, next/previous navigation).
- `guideline_line/shape_color_analyzer.py`
  Interactive polygon tool to analyze all colors inside a user-drawn shape.
- `guideline_line/train_cv_guideline.py`
  Non-LoRA dedicated cue-line segmentation model training (precision-first).
- `guideline_line/eval_cv_guideline.py`
  Non-LoRA checkpoint evaluation with threshold sweep and optional preview exports.
- `guideline_line/infer_cv_guideline.py`
  Non-LoRA tiled full-image inference for masks + overlays export.
- `guideline_line/cv_guideline_common.py`
  Shared model/data/metrics utilities for non-LoRA scripts.
- `guideline_line/config.json`
  Runtime/training defaults, threshold profiles, and gating policy.

## Dataset Layout

```text
guideline_line/data/
  images/
  masks/
  previews/
  annotations.jsonl
  splits.json
```

`annotations.jsonl` entries include:

- `id`
- `image_path`
- `mask_path`
- `prompt`
- `monitor_index`
- `roi_xyxy`
- `capture_upscale`
- `tiny_line`
- `refined`
- `created_at`

## Live Labeling

```powershell
.\.sam3_py312\Scripts\python.exe guideline_line\live_label_overlay.py `
  --monitor-index 1 `
  --prompt "white aiming line" `
  --profile precision `
  --mode hybrid `
  --tile-small-lines true
```

### Live Controls

- `Left click` (no active mask): propose line mask at click point
- `Left drag` (active mask): add brush
- `Right drag` (active mask): erase brush
- `[` / `]`: brush size down/up
- `S` or `Enter`: save sample
- `Esc`: cancel pending sample
- `A`: cycle mode (`label -> hybrid -> auto`)
- `R`: crop capture region
- `F`: reset full capture
- `Q`: quit

## Image Folder Labeling

```powershell
.\.sam3_py312\Scripts\python.exe guideline_line\image_click_labeler.py `
  --image-dir C:\My_Project\8_BALL_POOL\data\images `
  --profile precision `
  --mode label `
  --click-strategy dual `
  --auto-next-on-save true `
  --resume-last true `
  --dual-max-angle-deg 18 `
  --dual-min-span-ratio 0.78 `
  --dual-corridor-px 14 `
  --dual-min-corridor-ratio 0.60 `
  --dual-max-overshoot-ratio 0.95 `
  --dual-edge-expand false `
  --dual-force-connect false `
  --dual-min-white-ratio 0.50 `
  --dual-min-dark-border-ratio 0.08 `
  --dual-max-fill-ratio 0.42 `
  --dual-max-border-touch 0.40 `
  --dual-dark-border-ring-px 1 `
  --dual-dark-border-threshold 122 `
  --dual-short-span-px 46 `
  --dual-short-pos-hit-tol-px 5 `
  --dual-short-min-span-ratio 0.62 `
  --dual-short-min-corridor-ratio 0.48 `
  --dual-short-max-overshoot-ratio 1.00 `
  --dual-short-min-white-ratio 0.42 `
  --dual-short-max-fill-ratio 0.60 `
  --dual-roi-rescue true `
  --dual-roi-upscale 3.5 `
  --dual-adaptive-upscale true `
  --dual-adaptive-ref-span-px 56 `
  --dual-adaptive-gamma 1.25 `
  --dual-roi-upscale-max 10 `
  --dual-cv-upscale-max 12 `
  --dual-path-upscale-max 14 `
  --dual-roi-prompt-scale true `
  --dual-cv-rescue false `
  --dual-path-rescue true `
  --dual-path-upscale 6.0 `
  --dual-path-max-side 640 `
  --dual-path-corridor-ratio 0.44 `
  --dual-path-rail-offset-px 1.8 `
  --dual-path-rail-outer-offset-px 2.8 `
  --dual-path-min-rail-support 0.10 `
  --dual-path-straighten-min-span-px 18 `
  --dual-path-min-white-support 0.30 `
  --dual-path-min-straightness 0.82 `
  --ultra-tiny-mode true `
  --ultra-tiny-span-px 14 `
  --ultra-tiny-path-only true `
  --ultra-tiny-skip-full-sam true `
  --ultra-tiny-adaptive-ref-span-px 24 `
  --ultra-tiny-adaptive-gamma 2.0 `
  --ultra-tiny-roi-upscale 8 `
  --ultra-tiny-roi-upscale-max 18 `
  --ultra-tiny-path-upscale 12 `
  --ultra-tiny-path-upscale-max 24 `
  --ultra-tiny-path-max-side 1280 `
  --ultra-tiny-path-corridor-ratio 0.36 `
  --ultra-tiny-path-rail-offset-px 1.2 `
  --ultra-tiny-path-rail-outer-offset-px 2.0 `
  --ultra-tiny-path-min-rail-support 0.06 `
  --ultra-tiny-save-min-white-ratio 0.54 `
  --ultra-tiny-save-min-dark-border-ratio 0.03 `
  --ultra-tiny-save-min-span-ratio 0.80 `
  --ultra-tiny-save-min-corridor-ratio 0.62 `
  --save-min-dark-border-ratio 0.06 `
  --save-dark-border-threshold 122 `
  --save-precision-gate true
```

### Image Controls

- `Left click`: add prompt point in `dual` mode, or click-select in `single` mode
- `N`: toggle next click type `POS`/`NEG` (dual mode)
- `Backspace`: undo last prompt point (dual mode)
- `T`: toggle click strategy `dual <-> single`
- `U`: toggle ultra-tiny mode on/off during labeling
- `V`: toggle exact save-style preview (same blend/contour as saved preview image)
- `Right click`: clear all prompts and pending mask
- `Mouse wheel`: true viewport zoom in/out to cursor position
- `S` or `Enter`: save mask (auto-advance if enabled)
- `D`: next image
- `A`: previous image
- `Shift + D`: jump forward 25 images
- `Shift + A`: jump backward 25 images
- `M`: cycle mode (`label -> hybrid -> auto`)
- `0`: reset zoom
- `Q`: quit

### Two-Click Precision Behavior

- Default is `--click-strategy dual`.
- Add two positive clicks on the line endpoints.
- Optionally add one or more negative clicks on circles/balls to suppress bleed.
- Inference auto-runs after the second positive click and whenever prompts change.
- Strict reject is active: if no candidate passes all gates, no mask is shown and reject reason is displayed.
- Save gate is active by default: bad masks are blocked with a reason (`Blocked save (...)`).

### Resume Behavior

- Progress is auto-saved in `guideline_line/data/image_click_labeler_state.json` by default.
- On next launch, it automatically resumes at your last image (`--resume-last true`).
- You can override with `--start-index`.

### Single-Click Rescue Tuning (Optional Fallback)

- Applies mainly to `--click-strategy single`.
- Key flags:
  - `--rescue-min-score` (higher = stricter)
  - `--rescue-min-aspect` (higher = more line-like)
  - `--rescue-max-fill-ratio` (lower = reject blob-like masks)
  - `--rescue-min-white-ratio` (higher = enforce white-line support)
  - `--rescue-require-cv-overlap` and `--rescue-min-cv-iou`
- If valid line masks are rejected too often, lower `--rescue-min-white-ratio` first (for example `0.40`).

### Dual Precision Tuning

- `--prompt-box-pos-px` and `--prompt-box-neg-px`: click box size for SAM prompt encoding.
- `--dual-max-angle-deg`: max orientation mismatch vs endpoint direction.
- `--dual-min-span-ratio`: required span of mask along endpoint axis.
- `--dual-corridor-px`: max perpendicular distance from click-to-click segment.
- `--dual-endpoint-margin-px`: small tolerance around click endpoints for corridor check.
- `--dual-min-corridor-ratio`: required fraction of mask pixels near the click-to-click segment.
- `--dual-max-overshoot-ratio`: limits how far mask can extend beyond your two clicks.
- `--dual-edge-expand`: constrained edge fill for missing white-edge pixels.
- `--dual-edge-expand-iter`, `--dual-edge-expand-kernel`: strength/shape of constrained edge fill.
- `--dual-edge-expand-corridor-pad`: extra corridor padding used only by edge fill.
- `--dual-force-connect`: force-connects click 1 and click 2 with a corridor-limited bridge.
- `--dual-force-connect-thickness`: thickness of the forced bridge centerline.
- `--dual-force-connect-corridor-ratio`: bridge corridor width relative to click span.
- `--dual-min-white-ratio`: minimum white support inside mask.
- `--dual-min-dark-border-ratio`: minimum dark-border shell ratio around the white line mask.
- `--dual-max-fill-ratio`: rejects boxy blobs.
- `--dual-max-border-touch`: rejects masks touching too much image border.
- `--dual-dark-border-ring-px`, `--dual-dark-border-threshold`: dark-border shell settings used in strict reject.
- `--dual-short-*`: relaxed gates that apply only when click span is short.
- `--dual-long-*`: long-span gates (less strict than default) to avoid over-rejecting medium/long lines.
- `--dual-cv-rescue`: corridor-constrained white-pixel fallback for short/low-confidence cases.
- `--dual-cv-upscale`: local ROI upscale factor for CV rescue.
- `--dual-cv-s-relax`, `--dual-cv-v-relax`: relaxed white thresholding for anti-aliased short lines.
- `--dual-cv-corridor-ratio`: corridor width as fraction of click span in CV rescue.
- `--dual-path-rescue`: endpoint-constrained shortest-path fallback for thin/short lines (precision-focused).
- `--dual-path-upscale`, `--dual-path-max-side`: upscale and bound the ROI for path search.
- `--dual-path-corridor-ratio`: narrows path search to a segment corridor between your two clicks.
- `--dual-path-min-white-support`, `--dual-path-min-rail-support`: enforce white core plus dark side rails.
- `--dual-path-straighten-min-span-px`: draws a straight centerline for larger spans to prevent curvy rescue masks.
- `--dual-path-rail-*`: rail-sampling offsets and weighting for black-border awareness.
- `--dual-center-rescue`: when dual prompting fails, runs center-point rescue along the 2-click segment and still enforces dual gates.
- `--dual-center-rescue-points`: number of sampled points along the segment for center rescue.
- `--save-precision-gate`: blocks saving masks that fail strict geometric/whiteness checks.
- `--save-min-white-ratio`, `--save-min-dark-border-ratio`, `--save-min-span-ratio`, `--save-min-corridor-ratio`: key save-time precision checks.
- `--dual-require-neg-exclusion`: reject masks covering negative points.
- `--dual-roi-rescue`, `--dual-roi-margin`, `--dual-roi-upscale`: ROI rescue around endpoint segment.
- `--dual-adaptive-upscale`: automatically increases local upscale as two positive clicks get closer.
- `--dual-adaptive-ref-span-px`, `--dual-adaptive-gamma`: control how aggressively upscale grows for short spans.
- `--dual-roi-upscale-max`, `--dual-cv-upscale-max`, `--dual-path-upscale-max`: caps for adaptive upscale.
- `--dual-roi-prompt-scale`: scales prompt box size with ROI upscale so positive prompts remain stable.
- `--ultra-tiny-mode`: enables ultra-tiny logic when the last two positive clicks are very close.
- `--ultra-tiny-span-px`: distance threshold to activate ultra-tiny behavior.
- `--ultra-tiny-path-only`: use endpoint-constrained path rescue only for ultra-tiny spans.
- `--ultra-tiny-skip-full-sam`: skips broad full-frame SAM pass when ultra-tiny mode is active.
- `--ultra-tiny-*-upscale*`: much higher local upscale caps for close-point cases.
- `--ultra-tiny-save-*`: stricter save gate for tiny spans so noisy blobs are blocked.

## Shape Color Analysis

Use this to test whether the white guide line keeps stable colors across many images.

```powershell
.\.sam3_py312\Scripts\python.exe guideline_line\shape_color_analyzer.py `
  --image-dir C:\My_Project\8_BALL_POOL\data\images `
  --recursive false `
  --resume-last true `
  --output-root guideline_line\data\color_analysis `
  --train-data-root guideline_line\data_colortrain `
  --train-prompt "white aiming line" `
  --refine-line-mask true `
  --refine-required false `
  --refine-min-area 4 `
  --refine-min-elongation 1.25 `
  --refine-dark-threshold 122 `
  --quant-step-rgb 4 `
  --top-k 20
```

### Shape Analyzer Controls

- `Left click`: add polygon border point
- `Enter` or `F`: finalize polygon, fill inside, run color analysis
- `Backspace` or `Right click`: undo last border point
- `C`: clear polygon and pending analysis
- `R`: toggle polygon->line refinement on/off
- `V`: toggle exact save-style preview
- `S`: save current mask + per-sample color report
- `E`: rebuild aggregate report from all saved samples
- `D` / `A`: next/previous image
- `Shift + D` / `Shift + A`: jump +25 / -25 images
- `Mouse wheel`: zoom
- `0`: reset zoom
- `Q`: quit

### Shape Analyzer Output

- `guideline_line/data/color_analysis/masks/`: saved binary shape masks
- `guideline_line/data/color_analysis/previews/`: overlays of analyzed shape
- `guideline_line/data/color_analysis/samples.jsonl`: per-sample color stats
- `guideline_line/data/color_analysis/reports/aggregate_report.json`: global aggregated stats + `auto_thresholds` (suggested core/border thresholds and labeler flag hints)
- Optional direct train export: with `--train-data-root`, each save is also appended to `<train-data-root>/annotations.jsonl` as a train-ready sample.

When `--refine-line-mask true` is enabled, you can draw a rough polygon around the line and the tool will try to snap to a more precise white-line mask (using white-core + dark-border + elongation scoring) before save.

## LoRA Training

```powershell
.\.sam3_py312\Scripts\python.exe guideline_line\train_lora_guideline.py `
  --data-root guideline_line/data `
  --splits guideline_line/data/splits.json `
  --save-dir guideline_line/runs/run_20260215_0000 `
  --epochs 6 `
  --batch-size 1 `
  --lr 5e-5 `
  --pos-weight 4.0 `
  --tiny-oversample 2.0
```

Optional prompt file format for `--prompt-set`: newline-separated prompts or a JSON list.

## Evaluation + Promotion Gate

```powershell
.\.sam3_py312\Scripts\python.exe guideline_line\eval_lora_guideline.py `
  --data-root guideline_line/data `
  --adapter-path guideline_line/runs/run_20260215_0000/adapter `
  --baseline base `
  --profile precision `
  --report-out guideline_line/runs/run_20260215_0000/eval_report_precision.json `
  --eval-modes production,train_like `
  --gate-metric-source production
```

Gate defaults in `config.json`:

- Precision gain at least `+5 pp`
- Tiny-line F1 drop no worse than `-2 pp`
- False positives per frame increase at most `+0.03`

If gate passes, the candidate is marked promoted in `guideline_line/runs/registry.json`.

### Eval Modes

- `production`: full inference path used by auto-detection (`post_process_instance_segmentation` + profile gates + optional CV fusion).
- `train_like`: raw-logit scoring path aligned with training metrics (debugs whether LoRA learned the target signal).

Use `--gate-metric-source production` for deployment decisions.
Use `--gate-metric-source train_like` only for debugging model-learning quality.
For safety, non-production gate source does not auto-promote unless you pass:

```powershell
--allow-nonproduction-promotion
```

Optional debug switch:

```powershell
--disable-cv-fusion
```

This turns off CV mask fusion in `production` mode to isolate SAM-only behavior.

## Base SAM Zoom Probe (No LoRA)

Use this when you want SAM to see exactly your zoomed view and test if base SAM can segment the cue line at that local scale.

```powershell
.\.sam3_py312\Scripts\python.exe guideline_line\base_sam_zoom_probe.py `
  --image-dir C:\My_Project\8_BALL_POOL\data\images `
  --resume-last true `
  --profile precision `
  --dataset-root guideline_line\data_zoomprobe `
  --quarantine-root guideline_line\data_zoomprobe\quarantine `
  --draw-topk 0 `
  --candidate-max 64 `
  --probe-upscale 3.0 `
  --pos-soft-gate true `
  --pos-soft-radius-px 7 `
  --pos-hard-max-dist-px 18 `
  --pos-distance-penalty 0.12 `
  --pos-fallback-on-miss true `
  --pos-fallback-hard-scale 1.4 `
  --snap-positive-to-line true `
  --snap-radius-px 10 `
  --snap-min-white-score 0.15 `
  --refine-white-core true `
  --refine-white-core-threshold 0.45 `
  --refine-support-dilate-px 1 `
  --refine-min-area 6 `
  --map-threshold 0.40 `
  --map-smooth-kernel 3 `
  --map-area-preblur-sigma 0.6 `
  --cv-rescue-on-empty true `
  --cv-rescue-s-relax 40 `
  --cv-rescue-v-relax 35 `
  --fallback-polygon-enabled true `
  --fallback-polygon-min-points 3 `
  --fallback-refine-required true `
  --fallback-refine-min-area 4 `
  --fallback-refine-min-elongation 1.25 `
  --fallback-refine-dark-threshold 122 `
  --fallback-refine-ring-px 1 `
  --display-soft-mask true `
  --display-prob-overlay true `
  --display-prob-gamma 0.9 `
  --display-supersample 2 `
  --display-edge-aa true `
  --display-edge-thickness 0 `
  --display-alpha-selected 0.34 `
  --display-alpha-other 0.18 `
  --min-white-ratio 0.35 `
  --save-gate-mode strict `
  --save-min-white-ratio 0.38 `
  --save-min-elongation 1.20 `
  --save-max-fill-ratio 0.62 `
  --save-max-pos-dist 18 `
  --save-allow-force true `
  --force-save-route quarantine
```

### Base Probe Controls

- `G`: toggle auto detection for the current zoomed/panned view
- `Space`: run one-shot detection
- `P`: toggle active input mode (`PROMPT`/`POLYGON`)
- `Left click`: add prompt point (`PROMPT` mode) or polygon vertex (`POLYGON` mode)
- `N`: toggle prompt type (`POS`/`NEG`)
- `Backspace`: undo last prompt point/vertex for active mode
- `C`: clear polygon vertices only
- `F` / `Enter` (in polygon mode): finalize polygon fallback candidate
- `Right click`: clear prompts + polygon + candidates
- `Wheel`: zoom to cursor
- `Middle drag`: pan
- `+` / `-`: increase/decrease probe upscale
- `1` / `2` / `3`: runtime presets (`thin` / `balanced` / `strict`)
- `K`: cycle selected candidate when multiple masks are found
- `V`: toggle save-preview view (exact saved overlay style)
- `s` / `Enter` (in prompt mode): normal save (strict-gated)
- `Shift+S`: force-save blocked sample (to quarantine when configured)
- `D` / `A`: next/prev image (`Shift+D/A` jump 25)

Saved samples are written under `guideline_line/data_zoomprobe` by default and source images are never modified.

Notes:

- Positive clicks are snapped (by default) to the nearest white-line center candidate to reduce `miss_pos`.
- Positive gating is soft-distance by default: near-miss masks are penalized instead of hard-rejected.
- If prompt-gating rejects everything by position, text-only fallback is retried with relaxed distance cap.
- If SAM paths produce no usable mask, a CV rescue path runs (white-line/dark-border based), then the same pos/neg gates are applied.
- Optional polygon fallback (`P` + clicks + `F`) refines your rough shape into a line mask and injects it as `polygon_refine`.
- Save gate is strict by default; blocked saves can be force-saved with `Shift+S`.
- Force-saved samples can be routed to quarantine and excluded from default training.
- White-core refinement is enabled by default, so masks better follow actual white line pixels.
- Display uses probability overlays (smooth), while saved masks remain binary for training stability.
- `--display-supersample 2` reduces stair-casing in display only; save masks stay binary full-res.
- `--draw-topk 0` (default) draws all candidates every frame so you can place negatives against all visible options.

## Non-LoRA CV Detector Workflow

This workflow trains a dedicated segmentation model for cue-line masks and does not use LoRA.

### 1) Label Collection (unchanged)

Keep using `base_sam_zoom_probe.py` for sample collection. It remains a labeling tool and source images are never modified.

### 2) Train Dedicated Model

Default train data is `main + quarantine + legacy` from:

- `guideline_line/data_zoomprobe_merged/annotations.jsonl`
- `guideline_line/data_zoomprobe_merged/splits.json`

Canonical training command:

```powershell
& .\.sam3_py312\Scripts\python.exe guideline_line\train_cv_guideline.py `
  --annotations guideline_line\data_zoomprobe_merged\annotations.jsonl `
  --splits guideline_line\data_zoomprobe_merged\splits.json `
  --negative-image-dir C:\PATH\TO\NEGATIVE_IMAGES `
  --include-main true `
  --include-quarantine true `
  --include-legacy true `
  --img-size 512 `
  --batch-size 8 `
  --epochs 120 `
  --lr 3e-4 `
  --workers 8 `
  --neg-per-pos 0.5 `
  --amp true `
  --early-stop-patience 18 `
  --early-stop-min-delta 0.001 `
  --save-dir guideline_line\runs_cv\run_YYYYMMDD_HHMM
```

Shutdown-safe variant:

```powershell
& .\.sam3_py312\Scripts\python.exe guideline_line\train_cv_guideline.py `
  --annotations guideline_line\data_zoomprobe_merged\annotations.jsonl `
  --splits guideline_line\data_zoomprobe_merged\splits.json `
  --negative-image-dir C:\PATH\TO\NEGATIVE_IMAGES `
  --include-main true `
  --include-quarantine true `
  --include-legacy true `
  --img-size 512 `
  --batch-size 8 `
  --epochs 120 `
  --lr 3e-4 `
  --workers 8 `
  --neg-per-pos 0.5 `
  --amp true `
  --early-stop-patience 18 `
  --early-stop-min-delta 0.001 `
  --save-dir guideline_line\runs_cv\run_YYYYMMDD_HHMM
$code = $LASTEXITCODE
if ($code -eq 0) { shutdown /s /t 30 } else { Write-Host "Training failed with code $code" }
```

Training outputs:

- `best.pt`
- `last.pt`
- `metrics.json`
- `thresholds.json`
- `train_manifest.jsonl`
- `val_manifest.jsonl`
- `config_used.json`

### 3) Evaluate Checkpoint

```powershell
.\.sam3_py312\Scripts\python.exe guideline_line\eval_cv_guideline.py `
  --checkpoint guideline_line\runs_cv\run_YYYYMMDD_HHMM\best.pt `
  --annotations guideline_line\data_zoomprobe_merged\annotations.jsonl `
  --splits guideline_line\data_zoomprobe_merged\splits.json `
  --negative-image-dir C:\PATH\TO\NEGATIVE_IMAGES `
  --report-out guideline_line\runs_cv\run_YYYYMMDD_HHMM\eval_report.json `
  --save-previews true
```

### 4) Visual Inference QA (random images)

```powershell
.\.sam3_py312\Scripts\python.exe guideline_line\infer_cv_guideline.py `
  --checkpoint guideline_line\runs_cv\run_YYYYMMDD_HHMM\best.pt `
  --image-dir C:\My_Project\8_BALL_POOL\data\images `
  --output-root guideline_line\CV_Test `
  --num-images 70 `
  --seed 42
```

### Troubleshooting

- BOM/encoding issues in annotations are handled with `utf-8-sig` parsing.
- Missing image/mask rows are skipped and counted during loading.
- If false positives are high, add stronger negatives in `--negative-image-dir` and retrain.
- If recall is too low on tiny lines, add more tiny-line positives and retrain.
- If threshold is too strict/loose, check `thresholds.json` and rerun eval/infer with an explicit `--threshold`.
