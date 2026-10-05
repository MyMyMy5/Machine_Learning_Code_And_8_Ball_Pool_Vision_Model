# Experiment Log

Updated: 2026-03-06T23:30:00Z

## Current Promoted Checkpoints
- Balanced default: `guideline_line/runs_cv/overnight_custom_stratified_20260303_161242/ft_oldchamp_balanced_v1/run/best.pt`
  - Family: `custom`
  - Backbone: `resnet34`
  - Direct eval on overnight stratified split: pixel precision `0.8923`, pixel recall `0.7539`, pixel F1 `0.8173`, centerline F1 `0.9632`, FP/frame `0.0481`, composite `0.8744`
- Conservative low-FP: `guideline_line/runs_cv/overnight_20260301_162856/exp04_resnet34_640_neg1p0/run/best.pt`
  - Family: `custom`
  - Backbone: `resnet34`
  - Direct eval on overnight stratified split: pixel precision `0.9017`, pixel recall `0.6654`, pixel F1 `0.7657`, centerline F1 `0.9161`, FP/frame `0.0288`, composite `0.8580`
- Tiny-line specialist: `guideline_line/runs_cv/ft_balanced_ultratiny_replay_v1/best.pt`
  - Family: `custom`
  - Backbone: `resnet34`
  - Direct eval on the overnight split: pixel precision `0.8944`, pixel recall `0.7073`, pixel F1 `0.7899`, centerline F1 `0.9403`, FP/frame `0.0288`, composite `0.8665`
  - Fixed-slice strengths:
    - ultra-tiny `<12`: `P=0.2808`, `R=0.6465`, `F1=0.3915`, `cF1=0.9431`, `FP/frame=0.0288`
    - short `<20`: `P=0.6851`, `R=0.6716`, `F1=0.6783`, `cF1=0.9869`, `FP/frame=0.0288`
    - distractor negatives top48: `0/48` false-positive frames

## Dataset Snapshot
- Positives total: `629`
- Main: `590`
- Quarantine: `39`
- Negatives: `691`

## Failure Categories Observed
- Tiny thin line misses
- Long-line truncation
- Partial-line predictions
- Cue-circle / nearby white distractor confusion
- Rare table themes and low-contrast backgrounds

## 2026-03-03 Investigation
- Root cause found in the current positive split, not in SAM 3 itself.
- Existing `guideline_line/data_zoomprobe_merged/splits.json` is strongly skewed:
  - train major-length buckets: `0-8: 8`, `8-12: 30`, `12-20: 110`, `20-40: 163`, `40+: 224`
  - val major-length buckets: `0-8: 2`, `8-12: 6`, `12-20: 13`, `20-40: 1`, `40+: 72`
  - val source mix: `94/94 polygon_refine`
- Consequence:
  - model selection is biased toward long refined lines
  - short/noisy cases are under-represented in validation
  - architecture comparisons against this split are not trustworthy for the target problem

## 2026-03-03 Trainer Patch
- `guideline_line/cv_guideline_common.py`
  - added per-sample mask geometry (`area`, bbox, major length, refined flag)
  - added `stratified_annotation_split(...)`
- `guideline_line/train_cv_guideline.py`
  - added `--split-mode stratified`
  - added size-aware crop policies for short and ultra-short lines
  - added size-aware sample weighting and mild refined-label weighting
  - writes `split_summary.json` for every run

## 2026-03-03 Smoke Validation
- Run: `guideline_line/runs_cv/smoke_stratified_smallline`
- Purpose: verify stratified split + size-aware curriculum path executes end to end.
- Result:
  - train positive buckets: `<12: 39`, `<20: 105`, `<40: 139`, `>=40: 253`
  - val positive buckets: `<12: 7`, `<20: 18`, `<40: 25`, `>=40: 43`
  - train source mix: `polygon_refine 407`, `cv_rescue_whitecore 99`, others 30
  - val source mix: `polygon_refine 71`, `cv_rescue_whitecore 17`, others 5
- Smoke command completed successfully and produced checkpoints + metrics.

## 2026-03-03 Stratified Run Outcome
- Run: `guideline_line/runs_cv/resnet34_stratified_smallline_v1`
- Training-time best checkpoint:
  - epoch `15`
  - `P=0.8265`
  - `R=0.7223`
  - `F1=0.7709`
  - `cF1=0.9412`
  - `FP/frame=0.0577`
- Direct eval on the same generated stratified split:
  - new best `best.pt`: `P=0.8481`, `R=0.6896`, `F1=0.7607`, `cF1=0.9282`, `FP/frame=0.0481`
  - old champion `overnight_20260301_162856/exp04_resnet34_640_neg1p0/run/best.pt`: `P=0.9017`, `R=0.6654`, `F1=0.7657`, `cF1=0.9161`, `FP/frame=0.0288`
  - new best centerline `best_centerline.pt`: centerline improved, but balanced result was weaker than both candidates
- Conclusion:
  - stratified validation fixed the selection problem
  - the new run is a valid recall/centerline candidate
  - the old champion still narrowly wins as the balanced default on the fair split
  - next work should stay on the `custom/resnet34` path, not transformers

## 2026-03-03 Overnight Matrix Plan
- Runner updated: `guideline_line/run_overnight_cv_experiments.py`
  - manifest rebuild is now optional
  - evaluation now uses each run's generated split file when present
- Planned overnight batch:
  - `ft_oldchamp_balanced_v1`
  - `ft_oldchamp_tinyfocus_v1`
  - `ft_newbest_precision_recover_v1`
  - `scratch_stratified_seed17_v1`
- Goal:
  - test whether old-champion fine-tuning can pick up tiny lines without losing precision
  - test whether new-stratified fine-tuning can recover precision while keeping its centerline behavior
  - keep one fresh stratified seed as a control

## 2026-03-04 Overnight Matrix Outcome
- Run: `guideline_line/runs_cv/overnight_custom_stratified_20260303_161242`
- Winner on overnight leaderboard: `ft_oldchamp_balanced_v1`
- Clean direct comparison on the exact overnight split:
  - new balanced winner: `P=0.8923`, `R=0.7539`, `F1=0.8173`, `cF1=0.9632`, `FP/frame=0.0481`, composite `0.8744`
  - old champion: `P=0.9017`, `R=0.6654`, `F1=0.7657`, `cF1=0.9161`, `FP/frame=0.0288`, composite `0.8580`
- Promotion decision:
  - promote `ft_oldchamp_balanced_v1` as the new default
  - retain the prior champion as the conservative low-FP fallback
  - do not promote `ft_oldchamp_tinyfocus_v1`, `ft_newbest_precision_recover_v1`, or `scratch_stratified_seed17_v1`

## 2026-03-04 Fixed Subset Benchmarks
- Catalog: `guideline_line/benchmarks/subset_benchmark_catalog.json`
- Manifests:
  - `guideline_line/benchmarks/ultra_tiny_lt12_plus_valneg.json`
  - `guideline_line/benchmarks/short_lt20_plus_valneg.json`
  - `guideline_line/benchmarks/distractor_negatives_top48.json`
- Summary: `guideline_line/runs_cv/subset_benchmark_eval_20260304/subset_benchmark_summary.json`
- Counts:
  - ultra-tiny positives `<12`: `7`
  - short positives `<20`: `25`
  - validation negatives: `104`
  - distractor-negative slice: `48`
- Result at promoted operating points:
  - balanced default on ultra-tiny: `P=0.2293`, `R=0.6326`, `F1=0.3366`, `cF1=0.9481`, `FP/frame=0.0481`
  - conservative low-FP on ultra-tiny: `P=0.1787`, `R=0.2651`, `F1=0.2135`, `cF1=0.5934`, `FP/frame=0.0288`
  - balanced default on short `<20`: `P=0.6285`, `R=0.6601`, `F1=0.6439`, `cF1=0.9860`, `FP/frame=0.0481`
  - conservative low-FP on short `<20`: `P=0.6822`, `R=0.4700`, `F1=0.5565`, `cF1=0.8310`, `FP/frame=0.0288`
  - balanced default on distractor negatives: `FP/frame=0.0417` (`2/48` frames)
  - conservative low-FP on distractor negatives: `FP/frame=0.0000` (`0/48` frames)
- Conclusion:
  - the balanced default is clearly better on the actual short-line problem
  - the conservative model is only better on distractor suppression
  - next tuning should start from the balanced default and target distractor false positives without giving back tiny-line recall

## 2026-03-04 Hardcase Replay Tooling
- Trainer updated: `guideline_line/train_cv_guideline.py`
  - added `--hardcase-positive-manifest`
  - added `--hardcase-negative-manifest`
  - added per-manifest weight multipliers for train-time replay
  - writes `hardcase_replay_summary.json` per run
- Builder added: `guideline_line/build_hardcase_replay_manifests.py`
  - creates train-only replay manifests so validation samples are not leaked into training
- Built replay catalog: `guideline_line/hardcase_replay/overnight_split_v1/hardcase_replay_catalog.json`
  - train ultra-tiny positives `<12`: `39`
  - train short positives `<20`: `144`
  - train distractor negatives: `48`

## 2026-03-04 Hardcase Replay Smokes
- Mixed replay smoke:
  - run: `guideline_line/runs_cv/smoke_hardcase_replay_v1`
  - replay inputs: `39` train ultra-tiny positives, `48` train distractor negatives
  - full-split validation at epoch `1`: `P=0.8984`, `R=0.6898`, `F1=0.7804`, `cF1=0.9231`, `FP/frame=0.0865`
  - fixed-subset eval:
    - ultra-tiny `<12`: `P=0.4150`, `R=0.6930`, `F1=0.5192`, `cF1=0.9449`, `FP/frame=0.0865`
    - short `<20`: `P=0.7818`, `R=0.6988`, `F1=0.7379`, `cF1=0.9901`, `FP/frame=0.0865`
    - distractor negatives: `FP/frame=0.1458` (`7/48` frames)
- Negative-only replay smoke:
  - run: `guideline_line/runs_cv/smoke_negrecover_only_v1`
  - replay input: `48` train distractor negatives, no positive replay
  - full-split validation at epoch `1`: `P=0.8811`, `R=0.7470`, `F1=0.8086`, `cF1=0.9533`, `FP/frame=0.2308`
  - distractor-negative eval: `FP/frame=0.4167` (`20/48` frames)
- Conclusion:
  - ultra-tiny positive replay is a real improvement lever
  - simple distractor-negative replay is not helping and currently makes the false-positive slice worse
  - next run should continue from the balanced winner with positive replay only

## 2026-03-04 Ultra-Tiny Replay Outcome
- Run: `guideline_line/runs_cv/ft_balanced_ultratiny_replay_v1`
- Init checkpoint: `guideline_line/runs_cv/overnight_custom_stratified_20260303_161242/ft_oldchamp_balanced_v1/run/best.pt`
- Replay input: `39` train ultra-tiny positives from `guideline_line/hardcase_replay/overnight_split_v1/train_ultra_tiny_lt12.json`
- Full-split direct eval:
  - `P=0.8944`
  - `R=0.7073`
  - `F1=0.7899`
  - `cF1=0.9403`
  - `FP/frame=0.0288`
  - composite `0.8665`
- Fixed-slice direct eval:
  - ultra-tiny `<12`: `P=0.2808`, `R=0.6465`, `F1=0.3915`, `cF1=0.9431`, `FP/frame=0.0288`
  - short `<20`: `P=0.6851`, `R=0.6716`, `F1=0.6783`, `cF1=0.9869`, `FP/frame=0.0288`
  - distractor negatives top48: `0/48` FP frames
- Conclusion:
  - this run does not replace the balanced default, because its full-split composite stays below the champion
  - this run is still worth preserving as a third promoted profile because it is stronger on the fixed ultra-tiny and short-line slices and cleaner on the fixed distractor-negative slice

## 2026-03-04 Mining Pass Audit
- Initial mining run: `guideline_line/CV_Test_hard_mining/balanced_mining_400_seed42_20260304_051557`
- Audit result:
  - source-path overlap with labeled annotations: `151/400`
  - cause: the pass sampled random pool images without excluding `source_image_path` values already present in `annotations.jsonl`
  - consequence: this pass is still usable for review, but it is not a clean unlabeled mining set
- Fix applied:
  - added `guideline_line/build_clean_mining_manifest.py`
  - built clean manifest: `guideline_line/benchmarks/clean_mining_400_seed42_v1.json`
  - verified overlap against labeled annotation `source_image_path`: `0/400`
  - verified overlap against prior mining run image paths: `0/400`
- Next clean mining command:
  - `guideline_line/benchmarks/clean_mining_400_seed42_v1_command.txt`

## Recommended Next Run
- Retrain the proven `custom/resnet34` family first; do not spend another cycle on SegFormer until the split is fixed.
- Command:

```powershell
.\.sam3_py312\Scripts\python.exe guideline_line\train_cv_guideline.py `
  --annotations guideline_line\data_zoomprobe_merged\annotations.jsonl `
  --splits guideline_line\data_zoomprobe_merged\splits.json `
  --negative-image-dir C:\My_Project\8_BALL_POOL\data\images\negative_selected `
  --include-main true `
  --include-quarantine true `
  --include-legacy false `
  --img-size 640 `
  --batch-size 4 `
  --epochs 45 `
  --lr 3e-4 `
  --workers 8 `
  --neg-per-pos 1.0 `
  --amp true `
  --early-stop-patience 10 `
  --early-stop-min-delta 0.001 `
  --model-family custom `
  --backbone resnet34 `
  --split-mode stratified `
  --split-size-thresholds 12,20,40 `
  --split-stratify-source true `
  --pos-weight 4.0 `
  --crop-mode-mix tight:0.30,context:0.45,fullframe:0.25 `
  --crop-mode-mix-small tight:0.65,context:0.35,fullframe:0.00 `
  --crop-mode-mix-ultra-tiny tight:0.85,context:0.15,fullframe:0.00 `
  --small-line-major-thresh 20 `
  --ultra-tiny-major-thresh 12 `
  --small-line-weight-mult 1.35 `
  --ultra-tiny-weight-mult 1.75 `
  --refined-weight-mult 1.05 `
  --freeze-encoder-epochs 1 `
  --grad-accum-steps 1 `
  --scheduler cosine `
  --warmup-epochs 1.0 `
  --loss-ce-weight 0.45 `
  --loss-dice-weight 0.25 `
  --loss-cldice-weight 0.20 `
  --loss-edge-weight 0.10 `
  --selection-metric balanced `
  --tile-sizes 640,896 `
  --tile-overlaps 160,224 `
  --tta-scales 1.0,1.15 `
  --tta-hflip true `
  --axis-complete true `
  --axis-complete-max-gap 24 `
  --axis-complete-max-extension 160 `
  --axis-complete-min-white-ratio 0.42 `
  --axis-complete-min-dark-border-ratio 0.05 `
  --val-runtime async `
  --val-loader-workers 8 `
  --val-postprocess-workers 8 `
  --val-postprocess-backend process_shared `
  --val-prefetch-items 32 `
  --val-max-batch-tiles 8 `
  --val-max-wait-ms 8 `
  --val-bucket-by-tile true `
  --val-save-runtime-stats true `
  --save-dir guideline_line\runs_cv\resnet34_stratified_smallline_v1
```

## Completed Sweeps
- `seg_v1_main_quarantine`: initial custom segmentation baseline, positive-only validation.
- `seg_v2_hardneg_effi`: failed hard-negative custom run.
- `seg_v3_context`: first materially good custom run after context crop fix.
- `overnight_20260301_162856`: negative-aware multi-experiment custom sweep; `exp04_resnet34_640_neg1p0` promoted.
- `ft_balanced_after_clean400_v1`: warm-start fine-tune from the balanced default after the clean 400-image mining pass and 80 repaired additions; promoted as the new default.
- `ft_exp04_hardcases_v1`: hardcase fine-tune, not promoted.
- `retrain_full_after_repairs_v1`: fresh retrain on expanded dataset, not promoted.

## Promotion Decisions
- Promoted default: `guideline_line/runs_cv/ft_balanced_after_clean400_v1/best.pt`
- Promoted conservative low-FP fallback: `guideline_line/runs_cv/overnight_20260301_162856/exp04_resnet34_640_neg1p0/run/best.pt`
- Promoted tiny-line specialist: `guideline_line/runs_cv/ft_balanced_ultratiny_replay_v1/best.pt`
- Superseded previous default: `guideline_line/runs_cv/overnight_custom_stratified_20260303_161242/ft_oldchamp_balanced_v1/run/best.pt`
  - Reason: lost on the updated 722-positive dataset direct eval, especially on full-split recall/F1, short-line F1, and distractor-negative FP control.
- Not promoted: `guideline_line/runs_cv/overnight_custom_stratified_20260303_161242/ft_oldchamp_tinyfocus_v1/run/best.pt`
  - Reason: did not beat the balanced winner and did not justify replacing the conservative fallback.
- Not promoted: `guideline_line/runs_cv/overnight_custom_stratified_20260303_161242/ft_newbest_precision_recover_v1/run/best.pt`
  - Reason: precision-recovery fine-tune did not recover enough balanced quality.
- Not promoted: `guideline_line/runs_cv/overnight_custom_stratified_20260303_161242/scratch_stratified_seed17_v1/run/best_centerline.pt`
  - Reason: very low FP, but recall remained too low.
- Rejected: `guideline_line/runs_cv/ft_exp04_hardcases_v1/best.pt`
  - Reason: recall and centerline quality regressed.
- Rejected: `guideline_line/runs_cv/retrain_full_after_repairs_v1/best.pt`
  - Reason: precision/FP improved, but overall composite remained below champion.

## Fixed Benchmarks
- `guideline_line/benchmarks/seed42_300.json`
- `guideline_line/benchmarks/seed123_300.json`
- `guideline_line/benchmarks/hardcases_latest.json`

## Next Planned Architecture Sweep
- `segformer_b2_640_seed42`
- `segformer_b3_640_seed42`
- `segformer_b3_768_seed42`

## Current Transformer Notes
- `segformer_b3_640_seed42` was not promoted.
- Observed issue: validation operating point was unstable, with large swings in threshold, FP/frame, and recall across epochs.
- Stabilization patch applied to `guideline_line/train_cv_guideline.py`:
  - added `--encoder-lr-mult`
  - added `--grad-clip-norm`
  - intended next retry recipe uses:
    - lower base LR
    - lower encoder LR than decoder
    - longer encoder freeze
    - gradient accumulation
    - gradient clipping
- Overnight matrix updated to:
  - `custom_resnet34_seed17_refresh`
  - `segformer_b2_640_stable_seed42`
  - `segformer_b3_640_stable_seed42`
  - `segformer_b3_768_stable_seed42`

## Runtime Work
- Added V1 async inference runtime in `guideline_line/inference_runtime.py`
- Scope:
  - training-time validation
  - standalone eval
  - standalone infer
- Design:
  - in-process queue-based GPU batching
  - CPU loader overlap
  - CPU postprocess overlap
- Accuracy guardrail:
  - `direct` mode remains available
  - benchmark equivalence should be checked before making async the default promotion path
- Notes:
  - `guideline_line/RUNTIME_NOTES.md`

## Promotion Gate
- Composite score must exceed `0.8565`
- FP/frame must stay at or below `0.08`
- Centerline F1 must reach at least `0.92`
- Must pass both fixed 300-image benchmarks and the hardcase benchmark without broader semantic regression.

## 2026-03-04 Clean 400 Mining Result
- Clean mining pass reviewed: `400` images from `C:\My_Project\8_BALL_POOL\data\images`
- New repaired additions: `80`
  - main: `74`
  - quarantine: `6`
- Updated merged dataset:
  - positives total: `722`
  - main: `677`
  - quarantine: `45`
- Promotion run: `guideline_line/runs_cv/ft_balanced_after_clean400_v1`
- Direct-eval winner on updated full split:
  - precision: `0.9013`
  - recall: `0.7827`
  - F1: `0.8378`
  - centerline F1: `0.9697`
  - FP/frame: `0.0385`
  - threshold: `0.60`
- Old default on the same updated split:
  - precision: `0.8708`
  - recall: `0.6919`
  - F1: `0.7711`
  - centerline F1: `0.9311`
  - FP/frame: `0.0865`
  - threshold: `0.70`
- Short-line slice (`<20` major length) on the updated split:
  - new default: `P=0.8727 R=0.5389 F1=0.6664 cF1=0.8784 FP/frame=0.0192`
  - old default: `P=0.6051 R=0.5824 F1=0.5935 cF1=0.8959 FP/frame=0.0481`
- Ultra-tiny slice (`<12` major length) remained hard for both models:
  - new default: `P=0.3651 R=0.1704 F1=0.2323 cF1=0.3636 FP/frame=0.0192`
  - old default: `P=0.1587 R=0.3185 F1=0.2118 cF1=0.6250 FP/frame=0.0481`
- Distractor-negative slice (`top48`) on the updated split:
  - new default: `0/48` FP frames
  - old default: `2/48` FP frames

## 2026-03-04 Fresh 400 Mining Result
- Fresh mining pass reviewed: `400` new images after excluding the dataset, the contaminated mining run, and the first clean mining run.
- New repaired additions: `32`
  - main: `27`
  - quarantine: `5`
- Updated merged dataset after rebuild:
  - positives total: `754`
  - main: `704`
  - quarantine: `50`
- Next planned warm-start run:
  - init checkpoint: `guideline_line/runs_cv/ft_balanced_after_clean400_v1/best.pt`
  - save dir: `guideline_line/runs_cv/ft_balanced_after_clean800_v1`
  - command file: `guideline_line/runs_cv/ft_balanced_after_clean800_v1_command.txt`

## 2026-03-04 Clean 800 Warm-Start Eval Result
- Run: `guideline_line/runs_cv/ft_balanced_after_clean800_v1`
- Direct-eval comparison dir:
  - `guideline_line/runs_cv/compare_ft_balanced_after_clean800_v1_20260304`
- Promotion decision: **not promoted** (all pointers unchanged)

- Full split (new candidate `best.pt`):
  - `P=0.8981 R=0.8206 F1=0.8576 cF1=0.9877 FP/frame=0.0385 thr=0.65 comp=0.8962`
- Full split (current default `ft_balanced_after_clean400_v1/best.pt`):
  - `P=0.8990 R=0.8300 F1=0.8631 cF1=0.9865 FP/frame=0.0385 thr=0.60 comp=0.8981`
- Full split (new candidate `best_centerline.pt` sanity check):
  - `P=0.9267 R=0.7074 F1=0.8024 cF1=0.9481 FP/frame=0.0288 thr=0.80 comp=0.8879`

- Ultra-tiny subset (`<12`, plus val negatives):
  - new candidate: `P=0.3811 R=0.3733 F1=0.3772 cF1=0.7702 FP/frame=0.0192`
  - current default: `P=0.5622 R=0.3562 F1=0.4361 cF1=0.6667 FP/frame=0.0192`

- Short subset (`<20`, plus val negatives):
  - new candidate: `P=0.7912 R=0.5516 F1=0.6500 cF1=0.9048 FP/frame=0.0192`
  - current default: `P=0.8844 R=0.5539 F1=0.6812 cF1=0.8902 FP/frame=0.0192`

- Distractor-negative subset (`top48`):
  - new candidate: `1/48` FP frames (`FP/frame=0.0208`)
  - current default: `0/48` FP frames (`FP/frame=0.0000`)

- Conclusion:
  - The `clean800` checkpoint is close but does not beat the current default on full split composite.
  - It also fails to beat the current default on short-line F1 and distractor-negative FP suppression.
  - Keep:
    - default: `guideline_line/runs_cv/ft_balanced_after_clean400_v1/best.pt`
    - conservative: `guideline_line/runs_cv/overnight_20260301_162856/exp04_resnet34_640_neg1p0/run/best.pt`
    - tiny specialist: `guideline_line/runs_cv/ft_balanced_ultratiny_replay_v1/best.pt`

## 2026-03-04 Disagreement Mining (Balanced vs Tiny Specialist)
- Goal: targeted mining instead of another random-400 pass.
- Shared image pool: `guideline_line/benchmarks/clean_mining_400_seed42_v2.json`
- Balanced summary:
  - `guideline_line/CV_Test_hard_mining/clean_mining_400_seed42_v2/summary.json`
- Tiny-specialist summary:
  - `guideline_line/CV_Test_hard_mining/clean_mining_400_seed42_v2_tinyline/summary.json`
- New script:
  - `guideline_line/build_disagreement_review_summary.py`
- Ranked disagreement outputs:
  - `guideline_line/CV_Test_hard_mining/clean_mining_400_seed42_v2_disagreement/disagreement_ranked_top200.json`
  - `guideline_line/CV_Test_hard_mining/clean_mining_400_seed42_v2_disagreement/disagreement_review_summary_top200.json`
- Case mix in the top-200 shortlist:
  - `secondary_stronger`: 175
  - `partial_disagreement`: 7
  - `weak_disagreement`: 18
- Review launch command saved at:
  - `guideline_line/CV_Test_hard_mining/clean_mining_400_seed42_v2_disagreement/review_command.txt`

## 2026-03-05 Clean 1400 + Scheduler Fix + Direct Re-Compare
- Applied training fix in `guideline_line/train_cv_guideline.py`:
  - AMP path now steps the LR scheduler only when optimizer actually stepped (prevents skipped-step warning behavior under GradScaler overflow).
- Run:
  - `guideline_line/runs_cv/ft_balanced_after_clean1400_bugfix_v1`
  - best training epoch summary: `P=0.8822 R=0.8449 F1=0.8631 cF1=0.9847 FP/frame=0.0288 thr=0.65 comp=0.8991`
- Direct re-compare on the updated 796-positive split (`120` val positives + `104` val negatives):
  - compare dir: `guideline_line/runs_cv/compare_ft_balanced_after_clean1400_bugfix_v1_20260305`
  - bugfix candidate (`fullsplit_bugfix_best_direct.json`):
    - `P=0.8888 R=0.8253 F1=0.8559 cF1=0.9828 FP/frame=0.0288 thr=0.70 comp=0.8919`
  - old champion (`fullsplit_oldchamp_direct.json`):
    - `P=0.9007 R=0.8151 F1=0.8558 cF1=0.9813 FP/frame=0.0385 thr=0.60 comp=0.8957`
  - clean800 candidate (`fullsplit_clean800_best_direct.json`):
    - `P=0.8955 R=0.7904 F1=0.8397 cF1=0.9744 FP/frame=0.0385 thr=0.65 comp=0.8868`
- Promotion decision:
  - **No balanced default promotion** (old champion keeps highest full-split composite).
  - Keep default pointer at `guideline_line/runs_cv/ft_balanced_after_clean400_v1/best.pt`.

## 2026-03-05 Hardcase Replay Setup (From Fresh 600 Pass)
- Built repaired-hardcase positive manifest from `clean_mining_600_seed99_v1` review:
  - `guideline_line/hardcase_replay/clean_mining_600_seed99_v1_repaired_pos.json`
  - matched repaired positives: `42`
- Prepared next targeted replay command:
  - `guideline_line/runs_cv/ft_balanced_hardcase_replay_clean600_v1_command.txt`

## 2026-03-05 Hardcase Replay v2 Result (Direct Re-Compare)
- Run:
  - `guideline_line/runs_cv/ft_balanced_hardcase_replay_clean600_v2`
- Direct-eval comparison dir:
  - `guideline_line/runs_cv/compare_ft_balanced_hardcase_replay_clean600_v2_20260305`
- Full split (`119` val positives + `104` val negatives):
  - replay v2 candidate (`fullsplit_v2_best_direct.json`):
    - `P=0.9317 R=0.6976 F1=0.7978 cF1=0.9020 FP/frame=0.1923 thr=0.65 comp=0.8471`
  - replay v2 centerline checkpoint (`fullsplit_v2_best_centerline_direct.json`):
    - `P=0.9310 R=0.6861 F1=0.7900 cF1=0.9046 FP/frame=0.2115 thr=0.70 comp=0.8411`
  - current champion (`fullsplit_oldchamp_direct.json`):
    - `P=0.9568 R=0.6640 F1=0.7840 cF1=0.8890 FP/frame=0.0385 thr=0.60 comp=0.8837`
- Decision:
  - **Not promoted**.
  - Replay v2 improves recall/line coverage slightly, but FP/frame is too high for deployment.
  - Keep default pointer at `guideline_line/runs_cv/ft_balanced_after_clean400_v1/best.pt`.
- Next targeted run prepared:
  - `guideline_line/runs_cv/ft_balanced_hardcase_precision_recover_v1_command.txt`
  - intent: start from replay-v2 and recover precision using stronger hard-negative replay.

## 2026-03-05 Hardcase Precision-Recover v1 Result (Direct Re-Compare)
- Run:
  - `guideline_line/runs_cv/ft_balanced_hardcase_precision_recover_v1`
- Direct-eval comparison dir:
  - `guideline_line/runs_cv/compare_ft_balanced_hardcase_precision_recover_v1_20260305`
- Full split (`119` val positives + `104` val negatives):
  - precision-recover candidate (`fullsplit_precision_recover_best_direct.json`):
    - `P=0.9327 R=0.7013 F1=0.8007 cF1=0.9197 FP/frame=0.0673 thr=0.70 comp=0.8770`
  - current champion (`fullsplit_oldchamp_direct.json`):
    - `P=0.9568 R=0.6640 F1=0.7840 cF1=0.8890 FP/frame=0.0385 thr=0.60 comp=0.8837`
- Decision:
  - **Not promoted as balanced default**.
  - Candidate improves recall and centerline quality but still loses on composite due higher FP/frame.
  - Keep default pointer at `guideline_line/runs_cv/ft_balanced_after_clean400_v1/best.pt`.

## 2026-03-05 Clean 600 (seed123 v2) Review Complete + Rebuild
- Review summary:
  - `guideline_line/CV_Test_hard_mining/clean_mining_600_seed123_v2/summary.json`
  - resume state: `last_index=599` (completed)
  - relabeled indices: `126`
- Added labels from this review:
  - total: `126`
  - main: `120`
  - quarantine: `6`
  - duplicate saves on same source image: `0`
- Rebuilt merged dataset:
  - merged positives total: `926`
  - main: `865`
  - quarantine: `61`
  - train: `787`
  - val: `139`
- Next warm-start run prepared:
  - command file: `guideline_line/runs_cv/ft_balanced_after_clean2000_v1_command.txt`
  - init checkpoint: `guideline_line/runs_cv/ft_balanced_after_clean400_v1/best.pt`
  - save dir: `guideline_line/runs_cv/ft_balanced_after_clean2000_v1`

## 2026-03-05 Clean 2000 Warm-Start Eval Result
- Run:
  - `guideline_line/runs_cv/ft_balanced_after_clean2000_v1`
- Direct-eval comparison dir:
  - `guideline_line/runs_cv/compare_ft_balanced_after_clean2000_v1_20260305`
- Full split (`139` val positives + `104` val negatives):
  - clean2000 candidate (`fullsplit_clean2000_best_direct.json`):
    - `P=0.9262 R=0.7793 F1=0.8464 cF1=0.9489 FP/frame=0.0577 thr=0.65 comp=0.8952`
  - clean2000 centerline checkpoint (`fullsplit_clean2000_best_centerline_direct.json`):
    - `P=0.9333 R=0.7064 F1=0.8042 cF1=0.9273 FP/frame=0.2115 thr=0.75 comp=0.8511`
  - current champion (`fullsplit_oldchamp_direct.json`):
    - `P=0.9469 R=0.7420 F1=0.8320 cF1=0.9318 FP/frame=0.0385 thr=0.60 comp=0.9017`
- Decision:
  - **Not promoted as balanced default**.
  - clean2000 candidate improves recall/F1/cF1, but still loses on composite due higher FP/frame.
  - Keep default pointer at `guideline_line/runs_cv/ft_balanced_after_clean400_v1/best.pt`.

## 2026-03-05 Clean 2000 Fixed-Split Audit
- Reports:
  - `guideline_line/benchmarks/analysis_20260305/ft_balanced_after_clean2000_v1_on_clean400split.json`
  - `guideline_line/benchmarks/analysis_20260305/ft_balanced_after_clean2000_v1_on_overnightsplit.json`
  - `guideline_line/benchmarks/analysis_20260305/ft_balanced_after_clean2000_v1_ultra_tiny_overnightsplit.json`
  - `guideline_line/benchmarks/analysis_20260305/ft_balanced_after_clean2000_v1_short_overnightsplit.json`
  - `guideline_line/benchmarks/analysis_20260305/ft_balanced_after_clean2000_v1_distractor_overnightsplit.json`
- Clean400 fixed split (`108` val positives + `104` val negatives):
  - clean2000 best:
    - `P=0.8851 R=0.7565 F1=0.8157 cF1=0.9585 FP/frame=0.0385 thr=0.70`
  - current balanced default (`guideline_line/runs_cv/ft_balanced_after_clean400_v1/best.pt`):
    - `P=0.9013 R=0.7827 F1=0.8378 cF1=0.9697 FP/frame=0.0385 thr=0.60`
- Overnight fixed split (`93` val positives + `104` val negatives):
  - clean2000 best:
    - `P=0.8922 R=0.7436 F1=0.8112 cF1=0.9523 FP/frame=0.0288 thr=0.80`
  - balanced default:
    - `P=0.8923 R=0.7539 F1=0.8173 cF1=0.9632 FP/frame=0.0481`
  - conservative low-FP:
    - `P=0.9017 R=0.6654 F1=0.7657 cF1=0.9161 FP/frame=0.0288`
- Fixed subset benchmarks on the overnight split:
  - ultra-tiny `<12`:
    - clean2000 best: `P=0.2176 R=0.6884 F1=0.3307 cF1=0.9489 FP/frame=0.0288`
    - balanced default: `P=0.2293 R=0.6326 F1=0.3366 cF1=0.9481 FP/frame=0.0481`
    - tiny specialist: `P=0.2808 R=0.6465 F1=0.3915 cF1=0.9431 FP/frame=0.0288`
  - short `<20`:
    - clean2000 best: `P=0.5984 R=0.6683 F1=0.6314 cF1=0.9886 FP/frame=0.0288`
    - balanced default: `P=0.6285 R=0.6601 F1=0.6439 cF1=0.9860 FP/frame=0.0481`
    - tiny specialist: `P=0.6851 R=0.6716 F1=0.6783 cF1=0.9869 FP/frame=0.0288`
  - distractor negatives top48:
    - clean2000 best: `FP/frame=0.0208` (`1/48`)
    - balanced default: `FP/frame=0.0417` (`2/48`)
    - conservative low-FP: `FP/frame=0.0000` (`0/48`)
    - tiny specialist: `FP/frame=0.0000` (`0/48`)
- Decision:
  - **Not promoted as balanced default**.
  - clean2000 best is a stronger **low-FP candidate** than the old conservative fallback because it keeps the `0.0288` FP regime while recovering materially more recall and centerline quality.
  - clean2000 best still does **not** beat the balanced default or the tiny specialist on the short-line target slices, so the next run should target replay rather than another plain warm-start.

## 2026-03-05 Current 926-Split Replay Refresh
- Built new train-only replay manifests from `guideline_line/runs_cv/ft_balanced_after_clean2000_v1/splits_stratified.json`:
  - `guideline_line/hardcase_replay/after_clean2000_split_v1/hardcase_replay_catalog.json`
  - `guideline_line/hardcase_replay/after_clean2000_split_v1/train_ultra_tiny_lt12.json`
  - `guideline_line/hardcase_replay/after_clean2000_split_v1/train_short_lt20.json`
  - `guideline_line/hardcase_replay/after_clean2000_split_v1/train_distractor_negatives_top48.json`
- Counts:
  - train positives: `788`
  - train negatives: `587`
  - ultra-tiny train positives `<12`: `82`
  - short train positives `<20`: `194`
  - distractor negatives top48: `48`
- Next ablation prepared:
  - command file: `guideline_line/runs_cv/ft_after_clean2000_ultratiny_replay_v1_command.txt`
  - hypothesis: preserve the clean2000 low-FP gains and recover short-line F1 with positive replay only
  - failure criteria: do not promote if the overnight full-split result stays below the balanced default or if distractor negatives regress above `2/48` FP frames

## 2026-03-06 Ultra-Tiny Replay v1 Audit
- Run:
  - `guideline_line/runs_cv/ft_after_clean2000_ultratiny_replay_v1`
- Training-time best on the current 926 split:
  - epoch `2`
  - `P=0.9102 R=0.7763 F1=0.8379 cF1=0.9616 FP/frame=0.0577 thr=0.75`
- Fixed direct eval reports:
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_ultratiny_replay_v1_on_clean400split.json`
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_ultratiny_replay_v1_on_overnightsplit.json`
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_ultratiny_replay_v1_ultra_tiny_overnightsplit.json`
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_ultratiny_replay_v1_short_overnightsplit.json`
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_ultratiny_replay_v1_distractor_overnightsplit.json`
- Fixed eval outcome:
  - clean400 split: `P=0.9084 R=0.6774 F1=0.7761 cF1=0.9280 FP/frame=0.0481`
  - overnight split: `P=0.8888 R=0.7646 F1=0.8221 cF1=0.9629 FP/frame=0.0481`
  - ultra-tiny `<12`: `P=0.1946 R=0.6744 F1=0.3021 cF1=0.9481 FP/frame=0.0481`
  - short `<20`: `P=0.5693 R=0.6724 F1=0.6166 cF1=0.9902 FP/frame=0.0481`
  - distractor negatives top48: `FP/frame=0.0208` (`1/48`)
- Decision:
  - **Not promoted**.
  - This checkpoint is slightly stronger than the balanced default on the overnight full split, but it fails the robustness test because it regresses hard on the clean400 split and still underperforms on the short/ultra-tiny target slices.

## 2026-03-06 Old-Core Replay Retry
- Prepared command:
  - `guideline_line/runs_cv/ft_after_clean2000_ultratiny_replay_oldcore_v1_command.txt`
- Run:
  - `guideline_line/runs_cv/ft_after_clean2000_ultratiny_replay_oldcore_v1`
- Purpose:
  - test whether the older proven 39-sample ultra-tiny replay core transfers better than the expanded 82-sample replay set
- Match count on the current split:
  - old-core replay positives matched into train: `35`
- Fixed overnight eval:
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_ultratiny_replay_oldcore_v1_on_overnightsplit.json`
  - result: `P=0.8918 R=0.7473 F1=0.8132 cF1=0.9561 FP/frame=0.0385`
- Decision:
  - **Not promoted**.
  - Cleaner replay did not recover the target slices and still lost to the stronger replay-v1 candidate on the overnight full split.

## 2026-03-06 Replay v1 Centerline Candidate
- Candidate:
  - `guideline_line/runs_cv/ft_after_clean2000_ultratiny_replay_v1/best_centerline.pt`
- Fixed eval reports:
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_ultratiny_replay_v1_best_centerline_on_overnightsplit.json`
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_ultratiny_replay_v1_best_centerline_ultra_overnightsplit.json`
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_ultratiny_replay_v1_best_centerline_short_overnightsplit.json`
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_ultratiny_replay_v1_best_centerline_distractor_overnightsplit.json`
- Outcome:
  - overnight split: `P=0.8126 R=0.7695 F1=0.7905 cF1=0.9657 FP/frame=0.2596`
  - ultra-tiny `<12`: `P=0.0759 R=0.6837 F1=0.1366 FP/frame=0.2596`
  - short `<20`: `P=0.3140 R=0.6823 F1=0.4301 FP/frame=0.2596`
  - distractor negatives top48: `FP/frame=0.4583`
- Decision:
  - **Rejected**.
  - The centerline checkpoint is too unstable for any deployment profile.

## 2026-03-06 Postprocess Investigation
- Saved visual audit previews for the fixed ultra-tiny slice:
  - `guideline_line/benchmarks/analysis_20260306/eval_previews/`
- Axis-complete sweep on the strongest replay-v1 checkpoint:
  - summary: `guideline_line/benchmarks/axis_sweep_20260306/summary.json`
  - configs tested:
    - default
    - axis disabled
    - relaxed gap/extension/white-ratio gates
  - result:
    - all tested configs produced the same best metrics on both `ultra_tiny_lt12_plus_valneg` and `short_lt20_plus_valneg`
    - conclusion: current axis-complete settings are not the active bottleneck on these hard slices
- Additional one-off checks:
  - white-core intersection and skeleton-thinning variants on the replay-v1 checkpoint produced no material gain on ultra-tiny or short fixed benchmarks
- Interpretation:
  - the line center is often being localized correctly (`cF1` remains very high), but automated replay and light postprocess changes are not converting that into a robust pixel-mask gain

## 2026-03-06 Blocker Status
- Overnight result:
  - no new checkpoint was robustly better than the promoted balanced default
  - no new specialist clearly beat `guideline_line/runs_cv/ft_balanced_ultratiny_replay_v1/best.pt`
- Strongest remaining hypothesis:
  - the expanded ultra-tiny replay pool added too many weak or distribution-shifted short-line masks
  - the current bottleneck is now label quality / target definition on the hardest tiny-line cases, not a missing training flag
- Human-review artifact prepared:
  - `guideline_line/hardcase_replay/after_clean2000_split_v1/train_ultra_tiny_new_only_review_manifest.json`
  - contents: the `47` ultra-tiny replay samples that are present in the newer 82-sample replay set but absent from the older proven core set
- Recommended next human step:
  - manually audit those `47` samples first
  - if many are overly thick, partial, or target the wrong white structure, rebuild the ultra-tiny replay manifest from only the verified-good subset before launching the next fine-tune

## 2026-03-06 Soft-Distance Fine-Tune Result
- Trainer patch:
  - `guideline_line/train_cv_guideline.py`
  - added `--target-mode soft_distance`
  - added `--target-soft-distance-scale`
  - training loss now uses a tolerant distance-based mask target while keeping hard centerline supervision
- Smoke validation:
  - run: `guideline_line/runs_cv/smoke_softdistance_afterclean2000_v1`
  - purpose: verify the new target mode trains end to end from the current warm-start checkpoint
  - result: completed successfully and produced checkpoints + metrics
- Full run:
  - `guideline_line/runs_cv/ft_after_clean2000_softdistance_v1`
  - best epoch on the current 926 split:
    - `P=0.8495 R=0.8509 F1=0.8502 cF1=0.9574 FP/frame=0.0481 thr=0.70`
- Fixed direct eval reports:
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_softdistance_v1_on_clean400split.json`
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_softdistance_v1_on_overnightsplit.json`
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_softdistance_v1_ultra_overnightsplit.json`
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_softdistance_v1_short_overnightsplit.json`
  - `guideline_line/benchmarks/analysis_20260306/ft_after_clean2000_softdistance_v1_distractor_overnightsplit.json`
- Fixed eval outcome:
  - clean400 split:
    - soft-distance: `P=0.8305 R=0.8441 F1=0.8373 cF1=0.9625 FP/frame=0.0481 comp=0.8493`
    - prior balanced default: `P=0.9013 R=0.7827 F1=0.8378 cF1=0.9697 FP/frame=0.0385`
  - overnight split:
    - soft-distance: `P=0.8336 R=0.8403 F1=0.8369 cF1=0.9592 FP/frame=0.0385 comp=0.8520`
    - prior balanced default: `P=0.8923 R=0.7539 F1=0.8173 cF1=0.9632 FP/frame=0.0481`
  - ultra-tiny `<12`:
    - soft-distance: `P=0.2186 R=0.7767 F1=0.3412 cF1=0.9548 FP/frame=0.0385`
    - prior balanced default: `P=0.2293 R=0.6326 F1=0.3366 cF1=0.9481 FP/frame=0.0481`
    - prior tiny specialist: `P=0.2808 R=0.6465 F1=0.3915 cF1=0.9431 FP/frame=0.0288`
  - short `<20`:
    - soft-distance: `P=0.5912 R=0.7844 F1=0.6742 cF1=0.9911 FP/frame=0.0385`
    - prior balanced default: `P=0.6285 R=0.6601 F1=0.6439 cF1=0.9860 FP/frame=0.0481`
    - prior tiny specialist: `P=0.6851 R=0.6716 F1=0.6783 cF1=0.9869 FP/frame=0.0288`
  - distractor negatives top48:
    - soft-distance: `FP/frame=0.0208` (`1/48`)
    - prior balanced default: `FP/frame=0.0417` (`2/48`)
- Interpretation:
  - this is the first run that clearly improves the real short-line problem without giving back distractor control
  - fine-tuning is still viable; scratch training is **not** the immediate next step anymore
  - the remaining gap is mostly precision and exact tiny-line mask quality, not line localization

## 2026-03-06 Clean 600 Inference Sanity Check
- A/B inference summaries:
  - baseline: `guideline_line/CV_Test_hard_mining/clean_mining_600_seed123_v2_balanced_default/summary.json`
  - soft-distance: `guideline_line/CV_Test_hard_mining/clean_mining_600_seed123_v2_softdistance/summary.json`
- Disagreement outputs:
  - `guideline_line/CV_Test_hard_mining/clean_mining_600_seed123_v2_softdistance_disagreement/disagreement_review_summary_top21.json`
  - `guideline_line/CV_Test_hard_mining/clean_mining_600_seed123_v2_softdistance_disagreement/disagreement_ranked_top21.json`
- Result:
  - only `21` strong disagreements remained across `600` clean mining images after excluding already-labeled source images
  - disagreement mix in the top-21 shortlist:
    - `partial_disagreement`: `3`
    - `secondary_stronger`: `9`
    - `weak_disagreement`: `9`
  - spot-check of the top disagreement overlays showed the soft-distance model recovering missing line segments near the cue ball rather than introducing obviously unrelated structures
- Promotion decision:
  - promote `guideline_line/runs_cv/ft_after_clean2000_softdistance_v1/best.pt` as the new main checkpoint candidate
  - keep `guideline_line/runs_cv/overnight_20260301_162856/exp04_resnet34_640_neg1p0/run/best.pt` as the conservative low-FP fallback
  - keep `guideline_line/runs_cv/ft_balanced_ultratiny_replay_v1/best.pt` as the tiny-line specialist until a later soft-distance specialist cleanly beats it on the ultra-tiny slice

## 2026-03-08 Hardcase600 Replay Debug + Closure
- Important bug found in the first gentle replay attempt:
  - run: `guideline_line/runs_cv/ft_after_clean2000_softdistance_hardcase600_gentle_v1`
  - trainer log showed `Hardcase replay: pos=0`
  - cause: the repaired-positive manifest was initially built from source image paths instead of saved dataset sample ids, so the replay set never matched the current train split
- Fixes applied:
  - cleaned exact duplicate saves from the disagreement review batch (`MyScreen_000315.png`)
  - patched `guideline_line/review_infer_repairs.py` to block future exact source-image duplicate saves
  - patched `guideline_line/build_repaired_positive_manifest.py` so it resolves saved dataset sample ids from `source_image_path`
  - rebuilt merged dataset:
    - `guideline_line/data_zoomprobe_merged/annotations.jsonl`
    - merged positives total: `944`
    - main: `882`
    - quarantine: `62`
  - rebuilt repaired-positive manifest:
    - `guideline_line/hardcase_replay/clean_mining_600_seed123_v2_softdistance_live_repaired_pos.json`
    - saved positives in manifest: `17`
- Corrected gentle replay run:
  - run: `guideline_line/runs_cv/ft_after_clean2000_softdistance_hardcase600_gentle_v2`
  - replay match count: `13` train positives
  - training best epoch: `1`
  - training composite peaked immediately and then degraded, which is a classic sign that this replay set is too narrow / too redundant for further warm-start improvement
- Fixed direct eval outcome:
  - clean400 split:
    - `P=0.8314 R=0.7433 F1=0.7849 cF1=0.9211 FP/frame=0.0865`
  - overnight split:
    - `P=0.7989 R=0.8451 F1=0.8213 cF1=0.9590 FP/frame=0.0865`
  - ultra-tiny `<12`:
    - `P=0.1265 R=0.7860 F1=0.2179 cF1=0.9560 FP/frame=0.0865`
  - short `<20`:
    - `P=0.4397 R=0.7959 F1=0.5665 cF1=0.9913 FP/frame=0.0865`
  - distractor negatives top48:
    - `FP/frame=0.1042`
- Decision:
  - **Do not promote** either hardcase600 replay run.
  - Low epoch count was the correct choice here; increasing epochs would not solve this branch because the corrected replay set already overfits almost immediately.
  - Keep `guideline_line/runs_cv/ft_after_clean2000_softdistance_v1/best.pt` as the main checkpoint.
  - Stop this replay branch and only resume targeted training when there is a larger, more diverse hardcase batch rather than near-duplicate corrections.
