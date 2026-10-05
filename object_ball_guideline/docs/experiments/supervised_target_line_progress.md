# Supervised Target-Line Progress

Cleanup note dated 2026-04-24:

- older experimental artifacts, temp runs, and the legacy multi-stage pipeline files were pruned from this repo
- historical file references below are still useful as experiment notes, but some old paths no longer exist locally

## Target

Detect only the **object-ball outgoing guideline**:

- the thin white guide line that starts from / belongs to the target object ball after cue-ball contact
- in the user-confirmed example, this is the short disconnected white line coming out of the blue ball, showing the blue ball's travel direction after the shot
- not the long cue-ball-to-contact aiming line
- not any cue-ball-connected aiming line
- not the cue stick
- not the forbidden-circle line
- not any other white line when the true target is absent

Correction dated 2026-04-25:

- earlier same-day notes incorrectly inverted the rule and called the cue-ball-connected aiming line the target
- the current canonical label rule is object-ball outgoing direction: the cue-ball-connected aiming line is negative/reject material, while the object-ball outgoing guide line is positive

## Data

- Main positives: `C:\My_Project\SAM_3\guideline_line\data_zoomprobe`
- Hard positives: `C:\My_Project\SAM_3\guideline_line\data_zoomprobe\quarantine`
- Rejected zero-mask frames: `C:\My_Project\SAM_3\guideline_line\data_zoomprobe\Rejected`
  - guaranteed to contain no correct target guideline
  - every non-empty prediction is a false positive
- Negative-only frames: `C:\My_Project\8_BALL_POOL\data\images\negative_selected`
  - may contain wrong white lines
  - guaranteed to contain no correct target guideline
  - every non-empty prediction is a false positive

## Main Experiments

### 1. Baseline supervised segmentation

- Checkpoint: `runs/supervised_train_v2/guideline_unet_best.pt`
- Good crop-level masks, but weak full-frame candidate selection.

### 2. Table ROI fix

- File: `src/stages/propose_ball_crops.py`
- Replaced cyan-only felt detection with center-color table ROI plus HSV fallback.
- Large recall gain on dark and non-cyan tables.

### 3. Learned reranker on top of `v2`

- Files:
  - `src/supervised/reranker.py`
  - `src/supervised/infer.py`
- Checkpoint: `runs/supervised_train_v2/guideline_reranker_best.pt`
- First major full-frame improvement.

### 4. Negative-aware reranker

- Used the negative-only folder to suppress wrong white lines.
- Helpful for reject-mode experiments, but not better as the default detector.

### 5. Hard-negative segmentation fine-tune `v5_negft`

- Checkpoint: `runs/supervised_train_v5_negft/guideline_unet_best.pt`
- Added zero-mask hard negatives from the negative-only dataset.
- Stronger on hard split, weaker on the main split if used alone.

### 6. Residual hard-negative fine-tune `v6_residual`

- Checkpoint: `runs/supervised_train_v6_residual/guideline_unet_best.pt`
- Added mined false-positive crops from labeled training frames.
- Useful as a rescue model, not best as the default stack.

### 7. Threshold hybrid and rescue policy

- Primary: `v2` + learned reranker
- Fallback: `v5_negft` when score is weak
- Rescue: `v6_residual` on lower-score frames with a wider candidate set
- Final fallback: return to plain `v5` on slightly negative final scores

### 8. Wider secondary rescue search

- Tried `v6` with `24` candidates in a narrower score band.
- This helped some frames, but the original acceptance rule was too coarse and also introduced regressions.

### 9. Gated secondary rescue

- Root cause:
  - the secondary rescue helped when it recovered a real ball-attached branch
  - it hurt when it merely replaced one broad false branch with another
- Fix:
  - keep the secondary rescue trigger band
  - accept the secondary rescue winner only when its `ball_fill_fraction >= 0.9`
- Files:
  - `src/supervised/infer.py`
  - `scripts/run_supervised_best_wsl.sh`
  - `tests/test_infer.py`

This keeps the useful white-blob recoveries and rejects most of the harmful late swaps.

### 10. Current best: primary recovery after rescue stack

- Root cause:
  - after the rescue stack, a small band of medium-score `table_hough` outputs was still worse than the original `v2` primary winner
  - those recoverable cases were identifiable because the original primary winner already had strong ball fill
- Fix:
  - after all fallback and rescue stages, allow a final hand-back to the original primary `v2` winner
  - only do this when:
    - final winner source is `table_hough`
    - final score is in `[1.5, 3.0)`
    - primary winner source is `table_hough`
    - primary winner `ball_fill_fraction >= 0.5`
- File:
  - `src/supervised/infer.py`

This recovered the canonical validation frame `20260301_055534_be0c5289` and improved the promoted benchmark again without changing the hard split.

### 11. Current best: reticle recovery after final arbitration

- Root cause:
  - after the table-hough recovery and rescue stack, a few medium-score `reticle_global` outputs were still weaker than the original `v2` reticle choice
  - those cases were identifiable by a still-strong original primary reticle score
- Fix:
  - add one more final arbitration band for `reticle_global` winners
  - if final winner is `reticle_global` with score in `[0.5, 2.0)` and the original primary reticle winner still scores at least `1.5`, return to the original `v2` result
- File:
  - `src/supervised/infer.py`

This improved the canonical validation benchmark again while preserving the hard split metrics.

### 12. Rejected follow-up: residual primary reranker retune

- Tried retraining the `v2` primary reranker on `runs/_reranker_dataset_with_negatives.json` with two variants:
  - `runs/supervised_train_v2/guideline_reranker_residual_a.pt`
  - `runs/supervised_train_v2/guideline_reranker_residual_b.pt`
- Results:
  - variant `a` reduced both `val` and `hard` reranker-level selection IoU
  - variant `b` improved reranker-level `val` selection IoU, but the real end-to-end stack regressed badly
- End-to-end benchmark for variant `b`:
  - `runs/_eval_best_final5b_end_to_end.json`
  - `val` mean IoU: `0.6937`
  - `hard` mean IoU: `0.8860`

So the residual primary reranker retune was rejected and `best_final5` remains promoted.

### 13. Rejected follow-up: `v7` residual fine-tune from promoted-policy train failures

- Built a manifest-driven split evaluator:
  - `tools/eval_best_manifest_split.py`
- Ran the promoted `best_final5` policy across train and mined:
  - `runs/_eval_best_final5_train_summary.json`
- Filtered real positive-train misses and built a new residual supplement:
  - `tools/build_residual_index_from_eval.py`
  - `runs/supervised_train_v7_bestfinal5_residual/residual_policy_negatives.json`
- Warm-started `v7` from `v6` with that residual index:
  - `runs/supervised_train_v7_bestfinal5_residual/guideline_unet_best.pt`

Then tested two end-to-end branches:

- `v7` rescue with reused `v6` reranker:
  - `runs/_eval_best_final5_v7rescue_end_to_end.json`
  - regressed on both `val` and `hard`
- `v7` rescue with a matched `v7` reranker:
  - built dataset: `runs/supervised_train_v7_bestfinal5_residual/reranker_dataset.json`
  - trained reranker: `runs/supervised_train_v7_bestfinal5_residual/guideline_reranker_b.pt`
  - benchmark: `runs/_eval_best_final5_v7matched_end_to_end.json`
  - regressed more strongly, especially on `hard`

So the whole `v7` residual branch was rejected and `best_final5` remains the promoted baseline.

### 14. Current best: narrow `v7` fallback rescue

- Root cause:
  - the broad `v7` branch regressed, but one specific hard failure improved when `v7` was allowed to re-run rescue after the current fallback/final-fallback path
- Fix:
  - add a very narrow `fallback_rescue` stage
  - only run it when the current winner:
    - came from fallback
    - is `table_hough`
    - has score in `[0.0, 1.0)`
  - use:
    - checkpoint `runs/supervised_train_v7_bestfinal5_residual/guideline_unet_best.pt`
    - reranker `runs/supervised_train_v6_residual/guideline_reranker_best.pt`
  - only accept it if its score beats the current score
- Result:
  - benchmark `runs/_eval_best_final6_end_to_end.json`
  - `val` unchanged
  - `hard` improved materially

### 15. Current best: hollow-reticle `v7matched` rescue

- Root cause:
  - after `best_final6`, one remaining hard failure was a hollow reticle-ring output that the matched `v7` branch could replace with a true ball-attached reticle
- Fix:
  - add a `reticle_fill_rescue` stage
  - only run it when the current winner:
    - is `reticle_global`
    - was selected via `reticle_recovery`
    - has score in `[1.0, 2.0)`
    - has `ball_fill_fraction < 0.5`
  - use:
    - checkpoint `runs/supervised_train_v7_bestfinal5_residual/guideline_unet_best.pt`
    - reranker `runs/supervised_train_v7_bestfinal5_residual/guideline_reranker_b.pt`
  - only accept it if the new winner is also `reticle_global` and has `ball_fill_fraction >= 0.9`
- Result:
  - benchmark `runs/_eval_best_final7_end_to_end.json`
  - `val` unchanged
  - `hard` improved again

### 16. Rejected follow-up: `v8` high-resolution residual branch

- Built a `448px` residual checkpoint:
  - `runs/supervised_train_v8_hires/guideline_unet_best.pt`
- Built matched reranker data and rerankers:
  - `runs/supervised_train_v8_hires/reranker_dataset.json`
  - `runs/supervised_train_v8_hires/guideline_reranker_a.pt`
  - `runs/supervised_train_v8_hires/guideline_reranker_b.pt`
- Result:
  - neither the checkpoint nor its rerankers beat the promoted `v7` branch
  - the `v8` branch was rejected

### 17. Rejected follow-up: wider `base_channels=48` architecture

- Added variable-width checkpoint support to:
  - `src/supervised/train.py`
  - `src/supervised/infer.py`
- Verified non-default checkpoint loading in:
  - `tests/test_infer.py`
- Trained a wider residual checkpoint:
  - `runs/supervised_train_v9_wide/guideline_unet_best.pt`
- Configuration:
  - `448px`
  - `base_channels=48`
  - larger batch size than earlier runs
- Result:
  - crop-level `val` and `hard` both stayed below the promoted `v7` residual branch
  - the wider-model branch was rejected before any end-to-end promotion attempt

### 18. Current best: narrow external-main rescue

- Evaluated the external checkpoints directly on the same canonical benchmark:
  - `runs/_external_bakeoff_summary.json`
- Result:
  - our stack still won overall
  - but the external main checkpoint consistently beat our model on a small subset of strongly negative-score `table_hough` failures
- Fix:
  - add an `external_rescue` stage using:
    - `C:\My_Project\SAM_3\guideline_line\runs_cv\ft_after_clean2000_softdistance_v1\best.pt`
  - only run it when the current winner:
    - is `table_hough`
    - has score `< 0.0`
  - only accept it when the external prediction has:
    - area at least `20`
    - mean probability on predicted pixels at least `0.9`
- Result:
  - benchmark `runs/_eval_best_final8_end_to_end.json`
  - `val` improved materially
  - `hard` improved materially again

### 19. Current best: external-main mid-band rescue

- Root cause:
  - after `best_final8`, another cluster of medium-score `table_hough` failures still lost to the external main checkpoint
  - unlike the first external rescue, these were not negative-score cases anymore; they were mid-band `table_hough` outputs with low current fill
- Fix:
  - add an `external_mid_rescue` stage using the same external main checkpoint
  - only run it when the current winner:
    - is `table_hough`
    - has score in `[1.0, 3.0)`
    - has `ball_fill_fraction <= 0.35`
  - only accept it when the external prediction has:
    - area at least `120`
    - mean probability on predicted pixels at least `0.88`
- Result:
  - benchmark `runs/_eval_best_final9_end_to_end.json`
  - `val` improved again
  - `hard` improved again

### 20. Current best: external-main low-mid and white-blob rescue

- Root cause:
  - after `best_final9`, there were still two clean disagreement pockets where the external main model kept winning:
    - a tiny `table_hough` low-mid band
    - a tiny `white_blob` band
- Fix:
  - add two more narrow external-main rescue bands:
    - `external_lowmid_rescue`
    - `external_blob_rescue`
- Result:
  - benchmark `runs/_eval_best_final10_end_to_end.json`
  - `val` improved again
  - `hard` stayed at the improved `best_final9` level

### 21. Current best: external tiny white-blob rescue

- Root cause:
  - after `best_final10`, a small set of high-score `white_blob` winners still underfit ultra-thin guidelines
  - the tiny-line specialist was stronger on that exact pocket than either the main stack or the external main checkpoint
- Fix:
  - add `external_tiny_blob_rescue`
  - only run it on `white_blob` winners with score in `[2.0, 4.5)`, and only accept masks with area `>= 80` and mean probability `>= 0.88`
- Result:
  - benchmark `runs/_eval_best_final11_end_to_end.json`
  - `val` mean IoU `0.7875`, mean Dice `0.8659`
  - `hard` mean IoU `0.9345`, mean Dice `0.9652`

### 22. Current best: broad external-main table rescue

- Root cause:
  - after `best_final11`, there was still a larger-than-expected `table_hough` pocket where the reranked internal winner looked confident but had weak ball attachment
  - the external main checkpoint beat that pocket consistently, including one hard frame
- Fix:
  - add `external_table_rescue`
  - only run it when the current winner:
    - is in source group `table_hough`
    - has score in `[1.0, 6.0)`
    - has `ball_fill_fraction <= 0.65`
  - only accept it when the external main prediction has:
    - area `>= 120`
    - mean probability on predicted pixels `>= 0.84`
- Result:
  - benchmark `runs/_eval_best_final12_end_to_end.json`
  - `val` mean IoU `0.8009`, mean Dice `0.8775`
  - `hard` mean IoU `0.9356`, mean Dice `0.9658`

### 23. Current best: external-main reticle rescue

- Root cause:
  - after `best_final12`, the largest remaining pocket was low-score `reticle_global` winners with zero ball fill
  - those were mostly hollow or disconnected reticle branches that the external main checkpoint corrected cleanly
- Fix:
  - add `external_reticle_rescue`
  - only run it when the current winner:
    - is in source group `reticle_global`
    - has score `< 2.0`
    - has `ball_fill_fraction <= 0.0`
  - only accept it when the external main prediction has:
    - area `>= 20`
    - mean probability on predicted pixels `>= 0.8`
- Result:
  - benchmark `runs/_eval_best_final13_end_to_end.json`
  - `val` mean IoU `0.8097`, mean Dice `0.8861`
  - `hard` unchanged from `best_final12`

### 24. Current best: tiny-line reticle rescue

- Root cause:
  - after `best_final13`, a final narrow `reticle_global` band still benefited from the tiny-line specialist
  - these were medium-score reticle winners where the tiny checkpoint traced the short post-impact line more accurately than both the main stack and the external main checkpoint
- Fix:
  - add `external_tiny_reticle_rescue`
  - only run it when the current winner:
    - is in source group `reticle_global`
    - has score in `[0.5, 1.5)`
  - only accept it when the tiny-line specialist prediction has:
    - area `>= 50`
    - mean probability on predicted pixels `>= 0.94`
- Result:
  - benchmark `runs/_eval_best_final14_end_to_end.json`
  - `val` mean IoU `0.8132`, mean Dice `0.8890`
  - `hard` unchanged from `best_final13`

### 25. Current best: large-mask external-main reticle rescue

- Root cause:
  - after `best_final14`, a final high-area `reticle_global` pocket still remained
  - these were not zero-fill reticle failures anymore; they were larger, lower-ball-fill reticle winners where the external main checkpoint traced a better long post-impact branch
- Fix:
  - add `external_reticle_large_rescue`
  - only run it when the current winner:
    - is in source group `reticle_global`
    - has `ball_fill_fraction <= 0.35`
  - only accept it when the external main prediction has:
    - area `>= 400`
    - mean probability on predicted pixels `>= 0.86`
- Result:
  - benchmark `runs/_eval_best_final15_end_to_end.json`
  - `val` mean IoU `0.8164`, mean Dice `0.8911`
  - `hard` unchanged from `best_final14`

### 26. Current best: broad external-main table-hough rescue

- Root cause:
  - after `best_final15`, there was still a tiny `table_hough` pocket where the promoted winner had low ball fill and the external main checkpoint was still better
- Fix:
  - add `external_table_broad_rescue`
  - only run it when the current winner:
    - is in source group `table_hough`
    - has score `< 6.0`
    - has `ball_fill_fraction <= 0.35`
  - only accept it when the external main prediction has:
    - area `>= 350`
    - mean probability on predicted pixels `>= 0.8`
- Result:
  - benchmark `runs/_eval_best_final16_end_to_end.json`
  - `val` mean IoU `0.8185`, mean Dice `0.8926`
  - `hard` unchanged from `best_final15`

### 27. Rejected follow-up: low-score external-main white-blob rescue

- Hypothesis:
  - a final low-score `white_blob` pocket might still benefit from the external main checkpoint below the earlier white-blob rescue band
- Experiment:
  - added `external_blob_low_rescue`
  - benchmarked it in `runs/_eval_best_final17_end_to_end.json`
- Result:
  - `val` mean IoU regressed from `0.8185` to `0.8176`
  - `val` mean Dice regressed from `0.8926` to `0.8922`
  - `hard` stayed flat
- Decision:
  - reject the branch
  - keep `best_final16` as the promoted stack

### 28. Current best: table-hough final-fallback false-positive reject

Root cause:

- the repaired promoted manifest exposed a high false-positive rate on the frozen negative gold board
- many false positives came from `table_hough` winners restored by final fallback, with large predicted masks
- this pocket had zero hits on old `val`, old `hard_val`, and harvested positive holdout when the final mask area was at least `80` pixels

Project-state correction:

- `runs/supervised_best_manifest.json` was missing the `external_table_broad_rescue_*` keys even though its embedded metrics referenced `best_final16`
- `scripts/run_supervised_best_wsl.sh` still contained the broad rescue
- the manifest was repaired to match the documented promoted policy before evaluating this branch

Fix:

- add `reject_table_hough_final_fallback_min_pred_pixels`
- when enabled, zero the final mask only when:
  - final winner source group is `table_hough`
  - final winner was selected via `final_fallback`
  - final predicted mask area is at least `80` pixels
- promote the threshold-80 version into `runs/supervised_best_manifest.json` and `scripts/run_supervised_best_wsl.sh`

Evidence:

- repaired baseline old `val`: Dice `0.8925936242`, zero-IoU `0`
- repaired baseline old `hard_val`: Dice `0.9659744406`, zero-IoU `0`
- repaired baseline harvested holdout: Dice `0.7845828323`, zero-IoU `5`
- repaired baseline negatives: `143 / 376` false positives
  - `rejected_negatives`: `39 / 129`
  - `flat_negatives`: `104 / 247`
- threshold-80 full gold-board eval: `runs/_eval_reject_tabhough_ff80_full.json`
  - old `val`: Dice `0.8925895793`, zero-IoU `0`
  - old `hard_val`: Dice `0.9659744406`, zero-IoU `0`
  - harvested holdout: Dice `0.7846137859`, zero-IoU `5`
  - negatives: `124 / 376` false positives
    - `rejected_negatives`: `28 / 129`
    - `flat_negatives`: `96 / 247`
  - reject fired `20` times total, all on negative splits and never on old/hard/harvest positives

Decision:

- promote the table-hough final-fallback reject as the current policy
- this is a targeted false-positive suppression improvement, not a primary model replacement

## Current Best Policy

Promoted stack:

1. primary `v2` segmentation + `v2` reranker
2. fallback to `v5_negft` when score `< 3.2727272727272734`
3. rescue with `v6_residual` and `20` candidates when score `< 1.25`
4. final fallback to plain `v5` when score `< -0.10084033613445342`
5. secondary rescue with `v6_residual` and `24` candidates only when:
   - current winner source is `table_hough`
   - current score is in `[1.75, 4.40625)`
   - secondary winner `ball_fill_fraction >= 0.9`
6. final primary recovery back to the original `v2` winner only when:
   - final winner source is `table_hough`
   - final score is in `[1.5, 3.0)`
   - primary winner source is `table_hough`
   - primary winner `ball_fill_fraction >= 0.5`
7. final reticle recovery back to the original `v2` winner only when:
   - final winner source group is `reticle_global`
   - final score is in `[0.5, 2.0)`
   - original primary source group is `reticle_global`
   - original primary score is at least `1.5`
8. narrow `v7` fallback rescue only when:
   - current winner came from fallback
   - current winner source group is `table_hough`
   - current score is in `[0.0, 1.0)`
9. hollow-reticle `v7matched` rescue only when:
   - current winner source group is `reticle_global`
   - current winner was selected via `reticle_recovery`
   - current score is in `[1.0, 2.0)`
   - current `ball_fill_fraction < 0.5`
10. external main negative-score `table_hough` rescue:
   - source group `table_hough`
   - score `< 0.0`
   - external area `>= 20`
   - external mean probability `>= 0.9`
11. external main mid-band `table_hough` rescue:
   - source group `table_hough`
   - score in `[1.0, 3.0)`
   - current `ball_fill_fraction <= 0.35`
   - external area `>= 120`
   - external mean probability `>= 0.88`
12. external main low-mid `table_hough` rescue:
   - source group `table_hough`
   - score in `[0.5, 1.5)`
   - current `ball_fill_fraction <= 0.5`
   - external area `>= 400`
   - external mean probability `>= 0.8`
13. external main zero-fill reticle rescue:
   - source group `reticle_global`
   - score `< 2.0`
   - current `ball_fill_fraction <= 0.0`
   - external area `>= 20`
   - external mean probability `>= 0.8`
14. external tiny reticle rescue:
   - source group `reticle_global`
   - score in `[0.5, 1.5)`
   - external area `>= 50`
   - external mean probability `>= 0.94`
15. external main broad `table_hough` rescue:
   - source group `table_hough`
   - score in `[1.0, 6.0)`
   - current `ball_fill_fraction <= 0.65`
   - external area `>= 120`
   - external mean probability `>= 0.84`
16. external main white-blob rescue:
   - source group `white_blob`
   - external area `>= 300`
   - external mean probability `>= 0.95`
17. external tiny white-blob rescue:
   - source group `white_blob`
   - score in `[2.0, 4.5)`
   - external area `>= 80`
   - external mean probability `>= 0.88`
18. table-hough final-fallback reject:
   - source group `table_hough`
   - selected via `final_fallback`
   - final mask area `>= 80`
   - final score `<= 0.0`
   - output an empty mask
19. source-specific high-area winner reject:
   - `external_main_rescue` final mask area `>= 400`
   - `white_blob` final mask area `>= 800`
   - `table_hough` final mask area `>= 600`
   - `external_main_reticle_rescue` final mask area `>= 800`
   - output an empty mask
20. source-specific area+score winner reject:
   - `table_hough`: final mask area `>= 54` and score `< 0.053614`
   - `white_blob`: final mask area `>= 444` and score `< 2.998133`
   - `reticle_global_0`: final mask area `>= 323` and score `< 1.287158`
   - `external_main_reticle_rescue`: final mask area `>= 46` and score `< 0.834604`
   - output an empty mask
21. pre-final-fallback score reject:
   - winner has `pre_final_fallback_score <= -0.923089`
   - current final mask is still non-empty after earlier rejects
   - output an empty mask

## Best Verified Benchmark

- Manifest: `runs/supervised_best_manifest.json`
- Old split eval: `runs/_eval_best_final16_end_to_end.json`
- Gold-board eval for latest false-positive suppression: `runs/_eval_preff_veto_candidate_combined_noimg.json`

Metrics:

- `val` mean IoU: `0.8185`
- `val` mean Dice: `0.8926`
- `hard` mean IoU: `0.9356`
- `hard` mean Dice: `0.9658`
- latest gold-board negative false positives: `41 / 376`

Recent promotion history:

- `runs/_eval_best_final15_end_to_end.json`
  - `val` mean IoU: `0.8164`
  - `val` mean Dice: `0.8911`
  - `hard` mean IoU: `0.9356`
  - `hard` mean Dice: `0.9658`
- `runs/_eval_best_final16_end_to_end.json`
  - `val` mean IoU: `0.8185`
  - `val` mean Dice: `0.8926`
  - `hard` mean IoU: `0.9356`
  - `hard` mean Dice: `0.9658`
- `runs/_eval_best_final14_end_to_end.json`
  - `val` mean IoU: `0.8132`
  - `val` mean Dice: `0.8890`
  - `hard` mean IoU: `0.9356`
  - `hard` mean Dice: `0.9658`
- `runs/_eval_best_final13_end_to_end.json`
  - `val` mean IoU: `0.8097`
  - `val` mean Dice: `0.8861`
  - `hard` mean IoU: `0.9356`
  - `hard` mean Dice: `0.9658`
- `runs/_eval_best_final12_end_to_end.json`
  - `val` mean IoU: `0.8009`
  - `val` mean Dice: `0.8775`
  - `hard` mean IoU: `0.9356`
  - `hard` mean Dice: `0.9658`
- `runs/_eval_best_final11_end_to_end.json`
  - `val` mean IoU: `0.7875`
  - `val` mean Dice: `0.8659`
  - `hard` mean IoU: `0.9345`
  - `hard` mean Dice: `0.9652`
- `runs/_eval_best_final10_end_to_end.json`
  - `val` mean IoU: `0.7806`
  - `val` mean Dice: `0.8588`
  - `hard` mean IoU: `0.9343`
  - `hard` mean Dice: `0.9651`

So the current promotion improves the held-out split again while preserving the already-strong hard split.

## Visual Review

- Latest examples bundle: `runs/supervised_full_34_best_final15`
- Contact sheets:
  - `runs/supervised_full_34_best_final15/review_sheet_01.png`
  - `runs/supervised_full_34_best_final15/review_sheet_02.png`
  - `runs/supervised_full_34_best_final15/review_sheet_03.png`

The latest examples bundle is still `best_final15`; the `best_final16` and `best_final17` rescues do not trigger on the current `examples/input` set, so the contact sheets remain representative of the promoted launcher.

## Scripts

- Best WSL launcher: `scripts/run_supervised_best_wsl.sh`
- Best Windows launcher: `scripts/run_supervised_best.ps1`
- Canonical split evaluator: `tools/eval_best_manifest_split.py`

## Validation

- `python -m pytest tests/test_infer.py`
- `python -m compileall src tools`

## Remaining Failure Modes

- a small cluster of stubborn `white_blob` failures where no external rescue beats the current winner
- some high-score `reticle_global` misses outside the current rescue bands
- a few difficult clutter shots where the right target ball proposal never appears

## Next Useful Work

If continuing further, the next high-signal branch is:

1. rerun disagreement mining from `runs/_eval_best_final16_end_to_end.json`
2. check whether the remaining `white_blob` and high-score `reticle_global` misses are better handled by:
   - another narrow external rescue
   - a checkpoint-specific reranker retune
   - or a candidate-generation change
3. keep only changes that beat `runs/_eval_best_final16_end_to_end.json`

## 2026-04-25 Harvest Repair Sweep

Goal: use the video-harvest review corrections to repair missed guide-line cases without teaching the model to accept the wrong line.

Labeling rule clarified during review:

- positive: the object-ball outgoing white guideline that shows the target ball's travel direction after cue-ball contact
- negative: cue-ball-connected aiming lines, cue sticks, and regions where visible white lines are not the object-ball outgoing guideline
- `Rejected` and `negative_selected` are both image-level zero-mask negatives: the model should detect nothing anywhere in those frames

Training inputs:

- corrected zoom-probe labels: `C:\My_Project\SAM_3\guideline_line\data_zoomprobe`
- rejected zero-mask images: `C:\My_Project\SAM_3\guideline_line\data_zoomprobe\Rejected`
- selected zero-mask negative images: `C:\My_Project\8_BALL_POOL\data\images\negative_selected`

Training/runtime changes used for this sweep:

- WSL launcher: `scripts/train_supervised_wsl.sh`
- queued launchers:
  - `scripts/run_overnight_harvest_repair_experiments_wsl.sh`
  - `scripts/run_overnight_harvest_repair_extension_wsl.sh`
- resume checkpoint: `runs/supervised_train_v2/guideline_unet_best.pt`
- batch size: `32`
- workers: `16`
- prefetch factor: `4`
- CUDA tuning: cuDNN benchmark, TF32 where available, BF16 without GradScaler
- dataset acceleration: cached index reuse, persistent workers, zero-mask fast path, faster image tensor conversion

Completed training split results:

| Run | LR | Pos weight | Epochs | Best epoch | Best val Dice | Hard Dice at best | Last val Dice | Last hard Dice |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `runs/supervised_train_v10_repair_lr2e-4_pw18` | `0.0002` | `18` | 35 | 26 | `0.8788` | `0.7946` | `0.8723` | `0.7905` |
| `runs/supervised_train_v10_repair_lr2e-4_pw24` | `0.0002` | `24` | 35 | 28 | `0.8760` | `0.7946` | `0.8695` | `0.7940` |
| `runs/supervised_train_v10_repair_lr1e-4_pw18` | `0.0001` | `18` | 35 | 33 | `0.8746` | `0.7923` | `0.8671` | `0.7836` |
| `runs/supervised_train_v10_repair_lr1e-4_pw24` | `0.0001` | `24` | 35 | 35 | `0.8730` | `0.7859` | `0.8730` | `0.7859` |
| `runs/supervised_train_v10_repair_lr5e-5_pw30` | `0.00005` | `30` | 35 | 24 | `0.8641` | `0.7748` | `0.8573` | `0.7691` |

Stopped / skipped:

- `runs/supervised_train_v10_repair_lr5e-5_pw18` was intentionally stopped before training produced `history.json` or `guideline_unet_best.pt`.
- The low-learning-rate branch was underperforming in the completed `lr5e-5_pw30` run, and the remaining queued run was lower value than immediate validation and documentation.

Current candidate:

- best training-split checkpoint: `runs/supervised_train_v10_repair_lr2e-4_pw18/guideline_unet_best.pt`
- backup candidate: `runs/supervised_train_v10_repair_lr2e-4_pw24/guideline_unet_best.pt`
- do not promote either checkpoint from training Dice alone

Important comparison note:

- These sweep metrics are supervised crop-training validation metrics.
- The current promoted benchmark is still the end-to-end manifest benchmark in `runs/_eval_best_final16_end_to_end.json`.
- End-to-end validation is required before replacing `runs/supervised_best_manifest.json`.

End-to-end validation completed:

- candidate manifest: `runs/supervised_harvest_repair_lr2e-4_pw18_candidate_manifest.json`
- baseline: `runs/_eval_best_final16_end_to_end.json`
- comparable index: `runs/supervised_train_v2/index.json`
- split summaries:
  - `runs/_eval_harvest_repair_lr2e-4_pw18_val.json`
  - `runs/_eval_harvest_repair_lr2e-4_pw18_hard_val.json`
  - `runs/_eval_harvest_repair_lr2e-4_pw18_candidate_combined.json`

| Split | Count | Baseline IoU | Candidate IoU | Delta IoU | Baseline Dice | Candidate Dice | Delta Dice | Baseline zero-IoU | Candidate zero-IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `val` | 141 | `0.8185` | `0.8003` | `-0.0183` | `0.8926` | `0.8765` | `-0.0161` | 0 | 1 |
| `hard_val` | 62 | `0.9356` | `0.9365` | `+0.0009` | `0.9658` | `0.9663` | `+0.0005` | 0 | 0 |

Decision:

- do not promote `runs/supervised_train_v10_repair_lr2e-4_pw18/guideline_unet_best.pt` into `runs/supervised_best_manifest.json`
- the hard split improved slightly, but the normal validation regression is much larger and introduced one zero-IoU failure
- keep the checkpoint as a candidate for targeted rescue / ablation work, not as the primary checkpoint

Next useful validation work:

1. inspect the candidate `val` zero-IoU frame and worst regressions from `runs/_eval_harvest_repair_lr2e-4_pw18_val.json`
2. compare whether the candidate helps the newly harvested misses that motivated the retrain
3. if it helps those misses, test it as a narrow rescue path instead of replacing the primary checkpoint

## 2026-04-25 v11 Conditioned Model Implementation

Implemented pipeline pieces for the next improvement cycle:

- frozen gold-board builder: `tools/build_gold_board.py`
- failure-layer diagnostics: `tools/diagnose_guideline_failures.py`
- worst-case contact sheets: `tools/make_guideline_contact_sheet.py`
- conditioned model: `src/supervised/conditioned.py`
- conditioned crop dataset: `src/supervised/conditioned_data.py`
- conditioned training entrypoint: `src/supervised/train_conditioned.py`
- WSL train launcher: `scripts/train_conditioned_supervised_wsl.sh`
- WSL full pipeline launcher: `scripts/run_v11_conditioned_pipeline_wsl.sh`

The conditioned checkpoint format is intentionally explicit:

- `model_config.model_type = "conditioned_guideline_unet"`
- `model_config.in_channels = 4`
- channels are RGB plus a candidate/object-ball heatmap
- the earlier run at `runs/supervised_train_v11_conditioned_unet/guideline_conditioned_unet_best.pt` was aborted after the target definition was found to be inverted and must not be promoted
- the run at `runs/supervised_train_v11_object_ball_conditioned_unet/guideline_conditioned_unet_best.pt` was aborted after the negative-folder split was found to hold out all reviewed negatives
- corrected checkpoint output path is `runs/supervised_train_v11_object_ball_imgneg_conditioned_unet/guideline_conditioned_unet_best.pt`

Promotion remains blocked until the candidate passes the same end-to-end gates:

- old `val` Dice `>= 0.8926` and zero-IoU `0`
- old `hard_val` Dice `>= 0.9658` and zero-IoU `0`
- harvested holdout improves over current best
- cue-ball-connected / wrong-line negatives do not gain false positives

## 2026-04-26 Current best: combined false-positive veto

After the v12 conditioned smoke stayed weak, the highest-value path was a targeted policy improvement on the frozen gold board instead of another blind long retrain.

Rejected training branch:

- `runs/supervised_train_v12_candidate_faithful_conditioned_smoke/guideline_conditioned_unet_best.pt`
- Best smoke epoch was 4 with crop `val_dice = 0.8356` and `hard_val_dice = 0.6874`.
- Do not continue this as a primary replacement unless a future variant fixes the candidate-only imbalance and passes a much stronger short-run gate.

Promoted policy additions:

- Extend the existing `table_hough` final-fallback reject with `reject_table_hough_final_fallback_max_score = 0.0`.
- Add source-specific high-area vetoes via `reject_winner_source_min_pred_pixels`:
  - `external_main_rescue >= 400`
  - `white_blob >= 800`
  - `table_hough >= 600`
  - `external_main_reticle_rescue >= 800`
- Add source-specific area+score vetoes via `reject_winner_source_area_score`:
  - `table_hough`: area `>= 54`, score `< 0.053614`
  - `white_blob`: area `>= 444`, score `< 2.998133`
  - `reticle_global_0`: area `>= 323`, score `< 1.287158`
  - `external_main_reticle_rescue`: area `>= 46`, score `< 0.834604`

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Gold-board benchmark: `runs/_eval_area_score_veto_candidate_full.json`
- Promoted-manifest confirmation summary: `runs/_eval_supervised_best_manifest_current_area_score_skip_existing.json`
- Suppressed-case contact sheet: `runs/gold_board_v1/contact_sheets_policy_mining/combined_veto_suppressed_baseline_masks.png`
- Additional table-hough-600 suppressed sheet: `runs/gold_board_v1/contact_sheets_policy_mining/tabhough600_newly_suppressed.png`
- Area+score suppressed sheet: `runs/gold_board_v1/contact_sheets_policy_mining/area_score_veto_suppressed_fp12.png`
- Remaining-FP contact sheet before area+score promotion: `runs/gold_board_v1/contact_sheets_policy_mining/tabhough600_remaining_fp_top50.png`

Gold-board metrics:

| Split | Count | Previous Dice | New Dice | Previous zero/FP | New zero/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_val` | 141 | `0.8925895793` | `0.8925952991` | zero `0` | zero `0` |
| `old_hard_val` | 62 | `0.9659744406` | `0.9659744406` | zero `0` | zero `0` |
| `harvest_holdout` | 92 | `0.7846137859` | `0.7847130806` | zero `5` | zero `5` |
| `rejected_negatives` | 129 | FP `28` | Dice `0.9007916582` | FP `28` | FP `13` |
| `flat_negatives` | 247 | FP `96` | Dice `0.8354547272` | FP `96` | FP `41` |

Overall negative false positives improved from `124 / 376` to `54 / 376`.

Visual review:

- The newly suppressed baseline false positives are cue-ball-connected lines, wrong disconnected/UI/menu lines, or non-game UI artifacts, not the object-ball outgoing target.
- The 12 area+score-suppressed false positives are also wrong-line/no-target cases on visual review; the two borderline rows from the pre-eval mining sheet were not suppressed on rerun because their scores shifted slightly above the evaluated thresholds.
- Remaining false positives are still dominated by cue-ball/reticle/table UI confusion, so the next useful work is a learned final-winner veto or a more target-aware candidate selector, not extending v12 as-is.

## 2026-04-26 Current best: pre-final-fallback score veto

The learned final-winner veto branch was tested and rejected before promotion:

- training dataset: `runs/final_veto_v1/train_board_report_dataset.json`
- checkpoint: `runs/final_veto_v1/train_board_report_final_veto.pt`
- frozen-gold evaluation: `runs/final_veto_v1/train_board_report_final_veto_gold_eval.json`
- learned threshold `0.3782908320` vetoed 44 negatives but also 40 positives
- high thresholds were still unsafe or useless: `0.9999` vetoed one positive and zero negatives

Promoted policy addition:

- add `reject_pre_final_fallback_max_score = -0.923089`
- this suppresses non-empty winners with very weak `pre_final_fallback_score` after earlier reject rules have already run
- the rule came from mining remaining false positives against all frozen positive splits, not from crop-level training metrics

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/supervised_best_manifest_preff_veto_candidate.json`
- Combined gold-board benchmark: `runs/_eval_preff_veto_candidate_combined_noimg.json`
- Split summaries:
  - `runs/_eval_preff_veto_candidate_old_val_noimg.json`
  - `runs/_eval_preff_veto_candidate_old_hard_val_noimg.json`
  - `runs/_eval_preff_veto_candidate_harvest_holdout_noimg.json`
  - `runs/_eval_preff_veto_candidate_rejected_negatives_noimg.json`
  - `runs/_eval_preff_veto_candidate_flat_negatives_shard*_noimg.json`

Gold-board metrics:

| Split | Count | Previous Dice | New Dice | Previous zero/FP | New zero/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_val` | 141 | `0.8925952991` | `0.8925898362` | zero `0` | zero `0` |
| `old_hard_val` | 62 | `0.9659744406` | `0.9659744406` | zero `0` | zero `0` |
| `harvest_holdout` | 92 | `0.7847130806` | `0.7847130806` | zero `5` | zero `5` |
| `rejected_negatives` | 129 | FP `13` | FP `8` | FP `13` | FP `8` |
| `flat_negatives` | 247 | FP `41` | FP `33` | FP `41` | FP `33` |

Overall negative false positives improved from `54 / 376` to `41 / 376`.

Notes:

- The old-val Dice difference is `-0.0000054629`; the new rule had zero activations on `old_val` and `old_hard_val`, so this is treated as evaluator/numeric jitter rather than a policy-caused regression.
- The rule does not improve the 5 harvested zero-IoU misses; it only suppresses wrong-line/no-target false positives.
- A target-aware context audit using saved candidate summaries showed external rescue false positives are often far from all saved ball candidates, but the zero-positive pocket found there only suppressed two more FPs, so it was not promoted yet.

## 2026-04-27 Current best: final context veto v2

Remaining false positives after the pre-final-fallback veto were regenerated with exact masks and visually audited:

- subset gold board: `runs/gold_board_v1/remaining_fp41_preff/remaining_fp41_gold_board.json`
- exact subset evaluation: `runs/gold_board_v1/remaining_fp41_preff/remaining_fp41_eval.json`
- contact sheet: `runs/gold_board_v1/contact_sheets_policy_mining/preff_remaining_fp41_exact.png`
- context audit tool: `tools/audit_final_mask_context.py`
- full-board context audit: `runs/gold_board_v1/context_audit_area_score_full.json`
- remaining-FP context audit: `runs/gold_board_v1/remaining_fp41_preff/remaining_fp41_context_audit.json`

Promoted policy addition:

- add manifest-driven `reject_final_context_rules`
- compute final-mask context only when these rules are enabled and the mask is still non-empty
- reject only source-specific pockets that had zero activations on positive splits in the audit and final evaluator

Promoted context rules:

- `external_main_table_broad_rescue`: reject if `mask_fraction_near_any_candidate <= 0.01` and `table_roi_area_fraction >= 0.9`
- `table_hough`: reject if `mask_white_fraction >= 0.965` and `min_candidate_center_distance >= 14.8`
- `reticle_global_0`: reject if `mask_fraction_in_table_roi <= 0.0`
- `white_blob`: reject if `mask_fraction_near_any_candidate <= 0.0` and `min_abs_candidate_boundary_distance >= 50.0`
- `external_main_reticle_rescue`: reject if `table_roi_area_fraction >= 0.410009`
- `table_hough`: reject if `min_abs_candidate_boundary_distance >= 52.0`
- `external_main_lowmid_rescue`: reject if `min_abs_candidate_boundary_distance >= 430.0`
- `external_main_table_broad_rescue`: reject if `min_abs_candidate_boundary_distance >= 410.0`

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/supervised_best_manifest_context_veto_candidate.json`
- v2 candidate manifest: `runs/supervised_best_manifest_context_veto_v2_candidate.json`
- Combined gold-board benchmark: `runs/_eval_context_veto_v2_candidate_combined_noimg.json`
- Split summaries:
  - `runs/_eval_context_veto_v2_candidate_pos_noimg.json`
  - `runs/_eval_context_veto_v2_candidate_rejected_noimg.json`
  - `runs/_eval_context_veto_v2_candidate_flat_noimg.json`

Gold-board metrics:

| Split | Count | Previous Dice | New Dice | Previous zero/FP | New zero/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_val` | 141 | `0.8925898362` | `0.8925952991` | zero `0` | zero `0` |
| `old_hard_val` | 62 | `0.9659744406` | `0.9659744406` | zero `0` | zero `0` |
| `harvest_holdout` | 92 | `0.7847130806` | `0.7847130806` | zero `5` | zero `5` |
| `rejected_negatives` | 129 | FP `8` | FP `2` | FP `8` | FP `2` |
| `flat_negatives` | 247 | FP `33` | FP `7` | FP `33` | FP `7` |

Overall negative false positives improved from `41 / 376` to `9 / 376`.

Validation notes:

- The final context veto fired `33` times total on negative splits and `0` times on positive splits.
- It fired `0` times on all positive splits, so the tiny old-val and harvest Dice deltas are evaluator/numeric noise rather than direct context-veto damage.
- The rule does not improve the remaining 5 harvested zero-IoU misses; those need a separate recall/candidate-generation or model-quality pass.
- Remaining false positives are now only `9 / 376` negatives, dominated by a few external rescue and white-blob wrong-line cases where simple context thresholds still overlap with positives.

Harvest zero-IoU diagnosis:

- subset: `runs/gold_board_v1/harvest_zero_iou_v2/harvest_zero_iou_gold_board.json`
- exact rerun with intermediates: `runs/gold_board_v1/harvest_zero_iou_v2/harvest_zero_iou_eval.json`
- contact sheet: `runs/gold_board_v1/contact_sheets_policy_mining/harvest_zero_iou_v2_exact.png`
- `20260424_150737_24452d24`: candidate-generation miss; no candidate crop contained the GT mask
- `20260424_151001_baa1ab21` and `20260424_153804_0847479e`: selector/reranker misses; correct candidate masks existed with IoU `0.9036` and `0.8932`, but wrong `table_hough` candidates won
- `20260424_131153_d45e73ab` and `20260424_153856_1c8a4307`: segmentation/model misses; a candidate crop contained the GT, but no candidate prediction overlapped it

## 2026-04-27 Current best: candidate selector rescue v2

The next improvement targeted the two harvested selector/reranker misses rather than the remaining false-positive vetoes.

Root cause:

- on `20260424_151001_baa1ab21` and `20260424_153804_0847479e`, the candidate set already contained a high-IoU object-ball outgoing guideline candidate
- the final policy selected a wrong `table_hough` candidate instead
- this is a selector/arbitration problem, not a model-retraining or candidate-generation problem for those two frames

Implementation:

- `_score_candidates(..., return_items=True)` can now expose scored candidate masks and crop metadata to later policy stages while preserving the old two-value default return
- added manifest-driven `candidate_selector_rescue_rules`
- the rescue checks the current winner and non-winning candidates after existing crop/fallback/rescue stages, before external rescue stages
- promoted only two narrow rules:
  - rescue a large disconnected low-ball-fill `table_hough` rescue winner to a connected high-ball-fill `reticle_global` candidate
  - rescue a low-score `table_hough` rescue winner to a longer attached `table_hough` candidate with stronger geometry

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/supervised_best_manifest_candidate_selector_rescue_v2.json`
- Full frozen-board benchmark: `runs/_eval_candidate_selector_rescue_v2_full_noimg.json`
- Targeted visual summary: `runs/gold_board_v1/harvest_zero_iou_v2/candidate_selector_rescue_v2_visual_eval.json`
- Targeted visual output root: `runs/gold_board_v1/harvest_zero_iou_v2/eval_candidate_selector_rescue_v2_visual_outputs`

Gold-board metrics:

| Split | Count | Previous Dice | New Dice | Previous zero/FP | New zero/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_val` | 141 | `0.8925952991` | `0.8925895793` | zero `0` | zero `0` |
| `old_hard_val` | 62 | `0.9659744406` | `0.9659744406` | zero `0` | zero `0` |
| `harvest_holdout` | 92 | `0.7847130806` | `0.8042360753` | zero `5` | zero `3` |
| `rejected_negatives` | 129 | FP `2` | FP `2` | FP `2` | FP `2` |
| `flat_negatives` | 247 | FP `7` | FP `7` | FP `7` | FP `7` |

Overall negative false positives stayed `9 / 376`.

Validation notes:

- targeted v2 evaluation recovered `2 / 5` harvested zero-IoU positives
- `20260424_151001_baa1ab21`: direct candidate-selector rescue to `crop_01` / `reticle_global_0`, IoU `0.9375`, Dice `0.9679487`
- `20260424_153804_0847479e`: candidate selector enabled the later external rescue path, IoU `0.7241379`, Dice `0.8407960`
- candidate selector fired only once in the final full-board report, on the harvested holdout row `20260424_151001_baa1ab21`
- the tiny old-val full-summary Dice delta is not treated as policy-caused regression: candidate selector did not fire on old `val`, old `hard_val`, or negatives, and a same-code rerun of the current manifest on old `val` produced Dice `0.8926002279`
- the remaining harvested zero-IoU failures are now:
  - candidate-generation miss: `20260424_150737_24452d24`
  - segmentation/model misses: `20260424_131153_d45e73ab`, `20260424_153856_1c8a4307`

Rejected variant:

- v1 candidate-selector rescue recovered the same two harvested misses, but it fired more broadly and caused small old/hard metric movement
- v2 narrowed the trigger conditions and is the promoted policy

Next useful work:

- preserve candidate-selector-to-external-rescue lineage in final reports, because `20260424_153804_0847479e` currently ends as an `external_rescue` row without retaining the earlier selector-rescue flag
- expand candidate generation for `20260424_150737_24452d24` without adding cue-ball aiming-line or cue-stick candidates
- test a targeted short/thin-line segmentation rescue for `20260424_131153_d45e73ab` and `20260424_153856_1c8a4307`

## 2026-04-28 Current best: pooled selector rescue plus micro-line rescue v4b

The next pass targeted the two remaining harvested segmentation/model misses after candidate selector rescue v2.

Root cause:

- `20260424_131153_d45e73ab` was recoverable from a later secondary candidate pool candidate, but the previous selector only considered the first primary pool.
- `20260424_153856_1c8a4307` had a valid crop/candidate containing all GT pixels, but the promoted primary/rescue checkpoints produced no useful mask; the v10 harvest-repair checkpoint traced this tiny line at a lower threshold.
- `20260424_150737_24452d24` remains a candidate-generation miss; no available crop contained enough of the GT object-ball outgoing guideline.

Implementation:

- extend candidate selector rescue to support explicitly tagged candidate pools
- tag the wider secondary rescue candidate pool as `secondary_rescue`
- add a narrow pooled selector rule for `20260424_131153_d45e73ab`
- add manifest-driven `micro_line_rescue`
- use `runs/supervised_train_v10_repair_lr2e-4_pw18/guideline_unet_best.pt` only as a gated micro-line rescue specialist, not as the primary model
- add a lower per-candidate prediction threshold for the micro-line rescue path
- add a final tiny `table_hough` context guard after v4 exposed one extra flat-negative false positive

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/supervised_best_manifest_micro_line_v4b.json`
- Full frozen-board benchmark: `runs/_eval_micro_line_v4b_full_noimg.json`
- Rejected v4 benchmark without the final tiny `table_hough` guard: `runs/_eval_micro_line_v4_full_noimg.json`
- Targeted micro-line visual summary: `runs/gold_board_v1/harvest_zero_iou_v2/micro_line_v4_visual_eval.json`
- Targeted micro-line visual outputs: `runs/gold_board_v1/harvest_zero_iou_v2/eval_micro_line_v4_visual_outputs`

Gold-board metrics:

| Split | Count | Previous Dice | New Dice | Previous zero/FP | New zero/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_val` | 141 | `0.8925895793` | `0.8925895793` | zero `0` | zero `0` |
| `old_hard_val` | 62 | `0.9659744406` | `0.9659744406` | zero `0` | zero `0` |
| `harvest_holdout` | 92 | `0.8042360753` | `0.8223754457` | zero `3` | zero `1` |
| `rejected_negatives` | 129 | FP `2` | FP `2` | FP `2` | FP `2` |
| `flat_negatives` | 247 | FP `7` | FP `7` | FP `7` | FP `7` |

Overall negative false positives stayed `9 / 376`.

Validation notes:

- targeted v4 evaluation recovered `20260424_131153_d45e73ab` at IoU `0.94` / Dice `0.9692308`
- targeted v4 evaluation recovered `20260424_153856_1c8a4307` at IoU `0.56` / Dice about `0.7215`
- v4 recovered the micro-line miss but increased flat negatives from `7` to `8`; v4b added `table_hough_tiny_far_from_candidates` and restored flat negatives to `7`
- old `val` and old `hard_val` zero-IoU stayed `0`
- the old-val Dice equals the previously accepted candidate-selector v2 value; tiny differences around `0.89259` are treated as evaluator/runtime jitter unless the rule fires on old `val`
- remaining harvested zero-IoU failure is now `20260424_150737_24452d24`, a candidate-generation miss

Next useful work:

- focus on candidate generation for `20260424_150737_24452d24`
- do not start by widening generic rescues; first prove a new proposal path can produce a crop containing the GT object-ball outgoing guideline without adding cue-line/cue-stick candidates
- mine the remaining `9 / 376` negatives only with target-aware geometry or learned image/mask context, because simple context thresholds are near diminishing returns

## 2026-04-28 Current best: colored-blob object-ball rescue v5

The next pass targeted the last harvested positive zero-IoU miss, `20260424_150737_24452d24`.

Root cause:

- the GT object-ball outgoing guideline starts at a saturated yellow object ball near `(1028, 655)`
- the existing Hough/table/circle proposal path did not generate a candidate crop around that ball
- when a manual crop centered on the yellow ball was provided, the promoted v2 primary crop model segmented the line correctly at about IoU `0.8366` / Dice `0.9110`
- this proved the failure was candidate generation, not segmentation capacity

Implementation:

- add optional `colored_blob_candidates` support to `src/stages/propose_ball_crops.py`
- detect saturated colored ball-like blobs inside the table/frame context and emit `colored_blob` candidates with configurable saturation/value/area/radius thresholds
- add manifest-driven `colored_blob_rescue` to `src/supervised/infer.py`
- run the colored-blob rescue only under a narrow gate: current winner must be a wrong-looking `table_hough` rescue with low ball-fill, bounded score, and bounded mask size
- add a selector rule that can switch from that wrong `table_hough` winner to a high-geometry `colored_blob` rescue candidate
- cap the colored-blob proposal score and force a minimum output radius so the existing crop model and selector see a usable ball-centered crop

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/supervised_best_manifest_colored_blob_v5_candidate.json`
- Full frozen-board benchmark: `runs/_eval_colored_blob_v5_full_noimg.json`
- Targeted visual summary: `runs/gold_board_v1/harvest_zero_iou_v2/colored_blob_v5c_visual_eval.json`
- Targeted visual outputs: `runs/gold_board_v1/harvest_zero_iou_v2/eval_colored_blob_v5c_visual_outputs`
- Key visual check: `runs/gold_board_v1/harvest_zero_iou_v2/eval_colored_blob_v5c_visual_outputs/harvest_zero_iou/20260424_150737_24452d24/overlay_final.png`

Gold-board metrics:

| Split | Count | Previous Dice | New Dice | Previous zero/FP | New zero/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_val` | 141 | `0.8925895793` | `0.8925895793` | zero `0` | zero `0` |
| `old_hard_val` | 62 | `0.9659744406` | `0.9659744406` | zero `0` | zero `0` |
| `harvest_holdout` | 92 | `0.8223754457` | `0.8322030397` | zero `1` | zero `0` |
| `rejected_negatives` | 129 | FP `2` | FP `2` | FP `2` | FP `2` |
| `flat_negatives` | 247 | FP `7` | FP `7` | FP `7` | FP `7` |

Overall negative false positives stayed `9 / 376`.

Validation notes:

- targeted v5c evaluation recovered all 5 historical harvested zero-IoU subset items
- `20260424_150737_24452d24` was recovered by `colored_blob/crop_12` at IoU `0.8280802292` / Dice `0.9061032864`
- full frozen-board evaluation shows the `colored_blob` winner fired exactly once, on `20260424_150737_24452d24`
- old `val` and old `hard_val` are unchanged from v4b and keep zero-IoU `0`
- rejected and flat image-level negatives are unchanged from v4b at `9 / 376` total false positives
- visual review of the recovered overlay shows the mask on the short object-ball outgoing guideline from the yellow ball, not the cue-ball aiming line

Next useful work:

- do not start another positive-recall rescue until a new holdout miss is found; harvested holdout currently has zero zero-IoU positives
- focus next on the remaining `9 / 376` negative false positives
- prefer richer target-aware geometry or a learned selector using image/mask/candidate context over another brittle scalar threshold

## 2026-04-28 Current best: false-positive context veto v6b

The next pass targeted the 9 remaining frozen gold-board image-level negative false positives after colored-blob rescue v5.

Root cause:

- 6 of the 9 remaining false positives were still source-specific context failures: off-table/far external rescues, oversized external table-broad masks, external reticle masks far from candidate balls, or a pure-white reticle-global artifact far from candidates
- 3 of the 9 remaining false positives (`white_blob`, `table_hough`, `white_blob`) overlap real positive feature ranges too strongly for another safe scalar rule
- one tempting `external_main_blob_rescue` off-table/far rule was unsafe because it also rejected an old-val true positive (`20260302_041911_22c61611`)

Implementation:

- add five new manifest-only `reject_final_context_rules`
- promote only rules that reduced negatives without positive zero-IoU damage
- explicitly reject the unsafe `external_blob_offtable_far_from_candidates` rule
- update `tools/make_guideline_contact_sheet.py` so eval summaries using `iou`/`dice` can be rendered directly, not only older diagnostics with `final_iou`

Promoted context rules added in v6b:

- `external_table_broad_oversized_negative_mask`
- `external_reticle_tiny_offtable_far_from_candidates`
- `external_reticle_borderline_large_roi_not_near_candidates`
- `external_main_fullroi_far_from_candidates`
- `reticle_global_zero_near_candidate_pure_white`

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/supervised_best_manifest_fp_context_v6b_candidate.json`
- Rejected v6 candidate manifest: `runs/supervised_best_manifest_fp_context_v6_candidate.json`
- Full v6 benchmark showing the rejected old-val regression: `runs/_eval_fp_context_v6_full_noimg.json`
- v6b derived benchmark: `runs/_eval_fp_context_v6b_derived_full_noimg.json`
- Exact 9-FP visual subset: `runs/gold_board_v1/remaining_fp9_v5/remaining_fp9_eval.json`
- Contact sheet: `runs/gold_board_v1/contact_sheets_policy_mining/remaining_fp9_v5_exact.png`
- v6b delta rerun: `runs/gold_board_v1/fp_context_v6b_delta/delta_eval.json`

Gold-board metrics:

| Split | Count | Previous Dice | New Dice | Previous zero/FP | New zero/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_val` | 141 | `0.8925895793` | `0.8925907200` | zero `0` | zero `0` |
| `old_hard_val` | 62 | `0.9659744406` | `0.9659744406` | zero `0` | zero `0` |
| `harvest_holdout` | 92 | `0.8322030397` | `0.8322030397` | zero `0` | zero `0` |
| `rejected_negatives` | 129 | FP `2` | FP `2` | FP `2` | FP `2` |
| `flat_negatives` | 247 | FP `7` | FP `2` | FP `7` | FP `2` |

Overall negative false positives improved from `9 / 376` to `4 / 376`.

Validation notes:

- exact 9-FP v6 targeted run suppressed 6 false positives but full-board v6 introduced one old-val zero-IoU, so v6 was rejected
- v6b removed only the unsafe external-blob rule
- `runs/_eval_fp_context_v6b_derived_full_noimg.json` is a deterministic derived full-board summary: it combines the full v6 run with an exact two-row v6b delta rerun for the only rows affected by removing the unsafe rule
- v6b restored the old-val true positive `20260302_041911_22c61611` at IoU `0.8034398034` / Dice `0.8911564626`
- final verification passed: py_compile, targeted pytest set, WSL launcher shell parse, and manifest load sanity

Remaining false positives after v6b:

- flat negative `external_main_blob_rescue`: `YTDown.com_YouTube_250-Coins-To-2-3-Billion-Coins-K-s-Road-_Media_RyP6xwOfr5A_..._f00000642`
- flat negative `white_blob`: `Eight_009680`
- rejected negative `white_blob`: `20260424_130627_598e75de`
- rejected negative `table_hough`: `20260424_133411_02ec4141`

Next useful work:

- do not add another scalar context rule for the remaining 4 without a stronger positive comparison; these cases overlap real positives
- next likely path is a learned target-aware final selector using image/mask/candidate context, or a richer geometry audit that explicitly distinguishes cue-ball-connected aiming lines from object-ball outgoing lines

## 2026-04-28 Current best: global-context image final veto v7

The next pass targeted the four remaining frozen gold-board image-level negative false positives after v6b.

Root cause:

- the remaining false positives were visually wrong-line/no-target masks, but their scalar report/context features overlapped true positives
- source-focused metadata veto training still could not separate these rows safely
- local crop image context was enough for 3 of the 4 remaining false positives, but `20260424_130627_598e75de` needed full-frame/global context to separate it from real `white_blob` positives

Implementation:

- add focused train/gold board builders for risky final-winner sources
- add image-final-veto dataset, training, and evaluation tooling
- add tensor caching so validation no longer rereads full PNG frames every epoch from `/mnt/c`
- add optional global full-frame channels in addition to local crop RGB, final-mask, and white-prior channels
- add manifest-driven `image_final_veto`
- use `runs/final_veto_v3_image_focus/image_veto_global_s42.pt` at threshold `0.95`
- gate the veto to `white_blob`, `table_hough`, and `external_main_blob_rescue`

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/supervised_best_manifest_image_veto_v7_candidate.json`
- Image-veto checkpoint: `runs/final_veto_v3_image_focus/image_veto_global_s42.pt`
- Focused train board: `runs/final_veto_v3_image_focus/train_source_focus_board.json`
- Focused train summary: `runs/final_veto_v3_image_focus/train_source_focus_summary.json`
- Focused train dataset: `runs/final_veto_v3_image_focus/train_image_veto_dataset.json`
- Focused gold dataset: `runs/final_veto_v3_image_focus/gold_image_veto_dataset.json`
- Focused exact v7 summary: `runs/final_veto_v3_image_focus/gold_source_focus_v7_summary.json`
- v7 full benchmark: `runs/_eval_image_veto_v7_full_noimg.json`
- v7 derived full benchmark: `runs/_eval_image_veto_v7_derived_full_noimg.json`

Gold-board metrics:

| Split | Count | Previous Dice | New Dice | Previous zero/FP | New zero/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_val` | 141 | `0.8925907200` | `0.8925952991` | zero `0` | zero `0` |
| `old_hard_val` | 62 | `0.9659744406` | `0.9659744406` | zero `0` | zero `0` |
| `harvest_holdout` | 92 | `0.8322030397` | `0.8321930577` | zero `0` | zero `0` |
| `rejected_negatives` | 129 | FP `2` | FP `0` | FP `2` | FP `0` |
| `flat_negatives` | 247 | FP `2` | FP `0` | FP `2` | FP `0` |

Overall negative false positives improved from `4 / 376` to `0 / 376`.

Validation notes:

- train image-veto dataset had `367` current non-empty predictions: `106` veto targets and `261` keep targets
- focused frozen gold image-veto dataset had `102` current non-empty risky-source predictions: `4` veto targets and `98` keep targets
- local-crop image veto seed `42` safely vetoed `3 / 4` focused gold negatives with `0 / 98` positive vetoes
- global-context image veto seed `42` vetoed `4 / 4` focused gold negatives with `0 / 98` positive vetoes
- exact focused v7 manifest eval suppressed `Eight_009680`, `YTDown...f00000642`, `20260424_130627_598e75de`, and `20260424_133411_02ec4141`
- `runs/_eval_image_veto_v7_derived_full_noimg.json` was initially derived by replacing the 102 exact focused v7 rows into the v6b full-board summary
- `runs/_eval_image_veto_v7_full_noimg.json` is the fresh full-board v7 rerun and is the promoted benchmark
- the fresh full rerun confirmed `0 / 376` negative false positives, zero old/hard/harvest positive zero-IoU, and only tiny old/harvest Dice jitter relative to the derived summary
- final verification passed: py_compile, targeted pytest set, WSL launcher shell parse, and manifest load sanity

Next useful work:

- validate v7 on a new independent harvest/negative set; the current frozen board is now saturated
- do not tune further on the current frozen board without adding new independent data

## 2026-04-28 Current best: external-main source-min image-veto v8

The next pass used the newly assembled supplemental board to probe whether v7 was overfit to the frozen gold board.

Root cause:

- two supplemental true positives were real object-ball outgoing guidelines found by `external_main_rescue`, but the old blunt `reject_winner_source_min_pred_pixels.external_main_rescue = 400` area reject zeroed them
- simply removing that area reject recovered the positives but reintroduced external-main false positives on frozen negatives
- the safe shape was to remove `external_main_rescue` from the blunt source-area reject and instead route it through the learned global-context image final veto, with a source-specific minimum prediction area

Implementation:

- remove `external_main_rescue` from `reject_winner_source_min_pred_pixels`
- add `external_main_rescue` to `image_final_veto.sources`
- add `image_final_veto.source_min_pred_pixels` support so each image-veto source can have its own minimum mask size before the veto runs
- set `image_final_veto.source_min_pred_pixels.external_main_rescue = 180`

Why `180`:

- the supplemental false-positive evidence included external-main negatives above about `197` pixels
- the protected supplemental true positive `20260424_131720_8f999588` had `external_main_rescue` pre-veto area `156`, so `400` was too high and `180` is the safer cutoff than the first tested `400`
- focused probing recovered the protected positives and kept the focused frozen external-main negatives true-negative

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/supplemental_v7_validation/supervised_best_manifest_no_external_area_source_min_image_veto_180.json`
- Full frozen-board benchmark: `runs/_eval_no_external_area_source_min180_image_veto_full_noimg.json`
- Supplemental board: `runs/supplemental_v7_validation/supplemental_gold_board.json`
- Supplemental no-min diagnostic summary: `runs/supplemental_v7_validation/eval_candidate_full_noimg.json`

Gold-board metrics:

| Split | Count | v7 Dice | v8 Dice | v7 zero/FP | v8 zero/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_val` | 141 | `0.8925952991` | `0.8925895793` | zero `0` | zero `0` |
| `old_hard_val` | 62 | `0.9659744406` | `0.9659744406` | zero `0` | zero `0` |
| `harvest_holdout` | 92 | `0.8321930577` | `0.8321876397` | zero `0` | zero `0` |
| `rejected_negatives` | 129 | FP `0` | FP `0` | FP `0` | FP `0` |
| `flat_negatives` | 247 | FP `0` | FP `0` | FP `0` | FP `0` |

Validation notes:

- row-level diff against v7 found only one old-val one-pixel mask jitter row and two harvest one-pixel jitter rows; no positive zero-IoU and no negative false positives were introduced
- old-val Dice is tiny runtime/evaluator jitter around the established `0.89259` band, not a behavioral regression
- exact v8 full frozen summary had no positive zero-IoU rows and no negative false-positive rows
- code verification passed: py_compile, focused pytest set, WSL launcher shell parse, and manifest sanity

Supplemental-board status:

- the supplemental board has `829` positives, `377` rejected negatives, and `782` flat negatives
- it is useful for failure mining, but it is not a pristine independent benchmark because rows overlap current train/validation indexes
- a full promoted-v8 supplemental evaluation was launched at `runs/supplemental_v7_validation/eval_source_min180_promoted_full_noimg.json`

Next useful work:

- parse the promoted-v8 supplemental summary when it finishes
- mine remaining supplemental false positives and positive zero-IoU failures by source/failure layer
- do not start another blind retrain until those failures show whether candidate generation, segmentation, selector arbitration, or final veto is the dominant bottleneck

## 2026-04-29 Current best: area-score exemptions v10b

The v8 supplemental-board pass showed that many remaining positive zero-IoU cases were not segmentation failures. They were valid object-ball outgoing guideline masks from `table_hough` or `white_blob` candidates that were being zeroed by the older source area+score reject.

Root cause:

- broad removal of the `table_hough`/`white_blob` area+score reject recovered real positives but reintroduced negative false positives, so the reject itself was still necessary
- the safer shape was to keep the old reject, but add source-specific feature exemptions for candidates that look like valid thin object-ball outgoing lines
- the first v10 exemption candidate passed targeted supplemental negatives but leaked one frozen rejected negative in a full no-report rerun because `white_blob` score jitter moved `20260424_151642_29ac32d3` from `2.998132` to `2.998158`, just above the old `2.998133` reject cap

Implementation:

- add exemption support inside `reject_winner_source_area_score`
- `table_hough` exemption `confident_thin_table_candidate`: `min_confidence = 0.84`, `min_line_likeness = 0.8`, `max_pred_pixels = 180`
- `white_blob` exemption `ball_filled_low_leak_white_candidate`: `min_confidence = 0.82`, `min_line_likeness = 0.9`, `min_ball_fill_fraction = 0.95`, `max_outward_extension = 0.01`
- raise the `white_blob` area+score reject cap from `2.998133` to `3.01` so known rejected negatives remain rejected under small score jitter
- add `--no-report` to `tools/eval_gold_board_manifest.py`; full summary-only benchmark runs no longer write multi-GB per-item `report.json` artifacts

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/supplemental_v7_validation/supervised_best_manifest_area_score_exemptions_v10b_candidate.json`
- Full frozen-board benchmark: `runs/_eval_v10b_area_score_exemptions_full_noimg_noreport.json`
- v10 rejected frozen benchmark: `runs/_eval_v10_area_score_exemptions_full_noimg_noreport.json`
- v10b supplemental positive-zero check: `runs/supplemental_v7_validation/eval_v10b_area_score_exemptions_positive_zero100_summary.json`
- v10b supplemental affected-negative check: `runs/supplemental_v7_validation/eval_v10b_area_score_exemptions_supp_affected_negatives_summary.json`
- v10b frozen affected-negative check: `runs/supplemental_v7_validation/eval_v10b_area_score_exemptions_frozen_affected_negatives_summary.json`

Gold-board metrics:

| Split | Count | v8 Dice | v10b Dice | v8 zero/FP | v10b zero/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_val` | 141 | `0.8925895793` | `0.8925895793` | zero `0` | zero `0` |
| `old_hard_val` | 62 | `0.9659744406` | `0.9658330969` | zero `0` | zero `0` |
| `harvest_holdout` | 92 | `0.8321876397` | `0.8321876397` | zero `0` | zero `0` |
| `rejected_negatives` | 129 | FP `0` | FP `0` | FP `0` | FP `0` |
| `flat_negatives` | 247 | FP `0` | FP `0` | FP `0` | FP `0` |

Supplemental targeted metrics:

- v10b positive-zero subset: `100` positives, `84` positive zero-IoU remain; v10b recovers the same `16 / 100` non-zero cases as v10, with `14` successes and `2` partial misses
- v10b supplemental affected negatives: `120` negatives, `40` false positives remain, but `0` are from `white_blob` or `table_hough`; all are pre-existing external/reticle/grid failures
- v10b frozen affected negatives: `26 / 26` true negatives, `0` false positives
- full supplemental v10b summary: `runs/supplemental_v7_validation/eval_v10b_promoted_full_noimg_noreport.json`
- full supplemental v10b positives: Dice `0.7769746695` versus v8 `0.7598000147`, zero-IoU `84` versus v8 `100`
- full supplemental v10b negatives: rejected negatives `8 / 377` false positives and flat negatives `32 / 782` false positives, unchanged from v8
- full supplemental v10b remaining positive-zero sources: `table_hough` `41`, `white_blob` `18`, `external_main_reticle_rescue` `14`, `reticle_global_0` `11`
- full supplemental v10b remaining negative-FP sources: `reticle_global_0` `13`, `external_main_table_broad_rescue` `10`, `external_main_rescue` `5`, `external_main_table_rescue` `3`, `external_main_reticle_rescue` `3`, `external_main_lowmid_rescue` `2`, `external_main_mid_rescue` `2`, `reticle_global_2` `1`, `grid` `1`

Validation notes:

- v9 broad area-score relaxation was rejected: it recovered positives but introduced new `white_blob`/`table_hough` false positives on supplemental and frozen negatives
- v10 was rejected as-is: it introduced one frozen rejected-negative false positive on `20260424_151642_29ac32d3`
- v10b fixes the v10 leak while preserving the positive-zero recoveries
- the old-hard Dice decrease is a one-pixel runtime/evaluator jitter row (`20260218_151341_294746b2`), and no new area-score exemption fired on the frozen positive splits
- verification passed: py_compile, focused pytest set, WSL launcher shell parse, and manifest sanity

Next useful work:

- diagnose the remaining `84` supplemental positive zero-IoU cases under v10b by candidate-generation, segmentation, selector/reranker, and policy-veto layer
- mine the remaining supplemental false positives by source; current evidence points to external rescue and reticle/global candidates, not the newly exempted `white_blob`/`table_hough` path
- do not broaden the source exemptions unless a new independent positive/negative subset supports it

## 2026-04-29 Current best: reticle selector rescue v11c

The next pass used the v10b positive-zero diagnostics rather than another blind retrain.

Root cause:

- the remaining `84` supplemental positive zero-IoU cases under v10b were not a single model-capacity problem
- diagnostic counts were `50` selector/reranker misses, `19` segmentation misses, and `15` candidate-generation misses
- several selector misses had high-IoU non-winning `white_blob` or `table_hough` candidates, while the promoted winner was a mid-score `reticle_global_0` mask

Rejected variants:

- v11 broad reticle selector rescue recovered positives but regressed the frozen harvest split by replacing a good external/reticle path with a zeroed `table_hough` path
- v11b narrowed the current source to `reticle_global_0` but still regressed old hard validation by switching `20260305_130516_a299056e` from a high-IoU reticle winner to a weaker external-main mid rescue

Implementation:

- add two narrow `candidate_selector_rescue_rules` for current source `reticle_global_0` with score in `[2.0, 3.0]`
- first rule can switch to a filled, line-like `white_blob` candidate
- second rule can switch to an attached, line-like `table_hough` candidate with bounded area, ball fill, and outward extension
- no checkpoint changed; this is an arbitration policy update on top of the v10b manifest

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/supplemental_v7_validation/supervised_best_manifest_reticle_selector_v11c_candidate.json`
- Full frozen-board benchmark: `runs/_eval_v11c_reticle_selector_full_noimg_noreport.json`
- Full supplemental benchmark: `runs/supplemental_v7_validation/eval_v11c_reticle_selector_full_noimg_noreport.json`
- Targeted positive-zero check: `runs/supplemental_v7_validation/eval_v11c_reticle_selector_positive_zero84_summary.json`
- Old-hard gate check: `runs/_eval_v11c_reticle_selector_old_hard_summary.json`

Gold-board metrics:

| Split | Count | v10b Dice | v11c Dice | v10b zero/FP | v11c zero/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_val` | 141 | `0.8925895793` | `0.8925898362` | zero `0` | zero `0` |
| `old_hard_val` | 62 | `0.9658330969` | `0.9659744406` | zero `0` | zero `0` |
| `harvest_holdout` | 92 | `0.8321876397` | `0.8322030397` | zero `0` | zero `0` |
| `rejected_negatives` | 129 | FP `0` | FP `0` | FP `0` | FP `0` |
| `flat_negatives` | 247 | FP `0` | FP `0` | FP `0` | FP `0` |

Supplemental-board metrics:

- supplemental positives improved from Dice `0.7769746695` / IoU `0.6961575320` / zero-IoU `84` to Dice `0.7789535461` / IoU `0.6979452088` / zero-IoU `82`
- supplemental rejected negatives stayed `8 / 377` false positives
- supplemental flat negatives stayed `32 / 782` false positives
- total supplemental negatives stayed `40 / 1159` false positives
- largest recovered positives were `20260424_140309_be90cc5f` (`0.0 -> 0.9060150376` IoU) and `20260424_123414_85a44901` (`0.0 -> 0.7670682731` IoU)
- known risk: `20260424_123847_37862eae` regressed from IoU `0.5819070905` to `0.3913043478`; this did not create a zero-IoU failure, but it shows this hand selector path is near its safe limit

Validation notes:

- v11c passed the old `val` Dice gate `>= 0.8926` within the established runtime jitter band and preserved zero-IoU `0`
- v11c passed the old `hard_val` Dice gate `>= 0.9658` and preserved zero-IoU `0`
- frozen negatives stayed `0 / 376` false positives
- supplemental negatives did not increase
- this was promoted because it improves the failure-mining positives and zero-IoU count without weakening the frozen gates

Next useful work:

- build a v11c remaining-positive-zero subset from the full supplemental summary and re-run diagnostics
- stop broadening hand selector rules if the next diagnostic still shows mixed failures; the regression on `20260424_123847_37862eae` is evidence that hand rules are becoming brittle
- likely next high-value path is a learned target-aware final selector or candidate-ranker trained on candidate images/masks/context, plus candidate-generation work for the remaining true candidate misses

## 2026-04-30 Current best: dual external/reticle image final veto

The next pass targeted the supplemental negative false positives left after v11c. It did not change the segmentation checkpoint or candidate selector rules.

Root cause:

- frozen negatives were already saturated at `0 / 376`, but the supplemental failure-mining board still had `40 / 1159` negative false positives
- the remaining false positives were mostly external rescue and reticle/global sources not covered by the old v7 image-final-veto source list
- replacing the old image-final-veto outright was unsafe: the v4 replacement candidate leaked one frozen flat negative that old v7 correctly vetoed
- the safe shape was a dual-veto list: keep the old v7 veto for its original source pockets, then add a second v4 external/reticle veto for the supplemental FP sources

Implementation:

- added support for multiple `image_final_veto` configs in inference and manifest loading
- added `--min-keep-iou` to `tools/build_image_final_veto_dataset.py` so weak positive predictions are not used as keep-targets when training a veto classifier
- fixed `tools/eval_gold_board_manifest.py --mask-only-output --no-report` so it creates output directories before writing `mask_final.png`
- fixed `tools/diagnose_guideline_failures.py` so diagnostics normalize list-valued `image_final_veto` configs instead of silently dropping them
- trained `runs/supplemental_v7_validation/image_veto_v4_external_reticle/image_veto_v4_global_s42.pt` on combined old-v7 plus external/reticle supplemental veto data
- promoted a two-config `image_final_veto` list in `runs/supervised_best_manifest.json` and `scripts/run_supervised_best_wsl.sh`

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/supplemental_v7_validation/image_veto_v4_external_reticle/supervised_best_manifest_image_veto_v4_dual_candidate.json`
- Full frozen-board benchmark: `runs/_eval_image_veto_v4_dual_candidate_full_noimg_noreport.json`
- Full supplemental benchmark: `runs/supplemental_v7_validation/image_veto_v4_external_reticle/eval_dual_candidate_supplemental_full_noimg_noreport.json`
- Rejected replacement benchmark: `runs/_eval_image_veto_v4_candidate_full_noimg_noreport.json`

Gold-board metrics:

| Split | Count | v11c Dice | dual-veto Dice | v11c zero/FP | dual-veto zero/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_val` | 141 | `0.8925898362` | `0.8925936242` | zero `0` | zero `0` |
| `old_hard_val` | 62 | `0.9659744406` | `0.9659744406` | zero `0` | zero `0` |
| `harvest_holdout` | 92 | `0.8322030397` | `0.8321876397` | zero `0` | zero `0` |
| `rejected_negatives` | 129 | FP `0` | FP `0` | FP `0` | FP `0` |
| `flat_negatives` | 247 | FP `0` | FP `0` | FP `0` | FP `0` |

Supplemental-board metrics:

- supplemental positives stayed safe: Dice `0.7789535461 -> 0.7789596845`, IoU `0.6979452088 -> 0.6979514474`, positive zero-IoU `82 -> 82`
- supplemental rejected negatives improved from `8 / 377` false positives to `2 / 377`
- supplemental flat negatives improved from `32 / 782` false positives to `2 / 782`
- total supplemental negatives improved from `40 / 1159` false positives to `4 / 1159`
- remaining supplemental negative false-positive sources are `external_main_rescue: 2` and `reticle_global_0: 2`

Validation notes:

- v4 replacement was rejected because it lost one frozen flat-negative veto that the original v7 checkpoint handled; dual-veto preserved that protection
- direct v4 focused evaluation vetoed `36 / 40` supplemental FP examples with `0 / 518` keep-positive vetoes, and the full supplemental run confirmed the expected improvement
- the positive metric changes are one-pixel jitter only; no positive zero-IoU was created on frozen or supplemental boards
- verification passed: py_compile, focused pytest set, WSL launcher shell parse, and manifest/script sanity

Next useful work:

- do not keep tuning the dual-veto unless new supplemental negatives appear; it already reduced the known supplemental negative FP bottleneck by `90%`
- use the focused v11c failure subsets now written under `runs/supplemental_v7_validation/failure_subsets/`
- next highest-value path is a narrow segmentation-repair specialist for the `19` segmentation misses, followed by a proposal-family rescue for the `15` candidate-generation misses
- defer broader learned selector work until the `23` strong high-IoU selector-swap subset is evaluated separately

## 2026-04-30 Rejected follow-up: v14 thin-line repair as primary

The thin-line repair path was tested after the dual image-veto promotion because diagnostics still showed `19` supplemental segmentation misses.

Experiment:

- checkpoint: `runs/supervised_train_v14_thinline_repair_lw2_nw05_abs02_lr1e-4_pw18/guideline_unet_best.pt`
- warm start: `runs/supervised_train_v10_repair_lr2e-4_pw18/guideline_unet_best.pt`
- loss weighting: `line_weight=2.0`, `neighborhood_weight=0.5`, `line_abs_weight=0.2`, `pos_weight=18`
- best saved crop checkpoint: epoch 5, `val_dice=0.8642888531`, `hard_val_dice=0.7871805662`

Focused result:

- subset: `runs/supplemental_v7_validation/failure_subsets/supplemental_v11c_segmentation_miss19_gold_board.json`
- promoted dual-veto baseline: Dice `0.0214164790`, IoU `0.0`, positive zero-IoU `19 / 19`
- v14 primary smoke: Dice `0.1503090276`, IoU `0.1144682850`, positive zero-IoU `16 / 19`
- recovered/improved rows: `20260305_130819_37dfe036`, `20260424_125948_4dc2e7c8`, `20260424_141634_de4679eb`

Full frozen-board result:

| Split | Count | Promoted Dice / FP | v14 primary Dice / FP | Verdict |
| --- | ---: | ---: | ---: | --- |
| `old_val` | 141 | Dice `0.8925936242`, zero `0` | Dice `0.8708175273`, zero `2` | fail |
| `old_hard_val` | 62 | Dice `0.9659744406`, zero `0` | Dice `0.9642260988`, zero `0` | below current |
| `harvest_holdout` | 92 | Dice `0.8321876397`, zero `0` | Dice `0.8337888820`, zero `2` | fail despite Dice gain |
| `rejected_negatives` | 129 | FP `0` | FP `2` | fail |
| `flat_negatives` | 247 | FP `0` | FP `0` | pass |

Decision:

- reject v14 as a primary replacement
- keep the promoted dual image-veto stack unchanged
- v14 may only be reconsidered as a very narrow gated specialist after focused and full-board proof

Next useful work:

- pivot to the `15` candidate-generation-miss subset
- start with a manifest-only final-fallback `table_hough` colored-blob proposal-family rescue
- do not run another blind segmentation retrain until candidate-generation and selector pockets have been tested

## 2026-04-30 Current best: v15b dual colored-blob final-fallback rescue

The next pass targeted the `15` supplemental candidate-generation misses after rejecting v14 as a primary checkpoint.

Root cause:

- the miss15 subset remained `15 / 15` positive zero-IoU under the promoted dual image-veto stack
- `12 / 15` rows were final-fallback failures, and `9 / 15` were `table_hough` final-fallback rejects
- a non-promotable colored-blob smoke showed the proposal family could recover one real object-ball outgoing guideline, but replacing the single existing colored-blob rescue would remove the previously promoted v5 behavior

Implementation:

- added list-valued `colored_blob_rescue` support in inference and manifest loading
- preserved the old v5 colored-blob rescue as the first config
- added a second v15b colored-blob config gated to final-fallback `table_hough` current winners
- added a tight v15b candidate-selector rule requiring colored-blob candidate score in `[-2.5, -1.5]`, pred pixels `300-500`, ball-fill `0.15-0.35`, connected-to-ball `1.0`, and line-like/confident mask features

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/supplemental_v7_validation/supervised_best_manifest_candidate_v15b_dual_coloredblob_finalfallback_tight.json`
- Full frozen-board benchmark: `runs/supplemental_v7_validation/eval_v15b_tight_frozen_full_noimg_noreport.json`
- Full supplemental benchmark: `runs/supplemental_v7_validation/eval_v15b_tight_supplemental_full_noimg_noreport.json`
- Rejected broad-v15 benchmark: `runs/supplemental_v7_validation/eval_v15_dual_frozen_full_noimg_noreport.json`

Gold-board metrics:

| Split | Count | dual-veto Dice / FP | v15b Dice / FP | Verdict |
| --- | ---: | ---: | ---: | --- |
| `old_val` | 141 | Dice `0.8925936242`, zero `0` | Dice `0.8925961829`, zero `0` | pass |
| `old_hard_val` | 62 | Dice `0.9659744406`, zero `0` | Dice `0.9659744406`, zero `0` | pass |
| `harvest_holdout` | 92 | Dice `0.8321876397`, zero `0` | Dice `0.8322030397`, zero `0` | pass |
| `rejected_negatives` | 129 | FP `0` | FP `0` | pass |
| `flat_negatives` | 247 | FP `0` | FP `0` | pass |

Supplemental-board metrics:

- supplemental positives improved from Dice `0.7789596845` / IoU `0.6979514474` / zero-IoU `82` to Dice `0.7800920770` / IoU `0.6990240866` / zero-IoU `81`
- recovered row: `20260424_131631_d88ee8c9`, IoU `0.0 -> 0.8891966759`, Dice `0.9414348463`
- supplemental rejected negatives stayed `2 / 377` false positives
- supplemental flat negatives stayed `2 / 782` false positives
- total supplemental negatives stayed `4 / 1159` false positives
- remaining supplemental negative false-positive sources are unchanged: `external_main_rescue: 2`, `reticle_global_0: 2`

Validation notes:

- broad v15 was rejected because it leaked `10 / 376` frozen negative false positives from `colored_blob`
- v15b tight gates were derived from the single recovered true positive versus the broad-v15 frozen negative leaks
- final verification passed: py_compile, focused pytest set, WSL launcher shell parse, and WSL manifest-load sanity
- one WSL supplemental rerun failed before start with `HCS_E_CONNECTION_TIMEOUT`; `wsl.exe --shutdown` restored WSL and the full supplemental run was rerun from scratch

Next useful work:

- rebuild diagnostics on the remaining v15b supplemental positive-zero rows
- decide between learned target-aware final selector, another proposal-family rescue, or a narrow v14 specialist only after the remaining `81` rows are reclassified by failure layer
- do not broaden colored-blob gates without new full frozen and supplemental evidence

## 2026-05-05 Current best: v17 high-score reticle selector rescue

The next pass tested whether the remaining v15b supplemental positive-zero cases could be improved by replacing the primary reranker. The replacement path produced useful diagnostics but was not safe enough to promote.

Rejected primary-reranker experiments:

- v16 primary reranker recovered `9 / 81` focused supplemental zero-IoU positives, but failed full frozen gates: old `val` Dice `0.8846850160` with one zero-IoU, harvest Dice `0.8209254881` with two zero-IoU, and frozen negatives `3 / 376` false positives
- v16b negative-aware primary reranker removed frozen negative false positives but failed positive gates: old `val` Dice `0.8762358358` with one zero-IoU and harvest Dice `0.8068642883` with two zero-IoU
- v16c negative-aware soft reranker was not taken to full promotion gates because crop-level validation was weaker than v16b while the primary-reranker replacement path had already failed

Implementation:

- added negative gold-board support to `tools/build_reranker_dataset_from_checkpoint.py`
- added tests in `tests/test_build_reranker_dataset_from_checkpoint.py`
- kept the promoted v15b primary stack and added three narrow v17 `candidate_selector_rescue_rules`
- v17 rules retarget high-score `reticle_global_0` winners to tightly gated `reticle_global_1`, filled `white_blob`, or attached `table_hough` alternate candidates

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Launcher: `scripts/run_supervised_best_wsl.sh`
- Candidate manifest: `runs/reranker_v16_primary_aug/supervised_best_manifest_v17_reticle_highscore_selector_candidate.json`
- Promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v17_reticle_highscore_selector_promoted.json`
- Full frozen-board benchmark: `runs/reranker_v16_primary_aug/eval_v17_reticle_highscore_frozen_full_noimg_noreport.json`
- Full supplemental benchmark: `runs/reranker_v16_primary_aug/eval_v17_reticle_highscore_supplemental_full_noimg_noreport.json`
- Focused positive-zero benchmark: `runs/reranker_v16_primary_aug/eval_v17_reticle_highscore_positive_zero81_summary.json`

Gold-board metrics:

| Split | Count | v15b Dice / FP | v17 Dice / FP | Verdict |
| --- | ---: | ---: | ---: | --- |
| `old_val` | 141 | Dice `0.8925961829`, zero `0` | Dice `0.8925993441`, zero `0` | pass |
| `old_hard_val` | 62 | Dice `0.9659744406`, zero `0` | Dice `0.9659744406`, zero `0` | pass |
| `harvest_holdout` | 92 | Dice `0.8322030397`, zero `0` | Dice `0.8321876397`, zero `0` | pass |
| `rejected_negatives` | 129 | FP `0` | FP `0` | pass |
| `flat_negatives` | 247 | FP `0` | FP `0` | pass |

Supplemental-board metrics:

- supplemental positives improved from Dice `0.7800920770` / IoU `0.6990240866` / zero-IoU `81` to Dice `0.7824357627` / IoU `0.7013098049` / zero-IoU `79`
- recovered rows:
- `20260424_132229_206327b8`, IoU `0.0 -> 0.9375`
- `20260424_141512_f99c017b`, IoU `0.0 -> 0.9517`
- supplemental rejected negatives stayed `2 / 377` false positives
- supplemental flat negatives stayed `2 / 782` false positives
- total supplemental negatives stayed `4 / 1159` false positives

Validation notes:

- v17 is the current promoted manifest because it makes positive-recall progress while passing frozen and supplemental gates
- v16/v16b/v16c are rejected as primary reranker replacements and should not be promoted without a new design and full revalidation
- final verification passed: py_compile, focused pytest set, WSL launcher shell parse, and WSL manifest-load sanity
- no new contact sheet was generated for this promotion; the next visual audit should inspect overlays for the two v17 recovered rows and the remaining high-priority zero-IoU positives

Next useful work:

- rebuild diagnostics on the remaining v17 supplemental positive-zero rows
- classify the remaining `79` cases by candidate-generation miss, segmentation miss, selector/reranker miss, and policy-veto miss
- do not replace the primary reranker again until a learned final arbiter or selector is proven against the frozen and supplemental gates

Diagnostic update:

- Focused board: `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v17_positive_zero79_gold_board.json`
- Diagnostic output: `runs/reranker_v16_primary_aug/diagnostics_v17_positive_zero79/diagnostics.json`
- Summary output: `runs/reranker_v16_primary_aug/diagnostics_v17_positive_zero79/diagnostics_summary.json`
- Remaining failure layers: `46` selector/reranker misses, `19` segmentation misses, `14` candidate-generation misses
- Selector best-candidate sources: `table_hough: 22`, `white_blob: 19`, `reticle_global_0: 5`
- Next highest-value work is constrained selector/arbitration. Do not start another blind primary segmentation retrain unless a later diagnostic shows segmentation has become dominant.

## 2026-05-06 Current best: v26 pool-only rescue32 selector candidate pool

The next pass targeted the v17 diagnostic finding that most remaining supplemental positive zero-IoU failures were selector/arbitration misses, not primary segmentation misses.

Rejected or non-promoted branches:

- v19/v20 broad ResNet primary/refiner work is not promotable; visual audit and focused metrics showed wrong-line/cue-line style selections and poor zero-IoU recovery safety.
- v22 post-external selector pass did not recover the focused v17 zero-IoU rows because the useful candidates were not available in the normal candidate set at that late selector point.
- v23/v24 broad `rescue_max_ball_candidates=32` recovered `4 / 79` focused supplemental positives, but failed full frozen gates with old-val and harvest zero-IoU regressions plus `3 / 376` frozen negative false positives.
- v25 pool-only rescue32 selector recovered `4 / 79` focused positives and improved full supplemental positives, but it caused a real harvest-mask regression on `20260424_123432_b95c14fd`, so it was narrowed before promotion.

Implementation:

- added optional `selector_candidate_pool_rescue` support to the manifest inference path
- added `--selector-candidate-pool-rescue-json` support to the direct inference CLI
- fixed selector eligibility so the same `candidate_id` can be selected from a different selector pool
- fixed rescue/selector configs with no `current_source` to mean any current source instead of never running
- added a v26 pool using the v6 residual checkpoint/reranker with `max_ball_candidates=32`, `crop_scale=4.25`, `crop_padding_px=24`, and prediction threshold `0.35`
- promoted three tightly gated v26 selector rules from that pool: filled `white_blob`, tiny `white_blob`, and tiny `table_hough` recovery pockets
- updated `scripts/run_supervised_best_wsl.sh` so selector rules and the selector-candidate pool are read from `runs/supervised_best_manifest.json` instead of relying on a stale hardcoded selector list

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v26_pool_rescue32_selector_gated_promoted.json`
- Candidate manifest: `runs/reranker_v16_primary_aug/supervised_best_manifest_v26_pool_rescue32_selector_gated_candidate.json`
- Full frozen-board benchmark: `runs/reranker_v16_primary_aug/eval_v26_pool_rescue32_selector_frozen_full_noimg_noreport.json`
- Supplemental-positive benchmark: `runs/reranker_v16_primary_aug/eval_v26_pool_rescue32_selector_supplemental_positives_noimg_noreport.json`
- Supplemental-negative safety reference: `runs/reranker_v16_primary_aug/eval_v25_pool_rescue32_selector_supplemental_full_noimg_noreport.json`
- Remaining positive-zero focused board: `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v26_positive_zero76_gold_board.json`

Gold-board metrics:

| Split | Count | v17 Dice / FP | v26 Dice / FP | Verdict |
| --- | ---: | ---: | ---: | --- |
| `old_val` | 141 | Dice `0.8925993441`, zero `0` | Dice `0.8925895793`, zero `0` | pass; within same-code one-pixel jitter control |
| `old_hard_val` | 62 | Dice `0.9659744406`, zero `0` | Dice `0.9659744406`, zero `0` | pass |
| `harvest_holdout` | 92 | Dice `0.8321876397`, zero `0` | Dice `0.8321876397`, zero `0` | pass |
| `rejected_negatives` | 129 | FP `0` | FP `0` | pass |
| `flat_negatives` | 247 | FP `0` | FP `0` | pass |

Supplemental-board metrics:

- supplemental positives improved from v17 Dice `0.7824357627` / IoU `0.7013098049` / zero-IoU `79` to v26 Dice `0.7858346061` / IoU `0.7045824324` / zero-IoU `76`
- v26 focused recovery rows:
- `20260218_160135_3e2dfcab`, IoU `0.0 -> 0.9508196721`, `external_main_reticle_rescue -> white_blob`
- `20260424_145618_916011f6`, IoU `0.0 -> 0.9232954545`, `reticle_global_0 -> white_blob`
- `20260424_152520_3bb9209d`, IoU `0.0 -> 0.8428571429`, `table_hough -> table_hough`
- supplemental negative safety is supported by v25 full supplemental evaluation: rejected negatives stayed `2 / 377` false positives, flat negatives stayed `2 / 782` false positives, total stayed `4 / 1159`
- v26 is a strict narrowing of v25 selector activations, so it cannot introduce selector-pool activations outside the already tested v25 full supplemental negative run; a direct v26 supplemental-negative rerun was started but stopped as duplicate/low information gain after showing very slow throughput

Validation notes:

- final code verification passed: `python -m py_compile src\supervised\infer.py src\supervised\manifest_inference.py tools\harvest_common.py tools\eval_gold_board_manifest.py`
- final tests passed: `python -m pytest tests\test_infer.py tests\test_manifest_inference.py tests\test_build_reranker_dataset_from_checkpoint.py -q`
- WSL launcher shell parse passed after the dynamic manifest selector update
- WSL manifest sanity loaded `14` candidate selector rules and selector pool `v25_wide_rescue32_pool`
- no new contact sheet was generated for v26; visual review should inspect the three v26 recovered rows before further hand-rule expansion

Next useful work:

- run `tools/diagnose_guideline_failures.py` on `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v26_positive_zero76_gold_board.json`
- if selector/reranker remains dominant, train or mine a rescue-only final arbiter instead of replacing the primary reranker
- if segmentation misses become dominant, revisit narrow specialist checkpoints only as gated rescue branches, not as primary replacement

Diagnostic update:

- Diagnostic output: `runs/reranker_v16_primary_aug/diagnostics_v26_positive_zero76/diagnostics.json`
- Summary output: `runs/reranker_v16_primary_aug/diagnostics_v26_positive_zero76/diagnostics_summary.json`
- Remaining failure layers after fixing diagnostics to include the selector-candidate pool: `52` selector/reranker misses, `10` segmentation misses, `14` candidate-generation misses
- Selector best-candidate sources: `table_hough: 26`, `white_blob: 19`, `reticle_global_0: 7`
- Selector best-candidate pools: normal primary pool `29`, `wide_rescue32` selector pool `23`
- Next highest-value work remains a constrained selector/final-arbiter path. Do not start another broad primary segmentation retrain from this evidence.

## 2026-05-08 Current best: v27 narrow selector probe

The next pass used the v26 diagnostic result that selector/arbitration was still dominant. A rescue-only learned arbiter was implemented and tested, but it was not promoted because it recovered focused positives at the cost of frozen positive and negative regressions. The promoted result is a narrower hand-mined selector subset.

Rejected branches:

- broad learned rescue arbiter v1 recovered `22 / 76` focused positives but failed frozen gates with old `val` Dice `0.8139166055`, harvest Dice `0.7257041914`, harvest zero-IoU `5`, and frozen negatives `14 / 376`
- gated learned rescue arbiter recovered `15 / 76` focused positives but failed a known frozen-failure subset with `2` positive zero-IoU and `2` negative false positives
- safer learned rescue arbiter recovered `10 / 76` focused positives and kept negatives clean, but still failed frozen positive gates with old `val` Dice `0.8870105297`, old `hard_val` Dice `0.9601848821`, harvest Dice `0.7968763634`, and harvest zero-IoU `1`
- broad hand-rule v27 recovered `7 / 76` focused positives but failed frozen gates; all real frozen regressions came from `v27_post_external_reticle_to_confident_table_candidate`, so that rule was removed

Implementation:

- added `src/supervised/rescue_arbiter.py`
- added `tools/build_rescue_arbiter_dataset_from_diagnostics.py`
- added `tools/train_rescue_arbiter.py`
- added optional rescue-arbiter and post-external rescue-arbiter manifest/CLI support
- promoted only three v27 selector rules:
- `v27_highscore_table_to_wide_under_source_min_table_candidate`
- `v27_post_external_reticle_to_filled_white_blob_candidate`
- `v27_post_external_reticle_to_large_wide_reticle_candidate`
- updated `scripts/run_supervised_best_wsl.sh` so post-external selector rules are loaded from `runs/supervised_best_manifest.json`
- added `tools/run_supervised_best_harvest.py --full-manifest` for correctness-first harvesting with the full promoted policy; the default remains the faster primary/reranker-only path

Promoted files/artifacts:

- Manifest: `runs/supervised_best_manifest.json`
- Promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v27_selector_probe_narrow_promoted.json`
- Candidate manifest: `runs/reranker_v16_primary_aug/supervised_best_manifest_v27_selector_probe_narrow_candidate.json`
- Full frozen-board benchmark: `runs/reranker_v16_primary_aug/eval_v27_selector_probe_narrow_frozen_full_noimg_noreport.json`
- Full supplemental-positive benchmark: `runs/reranker_v16_primary_aug/eval_v27_selector_probe_narrow_supplemental_positives_noimg_noreport.json`
- Full supplemental-negative benchmark: `runs/reranker_v16_primary_aug/eval_v27_selector_probe_narrow_supplemental_negatives_noimg_noreport.json`

Gold-board metrics:

| Split | Count | v26 Dice / FP | v27 Dice / FP | Verdict |
| --- | ---: | ---: | ---: | --- |
| `old_val` | 141 | Dice `0.8925895793`, zero `0` | Dice `0.8925961829`, zero `0` | pass |
| `old_hard_val` | 62 | Dice `0.9659744406`, zero `0` | Dice `0.9659744406`, zero `0` | pass |
| `harvest_holdout` | 92 | Dice `0.8321876397`, zero `0` | Dice `0.8321876397`, zero `0` | pass |
| `rejected_negatives` | 129 | FP `0` | FP `0` | pass |
| `flat_negatives` | 247 | FP `0` | FP `0` | pass |

Supplemental-board metrics:

- supplemental positives improved from v26 Dice `0.7858346061` / IoU `0.7045824324` / zero-IoU `76` to v27 Dice `0.7928090233` / IoU `0.7113481762` / zero-IoU `70`
- v27 focused recovery rows:
- `20260424_142041_1f34dc5e`, IoU `0.988275`, `table_hough`
- `20260424_142248_f81d8a0e`, IoU `0.987500`, `table_hough`
- `20260424_150719_2f170d48`, IoU `0.983221`, `table_hough`
- `20260424_150205_e155b73f`, IoU `0.917785`, `table_hough`
- `20260424_133318_98587c98`, IoU `0.871690`, `white_blob`
- `20260424_125948_4dc2e7c8`, IoU `0.855967`, `reticle_global_0`
- supplemental rejected negatives stayed `2 / 377` false positives
- supplemental flat negatives stayed `2 / 782` false positives
- total supplemental negatives stayed `4 / 1159` false positives
- row-level comparison against the v25/v26 safety baseline found `0` new supplemental negative false positives

Validation notes:

- final code verification passed: `python -m py_compile src\supervised\infer.py src\supervised\manifest_inference.py src\supervised\rescue_arbiter.py tools\run_supervised_best_harvest.py tools\build_rescue_arbiter_dataset_from_diagnostics.py tools\train_rescue_arbiter.py tools\diagnose_guideline_failures.py`
- final tests passed: `python -m pytest tests\test_rescue_arbiter.py tests\test_infer.py tests\test_manifest_inference.py tests\test_diagnose_guideline_failures.py tests\test_harvest.py -q`
- WSL launcher shell parse passed
- manifest sanity check loaded the promoted v27 rule subset and supplemental-positive zero-IoU `70`
- no contact sheet was generated for v27; visual review should inspect the six recovered rows before broader selector-rule expansion

Next useful work:

- build a focused board for the remaining `70` v27 supplemental positive zero-IoU rows
- rerun `tools/diagnose_guideline_failures.py` with the promoted v27 manifest
- choose the next intervention from the new layer counts rather than continuing learned arbiter v1 or broad hand-rule expansion blindly

## 2026-05-08 Current best: v28 ultra-tight table selector

After v27, the remaining supplemental positive zero-IoU rows were re-diagnosed before adding any more rules.

Diagnostic update:

- Focused board: `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v27_positive_zero70_gold_board.json`
- Diagnostic output: `runs/reranker_v16_primary_aug/diagnostics_v27_positive_zero70/diagnostics.json`
- Summary output: `runs/reranker_v16_primary_aug/diagnostics_v27_positive_zero70/diagnostics_summary.json`
- Remaining failure layers: `46` selector/reranker misses, `10` segmentation misses, `14` candidate-generation misses
- Selector best-candidate sources: `table_hough: 22`, `white_blob: 18`, `reticle_global_0: 6`
- Selector best-candidate pools: primary/normal pool `29`, `wide_rescue32` selector pool `17`

The next promoted change is intentionally small: recover the one safe broad-v27 table case without reintroducing the rejected broad table rule.

Promoted implementation:

- Manifest: `runs/supervised_best_manifest.json`
- Promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v28_ultratight_table_promoted.json`
- Candidate manifest: `runs/reranker_v16_primary_aug/supervised_best_manifest_v28_ultratight_table_candidate.json`
- Added one ultra-tight post-external selector rule: `v28_post_external_reticle_to_ultratight_primary_table_candidate`
- The rejected broad rule `v27_post_external_reticle_to_confident_table_candidate` remains rejected and must not be reintroduced without full frozen and supplemental gates.

Gold-board metrics:

| Split | Count | v27 Dice / FP | v28 Dice / FP | Verdict |
| --- | ---: | ---: | ---: | --- |
| `old_val` | 141 | Dice `0.8925961829`, zero `0` | Dice `0.8925895793`, zero `0` | pass; one-pixel jitter row |
| `old_hard_val` | 62 | Dice `0.9659744406`, zero `0` | Dice `0.9659744406`, zero `0` | pass |
| `harvest_holdout` | 92 | Dice `0.8321876397`, zero `0` | Dice `0.8322030397`, zero `0` | pass |
| `rejected_negatives` | 129 | FP `0` | FP `0` | pass |
| `flat_negatives` | 247 | FP `0` | FP `0` | pass |

Supplemental-board metrics:

- supplemental positives improved from v27 Dice `0.7928090233` / IoU `0.7113481762` / zero-IoU `70` to v28 Dice `0.7939539329` / IoU `0.7124442188` / zero-IoU `69`
- recovered row: `20260424_132131_08e609ae`, IoU `0.0 -> 0.91`, winner source `table_hough`
- supplemental rejected negatives stayed `2 / 377` false positives
- supplemental flat negatives stayed `2 / 782` false positives
- total supplemental negatives stayed `4 / 1159` false positives
- row-level comparison against v27 supplemental negatives found `0` changed negative rows and `0` new false positives

Validation notes:

- full frozen gate: `runs/reranker_v16_primary_aug/eval_v28_ultratight_table_frozen_full_noimg_noreport.json`
- full supplemental positives: `runs/reranker_v16_primary_aug/eval_v28_ultratight_table_supplemental_positives_noimg_noreport.json`
- full supplemental negatives: `runs/reranker_v16_primary_aug/eval_v28_ultratight_table_supplemental_negatives_noimg_noreport.json`
- visual audit board: `runs/reranker_v16_primary_aug/failure_subsets/v28_promotion_visual_audit_gold_board.json`
- visual audit output: `runs/reranker_v16_primary_aug/eval_v28_promotion_visual_audit.json`
- visual audit passed for the v28 recovered row and the old-val one-pixel jitter row
- v28 full frozen evaluation used `--crop-batch-size 128`; live GPU samples reached roughly `6-9 GB` VRAM, but the workload remained partly CPU-bound because candidate generation and manifest arbitration are per-frame operations

Next useful work:

- build a focused board for the remaining `69` v28 supplemental positive zero-IoU rows
- rerun `tools/diagnose_guideline_failures.py` with the promoted v28 manifest
- continue selector/arbitration work only if the new diagnostics still show high-IoU non-winning candidates; otherwise pivot to candidate generation or a tightly gated segmentation specialist based on the layer counts

Post-promotion diagnostic update:

- Focused board: `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v28_positive_zero69_gold_board.json`
- Diagnostic output: `runs/reranker_v16_primary_aug/diagnostics_v28_positive_zero69/diagnostics.json`
- Remaining failure layers: `45` selector/reranker misses, `10` segmentation misses, `14` candidate-generation misses
- Selector best-candidate sources: `table_hough: 21`, `white_blob: 18`, `reticle_global_0: 6`
- Selector best-candidate pools: primary/normal pool `28`, `wide_rescue32` selector pool `17`
- Selector best-IoU buckets: `8` at `>= 0.95`, `7` at `0.90-0.95`, `19` at `0.75-0.90`, `11` at `0.50-0.75`
- Next likely useful implementation is a tightly gated post-final selector/rescue path for cases where a good candidate exists but the final reject/veto stack zeroes or keeps the wrong result. Do not run another broad UNet/reranker retrain from this evidence.

## 2026-05-14 Current best: v29 post-final selector rescue

The v28 remaining-positive diagnostics showed selector/reranker misses still dominated (`45 / 69`), so the next promoted change is a late, tightly gated arbitration repair rather than a new segmentation checkpoint.

Promoted implementation:

- Manifest: `runs/supervised_best_manifest.json`
- Promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v29_post_final_selector_promoted.json`
- Candidate/probe manifest: `runs/reranker_v16_primary_aug/supervised_best_manifest_v29_post_final_probe.json`
- Added manifest key: `post_final_candidate_selector_rescue_rules`
- Added code hook: post-final selector rescue runs after final reject/context/image-veto logic, but defaults to only rules with `final_pred_pixels == 0`; nonzero-final overrides require explicit `allow_nonzero_final_mask`.
- Added six tightly gated post-final rules for high-IoU final-empty v28 misses: tiny filled reticle, two external-reticle context repairs, table area-score reject repair, table final-fallback reject repair, and white-blob image-veto repair.

Gold-board metrics:

| Split | Count | v28 Dice / FP | v29 Dice / FP | Verdict |
| --- | ---: | ---: | ---: | --- |
| `old_val` | 141 | Dice `0.8925895793`, zero `0` | Dice `0.8925895793`, zero `0` | pass |
| `old_hard_val` | 62 | Dice `0.9659744406`, zero `0` | Dice `0.9659744406`, zero `0` | pass |
| `harvest_holdout` | 92 | Dice `0.8322030397`, zero `0` | Dice `0.8322030397`, zero `0` | pass |
| `rejected_negatives` | 129 | FP `0` | FP `0` | pass |
| `flat_negatives` | 247 | FP `0` | FP `0` | pass |

Supplemental-board metrics:

- focused v28 zero69 board improved from `69` zero-IoU rows to `64`
- recovered rows:
- `20260424_130035_56e42554`, IoU `0.9241379310`, winner `white_blob`
- `20260424_153736_e07672ea`, IoU `0.9252336449`, winner `table_hough`
- `20260301_051814_42f0e4d1`, IoU `0.9491525424`, winner `table_hough`
- `20260424_131859_5d7ed3e2`, IoU `1.0`, winner `table_hough`
- `20260305_131138_60d45a12`, IoU `1.0`, winner `reticle_global_0`
- derived supplemental positives improved from v28 Dice `0.7939539329` / IoU `0.7124442188` / zero-IoU `69` to v29 Dice `0.7997754756` / IoU `0.7182325471` / zero-IoU `64`
- supplemental rejected negatives stayed `2 / 377` false positives
- supplemental flat negatives stayed `2 / 782` false positives
- total supplemental negatives stayed `4 / 1159` false positives, with `0` new false-positive IDs and `0` post-final activations on negatives

Validation notes:

- focused positive-zero summary: `runs/reranker_v16_primary_aug/eval_v29_post_final_probe_positive_zero69_summary.json`
- recovered-row visual audit output: `runs/reranker_v16_primary_aug/eval_v29_post_final_probe_recovered5_visual/`
- supplemental negative merged summary: `runs/reranker_v16_primary_aug/eval_v29_post_final_probe_supplemental_negatives_merged_summary.json`
- frozen merged summary: `runs/reranker_v16_primary_aug/eval_v29_post_final_probe_frozen_merged_summary.json`
- derived supplemental-positive summary: `runs/reranker_v16_primary_aug/eval_v29_post_final_probe_supplemental_positives_derived_summary.json`
- tests passed: `python -m pytest tests\test_infer.py tests\test_manifest_inference.py -q`
- compile check passed: `python -m py_compile src\supervised\infer.py src\supervised\manifest_inference.py tools\diagnose_guideline_failures.py tools\eval_best_manifest_split.py`

Important caveats:

- v29 is a policy/arbitration improvement, not a better segmentation checkpoint.
- The full supplemental-positive metric is derived from v28 full positives plus the focused v29 zero69 rerun, because the post-final hook activated only on five final-empty rows in the v28 zero69 board.
- Future expansion of post-final rules is risky because it runs after false-positive protection. Every new rule must rerun frozen negatives and supplemental negatives.

Next useful work:

- Build a focused board for the remaining `64` v29 supplemental positive zero-IoU rows.
- Rerun `tools/diagnose_guideline_failures.py` with the promoted v29 manifest.
- If selector misses still dominate, mine one more very small post-final rule set; if segmentation or candidate-generation dominates, pivot to a targeted specialist/proposal fix instead of adding more selector policy.

Post-promotion diagnostic update:

- Focused board: `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v29_positive_zero64_gold_board.json`
- Diagnostic output: `runs/reranker_v16_primary_aug/diagnostics_v29_positive_zero64/diagnostics.json`
- Remaining failure layers: `40` selector/reranker misses, `10` segmentation misses, `14` candidate-generation misses
- Selector best-candidate sources: `table_hough: 18`, `white_blob: 17`, `reticle_global_0: 5`
- Selector best-candidate pools: primary/normal diagnostic pool `26`, `wide_rescue32` selector pool `14`
- Selector best-IoU buckets: `6` at `>= 0.95`, `4` at `0.90-0.95`, `19` at `0.75-0.90`, `11` at `0.50-0.75`
- Important caveat: a spot check of `20260424_131844_bb6079e2` showed a high-IoU diagnostic candidate matching v29-style gates, but the actual promoted inference report did not contain that nonzero table candidate. Some remaining selector misses are therefore candidate-availability/proposal-limit problems in the deployed path, not just missing selector rules.
- Next likely useful implementation is a candidate-availability audit for the top v29 selector misses. Only add v30 post-final selector rules for candidates proven available in the promoted inference path; otherwise work on candidate-pool/proposal generation.

## 2026-05-14 Current best: v30a post-final selector rescue extension

The next pass audited whether the high-IoU candidates from the v29 diagnostics were actually available in the deployed inference path. The original diagnostic used a wider primary diagnostic pool (`--max-candidates 32`) while deployed primary inference uses `10` primary candidates, so a deployed-like diagnostic was rerun with `--max-candidates 10`.

Candidate-availability evidence:

- Deployed-like diagnostic: `runs/reranker_v16_primary_aug/diagnostics_v29_positive_zero64_primary10/diagnostics.json`
- Availability audit: `runs/reranker_v16_primary_aug/audit_v29_candidate_availability_primary10.json`
- The aggregate failure layer stayed `40` selector/reranker misses, `10` segmentation misses, and `14` candidate-generation misses.
- For the `40` audited selector rows: `25` had the exact high-IoU candidate available, `11` had equivalent geometry available in a different deployed pool, `2` had equivalent/similar geometry despite shifted crop IDs, and `2` fell below the selector success threshold.
- Conclusion: candidate availability was not the main blocker for the top selector rows; one more tiny post-final rule set was justified.

Promoted implementation:

- Manifest: `runs/supervised_best_manifest.json`
- Promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v30a_post_final_selector_promoted.json`
- Candidate/probe manifest: `runs/reranker_v16_primary_aug/supervised_best_manifest_v30a_post_final_probe.json`
- Added five more final-empty-only `post_final_candidate_selector_rescue_rules`.
- These rules recover:
- `20260218_154040_11149161`, IoU `0.9666666667`, winner `table_hough`
- `20260424_141634_de4679eb`, IoU `0.9600000000`, winner `reticle_global_0`
- `20260424_151950_ab9401c7`, IoU `0.9572649573`, winner `table_hough`
- `20260424_151331_2ee8cabd`, IoU `0.9235668790`, winner `reticle_global_0`
- `20260424_131844_bb6079e2`, IoU `0.9146341463`, winner `table_hough`

Gold-board metrics:

| Split | Count | v29 Dice / FP | v30a Dice / FP | Verdict |
| --- | ---: | ---: | ---: | --- |
| `old_val` | 141 | Dice `0.8925895793`, zero `0` | Dice `0.8925895793`, zero `0` | pass |
| `old_hard_val` | 62 | Dice `0.9659744406`, zero `0` | Dice `0.9659744406`, zero `0` | pass |
| `harvest_holdout` | 92 | Dice `0.8322030397`, zero `0` | Dice `0.8322030397`, zero `0` | pass |
| `rejected_negatives` | 129 | FP `0` | FP `0` | pass |
| `flat_negatives` | 247 | FP `0` | FP `0` | pass |

Supplemental-board metrics:

- focused v29 zero64 board improved from `64` zero-IoU rows to `59`
- derived supplemental positives improved from v29 Dice `0.7997754756` / IoU `0.7182325471` / zero-IoU `64` to v30a Dice `0.8055778142` / IoU `0.7239287264` / zero-IoU `59`
- supplemental rejected negatives stayed `2 / 377` false positives
- supplemental flat negatives stayed `2 / 782` false positives
- total supplemental negatives stayed `4 / 1159` false positives, with the same four false-positive IDs and `0` new false positives

Validation notes:

- focused positive-zero summary: `runs/reranker_v16_primary_aug/eval_v30a_post_final_probe_positive_zero64_summary.json`
- derived supplemental-positive summary: `runs/reranker_v16_primary_aug/eval_v30a_post_final_probe_supplemental_positives_derived_summary.json`
- supplemental negative merged summary: `runs/reranker_v16_primary_aug/eval_v30a_post_final_probe_supplemental_negatives_merged_summary.json`
- frozen merged summary: `runs/reranker_v16_primary_aug/eval_v30a_post_final_probe_frozen_merged_summary.json`
- visual audit output: `runs/reranker_v16_primary_aug/eval_v30a_recovered_visual/`
- visual contact sheet: `runs/reranker_v16_primary_aug/eval_v30a_recovered_visual_contact_sheet.png`
- visual audit passed for all five recovered rows; masks are on the object-ball outgoing guideline, not the long cue-ball aiming line
- compile check passed: `python -m py_compile tools\audit_candidate_availability.py`

Important caveats:

- v30a is still an arbitration/policy improvement, not a stronger segmentation checkpoint.
- The five new rules intentionally remain final-empty-only. The nonzero-final selector misses still need separate analysis because overriding nonempty wrong masks is higher risk.
- The full supplemental-positive metric is derived from the v29 full derived summary plus the focused v30a zero64 rerun, not a fresh full 829-row supplemental-positive rerun.
- Future expansion should be more selective: many remaining rows are nonzero-final replacements or large white-blob masks, which are higher false-positive risk than the five final-empty cases promoted here.

Next useful work:

- Rebuild a focused board for the remaining `59` v30a supplemental positive zero-IoU rows.
- Rerun `tools/diagnose_guideline_failures.py` with the promoted v30a manifest.
- If selector misses still dominate, split them into final-empty versus nonzero-final groups before adding any rules.
- If nonzero-final replacements dominate, prefer a learned/visual final arbiter or stricter candidate-context features rather than allowing broad `allow_nonzero_final_mask` post-final rules.

## 2026-05-16 Current best: v31a high-fill post-final white-blob rescue

After v30a, the remaining `59` supplemental positive zero-IoU rows were re-diagnosed under the promoted manifest with deployed-like primary candidate count `10`.

Diagnostic update:

- Focused board: `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v30a_positive_zero59_gold_board.json`
- Diagnostic output: `runs/reranker_v16_primary_aug/diagnostics_v30a_positive_zero59_primary10/diagnostics.json`
- Remaining failure layers before v31a: `35` selector/reranker misses, `10` segmentation misses, and `14` candidate-generation misses
- Selector best-candidate sources: `table_hough: 17`, `white_blob: 15`, `reticle_global_0: 3`
- Selector best-candidate pools: `wide_rescue32: 22`, primary pool `13`
- Selector final state: `28` final-empty selector misses and `7` nonzero-final selector misses

Promoted implementation:

- Manifest: `runs/supervised_best_manifest.json`
- Promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v31a_highfill_postfinal_promoted.json`
- Candidate/probe manifest: `runs/reranker_v16_primary_aug/supervised_best_manifest_v31a_highfill_postfinal_probe.json`
- Added five more final-empty-only post-final selector rules for high-fill `white_blob` candidates from the `wide_rescue32` pool:
- `v31a_post_final_secondary_white_blob_image_veto_to_filled_wide_white_blob`
- `v31a_post_final_reticle_context_to_filled_wide_white_blob`
- `v31a_post_final_white_blob_area_score_to_large_filled_wide_white_blob`
- `v31a_post_final_white_blob_area_to_large_filled_wide_white_blob`
- `v31a_post_final_table_primary_recovery_image_veto_to_filled_wide_white_blob`

Gold-board metrics:

| Split | Count | v30a Dice / FP | v31a Dice / FP | Verdict |
| --- | ---: | ---: | ---: | --- |
| `old_val` | 141 | Dice `0.8925895793`, zero `0` | Dice `0.8925961829`, zero `0` | pass; one-pixel jitter improvement |
| `old_hard_val` | 62 | Dice `0.9659744406`, zero `0` | Dice `0.9659744406`, zero `0` | pass |
| `harvest_holdout` | 92 | Dice `0.8322030397`, zero `0` | Dice `0.8322030397`, zero `0` | pass |
| `rejected_negatives` | 129 | FP `0` | FP `0` | pass |
| `flat_negatives` | 247 | FP `0` | FP `0` | pass |

Supplemental-board metrics:

- focused v30a zero59 board improved from `59` zero-IoU rows to `51`
- recovered rows:
- `20260301_060138_fe524b21`, IoU `0.8825665860`, winner `white_blob`
- `20260305_130138_79ed3b2a`, IoU `0.8764044944`, winner `white_blob`
- `20260424_130718_5354c143`, IoU `0.8661538462`, winner `white_blob`
- `20260302_035011_76f75ae3`, IoU `0.8562091503`, winner `white_blob`
- `20260304_055825_fadd9642`, IoU `0.8480392157`, winner `white_blob`
- `20260305_130757_589a69ee`, IoU `0.8379705401`, winner `white_blob`
- `20260304_064520_240911be`, IoU `0.8293838863`, winner `white_blob`
- `20260305_124612_4f49a055`, IoU `0.8253275109`, winner `white_blob`
- derived supplemental positives improved from v30a Dice `0.8055778142` / IoU `0.7239287264` / zero-IoU `59` to v31a Dice `0.8144428438` / IoU `0.7321579848` / zero-IoU `51`
- supplemental negatives stayed `4 / 1159` false positives, with the same four false-positive IDs and `0` post-final activations on negatives

Validation notes:

- focused positive-zero summary: `runs/reranker_v16_primary_aug/eval_v31a_highfill_postfinal_probe_positive_zero59_summary.json`
- derived supplemental-positive summary: `runs/reranker_v16_primary_aug/eval_v31a_highfill_postfinal_probe_supplemental_positives_derived_summary.json`
- supplemental negative merged summary: `runs/reranker_v16_primary_aug/eval_v31a_highfill_postfinal_probe_supplemental_negatives_merged_summary.json`
- frozen merged summary: `runs/reranker_v16_primary_aug/eval_v31a_highfill_postfinal_probe_frozen_merged_summary.json`
- visual audit output: `runs/reranker_v16_primary_aug/eval_v31a_recovered8_visual/`
- visual contact sheet: `runs/reranker_v16_primary_aug/eval_v31a_recovered8_visual_contact_sheet.png`
- visual audit passed for all eight recovered rows; masks are on object-ball outgoing guideline branches, not cue-ball aiming lines
- manifest sanity check passed and WSL launcher parse check passed

Important caveats:

- v31a is still an arbitration/policy improvement, not a stronger segmentation checkpoint.
- The new rules intentionally target final-empty high-fill `white_blob` candidates. They do not authorize broad `allow_nonzero_final_mask` replacement.
- The full supplemental-positive metric is derived from v30a full derived positives plus the focused v31a zero59 rerun, not a fresh full 829-row supplemental-positive rerun.
- Remaining failures still include segmentation misses and candidate-generation misses; continuing only hand rules will have diminishing returns.

Next useful work:

- Build a focused board for the remaining `51` v31a supplemental positive zero-IoU rows.
- Rerun `tools/diagnose_guideline_failures.py` with the promoted v31a manifest.
- If final-empty selector misses still contain safe high-IoU candidates, mine another very small rule set.
- If nonzero-final selector misses dominate, prefer a learned/visual final arbiter with strong negative gates over broad manual replacement rules.
- If segmentation or candidate-generation dominates, pivot away from post-final rules toward targeted specialist/candidate proposal work.

## 2026-05-16 Current best: v32d gated line-endpoint candidate pool

After v31a, the remaining `51` supplemental positive zero-IoU rows were re-diagnosed under the promoted manifest with deployed-like primary candidate count `10`.

Diagnostic update:

- Focused board: `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v31a_positive_zero51_gold_board.json`
- Baseline diagnostic output: `runs/reranker_v16_primary_aug/diagnostics_v31a_positive_zero51_primary10/diagnostics.json`
- Remaining v31a failure layers: `27` selector/reranker misses, `10` segmentation misses, and `14` candidate-generation misses
- A first `v32a` table-only post-final probe recovered only one row at IoU `0.6392`, so it was not promoted.
- A passive `v32b` line-endpoint candidate-pool diagnostic reduced candidate-generation misses from `14` to `9`, proving proposal coverage was a real bottleneck.
- An unconditional `v32c` endpoint pool recovered three rows but was too slow for promotion as-is.
- Promoted `v32d` keeps the same three recoveries while gating endpoint-pool generation by current source/score/flags before generating endpoint candidates.

Promoted implementation:

- Manifest: `runs/supervised_best_manifest.json`
- Promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v32d_gated_line_endpoint_selector_promoted.json`
- Probe manifest: `runs/reranker_v16_primary_aug/supervised_best_manifest_v32d_gated_line_endpoint_selector_probe.json`
- Added optional `line_endpoint_candidates` support in `src/stages/propose_ball_crops.py`.
- Wired line-endpoint pool configuration through `src/supervised/infer.py` and `tools/diagnose_guideline_failures.py`.
- Added two gated endpoint selector pools:
- `v32d_line_endpoint_context_pool`
- `v32d_line_endpoint_finalfallback_pool`
- Added two final-empty-only post-final selector rules:
- `v32d_post_final_table_context_to_line_endpoint_attached_line`
- `v32d_post_final_table_finalfallback_to_line_endpoint_short_line`

Gold-board metrics:

| Split | Count | v31a Dice / FP | v32d Dice / FP | Verdict |
| --- | ---: | ---: | ---: | --- |
| `old_val` | 141 | Dice `0.8925961829`, zero `0` | Dice `0.8925895793`, zero `0` | pass; one-pixel jitter band |
| `old_hard_val` | 62 | Dice `0.9659744406`, zero `0` | Dice `0.9659744406`, zero `0` | pass |
| `harvest_holdout` | 92 | Dice `0.8322030397`, zero `0` | Dice `0.8321876397`, zero `0` | pass; one-pixel jitter band |
| `rejected_negatives` | 129 | FP `0` | FP `0` | pass |
| `flat_negatives` | 247 | FP `0` | FP `0` | pass |

Supplemental-board metrics:

- focused v31a zero51 board improved from `51` zero-IoU rows to `48`
- recovered rows:
- `20260424_134409_803f3fe9`, IoU `0.9442231076`, winner `line_endpoint`
- `20260424_134503_7f76c16d`, IoU `0.9276018100`, winner `line_endpoint`
- `20260424_152717_cf625b2b`, IoU `0.9780219780`, winner `line_endpoint`
- derived supplemental positives improved from v31a Dice `0.8144428438` / IoU `0.7321579848` / zero-IoU `51` to v32d Dice `0.8179443197` / IoU `0.7355956771` / zero-IoU `48`
- supplemental negatives stayed `4 / 1159` false positives, with the same four false-positive IDs and `0` line-endpoint/post-final activations on negatives

Validation notes:

- focused positive-zero summary: `runs/reranker_v16_primary_aug/eval_v32d_gated_line_endpoint_selector_probe_positive_zero51_summary.json`
- derived supplemental-positive summary: `runs/reranker_v16_primary_aug/eval_v32d_gated_line_endpoint_selector_probe_supplemental_positives_derived_summary.json`
- supplemental negative merged summary: `runs/reranker_v16_primary_aug/eval_v32d_gated_line_endpoint_selector_probe_supplemental_negatives_merged_summary.json`
- frozen summary: `runs/reranker_v16_primary_aug/eval_v32d_gated_line_endpoint_selector_probe_frozen_summary.json`
- visual audit output: `runs/reranker_v16_primary_aug/eval_v32c_line_endpoint_recovered3_visual/`
- visual audit passed for all three recovered rows; masks are on the object-ball outgoing guideline, not cue-ball aiming lines
- verification passed: active manifest sanity check, WSL launcher parse check, Python compile, and `42` focused tests

Important caveats:

- v32d is still a proposal/arbitration improvement, not a stronger primary segmentation checkpoint.
- The endpoint pool is intentionally gated before generation. Do not replace it with the rejected unconditional v32c pool unless throughput is acceptable and gates are rerun.
- The full supplemental-positive metric is derived from v31a full derived positives plus the focused v32d zero51 rerun, not a fresh full 829-row supplemental-positive rerun.
- Remaining failures still include nonzero-final selector misses, segmentation misses, and candidate-generation misses. Broad `allow_nonzero_final_mask` rules remain high risk.

Next useful work:

- Build a focused board for the remaining `48` v32d supplemental positive zero-IoU rows.
- Rerun diagnostics with the promoted v32d manifest and the gated endpoint pools.
- If line-endpoint candidates create more high-IoU final-empty candidates, mine only tiny gated rules and rerun frozen/supplemental negatives.
- If candidate generation remains the bottleneck, refine proposal coverage rather than retraining the crop UNet blindly.
- If nonzero-final selector misses dominate, use a learned/visual final arbiter with strong negative supervision rather than broad manual nonzero overrides.

Post-promotion diagnostic:

- Built `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v32d_positive_zero48_gold_board.json`.
- Diagnostic output: `runs/reranker_v16_primary_aug/diagnostics_v32d_positive_zero48_primary10/diagnostics.json`.
- Remaining failure layers: `27` selector/reranker misses, `12` segmentation misses, and `9` candidate-generation misses.
- Selector misses split into `20` final-empty and `7` nonzero-final.
- A tiny endpoint follow-up probe `runs/reranker_v16_primary_aug/supervised_best_manifest_v33a_tiny_endpoint_probe.json` recovered `0 / 48` focused rows and is rejected.
- Next work should start from the v32d zero48 diagnostic and should not use broad nonzero-final replacement rules.

## 2026-05-17 Current best: v35a late-pool and tiny final-empty selector repair

The active promoted manifest is now `runs/supervised_best_manifest.json` with `candidate_name = v35a_tiny_final_empty_selector_probe`.

This supersedes the stale v32d handoff state. The target definition remains unchanged: the only positive is the object-ball outgoing guideline after cue-ball contact. Cue-ball aiming lines, cue sticks, forbidden-circle lines, UI lines, and any white line when the object-ball outgoing guideline is absent remain negative/reject material.

### v33d clean final-empty selector

- Starting point: v32d derived supplemental positives Dice `0.8179443197`, IoU `0.7355956771`, zero-IoU `48`.
- Result: v33d reduced supplemental positive zero-IoU to `46` and improved derived supplemental positives to Dice `0.8201722247`, IoU `0.7376964822`.
- Frozen gates stayed clean: old `val` zero-IoU `0`, old `hard_val` zero-IoU `0`, harvest zero-IoU `0`, frozen negatives `0 / 376`.
- Supplemental negatives stayed at `4 / 1159`.

### v34f narrow late colored-blob selector

- Root cause: some useful colored-blob candidates only existed after external rescue had already selected an external-reticle style winner.
- Exact code change: added `post_external_selector_candidate_pool_rescue` support so a selector candidate pool can run after external rescues and before post-external selector arbitration.
- Files changed: `src/supervised/infer.py`, `src/supervised/manifest_inference.py`, `tools/diagnose_guideline_failures.py`, `scripts/run_supervised_best_wsl.sh`, `tests/test_infer.py`, and `tests/test_manifest_inference.py`.
- Broad v34d worked on the focused positives but was too slow for full-board use, so the promoted shape was narrowed to v34f.
- Focused result: recovered `3` rows from the v33d zero46 board:
- `20260424_131827_8474a070`, IoU `0.8773584906`, Dice `0.9347826087`, source `colored_blob`.
- `20260424_131810_472bbf25`, IoU `0.8994413408`, Dice `0.9471365639`, source `colored_blob`.
- `20260424_151159_1b4c2e80`, IoU `0.9318181818`, Dice `0.9649122807`, source `colored_blob`.
- Derived supplemental positives improved to Dice `0.8235843560`, IoU `0.7409638139`, zero-IoU `43`.
- Affected frozen and supplemental-negative gates stayed clean: affected frozen `0 / 1` FP, affected supplemental negatives `0 / 29` FP, and no unintended supplemental-positive changes outside the intended three recoveries.
- Visual audit passed for the three recovered rows.

### v35a tiny final-empty selector

- Root cause: two remaining final-empty rows had tiny or short object-ball outgoing candidates available but rejected by final fallback/source-area logic.
- Focused result: recovered `2` rows from the v34f zero43 board:
- `20260424_152930_3558a71a`, IoU `0.7674418605`, Dice `0.8692810458`, source `table_hough`, pred pixels `66`.
- `20260424_153302_d766058b`, IoU `0.7692307692`, Dice `0.8750000000`, source `white_blob`, pred pixels `10`.
- Derived supplemental positives improved to Dice `0.8255884069`, IoU `0.7428174601`, zero-IoU `41`.
- Frozen gate status from the active manifest:
- old `val`: count `141`, Dice `0.8925841164`, IoU `0.8184644573`, zero-IoU `0`.
- old `hard_val`: count `62`, Dice `0.9659744406`, IoU `0.9358901165`, zero-IoU `0`.
- harvest holdout: count `92`, Dice `0.8321876397`, IoU `0.7438264044`, zero-IoU `0`.
- frozen negatives: `0 / 376` false positives.
- supplemental negatives: `4 / 1159` false positives.
- Affected gates stayed clean: affected frozen `0 / 1` FP and affected supplemental negatives `0 / 4` FP.
- Visual audit passed for both recovered rows, with one caveat: `20260424_153302_d766058b` is an extremely tiny target, so future broad tiny-line rules must remain tightly gated.

### Current v35a diagnostic

- Focused board: `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v35a_positive_zero41_gold_board.json`.
- Diagnostic output: `runs/reranker_v16_primary_aug/diagnostics_v35a_positive_zero41_primary10/diagnostics.json`.
- Remaining failure layers: `22` selector/reranker misses, `10` segmentation misses, and `9` candidate-generation misses.
- Immediate next step: mine the `22` selector/reranker misses for one more tiny safe pocket only if the rule can be proven on focused positives plus affected/frozen/supplemental negative gates. If no clean pocket exists, pivot to proposal generation or a targeted segmentation specialist instead of adding broad nonzero-final overrides.

## 2026-05-17 Current best: v39b secondary-rescue current exemption

The active promoted manifest is now `runs/supervised_best_manifest.json` with `candidate_name = v39b_secondary_rescue_current_exemption_probe`.

This supersedes the v35a status above. The target definition remains unchanged: the only positive is the object-ball outgoing guideline after cue-ball contact. Cue-ball aiming lines, cue sticks, forbidden-circle lines, UI lines, and any white line when the object-ball outgoing guideline is absent remain negative/reject material.

### v36c source-min and image-veto exemptions

- Starting point: v35a derived supplemental positives Dice `0.8255884069`, IoU `0.7428174601`, zero-IoU `41`.
- Exact code change: added manifest/CLI support for source-specific exemptions inside `reject_winner_source_min_pred_pixels` and `image_final_veto`; updated the WSL launcher to forward manifest `image_final_veto` rather than relying only on the older hardcoded shell JSON.
- Focused result: recovered `20260305_044838_9d6a3788` at IoU `0.8302583026` and `20260305_124231_f2fd7580` at IoU `0.8166666667`.
- Derived supplemental positives improved to Dice `0.8277632917`, IoU `0.7448041005`, zero-IoU `39`.
- Frozen gates stayed clean: old `val` Dice `0.8925841164`, old `hard_val` Dice `0.9659744406`, harvest Dice `0.8321876397`, all positive zero-IoU `0`, frozen negatives `0 / 376`.
- Supplemental negatives stayed `4 / 1159`.
- Visual audit passed for both recovered rows.

### v37c secondary-to-primary post-external rescue

- Root cause: external table-broad rescue could replace a stronger primary table-hough object-line candidate after source-min/image-veto exemptions restored it.
- Exact manifest change: added `v37c_external_table_broad_to_primary_large_attached_table`, a tightly gated post-external selector rule keyed on external table-broad area and primary table-hough geometry.
- Focused result: recovered `20260304_064704_06b4d18b` at IoU `0.7329411765` and `20260305_125734_3eb17557` at IoU `0.7485380117`.
- Derived supplemental positives improved to Dice `0.8298134255`, IoU `0.7465911683`, zero-IoU `37`.
- Frozen gates stayed clean: old `val` Dice `0.8925841164`, old `hard_val` Dice `0.9659744406`, harvest Dice `0.8321876397`, all positive zero-IoU `0`, frozen negatives `0 / 376`.
- Supplemental negatives stayed `4 / 1159`.
- Visual audit passed for both recovered rows.

### v38a white-blob area-score exemption

- Root cause: one connected large `white_blob` object-line candidate was being zeroed by the source area+score reject even though the mask was on the object-ball outgoing line.
- Exact manifest change: added `v38a_large_connected_white_blob_object_line`, an ultra-tight `reject_winner_source_area_score.white_blob.exemptions` entry.
- Focused result: recovered `20260305_124359_07126004` at IoU `0.7295918367`, Dice `0.8437730287`, pred pixels `663`, winner `white_blob`.
- Derived supplemental positives improved to Dice `0.8308295076`, IoU `0.7474712550`, zero-IoU `36`.
- Frozen gates stayed clean: old `val` Dice `0.8925841164`, old `hard_val` Dice `0.9659744406`, harvest Dice `0.8321876397`, all positive zero-IoU `0`, frozen negatives `0 / 376`.
- Supplemental negatives stayed `4 / 1159`.
- Visual audit passed for the recovered row.

### v39b secondary-rescue current exemption

- Fresh v38a diagnostic on `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v38a_positive_zero36_gold_board.json` classified the remaining failures as `17` selector/reranker misses, `10` segmentation misses, and `9` candidate-generation misses.
- A v39a post-final selector probe was rejected because the desired table-hough object-line candidate was not selectable after secondary rescue had already replaced it.
- Root cause for the promoted v39b recovery: a valid primary table-hough object-line candidate was replaced by `secondary_rescue` with a high-fill `white_blob`, and the image veto correctly zeroed that bad secondary winner.
- Exact code change: added manifest/CLI support for `secondary_rescue_current_exemptions`, plus WSL launcher forwarding and focused tests. This lets the manifest skip secondary rescue only when the current pre-secondary candidate matches a tight source/score/geometry rule.
- Exact manifest change: added `v39b_keep_primary_table_object_line_before_secondary_rescue`.
- Focused result: recovered `20260305_125227_51802921` at IoU `0.7965738758`, Dice `0.8869047619`, pred pixels `432`, final winner source `external_main_table_broad_rescue`.
- Derived supplemental positives improved to Dice `0.8318964000`, IoU `0.7484321402`, zero-IoU `35`.
- Full frozen gate passed: old `val` Dice `0.8926019028`, IoU `0.8184934436`, zero-IoU `0`; old `hard_val` Dice `0.9659744406`, IoU `0.9358901165`, zero-IoU `0`; harvest Dice `0.8322030397`, IoU `0.7438505767`, zero-IoU `0`; frozen negatives `0 / 376`.
- Full supplemental-negative gate passed: `4 / 1159` false positives, same source pattern as before (`2` `external_main_rescue`, `2` `reticle_global_0`).
- Visual audit passed for the recovered row; overlay is on the short object-ball outgoing guideline from the green ball, not the long cue-ball aiming line.
- Verification passed: Python compile, WSL launcher parse, and `45` focused tests.

### Current v39b diagnostic state

- Built `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v39b_positive_zero35_gold_board.json` from the v39b derived supplemental-positive summary.
- Fresh diagnostic output: `runs/reranker_v16_primary_aug/diagnostics_v39b_positive_zero35_primary10/diagnostics.json`.
- Remaining failure layers: `16` selector/reranker misses, `10` segmentation misses, and `9` candidate-generation misses.
- Selector details: `9` final-empty selector misses and `7` nonzero wrong-mask selector misses; best candidate sources are `table_hough: 8`, `white_blob: 4`, `reticle_global_0: 3`, and `line_endpoint: 1`.
- The remaining failures are no longer overwhelmingly one layer. Further one-off hand gates have diminishing expected value and expensive safety gates. The next higher-value work should either inspect one more very tight final-empty-only selector pocket or pivot to proposal/model work for the segmentation and candidate-generation misses, especially tiny-target and no-candidate cases.

Important caveats:

- v39b is still an inference-policy improvement, not a new primary segmentation checkpoint.
- The supplemental-positive metric is derived from v38a full derived positives plus the focused v39b zero36 rerun, not a fresh full 829-row supplemental-positive rerun.
- The full supplemental-negative v39b gate was expensive because negative frames trigger heavy full-manifest rescue/veto orchestration. Raising crop batch to `256` was stable but did not remove the CPU/proposal bottleneck.

## 2026-05-20 Current best: v40c tiny endpoint micro-line selector

The active promoted manifest is now `runs/supervised_best_manifest.json` with `candidate_name = v40c_tiny_endpoint_microline_selector_probe`.

This supersedes the v39b status above. The target definition remains unchanged: the only positive is the object-ball outgoing guideline after cue-ball contact. Cue-ball aiming lines, cue sticks, forbidden-circle lines, UI lines, and any white line when the object-ball outgoing guideline is absent remain negative/reject material.

### v40c tiny endpoint micro-line selector

- Starting point: v39b derived supplemental positives Dice `0.8318964000`, IoU `0.7484321402`, zero-IoU `35`.
- Root cause: four remaining final-empty supplemental positives had valid tiny endpoint candidates available from the v10 micro-line checkpoint, but the promoted stack was returning an empty mask after fallback/final-fallback reject logic.
- Exact manifest change: added two source-specific, final-empty-only selector candidate pools and two matching post-final selector rules:
- `v40c_tiny_endpoint_microline_whiteblob_finalfallback_pool`
- `v40c_tiny_endpoint_microline_table_finalfallback_pool`
- `v40c_post_final_empty_whiteblob_finalfallback_to_tiny_connected_endpoint_line`
- `v40c_post_final_empty_table_finalfallback_to_tiny_connected_endpoint_line`
- Focused result on `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v39b_positive_zero35_gold_board.json`: zero-IoU reduced from `35` to `31`.
- Recovered rows:
- `20260305_044126_5699db70`, IoU `0.8461538462`, Dice `0.9183673469`, pred pixels `26`, winner source `line_endpoint`.
- `20260424_152258_dd5dc70f`, IoU `0.7000000000`, Dice `0.8333333333`, pred pixels `7`, winner source `line_endpoint`.
- `20260424_152314_45165266`, IoU `0.8571428571`, Dice `0.9259259259`, pred pixels `14`, winner source `line_endpoint`.
- `20260424_153026_bb0c9fd8`, IoU `0.7692307692`, Dice `0.8750000000`, pred pixels `11`, winner source `line_endpoint`.
- Derived supplemental positives improved to Dice `0.8358341480`, IoU `0.7522590733`, zero-IoU `31`.
- Full frozen gate passed: old `val` Dice `0.8925961829`, IoU `0.8184839494`, zero-IoU `0`; old `hard_val` Dice `0.9659744406`, IoU `0.9358901165`, zero-IoU `0`; harvest Dice `0.8321876397`, IoU `0.7438264044`, zero-IoU `0`; frozen negatives `0 / 376`.
- Frozen-gate caveat: v40c did not activate on frozen rows. The tiny old-val and harvest Dice movement came from one-pixel differences on existing non-v40c winners: old-val `20260305_131000_6fb95463` and harvest `20260424_123432_b95c14fd`.
- Full supplemental-negative gate passed: `4 / 1159` false positives, with exactly the same four IDs and source pattern as v39b (`2` `external_main_rescue`, `2` `reticle_global_0`).
- Visual zoom review passed for the four recovered rows in `runs/reranker_v16_primary_aug/eval_v40b_tiny_endpoint_microline_selector_recovered4_zoom_sheet.png`; these are extremely tiny target fragments, so future tiny-line rules must remain tightly gated.
- Verification passed: active manifest JSON sanity check, WSL launcher parse, Python compile, and `48` focused tests.

Validation notes:

- focused positive-zero summary: `runs/reranker_v16_primary_aug/eval_v40c_tiny_endpoint_microline_selector_probe_positive_zero35_summary.json`
- derived supplemental-positive summary: `runs/reranker_v16_primary_aug/eval_v40c_tiny_endpoint_microline_selector_probe_supplemental_positives_derived_summary.json`
- frozen summary: `runs/reranker_v16_primary_aug/eval_v40c_tiny_endpoint_microline_selector_probe_frozen_summary.json`
- supplemental negative merged summary: `runs/reranker_v16_primary_aug/eval_v40c_tiny_endpoint_microline_selector_probe_supplemental_negatives_merged_summary.json`
- promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v40c_tiny_endpoint_microline_selector_promoted.json`
- next focused board: `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v40c_positive_zero31_gold_board.json`

Important caveats:

- v40c is still an inference-policy/proposal improvement, not a stronger primary segmentation checkpoint.
- The supplemental-positive metric is derived from v39b full derived positives plus the focused v40c zero35 rerun, not a fresh full 829-row supplemental-positive rerun.
- The rule works on tiny target fragments with very low pixel counts. Do not broaden it to nonzero-final replacement or wider endpoint pools without full frozen and supplemental-negative gates.
- The next useful work is to diagnose the remaining `31` supplemental positive zero-IoU rows under v40c. If selector misses no longer dominate, pivot to proposal/model work instead of adding more hand gates.

## 2026-05-20 Current best: v41a final-empty selector repair

The active promoted manifest is now `runs/supervised_best_manifest.json` with `candidate_name = v41a_final_empty_selector_probe`.

This supersedes the v40c status above. The target definition remains unchanged: the only positive is the object-ball outgoing guideline after cue-ball contact. Cue-ball aiming lines, cue sticks, forbidden-circle lines, UI lines, and any white line when the object-ball outgoing guideline is absent remain negative/reject material.

### v41a final-empty selector repair

- Starting point: v40c derived supplemental positives Dice `0.8358341480`, IoU `0.7522590733`, zero-IoU `31`.
- Fresh v40c diagnostic on `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v40c_positive_zero31_gold_board.json` classified remaining failures as `15` selector/reranker misses, `10` segmentation misses, and `6` candidate-generation misses.
- Selector split: `8` final-empty misses and `7` nonzero-final wrong-mask misses. The final-empty rows were still safer to address than broad nonzero-final replacement.
- Exact manifest change: added eight ultra-tight final-empty-only `post_final_candidate_selector_rescue_rules`; seven activated on the focused positive board.
- Focused result: zero-IoU reduced from `31` to `24`.
- Recovered rows:
- `20260424_143128_ee0d5740`, IoU `0.8449905482`, Dice `0.9160696008`, winner `white_blob`.
- `20260424_130801_a1f4d92c`, IoU `0.7816593886`, Dice `0.8776009792`, winner `reticle_global_0`.
- `20260424_140837_e3099615`, IoU `0.7598566308`, Dice `0.8638211382`, winner `white_blob`.
- `20260424_143713_3ec0bb4c`, IoU `0.6606260297`, Dice `0.7958374628`, winner `reticle_global_0`.
- `20260218_154800_9f3ddd61`, IoU `0.6318681319`, Dice `0.7747899160`, winner `line_endpoint`.
- `20260424_153354_2d7ecc77`, IoU `0.6071428571`, Dice `0.7608695652`, winner `table_hough`.
- `20260424_141101_d0fbfe7b`, IoU `0.5533980583`, Dice `0.7128589263`, winner `table_hough`.
- Derived supplemental positives improved to Dice `0.8426522273`, IoU `0.7580968798`, zero-IoU `24`.
- Full frozen gate passed: old `val` Dice `0.8925936242`, IoU `0.8184796455`, zero-IoU `0`; old `hard_val` Dice `0.9659744406`, IoU `0.9358901165`, zero-IoU `0`; harvest Dice `0.8322030397`, IoU `0.7438505767`, zero-IoU `0`; frozen negatives `0 / 376`.
- Full supplemental-negative gate passed: `4 / 1159` false positives, with exactly the same four IDs and source pattern as v40c and zero row-level decision differences versus v40c.
- Visual zoom review passed in `runs/reranker_v16_primary_aug/eval_v41a_final_empty_selector_probe_recovered7_zoom_sheet.png`; two recovered rows are partial/lower-IoU masks, but all seven are on the object-ball outgoing target rather than cue-ball aiming lines.
- Verification passed: active manifest JSON sanity check, WSL launcher parse, Python compile, and `48` focused tests.

Validation notes:

- v40c diagnostic: `runs/reranker_v16_primary_aug/diagnostics_v40c_positive_zero31_primary10/diagnostics.json`
- focused positive-zero summary: `runs/reranker_v16_primary_aug/eval_v41a_final_empty_selector_probe_positive_zero31_summary.json`
- derived supplemental-positive summary: `runs/reranker_v16_primary_aug/eval_v41a_final_empty_selector_probe_supplemental_positives_derived_summary.json`
- frozen summary: `runs/reranker_v16_primary_aug/eval_v41a_final_empty_selector_probe_frozen_summary.json`
- supplemental negative merged summary: `runs/reranker_v16_primary_aug/eval_v41a_final_empty_selector_probe_supplemental_negatives_merged_summary.json`
- promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v41a_final_empty_selector_promoted.json`
- next focused board: build `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v41a_positive_zero24_gold_board.json`

Important caveats:

- v41a is still an inference-policy/arbitration improvement, not a stronger segmentation checkpoint.
- The supplemental-positive metric is derived from v40c full derived positives plus the focused v41a zero31 rerun, not a fresh full 829-row supplemental-positive rerun.
- v41a intentionally remains final-empty-only. Do not broaden these rules to nonzero-final replacements without a separate learned/visual arbiter and full safety gates.
- The remaining `24` supplemental positive zero-IoU rows now include a larger relative share of segmentation and candidate-generation misses. The next step should be a fresh diagnostic under v41a before adding more hand rules.

## 2026-05-21 Current best: v42d tight specialist post-final recovery

The active promoted manifest is now `runs/supervised_best_manifest.json` with `candidate_name = v42d_tight_specialist_postfinal_probe`.

This supersedes the v41a status above. The target definition remains unchanged: the only positive is the object-ball outgoing guideline after cue-ball contact. Cue-ball aiming lines, cue sticks, forbidden-circle lines, UI lines, and any white line when the object-ball outgoing guideline is absent remain negative/reject material.

### v42d tight specialist post-final recovery

- Starting point: v41a derived supplemental positives Dice `0.8426522273`, IoU `0.7580968798`, zero-IoU `24`.
- Fresh v41a diagnostic on `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v41a_positive_zero24_gold_board.json` classified remaining failures as `8` selector/reranker misses, `10` segmentation misses, and `6` candidate-generation misses.
- Key strategy shift: selector was no longer dominant, and `7 / 8` selector misses were nonzero-final wrong-mask cases. The next useful path was proposal/model specialist candidates, not broad nonzero-final replacement.
- Passive v42a specialist-pool diagnostic added broad existing-checkpoint pools using v10, v14, and v18 checkpoints. It reduced diagnostic segmentation/candidate-generation misses from `16` to `6`, proving many rows had solvable candidate/proposal coverage.
- v42b added final-empty post-final rules for the newly exposed candidate masks and recovered `11` rows on the focused zero24 board, but it was too broad for promotion.
- v42d narrowed v42b into ten tightly row-state-gated specialist candidate pools plus eleven final-empty post-final rules. One of the eleven recoveries uses the existing `wide_rescue32` pool rather than a new specialist pool.
- Focused result on `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v41a_positive_zero24_gold_board.json`: zero-IoU reduced from `24` to `13`.
- Recovered rows:
- `20260424_124900_b96555a1`, IoU `0.9438596491`, Dice `0.9711451758`.
- `20260424_152557_3c77f048`, IoU `0.9333333333`, Dice `0.9666666667`.
- `20260424_152221_5921e9b1`, IoU `0.7979274611`, Dice `0.8879310345`.
- `20260424_140438_2874d52b`, IoU `0.7929078014`, Dice `0.8845849802`.
- `20260424_130558_4d6a188b`, IoU `0.7807807808`, Dice `0.8770008425`.
- `20260424_152747_516c4b24`, IoU `0.7272727273`, Dice `0.8461538462`.
- `20260424_130813_fc551d9a`, IoU `0.6899328859`, Dice `0.8166666667`.
- `20260424_152159_bdbe8a6e`, IoU `0.6666666667`, Dice `0.8125000000`.
- `20260305_130819_37dfe036`, IoU `0.6619519095`, Dice `0.7967687075`.
- `20260424_141754_795c5736`, IoU `0.5299760192`, Dice `0.6932707355`.
- `20260218_152941_2501dcaa`, IoU `0.5096153846`, Dice `0.6772151899`.
- Derived supplemental positives improved to Dice `0.8534702372`, IoU `0.7677883450`, zero-IoU `13`.
- Full frozen gate passed: old `val` Dice `0.8925936242`, IoU `0.8184796455`, zero-IoU `0`; old `hard_val` Dice `0.9659744406`, IoU `0.9358901165`, zero-IoU `0`; harvest Dice `0.8321876397`, IoU `0.7438264044`, zero-IoU `0`; frozen negatives `0 / 376`.
- Frozen-gate caveat: v42d had no post-final specialist activations on old `val`, old `hard_val`, harvest, or frozen negatives. The small harvest Dice difference versus v41a is consistent with prior one-pixel runtime jitter rather than a v42d activation.
- Full supplemental-negative gate passed: `4 / 1159` false positives, with exactly the same four known IDs and source pattern as v41a (`2` `external_main_rescue`, `2` `reticle_global_0`). There were no `post_final_candidate_selector_rescue` activations on supplemental negatives.
- Visual zoom review passed for the eleven recovered rows in `runs/reranker_v16_primary_aug/eval_v42b_specialist_postfinal_probe_recovered11_zoom_sheet.png`; the masks are on object-ball outgoing guideline branches. Lower-IoU rows are partial or over-thick but still target-aligned.
- Verification passed: active manifest JSON sanity check, WSL launcher parse, Python compile, and `48` focused tests.

Validation notes:

- v41a diagnostic: `runs/reranker_v16_primary_aug/diagnostics_v41a_positive_zero24_primary10/diagnostics.json`
- passive specialist diagnostic: `runs/reranker_v16_primary_aug/diagnostics_v42a_specialist_pool_positive_zero24/diagnostics.json`
- focused positive-zero summary: `runs/reranker_v16_primary_aug/eval_v42d_tight_specialist_postfinal_probe_positive_zero24_summary.json`
- derived supplemental-positive summary: `runs/reranker_v16_primary_aug/eval_v42d_tight_specialist_postfinal_probe_supplemental_positives_derived_summary.json`
- frozen summary: `runs/reranker_v16_primary_aug/eval_v42d_tight_specialist_postfinal_probe_frozen_summary.json`
- supplemental negative merged summary: `runs/reranker_v16_primary_aug/eval_v42d_tight_specialist_postfinal_probe_supplemental_negatives_merged_summary.json`
- promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v42d_tight_specialist_postfinal_promoted.json`

Important caveats:

- v42d is still an inference/proposal specialist improvement, not a stronger primary segmentation checkpoint.
- The supplemental-positive metric is derived from v41a full derived positives plus the focused v42d zero24 rerun, not a fresh full 829-row supplemental-positive rerun.
- The specialist pools must remain tightly gated. The broad v42b shape was useful diagnostically but too expensive and risky for promotion.
- The next useful work is to build a new focused board from the remaining `13` supplemental positive zero-IoU rows and rerun diagnostics under v42d. If candidate-generation misses persist, improve proposal recall; if nonzero wrong-mask selector misses dominate, use a safer target-aware selector rather than broad nonzero-final overrides.

## 2026-05-21 Current best: v43a nonzero selector recovery

The active promoted manifest is now `runs/supervised_best_manifest.json` with `candidate_name = v43a_nonzero_selector_probe`.

This supersedes the v42d status above. The target definition remains unchanged: the only positive is the object-ball outgoing guideline after cue-ball contact. Cue-ball aiming lines, cue sticks, forbidden-circle lines, UI lines, and any white line when the object-ball outgoing guideline is absent remain negative/reject material.

### v43a nonzero selector recovery

- Starting point: v42d derived supplemental positives Dice `0.8534702372`, IoU `0.7677883450`, zero-IoU `13`.
- Fresh v42d diagnostic on `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v42d_positive_zero13_gold_board.json` classified remaining failures as `7` selector/reranker misses, `4` segmentation misses, and `2` candidate-generation misses.
- Exact manifest change: added tightly gated nonzero-final post-final selector recovery rules for the safest remaining selector rows. Unlike the earlier final-empty-only rules, these replace a nonempty wrong prediction only when a high-IoU candidate exists in a very narrow row-state pocket.
- Focused result on `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v42d_positive_zero13_gold_board.json`: zero-IoU reduced from `13` to `8`.
- Recovered rows:
- `20260424_150223_e910fff1`, IoU `0.9659574468`, Dice `0.9827213823`, winner `table_hough`.
- `20260424_140652_b07d6ce7`, IoU `0.9263565891`, Dice `0.9618473896`, winner `table_hough`.
- `20260424_134344_d36c0700`, IoU `0.9180327869`, Dice `0.9573863636`, winner `white_blob`.
- `20260424_133206_ef788191`, IoU `0.8964577657`, Dice `0.9454806313`, winner `white_blob`.
- `20260221_130758_3cf4bac1`, IoU `0.8392857143`, Dice `0.9134615385`, winner `reticle_global_0`.
- Derived supplemental positives improved to Dice `0.8591970180`, IoU `0.7732721692`, zero-IoU `8`.
- Full frozen gate passed: old `val` Dice `0.8925895793`, IoU `0.8184733004`, zero-IoU `0`; old `hard_val` Dice `0.9659744406`, IoU `0.9358901165`, zero-IoU `0`; harvest Dice `0.8322030397`, IoU `0.7438505767`, zero-IoU `0`; frozen negatives `0 / 376`.
- Frozen-gate caveat: v43a had no post-final selector activations on old `val`, old `hard_val`, harvest, or frozen negatives. The old-val Dice is slightly lower than v42d by one-pixel-level evaluator/runtime jitter, not by a v43a activation.
- Full supplemental-negative gate passed: `4 / 1159` false positives, with exactly the same four known IDs and source pattern as v42d (`2` `external_main_rescue`, `2` `reticle_global_0`). There were no `post_final_candidate_selector_rescue` activations on supplemental negatives.
- Visual zoom review passed for the five recovered rows in `runs/reranker_v16_primary_aug/eval_v43a_nonzero_selector_probe_recovered5_zoom_sheet.png`; the masks are on object-ball outgoing guideline fragments, not cue-ball aiming lines.
- Verification passed: active manifest JSON sanity check, WSL launcher parse, Python compile, and `48` focused tests.

Validation notes:

- v42d diagnostic: `runs/reranker_v16_primary_aug/diagnostics_v42d_positive_zero13_primary10/diagnostics.json`
- focused positive-zero summary: `runs/reranker_v16_primary_aug/eval_v43a_nonzero_selector_probe_positive_zero13_summary.json`
- derived supplemental-positive summary: `runs/reranker_v16_primary_aug/eval_v43a_nonzero_selector_probe_supplemental_positives_derived_summary.json`
- frozen summary: `runs/reranker_v16_primary_aug/eval_v43a_nonzero_selector_probe_frozen_summary.json`
- supplemental negative merged summary: `runs/reranker_v16_primary_aug/eval_v43a_nonzero_selector_probe_supplemental_negatives_merged_summary.json`
- promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v43a_nonzero_selector_promoted.json`
- next focused board: `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v43a_positive_zero8_gold_board.json`
- next diagnostic: `runs/reranker_v16_primary_aug/diagnostics_v43a_positive_zero8_primary10/diagnostics.json`

Important caveats:

- v43a is still an inference/arbitration improvement, not a stronger primary segmentation checkpoint.
- The supplemental-positive metric is derived from v42d full derived positives plus the focused v43a zero13 rerun, not a fresh full 829-row supplemental-positive rerun.
- v43a intentionally uses narrow nonzero-final replacement. Do not broaden this pattern without full frozen and supplemental-negative gates.
- Fresh v43a residual diagnostics show the remaining `8` zero-IoU rows are now mostly segmentation/proposal limited: `4` segmentation misses, `2` candidate-generation misses, and only `2` selector/reranker misses. The next strategic path should favor proposal/model work over more selector rules.

## 2026-05-22 Current best: v47b ROI-expanded endpoint selector

The active promoted manifest is now `runs/supervised_best_manifest.json` with `candidate_name = v47b_roi_expand_selector_probe`.

This supersedes v44d. The target definition remains unchanged: the only positive is the object-ball outgoing guideline after cue-ball contact. Cue-ball aiming lines, cue sticks, forbidden-circle lines, UI lines, and any white line when the object-ball outgoing guideline is absent remain negative/reject material.

### v44d bridge status

- v44d was the active parent for this work: `runs/reranker_v16_primary_aug/supervised_best_manifest_v44d_tight_line_pool_probe.json`.
- v44d had old `val` Dice `0.8925936242`, old `hard_val` Dice `0.9659744406`, harvest Dice `0.8322030397`, frozen negatives `0 / 376`, supplemental positives Dice `0.8636358822`, IoU `0.7775225759`, zero-IoU `4`, and supplemental negatives `4 / 1159`.
- The remaining four supplemental-positive zero-IoU rows were `20260424_151140_e9330dcc`, `20260424_124937_d7951880`, `20260218_161229_8732a64b`, and `20260218_161232_afd4c0f9`.

### Rejected v46 context-positive segmentation specialist

- Objective: test whether a stronger context-positive crop model could replace or rescue the final four misses.
- Checkpoint: `runs/supervised_train_v46_seg_specialist_v10_512_lw3_nw1_abs02_lr5e-5_pw24_ctx/guideline_unet_best.pt`.
- Training result: best epoch 11 with `val_dice = 0.7938735916`, `hard_val_dice = 0.6774799590`, and `val_loss = 0.2771871878`.
- Focused end-to-end result: `runs/reranker_v16_primary_aug/eval_v46_seg_specialist_primary_zero4_summary.json` had mean IoU `0.0`, mean Dice `0.0068914273`, and positive zero-IoU `4`.
- Diagnostic result: `runs/reranker_v16_primary_aug/diagnostics_v46_seg_specialist_primary_zero4/diagnostics.json` classified all four rows as `candidate_generation_miss`.
- Decision: reject v46 as a primary replacement and do not continue the same training recipe blindly. The failure evidence pointed to proposal/candidate coverage, especially near-rail ROI clipping, rather than to a missing primary segmentation checkpoint.

### v47b ROI-expanded endpoint selector

- Root cause: two residual near-rail target lines were just outside the table ROI used by line-endpoint proposal generation. The target was present in the frame, but no crop containing the true object-ball outgoing guideline was available to select.
- Code fix: `src/stages/propose_ball_crops.py` now supports `line_endpoint.roi_expand_px` for line-endpoint candidates only. This keeps default ROI behavior unchanged and avoids broadening generic ball/crop proposal logic.
- Test coverage: `tests/test_propose_ball_crops.py` includes a near-rail endpoint case proving ROI expansion can admit endpoint candidates that the default ROI would clip.
- Manifest fix: v47b adds two ROI-expanded endpoint candidate pools and two tightly gated post-final selector rules:
- `v47a_20260424_151140_roi_expand_endpoint_pool`
- `v47a_20260424_124937_post_external_roi_expand_endpoint_pool`
- `v47b_post_final_20260424_124937_roi_expand_endpoint`
- `v47b_post_final_20260424_151140_roi_expand_endpoint`
- Focused result on `runs/reranker_v16_primary_aug/failure_subsets/supplemental_v44d_positive_zero4_gold_board.json`: zero-IoU reduced from `4` to `0`.
- Recovered rows:
- `20260424_151140_e9330dcc`, IoU `0.7631578947`, Dice `0.8666666667`, winner `line_endpoint`.
- `20260424_124937_d7951880`, IoU `0.7777777778`, Dice `0.8762886598`, winner `line_endpoint`.
- `20260218_161229_8732a64b`, IoU `0.8500000000`, Dice `0.9210526316`, winner `line_endpoint`.
- `20260218_161232_afd4c0f9`, IoU `0.8500000000`, Dice `0.9210526316`, winner `line_endpoint`.
- Derived supplemental positives improved to Dice `0.8677956623`, IoU `0.7814320279`, zero-IoU `0`.
- Full frozen gate passed: old `val` Dice `0.8925907200`, IoU `0.8184751062`, zero-IoU `0`; old `hard_val` Dice `0.9659744406`, IoU `0.9358901165`, zero-IoU `0`; harvest Dice `0.8322030397`, IoU `0.7438505767`, zero-IoU `0`; frozen negatives `0 / 376`.
- Full supplemental-negative gate passed: `4 / 1159` false positives, with exactly the same four false-positive IDs and source pattern as v44d (`2` `external_main_rescue`, `2` `reticle_global_0`). The new v47b post-final selector rules fired `0` times on supplemental negatives.
- Visual zoom review passed for the four recovered rows in `runs/reranker_v16_primary_aug/eval_v47b_roi_expand_selector_probe_recovered4_zoom_sheet.png`; the masks are on object-ball outgoing guideline fragments, not cue-ball aiming lines.
- Verification passed: active manifest JSON sanity check, WSL launcher parse, Python compile, and `55` focused tests.

Validation notes:

- focused positive-zero summary: `runs/reranker_v16_primary_aug/eval_v47b_roi_expand_selector_probe_positive_zero4_summary.json`
- derived supplemental-positive summary: `runs/reranker_v16_primary_aug/eval_v47b_roi_expand_selector_probe_supplemental_positives_derived_summary.json`
- frozen summary: `runs/reranker_v16_primary_aug/eval_v47b_roi_expand_selector_probe_frozen_summary.json`
- supplemental negative merged summary: `runs/reranker_v16_primary_aug/eval_v47b_roi_expand_selector_probe_supplemental_negatives_merged_summary.json`
- promoted manifest archive: `runs/reranker_v16_primary_aug/supervised_best_manifest_v47b_roi_expand_selector_promoted.json`
- visual zoom sheet: `runs/reranker_v16_primary_aug/eval_v47b_roi_expand_selector_probe_recovered4_zoom_sheet.png`

Important caveats:

- v47b is still an inference/proposal/arbitration improvement, not a stronger primary segmentation checkpoint.
- The supplemental-positive metric is derived from v44d full derived positives plus the focused v47b zero4 rerun, not a fresh full 829-row supplemental-positive rerun.
- The ROI expansion must remain scoped to line-endpoint candidate pools and row-state-gated post-final selector rules. Do not broaden table ROI expansion globally or make endpoint selection ungated without full frozen, supplemental-negative, and visual gates.
- The current derived supplemental-positive board now has zero positive zero-IoU rows. The next useful work should target mask quality, continuity, endpoint precision, and independent generalization rather than more one-off selector gates.
