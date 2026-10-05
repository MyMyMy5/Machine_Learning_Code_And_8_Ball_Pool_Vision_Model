# Decisions

## Mandatory context

Before recording decisions, read `docs/experiments/Supervised_Target_Line_Progress.md`.

### YYYY-MM-DD — decision title

- Context:
- Options considered:
- Decision:
- Why:
- Evidence:
- Risks:
- Revisit when:

### 2026-04-25 - Promote table-hough final-fallback reject

- Context: the repaired promoted manifest preserved old `val`/`hard_val` quality but false-positive-tested poorly on the frozen negative gold board.
- Options considered: promote v11 conditioned primary, keep current policy unchanged, add a narrow table-hough final-fallback zero-mask reject, or start another blind retrain.
- Decision: reject v11 as primary and promote `reject_table_hough_final_fallback_min_pred_pixels = 80` as a targeted policy improvement.
- Why: v11 repaired keep-reranker regressed old `val` to Dice `0.8771027920` and introduced 1 zero-IoU; the reject rule reduced negative false positives without firing on old/hard/harvest positives.
- Evidence: `runs/_eval_reject_tabhough_ff80_full.json` reduced negatives from `143 / 376` false positives to `124 / 376`; old `hard_val` Dice stayed `0.9659744406`; old `val` had zero zero-IoU and no reject activations.
- Risks: the rule is calibrated on the current frozen gold board; unseen true positives that are large `table_hough` final-fallback masks could be suppressed.
- Revisit when: additional positive labels contain valid large `table_hough` final-fallback lines, or a learned validity/veto model beats the rule while preserving old `val`/`hard_val`.

### 2026-04-25 - Repair promoted manifest broad-rescue mismatch

- Context: `runs/supervised_best_manifest.json` embedded `best_final16` metrics but lacked `external_table_broad_rescue_*` keys; `scripts/run_supervised_best_wsl.sh` still had those settings.
- Options considered: trust the stale manifest, trust the launcher, or repair the manifest to match the documented current-best policy.
- Decision: repair the manifest and regenerate v11 candidate manifests from the repaired base.
- Why: manifest-based evaluation and candidate comparisons were under-running the documented current-best policy without the broad external table rescue.
- Evidence: repaired baseline old `val`/`hard_val` matched the stored `best_final16` benchmark within tiny evaluator differences and restored the expected `external_main_table_broad_rescue` activations.
- Risks: older validation files produced before the repair can be stale if they relied on `runs/supervised_best_manifest.json`.
- Revisit when: adding any future rescue key; verify both launcher scripts and JSON manifests contain the same policy.

### 2026-04-26 - Promote combined false-positive veto and stop v12 extension

- Context: v12 candidate-faithful conditioned smoke finished weak, while the frozen gold-board rows exposed narrow false-positive veto pockets with low positive risk.
- Options considered: continue v12 into a long run, promote source-area-only veto, promote combined low-score/table-area veto, or keep the previous `ff80` policy unchanged.
- Decision: do not extend v12; promote the combined veto in `runs/supervised_best_manifest.json` and `scripts/run_supervised_best_wsl.sh`.
- Why: v12 best smoke crop Dice was too weak (`val_dice = 0.8356`, `hard_val_dice = 0.6874`), while the combined veto preserved old/hard gates, slightly improved harvest Dice, and removed many negative false positives.
- Evidence: `runs/_eval_combined_veto_tabhough600_full.json` old `val` Dice `0.8925895793` with zero-IoU `0`; old `hard_val` Dice `0.9659744406` with zero-IoU `0`; harvest Dice `0.7847130806` with zero-IoU `5`; negative false positives improved from `124 / 376` to `66 / 376`.
- Evidence: visual review sheets `runs/gold_board_v1/contact_sheets_policy_mining/combined_veto_suppressed_baseline_masks.png` and `runs/gold_board_v1/contact_sheets_policy_mining/tabhough600_newly_suppressed.png` showed the newly suppressed masks were wrong-line or UI artifacts, not the object-ball outgoing guideline.
- Risks: this is a policy veto calibrated on the frozen gold board; unseen valid large masks from the same source buckets could be rejected if their morphology matches the veto thresholds.
- Revisit when: new positive labels contain valid large `external_main_rescue`, `white_blob`, `table_hough`, or `external_main_reticle_rescue` masks that the current veto would suppress, or a learned final-winner veto beats these hand-tuned rules.

### 2026-04-26 - Promote source-specific area-score false-positive veto

- Context: after the combined high-area veto, the frozen gold board still had `66 / 376` image-level negative false positives.
- Options considered: widen existing high-area vetoes, add a source-specific area+score veto, train a learned final-winner veto immediately, or leave the policy unchanged.
- Decision: promote the evaluated area+score veto map in `runs/supervised_best_manifest.json` and `scripts/run_supervised_best_wsl.sh`.
- Why: the rule removed 12 more false positives with no old `val`/`hard_val` zero-IoU regression and no harvest regression; visual review showed the removed masks were wrong-line/no-target cases.
- Evidence: `runs/_eval_area_score_veto_candidate_full.json` old `val` Dice `0.8925952991` with zero-IoU `0`; old `hard_val` Dice `0.9659744406` with zero-IoU `0`; harvest Dice `0.7847130806` with zero-IoU `5`; negative false positives improved from `66 / 376` to `54 / 376`.
- Evidence: visual review sheet `runs/gold_board_v1/contact_sheets_policy_mining/area_score_veto_suppressed_fp12.png` showed only cue-ball-connected, wrong-line, or no-target frames.
- Risks: this is still hand-tuned on the frozen gold board; source-specific score thresholds may suppress unseen valid low-score positives from similar source buckets.
- Revisit when: new positives include valid low-score, medium-area outputs from `table_hough`, `white_blob`, `reticle_global_0`, or `external_main_reticle_rescue`, or a learned final-winner validity/veto model beats the rule.

### 2026-04-26 - Reject metadata-only final veto and promote pre-final-fallback score veto

- Context: after the area+score veto, the frozen gold board still had `54 / 376` negative false positives.
- Options considered: train a learned final-winner veto on non-gold train-board report features, add a target-aware context selector, add another deterministic report-feature rule, or leave the policy unchanged.
- Decision: reject the metadata-only learned final-veto checkpoint and promote `reject_pre_final_fallback_max_score = -0.923089`.
- Why: the learned veto was unsafe on the frozen gold board, while the pre-final-fallback score rule reduced false positives without increasing old `val`, old `hard_val`, or harvest zero-IoU.
- Evidence: `runs/final_veto_v1/train_board_report_final_veto_gold_eval.json` at threshold `0.3782908320` vetoed 44 negatives but also 40 positives; thresholds `0.99`, `0.999`, and `0.9999` were also not promotable.
- Evidence: `runs/_eval_preff_veto_candidate_combined_noimg.json` reduced total negative false positives from `54 / 376` to `41 / 376`; old `val` zero-IoU stayed `0`, old `hard_val` zero-IoU stayed `0`, and harvest zero-IoU stayed `5`.
- Risks: the promoted rule is still calibrated on the frozen gold board and targets a policy-specific score field; if future rescue/fallback ordering changes, the threshold must be revalidated.
- Revisit when: new valid positives depend on low `pre_final_fallback_score` winners, or an enriched target-aware selector with image/mask/candidate geometry beats this rule while preserving old/hard/harvest gates.

### 2026-04-27 - Promote final context false-positive veto v2

- Context: after the pre-final-fallback score veto, the frozen gold board still had `41 / 376` image-level negative false positives, dominated by cue-ball/cue-stick/wrong-line, reticle, UI/menu, and external rescue artifacts.
- Options considered: train another model immediately, widen existing source/area thresholds, promote pure metadata rules, or add source-specific final-mask context rules using image/mask/candidate geometry.
- Decision: promote manifest-driven `reject_final_context_rules` v2 and keep further model training/retraining for the separate harvested zero-IoU recall problem.
- Why: offline context audit and exact evaluator runs found eight source-specific pockets with zero positive activations: external table-broad masks far from candidates in full-frame ROI failures, pure-white `table_hough` masks far from candidate centers, off-table `reticle_global_0` masks, `white_blob` masks far from candidate boundaries, external reticle large-ROI failures, `table_hough` masks far from candidate boundaries, external low-mid rescue masks far from candidates, and external table-broad masks far from candidates.
- Evidence: exact 41-FP subset evaluation first improved from `41` false positives to `20`, then v2 improved the exact remaining subset to `9`; full frozen-board summary `runs/_eval_context_veto_v2_candidate_combined_noimg.json` reduced negatives from `41 / 376` to `9 / 376`.
- Evidence: old `val` Dice `0.8925952991` with zero-IoU `0`; old `hard_val` Dice `0.9659744406` with zero-IoU `0`; harvest Dice `0.7847130806` with zero-IoU `5`; final context veto fired `0` times on positive splits.
- Risks: `table_hough` pure-white gating may be dataset-calibrated; future valid object-ball outgoing lines that are unusually pure white and far from detected candidate centers could be suppressed, so this rule must be revalidated when new positive labels are added.
- Revisit when: new positives contain valid masks matching any context-veto pocket, or a learned selector with richer object-ball/cue-ball geometry suppresses the remaining false positives without old/hard/harvest regression.

### 2026-04-27 - Promote candidate selector rescue v2

- Context: after final context veto v2, the frozen board still had 5 harvested positive zero-IoU misses; failure-layer diagnosis showed that two of them had high-IoU non-winning candidates, so the failure was selector/arbitration, not candidate recall or crop segmentation.
- Options considered: retrain another primary model, promote the broader v1 candidate-selector rescue, promote narrowed v2 candidate-selector rules, or leave the manifest unchanged.
- Decision: promote candidate selector rescue v2 in `runs/supervised_best_manifest.json` and `scripts/run_supervised_best_wsl.sh`.
- Why: v2 recovered the same two harvested selector misses as v1 while firing narrowly enough to avoid old/hard/negative activations in the full frozen-board report.
- Evidence: `runs/_eval_candidate_selector_rescue_v2_full_noimg.json` improved harvested holdout Dice from `0.7847130806` to `0.8042360753` and reduced harvested zero-IoU from `5` to `3`; old `val` zero-IoU stayed `0`, old `hard_val` zero-IoU stayed `0`, and negatives stayed `9 / 376` false positives.
- Evidence: targeted visual eval recovered `20260424_151001_baa1ab21` at IoU `0.9375` / Dice `0.9679487` and `20260424_153804_0847479e` at IoU `0.7241379` / Dice `0.8407960`.
- Evidence: candidate selector fired only once in the final full-board report; the small old-val Dice delta was traced to same-code one-pixel evaluator/runtime jitter, with `runs/_eval_current_code_old_val_noimg.json` producing old `val` Dice `0.8926002279` for the current manifest.
- Risks: this is still a hand-gated policy based on current candidate metadata; if candidate scoring or source naming changes, the rescue rules need full-board revalidation.
- Revisit when: new harvested positives show more selector/reranker misses, candidate-source metadata changes, or a learned selector with image/mask geometry beats the hand rules while preserving old `val`, old `hard_val`, harvested holdout, and negatives.

### 2026-04-28 - Promote micro-line rescue v4b

- Context: after candidate selector rescue v2, three harvested positives remained zero-IoU: one candidate-generation miss and two segmentation/model misses.
- Options considered: retrain another primary model, promote v10 harvest repair as the primary checkpoint, add a broad rescue branch, use v10 only as a narrow micro-line specialist, or leave the manifest unchanged.
- Decision: promote micro-line rescue v4b in `runs/supervised_best_manifest.json` and `scripts/run_supervised_best_wsl.sh`.
- Why: v10 as a primary checkpoint had already regressed old `val`, but as a gated micro-line specialist it recovered a tiny harvested object-ball outgoing guideline while preserving old/hard and negative gates.
- Evidence: `runs/_eval_micro_line_v4b_full_noimg.json` improved harvested holdout Dice from `0.8042360753` to `0.8223754457` and reduced harvested zero-IoU from `3` to `1`; old `val` zero-IoU stayed `0`, old `hard_val` zero-IoU stayed `0`, and negatives stayed `9 / 376` false positives.
- Evidence: targeted evaluation recovered `20260424_131153_d45e73ab` at IoU `0.94` / Dice `0.9692308` and `20260424_153856_1c8a4307` at IoU `0.56` / Dice about `0.7215`.
- Evidence: visual review of `runs/gold_board_v1/harvest_zero_iou_v2/eval_micro_line_v4_visual_outputs/harvest_zero_iou/20260424_153856_1c8a4307/overlay_final.png` showed the selected mask on the tiny object-ball outgoing line, not the cue-ball aiming line or reticle/circle.
- Risks: the micro-line rescue uses a lower threshold and a specialized checkpoint, so it must remain tightly gated; changing candidate source names, candidate order, or thresholding requires full frozen-board revalidation.
- Revisit when: new harvested misses show more tiny object-ball outgoing lines outside this gate, or a cleaner target-aware learned selector/conditioned model beats the rescue without old/hard/negative regression.

### 2026-04-28 - Promote colored-blob object-ball rescue v5

- Context: after micro-line rescue v4b, the only harvested positive zero-IoU miss was `20260424_150737_24452d24`, where no existing candidate crop covered the object-ball outgoing guideline from a saturated yellow ball.
- Options considered: widen generic Hough/table proposals, retrain another primary model, add a broad rescue branch, or add a narrow saturated colored-blob candidate branch gated on the specific wrong-winner pattern.
- Decision: promote colored-blob object-ball rescue v5 in `runs/supervised_best_manifest.json` and `scripts/run_supervised_best_wsl.sh`.
- Why: manual crop probing proved the existing v2 crop model could segment the target line if given the correct yellow-ball crop, so the best fix was candidate generation rather than another blind retrain.
- Evidence: `runs/_eval_colored_blob_v5_full_noimg.json` improved harvested holdout Dice from `0.8223754457` to `0.8322030397` and reduced harvested zero-IoU from `1` to `0`; old `val` zero-IoU stayed `0`, old `hard_val` zero-IoU stayed `0`, and negatives stayed `9 / 376` false positives.
- Evidence: targeted visual evaluation recovered `20260424_150737_24452d24` at IoU `0.8280802292` / Dice `0.9061032864`, and the `colored_blob` winner fired exactly once on the full frozen board.
- Evidence: visual review of `runs/gold_board_v1/harvest_zero_iou_v2/eval_colored_blob_v5c_visual_outputs/harvest_zero_iou/20260424_150737_24452d24/overlay_final.png` showed the selected mask on the short object-ball outgoing guideline from the yellow ball, not the cue-ball aiming line.
- Risks: the colored-blob rescue depends on saturation/area/radius thresholds and a narrow current-winner gate; if future videos include unusual colored UI blobs or differently colored balls, it must be revalidated on the full frozen board.
- Revisit when: new harvested positives expose more candidate-generation misses, or a learned target-aware selector/candidate generator beats the hand-gated colored-blob rescue while preserving old `val`, old `hard_val`, harvest holdout, and negatives.

### 2026-04-28 - Promote false-positive context veto v6b

- Context: after colored-blob rescue v5, harvested holdout had zero positive zero-IoU failures but the frozen board still had `9 / 376` image-level negative false positives.
- Options considered: add all six context rules from the 9-FP subset, promote no further vetoes, rerun blind training, or promote only the rules that survive full/delta validation.
- Decision: promote v6b final-context veto rules and reject the unsafe v6 external-blob rule.
- Why: v6 reduced the exact 9-FP subset from 9 to 3 but full-board validation showed one old-val positive was incorrectly vetoed; v6b removes that unsafe rule while preserving the other five useful context vetoes.
- Evidence: full v6 benchmark `runs/_eval_fp_context_v6_full_noimg.json` had old `val` Dice `0.8862899992` and one positive zero-IoU, so v6 was rejected.
- Evidence: the rejected rule was `external_blob_offtable_far_from_candidates`, which vetoed true positive `20260302_041911_22c61611`; the exact v6b delta rerun restored that row to IoU `0.8034398034` / Dice `0.8911564626`.
- Evidence: v6b derived full summary `runs/_eval_fp_context_v6b_derived_full_noimg.json` has old `val` Dice `0.8925907200` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8322030397` zero-IoU `0`, and negatives `4 / 376` false positives.
- Evidence: exact contact sheet `runs/gold_board_v1/contact_sheets_policy_mining/remaining_fp9_v5_exact.png` confirmed the suppressed negative masks were wrong-line/no-target cases under the object-ball outgoing guideline rule.
- Risks: `runs/_eval_fp_context_v6b_derived_full_noimg.json` is derived from full v6 plus a two-row exact delta, not a fresh hour-long full v6b rerun; the derivation is valid because v6b removes only one rule and that rule fired on exactly two rows in v6.
- Revisit when: a fresh full v6b rerun is needed for audit strictness, new positives match one of the v6b context pockets, or a learned target-aware selector can reduce the remaining four false positives without positive regression.

### 2026-04-28 - Promote global-context image final veto v7

- Context: after v6b, harvested positives had zero zero-IoU failures but the frozen negative board still had four false positives from `white_blob`, `table_hough`, and `external_main_blob_rescue`.
- Options considered: add more scalar context thresholds, use the failed source-focused metadata veto, promote local-crop image veto, promote global-context image veto, or leave v6b unchanged.
- Decision: promote global-context image final veto checkpoint `runs/final_veto_v3_image_focus/image_veto_global_s42.pt` at threshold `0.95`, gated to `white_blob`, `table_hough`, and `external_main_blob_rescue`.
- Why: scalar/report-only vetoes overlapped positives; local-crop image veto safely removed only `3 / 4` remaining false positives; global context removed all four while preserving all focused gold positives.
- Evidence: exact focused gold eval `runs/final_veto_v3_image_focus/gold_source_focus_v7_summary.json` had `0 / 4` negative false positives and `0` positive zero-IoU across `98` positive rows from the risky sources.
- Evidence: fresh full summary `runs/_eval_image_veto_v7_full_noimg.json` keeps old `val` Dice `0.8925952991` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8321930577` zero-IoU `0`, and improves negatives from `4 / 376` false positives to `0 / 376`.
- Risks: v7 uses a learned classifier trained on current final masks, so it should be revalidated after any upstream winner-source/candidate-selection change.
- Revisit when: new independent negatives/positives show image-veto false positives or false negatives.

### 2026-04-28 - Promote external-main source-min image-veto v8

- Context: supplemental failure mining found true object-ball outgoing guideline positives selected by `external_main_rescue` that were being zeroed by the old blunt `reject_winner_source_min_pred_pixels.external_main_rescue = 400` rule.
- Options considered: keep v7 unchanged, remove the external-main area reject outright, use a `400` source-specific image-veto minimum, use a lower source-specific image-veto minimum, or retrain another segmentation checkpoint.
- Decision: remove `external_main_rescue` from the blunt source-area reject, add `external_main_rescue` to `image_final_veto.sources`, and promote `image_final_veto.source_min_pred_pixels.external_main_rescue = 180`.
- Why: removing the area reject alone recovered positives but reintroduced external-main negative false positives; source-min `400` was still too conservative for supplemental positives; source-min `180` protects the known true positive with pre-veto area `156` while allowing the learned image veto to suppress larger external-main negatives.
- Evidence: exact full frozen summary `runs/_eval_no_external_area_source_min180_image_veto_full_noimg.json` has old `val` Dice `0.8925895793` with zero-IoU `0`, old `hard_val` Dice `0.9659744406` with zero-IoU `0`, harvest Dice `0.8321876397` with zero-IoU `0`, and negatives `0 / 376` false positives.
- Evidence: row-level diff versus v7 showed only one old-val one-pixel mask jitter row and two harvest one-pixel jitter rows; no positive zero-IoU and no negative false positives were introduced.
- Risks: `180` is calibrated from current supplemental/frozen evidence, and the supplemental board is failure-mining material rather than a pristine independent benchmark. The full supplemental promoted-v8 result must be parsed before deciding the next training or policy direction.
- Revisit when: new positives below the `180` image-veto floor are missed, new external-main negatives below `180` are false positives, or a learned target-aware selector replaces this hand-gated source policy.

### 2026-04-29 - Promote area-score exemptions v10b

- Context: promoted-v8 supplemental failure mining found that many remaining positive zero-IoU cases were valid `table_hough` or `white_blob` candidates zeroed by the existing source area+score reject.
- Options considered: keep v8 unchanged, remove `table_hough`/`white_blob` area+score rejects broadly, promote feature-gated exemptions, or start another blind segmentation retrain.
- Decision: promote v10b feature-gated area-score exemptions in `runs/supervised_best_manifest.json` and `scripts/run_supervised_best_wsl.sh`.
- Why: broad relaxation recovered positives but introduced new negatives; v10b recovers the same supplemental positive-zero cases as v10 while keeping frozen negatives at `0 / 376` and avoiding the v10 score-jitter leak.
- Evidence: `runs/_eval_v10b_area_score_exemptions_full_noimg_noreport.json` has old `val` Dice `0.8925895793` zero-IoU `0`, old `hard_val` Dice `0.9658330969` zero-IoU `0`, harvest Dice `0.8321876397` zero-IoU `0`, and negatives `0 / 376` false positives.
- Evidence: `runs/supplemental_v7_validation/eval_v10b_area_score_exemptions_positive_zero100_summary.json` keeps `16 / 100` previously zero-IoU supplemental positives recovered relative to v8; `runs/supplemental_v7_validation/eval_v10b_area_score_exemptions_supp_affected_negatives_summary.json` has `0` new `white_blob`/`table_hough` false positives.
- Evidence: v9 broad relaxation was rejected because it added new `white_blob`/`table_hough` false positives; v10 was rejected because score jitter leaked rejected negative `20260424_151642_29ac32d3`; v10b raises `white_blob.max_score` to `3.01` while preserving the positive-saving exemption.
- Risks: this is still hand-gated policy based on current candidate features; the old-hard Dice change is a one-pixel runtime jitter row, but future changes to scoring, candidate source names, or feature definitions require full frozen-board revalidation.
- Revisit when: new positives fail outside these exemptions, new negatives match the exemption feature pocket, or a learned target-aware final selector can replace hand-gated source exemptions.

### 2026-04-29 - Promote reticle selector rescue v11c

- Context: after v10b, supplemental failure mining still had `84` positive zero-IoU cases. Candidate diagnostics showed selector/reranker miss was the largest class, with high-IoU non-winning candidates available in many rows.
- Options considered: keep v10b unchanged, promote broad v11 reticle selector rules, promote narrowed v11b, promote narrower score-bounded v11c, or stop hand-rule work and start a learned selector immediately.
- Decision: promote v11c in `runs/supervised_best_manifest.json` and `scripts/run_supervised_best_wsl.sh`; reject v11 and v11b.
- Why: v11c recovers two supplemental positive zero-IoU rows, improves supplemental positive Dice/IoU, and preserves frozen old/hard/harvest and negative gates. v11 and v11b both caused unacceptable positive regressions.
- Evidence: `runs/_eval_v11c_reticle_selector_full_noimg_noreport.json` has old `val` Dice `0.8925898362` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8322030397` zero-IoU `0`, and frozen negatives `0 / 376` false positives.
- Evidence: `runs/supplemental_v7_validation/eval_v11c_reticle_selector_full_noimg_noreport.json` improves supplemental positives from Dice `0.7769746695` / zero-IoU `84` to Dice `0.7789535461` / zero-IoU `82`, while supplemental negatives remain `40 / 1159` false positives.
- Evidence: recovered positives include `20260424_140309_be90cc5f` (`0.0 -> 0.9060150376` IoU) and `20260424_123414_85a44901` (`0.0 -> 0.7670682731` IoU).
- Risks: one supplemental positive, `20260424_123847_37862eae`, regressed from IoU `0.5819070905` to `0.3913043478`; this does not violate promotion gates but shows that more hand selector rules may be brittle.
- Revisit when: new negative false positives appear from the v11c rules, more reticle-selector rescues regress true positives, or a learned candidate selector can recover the same misses with fewer partial-mask regressions.

### 2026-04-30 - Promote dual external/reticle image final veto

- Context: v11c preserved frozen gates but left `40 / 1159` supplemental negative false positives, mostly from external rescue and reticle/global sources outside the old v7 image-veto source list.
- Options considered: keep v11c unchanged, replace old v7 image-final-veto with a new v4 external/reticle checkpoint, promote a dual-veto list containing both v7 and v4, or continue hand selector/policy rules.
- Decision: promote the dual image-final-veto list in `runs/supervised_best_manifest.json` and `scripts/run_supervised_best_wsl.sh`; reject single-v4 replacement.
- Why: single-v4 replacement fixed many supplemental negatives but leaked one frozen flat negative that old v7 handled; dual-veto preserves the old frozen-board protection and adds the new external/reticle protection.
- Evidence: `runs/_eval_image_veto_v4_dual_candidate_full_noimg_noreport.json` keeps old `val` Dice `0.8925936242` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8321876397` zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: `runs/supplemental_v7_validation/image_veto_v4_external_reticle/eval_dual_candidate_supplemental_full_noimg_noreport.json` keeps supplemental positive zero-IoU at `82`, slightly improves positive Dice to `0.7789596845`, and reduces supplemental negatives from `40 / 1159` to `4 / 1159` false positives.
- Evidence: v4 direct focused eval suppressed `36 / 40` supplemental FP rows with `0 / 518` keep-positive vetoes, but full frozen replacement eval caught the single-v4 replacement leak.
- Risks: this adds another learned veto checkpoint and increases policy complexity; future upstream source/candidate changes require revalidating both veto configs as a pair.
- Revisit when: new supplemental negatives appear from the remaining `external_main_rescue` or `reticle_global_0` pockets, or a single learned target-aware selector/veto can replace both image-veto checkpoints while preserving frozen and supplemental gates.

### 2026-04-30 - Reject v14 thin-line repair as primary replacement

- Context: after the dual image-veto promotion, the next documented bottleneck was positive recall on `82` supplemental zero-IoU cases, including `19` rows classified as segmentation misses.
- Options considered: continue v14 training to 24 epochs, promote v14 as the primary checkpoint, use v14 as a narrow rescue/specialist later, or pivot to candidate-generation rescue work.
- Decision: stop v14 after the validation curve stopped improving and reject it as a primary replacement.
- Why: v14 recovered a few focused segmentation-miss rows but materially regressed frozen old `val`, introduced positive zero-IoU failures, and leaked rejected-negative false positives. That violates the project promotion policy.
- Evidence: focused `19` segmentation-miss eval improved from `zero-IoU=19` to `zero-IoU=16` and recovered `20260424_125948_4dc2e7c8` and `20260424_141634_de4679eb` strongly.
- Evidence: full frozen v14 primary summary `runs/supplemental_v7_validation/eval_v14_thinline_primary_frozen_full_noimg_noreport.json` had old `val` Dice `0.8708175273` with `2` zero-IoU, harvest Dice `0.8337888820` with `2` zero-IoU, and rejected negatives `2 / 129` false positives.
- Risks: v14 may still be useful as a tightly gated specialist, but promoting or broadening it would damage the main model. Any v14 rescue must be evaluated first on a focused subset, then on full frozen and supplemental boards.
- Revisit when: a narrow v14 rescue rule can capture the three focused recoveries without firing on frozen positives or rejected negatives, or after a separate candidate-generation rescue reduces the remaining failure board enough to make v14 unnecessary.

### 2026-04-30 - Promote v15b dual colored-blob final-fallback rescue

- Context: after v14 was rejected as a primary segmentation replacement, the next documented bottleneck was the `15` supplemental candidate-generation-miss subset. The promoted stack still had supplemental positive zero-IoU `82` and supplemental negatives `4 / 1159`.
- Options considered: keep the promoted dual-veto manifest unchanged, retarget the single existing colored-blob rescue slot, add list-valued colored-blob rescue support and a second final-fallback branch, or start another segmentation retrain.
- Decision: promote v15b: keep the old v5 colored-blob rescue, add support for multiple `colored_blob_rescue` configs, and add a second tightly gated final-fallback `table_hough` colored-blob rescue selected by a narrow candidate-selector rule.
- Why: candidate-generation diagnostics showed this is a proposal-recall pocket, not a primary segmentation-training problem. A focused smoke proved the colored-blob proposal family can recover one real target, while the broad v15 full frozen failure showed the branch must be tightly gated before promotion.
- Evidence: v15b recovered `20260424_131631_d88ee8c9` at IoU `0.8891966759`; focused miss15 zero-IoU improved from `15` to `14`.
- Evidence: v15b full frozen summary `runs/supplemental_v7_validation/eval_v15b_tight_frozen_full_noimg_noreport.json` passed all gates: old `val` Dice `0.8925961829` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8322030397` zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: v15b full supplemental summary `runs/supplemental_v7_validation/eval_v15b_tight_supplemental_full_noimg_noreport.json` improved positives from Dice `0.7789596845`, IoU `0.6979514474`, zero-IoU `82` to Dice `0.7800920770`, IoU `0.6990240866`, zero-IoU `81`, while negatives stayed `4 / 1159`.
- Evidence: broad v15 was rejected because `runs/supplemental_v7_validation/eval_v15_dual_frozen_full_noimg_noreport.json` leaked `10 / 376` frozen negative false positives, all from `colored_blob`.
- Risks: v15b is a hand-gated candidate-generation branch calibrated from one recovered supplemental row and the broad-v15 false positives. It should not be broadened without new focused and full-board evidence.
- Revisit when: new candidate-generation misses resemble the recovered row, new colored-blob false positives appear, or a learned target-aware selector/candidate generator can replace this hand-gated branch while preserving frozen and supplemental gates.

### 2026-05-05 - Promote v17 high-score reticle selector rescue and reject v16 primary reranker replacements

- Context: after v15b, supplemental positives still had `81` zero-IoU failures while frozen negatives were already saturated at `0 / 376` and supplemental negatives were `4 / 1159`.
- Options considered: promote the v16 primary reranker replacement, add negative-aware reranker training and promote v16b/v16c, add only narrow selector rescue rules derived from v16 diagnostics, or stop and rebuild diagnostics without changing the manifest.
- Decision: promote v17 manifest-only high-score reticle selector rescue rules in `runs/supervised_best_manifest.json` and `scripts/run_supervised_best_wsl.sh`; reject v16, v16b, and v16c as primary reranker replacements.
- Why: v16 and v16b recovered focused zero-IoU positives, but both failed frozen positive gates. v17 keeps only the narrow alternate-candidate pockets that pass full frozen and supplemental validation.
- Evidence: v16 full frozen summary `runs/reranker_v16_primary_aug/eval_v16_primary_reranker_frozen_full_noimg_noreport.json` regressed old `val` to Dice `0.8846850160` with one zero-IoU, regressed harvest to Dice `0.8209254881` with two zero-IoU, and introduced `3 / 376` frozen negative false positives.
- Evidence: v16b full frozen summary `runs/reranker_v16_primary_aug/eval_v16b_primary_reranker_frozen_full_noimg_noreport.json` removed frozen negative false positives but regressed old `val` to Dice `0.8762358358` with one zero-IoU and harvest to Dice `0.8068642883` with two zero-IoU.
- Evidence: v17 full frozen summary `runs/reranker_v16_primary_aug/eval_v17_reticle_highscore_frozen_full_noimg_noreport.json` passes gates: old `val` Dice `0.8925993441`, old `hard_val` Dice `0.9659744406`, harvest Dice `0.8321876397`, all positive zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: v17 full supplemental summary `runs/reranker_v16_primary_aug/eval_v17_reticle_highscore_supplemental_full_noimg_noreport.json` improves positives from Dice `0.7800920770`, IoU `0.6990240866`, zero-IoU `81` to Dice `0.7824357627`, IoU `0.7013098049`, zero-IoU `79`, while negatives stay `4 / 1159`.
- Risks: v17 adds more hand-gated selector policy and only recovers two rows. The next broad selector/model replacement must be justified by a fresh failure-layer diagnosis and must pass the same frozen plus supplemental gates.
- Revisit when: the remaining `79` supplemental zero-IoU rows are reclassified and show a large selector/arbitration pocket that can be handled by a learned final arbiter without old/hard/harvest regressions.

### 2026-05-05 - Prioritize constrained selector/arbitration after v17 diagnostics

- Context: v17 reduced supplemental positive zero-IoU to `79`, but the next action needed failure-layer evidence before more training or hand rules.
- Options considered: start another primary segmentation retrain, broaden proposal generation, continue hand selector rules immediately, or run a focused failure-layer diagnostic first.
- Decision: prioritize selector/arbitration work next, but only as constrained rescue or learned final arbiter evaluated against negatives. Do not replace the primary reranker again without a new design and full gates.
- Why: `runs/reranker_v16_primary_aug/diagnostics_v17_positive_zero79/diagnostics.json` shows `46 / 79` remaining failures are selector/reranker misses, while only `19 / 79` are segmentation misses and `14 / 79` are candidate-generation misses.
- Evidence: the selector-miss best candidates are mostly existing `table_hough`, `white_blob`, and `reticle_global_0` candidates, meaning the model often already produced a good mask but the final policy selected the wrong candidate or rejected it.
- Risks: the previous v16 primary reranker path also targeted selection and failed frozen gates. The next selector work must be rescue-only or heavily constrained, and must be checked against frozen and supplemental negatives before full promotion.
- Revisit when: feature mining shows no safe selector pocket, or a learned final arbiter cannot pass old `val`, old `hard_val`, harvest, frozen negatives, and supplemental negatives.

### 2026-05-06 - Promote v26 pool-only rescue32 selector and reject broad rescue32 variants

- Context: v17 diagnostics showed `46 / 79` remaining supplemental positive zero-IoU rows were selector/reranker misses. The next path needed to expose better candidates without repeating the v16 primary-reranker regressions.
- Options considered: promote broad v19/v20 ResNet primary/refiner work, add a post-external selector pass, widen rescue candidate counts globally, add a pool-only wide rescue32 candidate source with selector gates, or train another blind segmentation model.
- Decision: promote v26 pool-only rescue32 selector in `runs/supervised_best_manifest.json`; reject v19/v20 as broad model replacements, reject v22 as ineffective, reject v23/v24 broad candidate-count increases, and reject v25 as too broad because of a harvest regression.
- Why: v26 recovers three real supplemental positive zero-IoU rows while preserving frozen old `val`, old `hard_val`, harvest, and frozen negatives. It avoids v25's harvest regression by narrowing current-winner gates.
- Evidence: v26 full frozen summary `runs/reranker_v16_primary_aug/eval_v26_pool_rescue32_selector_frozen_full_noimg_noreport.json` has old `val` Dice `0.8925895793` with zero-IoU `0`, old `hard_val` Dice `0.9659744406` with zero-IoU `0`, harvest Dice `0.8321876397` with zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: current-code v17 old-val control `runs/reranker_v16_primary_aug/eval_current_code_v17_old_val_control_noimg_noreport.json` has old `val` Dice `0.8925936242`, so the v26 old-val delta is within observed one-pixel runtime jitter rather than a meaningful regression.
- Evidence: v26 supplemental positives `runs/reranker_v16_primary_aug/eval_v26_pool_rescue32_selector_supplemental_positives_noimg_noreport.json` improve from v17 Dice `0.7824357627` / zero-IoU `79` to Dice `0.7858346061` / zero-IoU `76`.
- Evidence: recovered rows are `20260218_160135_3e2dfcab` IoU `0.9508196721`, `20260424_145618_916011f6` IoU `0.9232954545`, and `20260424_152520_3bb9209d` IoU `0.8428571429`.
- Evidence: v25 full supplemental summary `runs/reranker_v16_primary_aug/eval_v25_pool_rescue32_selector_supplemental_full_noimg_noreport.json` kept supplemental negatives unchanged at `4 / 1159`, and v26 is a strict narrowing of v25 selector activations.
- Risks: v26 adds another hand-gated selector layer and the direct v26 supplemental-negative rerun was stopped as low information gain. The negative-safety argument is valid only while v26 remains a strict subset of v25 selector behavior.
- Revisit when: a direct v26 supplemental-negative rerun is required for audit strictness, new negatives match the v26 selector pool gates, or a learned rescue-only final arbiter can recover more rows with equal or better frozen/supplemental safety.

### 2026-05-08 - Promote v27 narrow selector probe and reject learned rescue-arbiter v1

- Context: after v26 diagnostics, `52 / 76` remaining supplemental positive zero-IoU rows were selector/reranker misses, including `23` where the best candidate was already present in the `wide_rescue32` selector pool.
- Options considered: promote a broad learned rescue arbiter, narrow/gate the learned arbiter, use a safer source-limited learned arbiter, promote the broad hand-mined v27 selector probe, promote only the safe hand-mined v27 subset, or stop and retrain segmentation again.
- Decision: promote the narrowed v27 selector subset in `runs/supervised_best_manifest.json`; reject broad/gated/safer learned rescue-arbiter v1 variants and reject the broad hand-rule selector probe.
- Why: the narrowed hand-mined v27 subset recovers six additional real supplemental positive zero-IoU rows while preserving frozen old `val`, old `hard_val`, harvest, frozen negatives, and full supplemental negative safety. The learned arbiter variants recovered more focused positives but caused unacceptable frozen positive regressions and/or negative false positives.
- Evidence: promoted v27 narrow full frozen summary `runs/reranker_v16_primary_aug/eval_v27_selector_probe_narrow_frozen_full_noimg_noreport.json` has old `val` Dice `0.8925961829` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8321876397` zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: promoted v27 narrow supplemental positives `runs/reranker_v16_primary_aug/eval_v27_selector_probe_narrow_supplemental_positives_noimg_noreport.json` improve from v26 Dice `0.7858346061` / zero-IoU `76` to Dice `0.7928090233` / zero-IoU `70`.
- Evidence: promoted v27 narrow supplemental negatives `runs/reranker_v16_primary_aug/eval_v27_selector_probe_narrow_supplemental_negatives_noimg_noreport.json` stayed at rejected negatives `2 / 377`, flat negatives `2 / 782`, total `4 / 1159`, with `0` new false positives versus the v25/v26 baseline.
- Evidence: broad hand-rule v27 failed frozen gates because `v27_post_external_reticle_to_confident_table_candidate` caused all real frozen regressions; row-level report reruns showed this rule recovered only one focused positive while causing five harvest regressions and one old-val regression.
- Evidence: broad learned arbiter recovered `22 / 76` focused positives but failed frozen gates with old `val` Dice `0.8139166055`, harvest Dice `0.7257041914`, and frozen negatives `14 / 376`; safer learned arbiter removed negative false positives but still failed frozen positive gates with old `val` Dice `0.8870105297` and harvest zero-IoU `1`.
- Risks: v27 adds more hand-gated selector policy and does not solve the remaining `70` zero-IoU positives. The current board is partly failure-mining material, so future independent negatives/positives can still expose overfitting.
- Revisit when: fresh v27 diagnostics show no remaining safe selector pockets, the six recovered rows fail visual audit, or a learned target-aware selector trained with stronger negative/frozen supervision beats v27 on the same gates.

### 2026-05-08 - Add explicit full-manifest harvest mode

- Context: `tools/run_supervised_best_harvest.py` intentionally prioritized high-throughput primary/reranker inference and did not run the full promoted manifest stack, which can be confusing when the user wants the absolute best promoted policy for harvesting.
- Options considered: silently switch harvest to full manifest by default, leave it as fast-only, or add an explicit flag.
- Decision: add `--full-manifest` to `tools/run_supervised_best_harvest.py` and keep the default fast primary/reranker path.
- Why: full manifest is more correct but much slower and more CPU/orchestration-heavy; keeping it explicit avoids surprising long harvest runs while giving a correctness-first option for audit-quality harvests.
- Evidence: `tests/test_harvest.py::test_run_harvest_full_manifest_uses_manifest_stack` verifies that `--full-manifest` routes through `run_manifest_inference` and preserves post-external selector rules; `python tools\run_supervised_best_harvest.py --help` shows the new flag.
- Risks: `--full-manifest` will likely show spiky GPU utilization because much of the promoted stack is CPU-side proposal/rescue/veto orchestration.
- Revisit when: full-manifest harvest throughput becomes a bottleneck; the next improvement should be batching/vectorizing proposal and rescue stages rather than launching duplicate eval processes.

### 2026-05-08 - Promote v28 ultra-tight table selector

- Context: v27 reduced supplemental positive zero-IoU to `70`, and fresh diagnostics showed selector/reranker misses still dominated with `46 / 70` remaining failures.
- Options considered: keep v27 unchanged, reintroduce the rejected broad v27 table selector, promote only an ultra-tight table-rule subset for the one safe recovered row, or start another broad model/reranker training pass.
- Decision: promote `v28_post_external_reticle_to_ultratight_primary_table_candidate` in `runs/supervised_best_manifest.json`; keep the broad `v27_post_external_reticle_to_confident_table_candidate` rejected.
- Why: v28 recovers one additional real supplemental positive zero-IoU row while preserving frozen old `val`, old `hard_val`, harvest, frozen negatives, and supplemental negatives. It is a narrow selector repair, not a broad policy expansion.
- Evidence: full frozen summary `runs/reranker_v16_primary_aug/eval_v28_ultratight_table_frozen_full_noimg_noreport.json` has old `val` Dice `0.8925895793` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8322030397` zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: supplemental positives `runs/reranker_v16_primary_aug/eval_v28_ultratight_table_supplemental_positives_noimg_noreport.json` improve from v27 Dice `0.7928090233` / zero-IoU `70` to Dice `0.7939539329` / zero-IoU `69`, recovering `20260424_132131_08e609ae` at IoU `0.91`.
- Evidence: supplemental negatives `runs/reranker_v16_primary_aug/eval_v28_ultratight_table_supplemental_negatives_noimg_noreport.json` stay unchanged at rejected negatives `2 / 377`, flat negatives `2 / 782`, total `4 / 1159`, with `0` changed negative rows versus v27.
- Evidence: visual audit output `runs/reranker_v16_primary_aug/eval_v28_promotion_visual_audit.json` passed for the recovered v28 row and the old-val one-pixel jitter row.
- Risks: the old-val mean Dice is slightly lower than v27 because one row moved by one predicted pixel; this is recorded as runtime/mask jitter, not a target-definition or policy failure. The supplemental board is failure-mining material rather than a pristine independent test set.
- Revisit when: future independent positives/negatives show new table-rule leakage, or the remaining `69` positive zero-IoU diagnostics show the selector layer no longer dominates.

### 2026-05-14 - Promote v29 post-final selector rescue

- Context: v28 diagnostics showed `45 / 69` remaining supplemental positive zero-IoU rows were selector/reranker misses, with many high-IoU candidates already available but final reject/veto logic returning an empty or wrong mask.
- Options considered: run another topology/thin-line segmentation retrain, train another learned final arbiter, continue post-external selector rules, or add a manifest-gated post-final selector rescue defaulting to final-empty rows only.
- Decision: promote v29 post-final selector rescue in `runs/supervised_best_manifest.json` and archive it at `runs/reranker_v16_primary_aug/supervised_best_manifest_v29_post_final_selector_promoted.json`.
- Why: the repository evidence pointed to late arbitration rather than segmentation as the immediate bottleneck; v29 recovers five high-IoU supplemental positives while preserving frozen old/hard/harvest and negative gates.
- Evidence: focused v28 zero69 rerun recovered `20260424_130035_56e42554` IoU `0.9241`, `20260424_153736_e07672ea` IoU `0.9252`, `20260301_051814_42f0e4d1` IoU `0.9492`, `20260424_131859_5d7ed3e2` IoU `1.0`, and `20260305_131138_60d45a12` IoU `1.0`.
- Evidence: frozen gates are unchanged versus v28: old `val` Dice `0.8925895793` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8322030397` zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: supplemental negatives are unchanged versus v28 at rejected `2 / 377`, flat `2 / 782`, total `4 / 1159`, with the same four false-positive IDs and `0` post-final activations on negatives.
- Evidence: derived supplemental positives improve from v28 Dice `0.7939539329`, IoU `0.7124442188`, zero-IoU `69` to v29 Dice `0.7997754756`, IoU `0.7182325471`, zero-IoU `64`.
- Evidence: visual overlay review of the five recovered rows passed; masks are on the object-ball outgoing guideline, not cue-ball aiming/long lines.
- Risks: this is a hand-gated policy layer after final veto/reject logic. The full supplemental-positive metric is derived from v28 full positives plus the focused zero69 rerun, not a fresh 829-row full rerun. Future changes to candidate metadata, source names, veto ordering, or score distributions require full gate revalidation.
- Revisit when: v29 diagnostics show selector misses are no longer dominant, new negatives trigger post-final activations, or a learned target-aware selector can replace the hand-gated rescue stack without frozen/supplemental regression.

### 2026-05-14 - Promote v30a final-empty post-final selector extension

- Context: v29 diagnostics left `64` supplemental positive zero-IoU rows, with `40` selector/reranker misses. A manual caveat suggested some high-IoU diagnostic candidates might be unavailable in deployed inference, so candidate availability needed to be checked before more rules.
- Options considered: pivot directly to candidate-generation fixes, add more post-final selector rules from the existing wide diagnostic, rerun deployed-like diagnostics with primary candidate count `10`, or train another segmentation/reranker model.
- Decision: rerun a deployed-like diagnostic, audit candidate availability, then promote only five additional final-empty post-final selector rules as v30a.
- Why: the availability audit showed the top selector misses were mostly real deployed-path candidate-selection failures, not diagnostic-only artifacts; the five final-empty rules recovered additional true positives while all frozen and supplemental negative gates stayed unchanged.
- Evidence: `runs/reranker_v16_primary_aug/audit_v29_candidate_availability_primary10.json` found `25` exact high-IoU deployed candidates, `11` equivalent-geometry candidates in a different pool, `2` similar geometry/crop-ID-shift candidates, and only `2` present-but-low-IoU rows among the `40` audited selector misses.
- Evidence: focused gate `runs/reranker_v16_primary_aug/eval_v30a_post_final_probe_positive_zero64_summary.json` recovered five rows and reduced focused positive zero-IoU from `64` to `59`.
- Evidence: v30a derived supplemental positives improved to Dice `0.8055778142`, IoU `0.7239287264`, zero-IoU `59`.
- Evidence: frozen gates stayed unchanged versus v29: old `val` Dice `0.8925895793` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8322030397` zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: supplemental negatives stayed unchanged at `4 / 1159`, with the same four false-positive IDs and no new false positives.
- Evidence: visual audit of the five recovered overlays passed; the masks land on the object-ball outgoing guideline.
- Risks: v30a is still hand-gated arbitration and not a better segmentation checkpoint. The remaining nonzero-final selector misses are higher risk because promoting them would override existing nonempty predictions, not just restore zeroed rows. The supplemental-positive metric is derived, not a fresh full rerun.
- Revisit when: a fresh v30a diagnostic shows selector misses are no longer dominant, nonzero-final replacement rules can be proven safe on frozen/supplemental negatives, or a learned target-aware arbiter can replace the hand-gated post-final rules without regressions.

### 2026-05-16 - Promote v31a high-fill post-final white-blob rescue

- Context: v30a reduced supplemental positive zero-IoU to `59`, and fresh v30a diagnostics showed `35 / 59` remaining rows were still selector/reranker misses. Most selector misses were final-empty after veto/reject logic, not segmentation-only failures.
- Options considered: add broad `allow_nonzero_final_mask` post-final rules, train another segmentation checkpoint, train a learned final arbiter immediately, or promote only the narrow high-fill final-empty `white_blob` pockets that passed all gates.
- Decision: promote v31a by adding five final-empty-only high-fill `white_blob` post-final selector rules in `runs/supervised_best_manifest.json`; archive it at `runs/reranker_v16_primary_aug/supervised_best_manifest_v31a_highfill_postfinal_promoted.json`.
- Why: v31a recovers eight real supplemental positives while preserving old `val`, old `hard_val`, harvest, frozen negatives, and supplemental negatives. It avoids the riskier nonzero-final override path.
- Evidence: focused v30a-zero59 rerun recovered eight rows, reducing focused zero-IoU from `59` to `51` with no focused regressions.
- Evidence: derived supplemental positives improved to Dice `0.8144428438`, IoU `0.7321579848`, zero-IoU `51`.
- Evidence: frozen gates passed with old `val` Dice `0.8925961829` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8322030397` zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: supplemental negatives stayed at `4 / 1159`, with the same four false-positive IDs as v30a and `0` post-final activations on negatives.
- Evidence: visual audit contact sheet `runs/reranker_v16_primary_aug/eval_v31a_recovered8_visual_contact_sheet.png` passed; predictions are on object-ball outgoing guideline branches.
- Risks: this is still hand-gated arbitration, not a new segmentation model. The full supplemental-positive metric is derived from the v30a full derived summary plus the focused zero59 rerun. Future broadening to nonzero-final masks could bypass false-positive protections and must rerun all gates.
- Revisit when: v31a diagnostics show final-empty selector misses no longer dominate, new negatives trigger high-fill white-blob post-final activations, or a learned target-aware final arbiter passes the same frozen and supplemental gates.

### 2026-05-16 - Promote v32d gated line-endpoint selector rescue

- Context: after v31a, supplemental positives still had `51` zero-IoU rows. Fresh diagnostics showed `27` selector/reranker misses, `10` segmentation misses, and `14` candidate-generation misses, so pure post-final selector rules had diminishing returns and proposal coverage needed to be tested.
- Options considered: continue table-only post-final selector rules, add an unconditional line-endpoint proposal pool, add a gated line-endpoint proposal pool, train another broad segmentation checkpoint, or attempt broad nonzero-final replacement rules.
- Decision: promote v32d in `runs/supervised_best_manifest.json`; reject v32a table-only as too weak and reject v32c unconditional endpoint pool as too slow for the promoted shape.
- Why: v32d recovers three additional real supplemental positive zero-IoU rows using endpoint proposals, while preserving frozen old `val`, old `hard_val`, harvest, frozen negatives, and supplemental negatives. It also gates endpoint-pool generation before proposal to avoid the heavy v32c throughput cost.
- Evidence: passive v32b endpoint diagnostics reduced candidate-generation misses from `14` to `9`, proving a proposal-coverage bottleneck existed.
- Evidence: focused v32d positive-zero51 summary `runs/reranker_v16_primary_aug/eval_v32d_gated_line_endpoint_selector_probe_positive_zero51_summary.json` recovered `20260424_134409_803f3fe9` IoU `0.9442`, `20260424_134503_7f76c16d` IoU `0.9276`, and `20260424_152717_cf625b2b` IoU `0.9780`.
- Evidence: derived supplemental positives improved from v31a Dice `0.8144428438`, IoU `0.7321579848`, zero-IoU `51` to v32d Dice `0.8179443197`, IoU `0.7355956771`, zero-IoU `48`.
- Evidence: frozen gates passed in `runs/reranker_v16_primary_aug/eval_v32d_gated_line_endpoint_selector_probe_frozen_summary.json`: old `val` Dice `0.8925895793` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8321876397` zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: supplemental negatives stayed at `4 / 1159` in `runs/reranker_v16_primary_aug/eval_v32d_gated_line_endpoint_selector_probe_supplemental_negatives_merged_summary.json`, with the same four false-positive IDs and `0` endpoint/post-final activations on negatives.
- Evidence: visual review of `runs/reranker_v16_primary_aug/eval_v32c_line_endpoint_recovered3_visual/v32c_line_endpoint_recovered3/*/overlay_final.png` passed; recovered masks are on the object-ball outgoing guideline.
- Risks: this adds another hand-gated proposal/arbitration branch and the supplemental-positive metric is derived. Future endpoint-pool broadening must rerun frozen and supplemental negatives and account for throughput.
- Revisit when: v32d diagnostics show no more endpoint-proposal gain, endpoint candidates activate on negatives, or a learned final arbiter can recover the same cases with lower policy complexity.

### 2026-05-16 - Reject v33a tiny endpoint probe

- Context: after v32d promotion, `runs/reranker_v16_primary_aug/diagnostics_v32d_positive_zero48_primary10/diagnostics.json` still showed one tiny endpoint-backed final-empty-looking candidate.
- Decision: reject `runs/reranker_v16_primary_aug/supervised_best_manifest_v33a_tiny_endpoint_probe.json`; do not promote and do not run full gates.
- Why: the focused deployed-path eval did not activate the rule and recovered `0 / 48` rows, so the expected benefit is zero.
- Evidence: `runs/reranker_v16_primary_aug/eval_v33a_tiny_endpoint_probe_positive_zero48_summary.json` has focused zero-IoU unchanged at `48`, no `line_endpoint` winner rows, and no post-final activations.
- Revisit when: a future diagnostic or report-level audit shows an endpoint candidate is actually available in the deployed path with stable features and worth a new gate.

### 2026-05-17 - Promote v35a as current best and continue from v35a zero41 diagnostics

- Context: after v32d, several small selector/pool probes reduced supplemental positive zero-IoU further without changing the canonical target definition.
- Decision: treat `runs/supervised_best_manifest.json` with `candidate_name = v35a_tiny_final_empty_selector_probe` as the active current best.
- Why: v35a preserves old `val`, old `hard_val`, harvest holdout, frozen-negative, and supplemental-negative safety while reducing supplemental positive zero-IoU to `41`.
- Evidence: active manifest metrics are old `val` Dice `0.8925841164` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8321876397` zero-IoU `0`, frozen negatives `0 / 376`, supplemental positives Dice `0.8255884069` zero-IoU `41`, and supplemental negatives `4 / 1159`.
- Evidence: v34f recovered three late colored-blob rows after adding post-external selector candidate-pool support; v35a recovered two tiny final-empty rows; visual review passed for both recovery groups.
- Rejected alternatives: broad v34d was too slow for full-board promotion, v34e candidate-cap variants lost intended recoveries, and v33a endpoint follow-up recovered `0 / 48`.
- Risks: v35a continues the hand-gated policy stack and uses derived supplemental-positive metrics. Future promotion still requires focused-positive, frozen-positive, frozen-negative, supplemental-negative, and visual gates.
- Revisit when: v35a zero41 diagnostics show no safe selector pocket, or a targeted model/proposal experiment beats v35a without regressing old/hard/harvest and negatives.

### 2026-05-17 - Promote v36c/v37c/v38a narrow exemptions and rescues

- Context: after v35a, focused supplemental positives still had `41` zero-IoU rows. Several rows had valid object-ball outgoing masks that were being removed by source-min/image-veto, post-external replacement, or source area+score reject policy.
- Decision: promote v36c, v37c, and v38a as narrow inference-policy improvements in sequence.
- Why: each step recovered real object-ball outgoing guideline masks and preserved old `val`, old `hard_val`, harvest holdout, frozen negatives, and supplemental negatives.
- Evidence: v36c recovered two rows and improved derived supplemental positives to Dice `0.8277632917`, IoU `0.7448041005`, zero-IoU `39`.
- Evidence: v37c recovered two rows and improved derived supplemental positives to Dice `0.8298134255`, IoU `0.7465911683`, zero-IoU `37`.
- Evidence: v38a recovered one row and improved derived supplemental positives to Dice `0.8308295076`, IoU `0.7474712550`, zero-IoU `36`.
- Rejected alternatives: broadening final/nonzero replacement was not used; only rows with tight feature pockets and clean affected/frozen/negative evidence were promoted.
- Risks: these remain hand-gated inference-policy changes, not a stronger primary segmentation model. Derived supplemental-positive metrics are not fresh full 829-row reruns.
- Revisit when: new negatives match these exemption feature pockets, or a target-aware learned selector can replace the hand-gated policy stack without regressions.

### 2026-05-17 - Promote v39b secondary-rescue current exemption; reject v39a post-final rescue

- Context: v38a diagnostics showed remaining failures split across `17` selector/reranker misses, `10` segmentation misses, and `9` candidate-generation misses. One valid object-line case was lost because secondary rescue replaced a good current candidate before post-final rescue could recover it.
- Decision: add `secondary_rescue_current_exemptions` infrastructure and promote `v39b_secondary_rescue_current_exemption_probe` as the active manifest. Reject v39a post-final rescue.
- Why: v39a did not activate because the desired candidate was no longer selectable after secondary rescue. v39b fixes the actual failure layer by skipping secondary rescue only for a tightly matched current table-hough object-line candidate.
- Evidence: v39b recovered `20260305_125227_51802921` at IoU `0.7965738758` and Dice `0.8869047619`.
- Evidence: v39b derived supplemental positives improved to Dice `0.8318964000`, IoU `0.7484321402`, zero-IoU `35`.
- Evidence: full frozen gate passed with old `val` Dice `0.8926019028`, old `hard_val` Dice `0.9659744406`, harvest Dice `0.8322030397`, all positive zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: full supplemental-negative gate stayed `4 / 1159` false positives with the same source pattern as before.
- Evidence: visual overlay for the recovered row is on the object-ball outgoing guideline, not the long cue-ball aiming line.
- Risks: this adds another hand-gated stage-specific policy. Because it changes behavior before secondary rescue, future edits to rescue ordering or score distributions require rerunning frozen and supplemental-negative gates.
- Revisit when: v39b remaining-zero diagnostics show selector misses are no longer dominant, or secondary rescue exemptions start matching independent negatives.

### 2026-05-20 - Promote v40c tiny endpoint micro-line selector

- Context: v39b left `35` supplemental positive zero-IoU rows. Saved diagnostics and visual review showed a small final-empty pocket where the v10 micro-line checkpoint had valid tiny `line_endpoint` candidates, but the promoted stack returned an empty mask.
- Options considered: keep v39b, simply increase primary candidate count, add an ungated tiny endpoint pool, promote v40b broad final-fallback endpoint rules, promote v40c source-specific final-empty endpoint rules, or pivot immediately to model training.
- Decision: promote v40c in `runs/supervised_best_manifest.json` and archive it as `runs/reranker_v16_primary_aug/supervised_best_manifest_v40c_tiny_endpoint_microline_selector_promoted.json`.
- Why: v40c recovers four additional real supplemental positives, reduces derived supplemental zero-IoU from `35` to `31`, keeps frozen positives and negatives safe, and keeps supplemental negatives exactly unchanged at `4 / 1159`.
- Evidence: focused v40c summary `runs/reranker_v16_primary_aug/eval_v40c_tiny_endpoint_microline_selector_probe_positive_zero35_summary.json` recovered `20260305_044126_5699db70`, `20260424_152258_dd5dc70f`, `20260424_152314_45165266`, and `20260424_153026_bb0c9fd8` with IoU `0.8462`, `0.7000`, `0.8571`, and `0.7692`.
- Evidence: derived supplemental positives improved to Dice `0.8358341480`, IoU `0.7522590733`, zero-IoU `31`.
- Evidence: frozen gate passed with old `val` Dice `0.8925961829` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8321876397` zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: supplemental-negative merged summary `runs/reranker_v16_primary_aug/eval_v40c_tiny_endpoint_microline_selector_probe_supplemental_negatives_merged_summary.json` stayed at `4 / 1159` false positives with exactly the same four IDs and source pattern as v39b.
- Evidence: visual zoom sheet `runs/reranker_v16_primary_aug/eval_v40b_tiny_endpoint_microline_selector_recovered4_zoom_sheet.png` passed target-rule review for the four recovered rows.
- Risks: this is another hand-gated proposal/arbitration improvement, not a stronger primary segmentation model. The old-val Dice is a few millionths below the rounded `0.8926` gate because of one unrelated one-pixel external-table-broad jitter row; v40c rules did not activate on frozen rows. The recovered masks are extremely tiny, so broadening this logic could leak false positives.
- Revisit when: the remaining v40c zero31 diagnostic shows no further final-empty endpoint pocket, endpoint rules activate on independent negatives, or a target-aware learned selector/proposal model can replace the hand-gated endpoint rules while preserving all gates.

### 2026-05-20 - Promote v41a final-empty selector repair

- Context: after v40c, fresh diagnostics on the remaining `31` supplemental positive zero-IoU rows still showed `15` selector/reranker misses, with `8` final-empty cases and `7` nonzero-final wrong-mask cases.
- Options considered: stop at v40c, add final-empty-only exact selector repairs, attempt broad nonzero-final replacement, or pivot immediately to training/proposal work.
- Decision: promote v41a final-empty-only selector repairs in `runs/supervised_best_manifest.json`; continue to reject broad nonzero-final replacement.
- Why: v41a recovered seven real positive rows, preserved old `val`, old `hard_val`, harvest, frozen negatives, and supplemental negatives, and caused zero row-level decision differences on supplemental negatives versus v40c.
- Evidence: focused v41a summary `runs/reranker_v16_primary_aug/eval_v41a_final_empty_selector_probe_positive_zero31_summary.json` reduced zero-IoU from `31` to `24`.
- Evidence: derived supplemental positives improved to Dice `0.8426522273`, IoU `0.7580968798`, zero-IoU `24`.
- Evidence: frozen gate passed with old `val` Dice `0.8925936242` zero-IoU `0`, old `hard_val` Dice `0.9659744406` zero-IoU `0`, harvest Dice `0.8322030397` zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: supplemental-negative merged summary `runs/reranker_v16_primary_aug/eval_v41a_final_empty_selector_probe_supplemental_negatives_merged_summary.json` stayed at `4 / 1159`, with the same four IDs/source pattern and no row-level decision differences versus v40c.
- Evidence: visual zoom sheet `runs/reranker_v16_primary_aug/eval_v41a_final_empty_selector_probe_recovered7_zoom_sheet.png` passed target-rule review; recovered masks are on object-ball outgoing guidelines, not cue-ball aiming lines.
- Risks: v41a deepens the hand-gated inference-policy stack and includes two lower-IoU partial recoveries. Remaining nonzero-final selector misses, segmentation misses, and candidate-generation misses need a new diagnostic before more rules.
- Revisit when: v41a remaining-zero diagnostics show selector misses are no longer dominant, new negatives activate v41a rules, or a target-aware learned selector/proposal model can replace this hand-gated branch with equal safety.

### 2026-05-21 - Promote v42d tight specialist post-final recovery

- Context: after v41a, the remaining `24` supplemental positive zero-IoU rows were no longer selector-dominant: diagnostics showed `8` selector/reranker misses, `10` segmentation misses, and `6` candidate-generation misses. Most selector misses were nonzero-final wrong-mask cases, which are higher risk than final-empty repairs.
- Options considered: continue final-empty hand selector rules, attempt broad nonzero-final replacement, train a new primary segmentation checkpoint, run passive specialist/proposal diagnostics with existing checkpoints, or stop at v41a.
- Decision: promote v42d in `runs/supervised_best_manifest.json`; reject broad v42b/v42c-style specialist pools as too expensive/risky for full-board promotion, and continue rejecting broad nonzero-final replacement.
- Why: passive v42a proved existing v10/v14/v18 specialist checkpoints can expose better candidate masks for many segmentation/candidate-generation misses. v42d narrows that gain into row-state-gated specialist pools and final-empty post-final rules that pass frozen and supplemental-negative gates.
- Evidence: focused v42d summary `runs/reranker_v16_primary_aug/eval_v42d_tight_specialist_postfinal_probe_positive_zero24_summary.json` reduced focused zero-IoU from `24` to `13`, recovering `11` real positives.
- Evidence: derived supplemental positives improved from v41a Dice `0.8426522273`, IoU `0.7580968798`, zero-IoU `24` to v42d Dice `0.8534702372`, IoU `0.7677883450`, zero-IoU `13`.
- Evidence: frozen summary `runs/reranker_v16_primary_aug/eval_v42d_tight_specialist_postfinal_probe_frozen_summary.json` passed: old `val` Dice `0.8925936242`, old `hard_val` Dice `0.9659744406`, harvest Dice `0.8321876397`, all positive zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: supplemental negatives summary `runs/reranker_v16_primary_aug/eval_v42d_tight_specialist_postfinal_probe_supplemental_negatives_merged_summary.json` stayed at `4 / 1159` false positives with the same four known IDs/source pattern as v41a and no post-final specialist activations.
- Evidence: visual zoom sheet `runs/reranker_v16_primary_aug/eval_v42b_specialist_postfinal_probe_recovered11_zoom_sheet.png` passed target-rule review; recovered masks are on object-ball outgoing guideline branches. Lower-IoU rows are partial/over-thick but target-aligned.
- Risks: v42d adds more hand-gated proposal/arbitration policy and uses derived supplemental-positive metrics. It is not a new primary checkpoint. Future edits to current-score windows, pool gates, source names, or checkpoint thresholds require rerunning frozen and supplemental-negative gates.
- Revisit when: v42d remaining-zero diagnostics show whether the final `13` rows are proposal misses, segmentation-quality misses, or nonzero wrong-mask selector misses; if nonzero selector dominates, use a target-aware arbiter rather than broad manual replacement.

### 2026-05-21 - Promote v43a nonzero selector recovery

- Context: after v42d, supplemental positives still had `13` zero-IoU rows. Diagnostics showed `7` selector/reranker misses, `4` segmentation misses, and `2` candidate-generation misses. Some selector misses had excellent candidate masks, but the final output was a nonempty wrong prediction.
- Options considered: stop at v42d, promote broad nonzero-final replacement, train another primary model, add ultra-tight nonzero-final selector rules for only high-confidence row-state pockets, or pivot immediately to proposal/model work.
- Decision: promote v43a in `runs/supervised_best_manifest.json`; continue rejecting broad nonzero-final replacement.
- Why: v43a recovers five real supplemental positives while preserving old `val`, old `hard_val`, harvest, frozen negatives, and supplemental negatives. The residual diagnostic after promotion shows selector is no longer dominant, so further work should pivot toward proposal/model fixes.
- Evidence: focused v43a summary `runs/reranker_v16_primary_aug/eval_v43a_nonzero_selector_probe_positive_zero13_summary.json` reduced focused zero-IoU from `13` to `8`.
- Evidence: derived supplemental positives improved from v42d Dice `0.8534702372`, IoU `0.7677883450`, zero-IoU `13` to v43a Dice `0.8591970180`, IoU `0.7732721692`, zero-IoU `8`.
- Evidence: frozen summary `runs/reranker_v16_primary_aug/eval_v43a_nonzero_selector_probe_frozen_summary.json` passed: old `val` Dice `0.8925895793`, old `hard_val` Dice `0.9659744406`, harvest Dice `0.8322030397`, all positive zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: supplemental negatives summary `runs/reranker_v16_primary_aug/eval_v43a_nonzero_selector_probe_supplemental_negatives_merged_summary.json` stayed at `4 / 1159` false positives with the same known source pattern as v42d and no post-final selector activations.
- Evidence: visual zoom sheet `runs/reranker_v16_primary_aug/eval_v43a_nonzero_selector_probe_recovered5_zoom_sheet.png` passed target-rule review; recovered masks are on object-ball outgoing guideline fragments.
- Risks: v43a is still hand-gated arbitration and uses derived supplemental-positive metrics. Nonzero-final replacement is inherently higher risk than final-empty repair, so future expansion must be narrow and fully gated.
- Revisit when: v43a zero8 residual diagnostics show a safe ultra-tight selector pocket worth the full gate, or a proposal/model experiment recovers the segmentation and candidate-generation misses without negative regressions.

### 2026-05-22 - Reject v46 context-positive segmentation specialist

- Context: after v44d, the remaining supplemental-positive board had four zero-IoU rows. A new context-positive training variant was tried to see whether a stronger primary/crop model could solve the residual tiny and near-rail misses.
- Options considered: promote v46 as a primary specialist, keep it only as a rescue source, continue the same training recipe longer, or reject it and diagnose proposal/candidate-generation instead.
- Decision: reject v46 as a primary replacement and do not continue the same recipe blindly.
- Why: v46 did not recover the focused zero4 board and its crop validation metrics were much weaker than the promoted stack's end-to-end gates. The failure layer remained candidate generation/proposal, not simply crop segmentation quality.
- Evidence: `runs/supervised_train_v46_seg_specialist_v10_512_lw3_nw1_abs02_lr5e-5_pw24_ctx/guideline_unet_best.pt` selected epoch 11 with `val_dice = 0.7938735916` and `hard_val_dice = 0.6774799590`.
- Evidence: focused v46 eval `runs/reranker_v16_primary_aug/eval_v46_seg_specialist_primary_zero4_summary.json` had mean IoU `0.0`, mean Dice `0.0068914273`, and positive zero-IoU `4`.
- Evidence: v46 diagnostics `runs/reranker_v16_primary_aug/diagnostics_v46_seg_specialist_primary_zero4/diagnostics.json` classified all four rows as candidate-generation misses.
- Risks: the v46 code support for context positives remains in the repo and could be useful later, but the checkpoint itself should not be treated as current best without a new experiment and full gates.
- Revisit when: a future dataset/split proves context-positive crops improve segmentation without hurting old/hard validation, or if a gated rescue branch needs a specialist checkpoint and passes focused, frozen, supplemental-negative, and visual gates.

### 2026-05-22 - Promote v47b ROI-expanded endpoint selector

- Context: v44d left four supplemental-positive zero-IoU rows. Two near-rail rows were candidate-generation misses because line-endpoint proposal search was clipped by the table ROI, and two tiny endpoint rows already had safe endpoint candidates under the existing specialist path.
- Options considered: continue v46 training, broaden generic candidate generation, expand the table ROI globally, add line-endpoint-only ROI expansion with strict row-state gates, or stop at v44d.
- Decision: promote `v47b_roi_expand_selector_probe` in `runs/supervised_best_manifest.json` and archive it as `runs/reranker_v16_primary_aug/supervised_best_manifest_v47b_roi_expand_selector_promoted.json`.
- Why: v47b recovers all four remaining supplemental-positive zero-IoU rows, preserves all frozen positive/negative gates, and does not create any new supplemental-negative false positives or post-final activations on negatives.
- Evidence: focused summary `runs/reranker_v16_primary_aug/eval_v47b_roi_expand_selector_probe_positive_zero4_summary.json` reduced positive zero-IoU `4 -> 0` and recovered all rows with IoU `0.7632`, `0.7778`, `0.8500`, and `0.8500`.
- Evidence: derived supplemental positives `runs/reranker_v16_primary_aug/eval_v47b_roi_expand_selector_probe_supplemental_positives_derived_summary.json` improved Dice from v44d `0.8636358822` to `0.8677956623`, improved IoU from `0.7775225759` to `0.7814320279`, and reduced zero-IoU `4 -> 0`.
- Evidence: frozen summary `runs/reranker_v16_primary_aug/eval_v47b_roi_expand_selector_probe_frozen_summary.json` passed with old `val` Dice `0.8925907200`, old `hard_val` Dice `0.9659744406`, harvest Dice `0.8322030397`, all positive zero-IoU `0`, and frozen negatives `0 / 376`.
- Evidence: supplemental negatives `runs/reranker_v16_primary_aug/eval_v47b_roi_expand_selector_probe_supplemental_negatives_merged_summary.json` stayed at `4 / 1159`, with exactly the same four false-positive IDs/source pattern as v44d and zero v47b post-final selector activations on negatives.
- Evidence: visual zoom sheet `runs/reranker_v16_primary_aug/eval_v47b_roi_expand_selector_probe_recovered4_zoom_sheet.png` passed target-rule review; recovered masks are on object-ball outgoing guideline fragments, not cue-ball aiming lines.
- Risks: v47b is another hand-gated inference/proposal repair, not a stronger primary segmentation model. The line-endpoint ROI expansion must stay tightly scoped and gated; broad ROI expansion or non-row-specific endpoint selection could leak wrong white lines on negatives.
- Revisit when: fresh independent positives expose new missed target lines near table boundaries, supplemental negatives activate v47b rules, or a target-aware learned selector/proposal model can replace the hand-gated endpoint stack with equal or better safety.
