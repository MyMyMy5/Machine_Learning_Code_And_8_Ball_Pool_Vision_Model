#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.wsl_envs/guideline_train/bin/python"
MANIFEST_PATH="$REPO_ROOT/runs/supervised_best_manifest.json"
PRIMARY_CHECKPOINT="$REPO_ROOT/runs/supervised_train_v2/guideline_unet_best.pt"
PRIMARY_RERANKER="$REPO_ROOT/runs/supervised_train_v2/guideline_reranker_best.pt"
FALLBACK_CHECKPOINT="$REPO_ROOT/runs/supervised_train_v5_negft/guideline_unet_best.pt"
FALLBACK_RERANKER="$REPO_ROOT/runs/supervised_train_v5_negft/guideline_reranker_best.pt"
FALLBACK_SCORE_THRESHOLD="3.2727272727272734"
RESCUE_CHECKPOINT="$REPO_ROOT/runs/supervised_train_v6_residual/guideline_unet_best.pt"
RESCUE_RERANKER="$REPO_ROOT/runs/supervised_train_v6_residual/guideline_reranker_best.pt"
RESCUE_SCORE_THRESHOLD="1.25"
RESCUE_MAX_BALL_CANDIDATES="20"
FINAL_FALLBACK_SCORE_THRESHOLD="-0.10084033613445342"
SECONDARY_RESCUE_CHECKPOINT="$REPO_ROOT/runs/supervised_train_v6_residual/guideline_unet_best.pt"
SECONDARY_RESCUE_RERANKER="$REPO_ROOT/runs/supervised_train_v6_residual/guideline_reranker_best.pt"
SECONDARY_RESCUE_MIN_SCORE="1.75"
SECONDARY_RESCUE_MAX_SCORE="4.40625"
SECONDARY_RESCUE_SOURCE="table_hough"
SECONDARY_RESCUE_MIN_BALL_FILL_FRACTION="0.9"
SECONDARY_RESCUE_MAX_BALL_CANDIDATES="24"
PRIMARY_RECOVERY_MIN_SCORE="1.5"
PRIMARY_RECOVERY_MAX_SCORE="3.0"
PRIMARY_RECOVERY_SOURCE="table_hough"
PRIMARY_RECOVERY_CANDIDATE_SOURCE="table_hough"
PRIMARY_RECOVERY_MIN_BALL_FILL_FRACTION="0.5"
RETICLE_RECOVERY_MIN_SCORE="0.5"
RETICLE_RECOVERY_MAX_SCORE="2.0"
RETICLE_RECOVERY_MIN_PRIMARY_SCORE="1.5"
FALLBACK_RESCUE_CHECKPOINT="$REPO_ROOT/runs/supervised_train_v7_bestfinal5_residual/guideline_unet_best.pt"
FALLBACK_RESCUE_RERANKER="$REPO_ROOT/runs/supervised_train_v6_residual/guideline_reranker_best.pt"
FALLBACK_RESCUE_MIN_SCORE="0.0"
FALLBACK_RESCUE_MAX_SCORE="1.0"
FALLBACK_RESCUE_SOURCE_GROUP="table_hough"
FALLBACK_RESCUE_MAX_BALL_CANDIDATES="20"
RETICLE_FILL_RESCUE_CHECKPOINT="$REPO_ROOT/runs/supervised_train_v7_bestfinal5_residual/guideline_unet_best.pt"
RETICLE_FILL_RESCUE_RERANKER="$REPO_ROOT/runs/supervised_train_v7_bestfinal5_residual/guideline_reranker_b.pt"
RETICLE_FILL_RESCUE_MIN_SCORE="1.0"
RETICLE_FILL_RESCUE_MAX_SCORE="2.0"
RETICLE_FILL_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION="0.5"
RETICLE_FILL_RESCUE_TARGET_MIN_BALL_FILL_FRACTION="0.9"
RETICLE_FILL_RESCUE_MAX_BALL_CANDIDATES="20"
EXTERNAL_RESCUE_CHECKPOINT="/mnt/c/My_Project/SAM_3/guideline_line/runs_cv/ft_after_clean2000_softdistance_v1/best.pt"
EXTERNAL_RESCUE_SOURCE_GROUP="table_hough"
EXTERNAL_RESCUE_MIN_SCORE="-999.0"
EXTERNAL_RESCUE_MAX_SCORE="0.0"
EXTERNAL_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION="1.1"
EXTERNAL_RESCUE_MIN_PRED_AREA="20"
EXTERNAL_RESCUE_MIN_PROB_MEAN="0.9"
EXTERNAL_MID_RESCUE_CHECKPOINT="/mnt/c/My_Project/SAM_3/guideline_line/runs_cv/ft_after_clean2000_softdistance_v1/best.pt"
EXTERNAL_MID_RESCUE_SOURCE_GROUP="table_hough"
EXTERNAL_MID_RESCUE_MIN_SCORE="1.0"
EXTERNAL_MID_RESCUE_MAX_SCORE="3.0"
EXTERNAL_MID_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION="0.35"
EXTERNAL_MID_RESCUE_MIN_PRED_AREA="120"
EXTERNAL_MID_RESCUE_MIN_PROB_MEAN="0.88"
EXTERNAL_LOWMID_RESCUE_CHECKPOINT="/mnt/c/My_Project/SAM_3/guideline_line/runs_cv/ft_after_clean2000_softdistance_v1/best.pt"
EXTERNAL_LOWMID_RESCUE_SOURCE_GROUP="table_hough"
EXTERNAL_LOWMID_RESCUE_MIN_SCORE="0.5"
EXTERNAL_LOWMID_RESCUE_MAX_SCORE="1.5"
EXTERNAL_LOWMID_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION="0.5"
EXTERNAL_LOWMID_RESCUE_MIN_PRED_AREA="400"
EXTERNAL_LOWMID_RESCUE_MIN_PROB_MEAN="0.8"
EXTERNAL_RETICLE_RESCUE_CHECKPOINT="/mnt/c/My_Project/SAM_3/guideline_line/runs_cv/ft_after_clean2000_softdistance_v1/best.pt"
EXTERNAL_RETICLE_RESCUE_SOURCE_GROUP="reticle_global"
EXTERNAL_RETICLE_RESCUE_MIN_SCORE="-999.0"
EXTERNAL_RETICLE_RESCUE_MAX_SCORE="2.0"
EXTERNAL_RETICLE_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION="0.0"
EXTERNAL_RETICLE_RESCUE_MIN_PRED_AREA="20"
EXTERNAL_RETICLE_RESCUE_MIN_PROB_MEAN="0.8"
EXTERNAL_TINY_RETICLE_RESCUE_CHECKPOINT="/mnt/c/My_Project/SAM_3/guideline_line/runs_cv/ft_balanced_ultratiny_replay_v1/best.pt"
EXTERNAL_TINY_RETICLE_RESCUE_SOURCE_GROUP="reticle_global"
EXTERNAL_TINY_RETICLE_RESCUE_MIN_SCORE="0.5"
EXTERNAL_TINY_RETICLE_RESCUE_MAX_SCORE="1.5"
EXTERNAL_TINY_RETICLE_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION="1.1"
EXTERNAL_TINY_RETICLE_RESCUE_MIN_PRED_AREA="50"
EXTERNAL_TINY_RETICLE_RESCUE_MIN_PROB_MEAN="0.94"
EXTERNAL_RETICLE_LARGE_RESCUE_CHECKPOINT="/mnt/c/My_Project/SAM_3/guideline_line/runs_cv/ft_after_clean2000_softdistance_v1/best.pt"
EXTERNAL_RETICLE_LARGE_RESCUE_SOURCE_GROUP="reticle_global"
EXTERNAL_RETICLE_LARGE_RESCUE_MIN_SCORE="-999.0"
EXTERNAL_RETICLE_LARGE_RESCUE_MAX_SCORE="999.0"
EXTERNAL_RETICLE_LARGE_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION="0.35"
EXTERNAL_RETICLE_LARGE_RESCUE_MIN_PRED_AREA="400"
EXTERNAL_RETICLE_LARGE_RESCUE_MIN_PROB_MEAN="0.86"
EXTERNAL_TABLE_BROAD_RESCUE_CHECKPOINT="/mnt/c/My_Project/SAM_3/guideline_line/runs_cv/ft_after_clean2000_softdistance_v1/best.pt"
EXTERNAL_TABLE_BROAD_RESCUE_SOURCE_GROUP="table_hough"
EXTERNAL_TABLE_BROAD_RESCUE_MIN_SCORE="-999.0"
EXTERNAL_TABLE_BROAD_RESCUE_MAX_SCORE="6.0"
EXTERNAL_TABLE_BROAD_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION="0.35"
EXTERNAL_TABLE_BROAD_RESCUE_MIN_PRED_AREA="350"
EXTERNAL_TABLE_BROAD_RESCUE_MIN_PROB_MEAN="0.8"
EXTERNAL_TABLE_RESCUE_CHECKPOINT="/mnt/c/My_Project/SAM_3/guideline_line/runs_cv/ft_after_clean2000_softdistance_v1/best.pt"
EXTERNAL_TABLE_RESCUE_SOURCE_GROUP="table_hough"
EXTERNAL_TABLE_RESCUE_MIN_SCORE="1.0"
EXTERNAL_TABLE_RESCUE_MAX_SCORE="6.0"
EXTERNAL_TABLE_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION="0.65"
EXTERNAL_TABLE_RESCUE_MIN_PRED_AREA="120"
EXTERNAL_TABLE_RESCUE_MIN_PROB_MEAN="0.84"
EXTERNAL_BLOB_RESCUE_CHECKPOINT="/mnt/c/My_Project/SAM_3/guideline_line/runs_cv/ft_after_clean2000_softdistance_v1/best.pt"
EXTERNAL_BLOB_RESCUE_SOURCE_GROUP="white_blob"
EXTERNAL_BLOB_RESCUE_MIN_SCORE="-999.0"
EXTERNAL_BLOB_RESCUE_MAX_SCORE="999.0"
EXTERNAL_BLOB_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION="1.1"
EXTERNAL_BLOB_RESCUE_MIN_PRED_AREA="300"
EXTERNAL_BLOB_RESCUE_MIN_PROB_MEAN="0.95"
EXTERNAL_TINY_BLOB_RESCUE_CHECKPOINT="/mnt/c/My_Project/SAM_3/guideline_line/runs_cv/ft_balanced_ultratiny_replay_v1/best.pt"
EXTERNAL_TINY_BLOB_RESCUE_SOURCE_GROUP="white_blob"
EXTERNAL_TINY_BLOB_RESCUE_MIN_SCORE="2.0"
EXTERNAL_TINY_BLOB_RESCUE_MAX_SCORE="4.5"
EXTERNAL_TINY_BLOB_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION="1.1"
EXTERNAL_TINY_BLOB_RESCUE_MIN_PRED_AREA="80"
EXTERNAL_TINY_BLOB_RESCUE_MIN_PROB_MEAN="0.88"
REJECT_TABLE_HOUGH_FINAL_FALLBACK_MIN_PRED_PIXELS="80"
REJECT_TABLE_HOUGH_FINAL_FALLBACK_MAX_SCORE="0.0"
REJECT_WINNER_SOURCE_MIN_PRED_PIXELS_JSON='{"white_blob":800,"table_hough":600,"external_main_reticle_rescue":800}'
REJECT_WINNER_SOURCE_AREA_SCORE_JSON='{"table_hough":{"min_pred_pixels":54,"max_score":0.053614,"exemptions":[{"name":"confident_thin_table_candidate","min_confidence":0.84,"min_line_likeness":0.8,"max_pred_pixels":180}]},"white_blob":{"min_pred_pixels":444,"max_score":3.01,"exemptions":[{"name":"ball_filled_low_leak_white_candidate","min_confidence":0.82,"min_line_likeness":0.9,"min_ball_fill_fraction":0.95,"max_outward_extension":0.01}]},"reticle_global_0":{"min_pred_pixels":323,"max_score":1.287158},"external_main_reticle_rescue":{"min_pred_pixels":46,"max_score":0.834604}}'
REJECT_PRE_FINAL_FALLBACK_MAX_SCORE="-0.923089"
REJECT_FINAL_CONTEXT_RULES_JSON='[{"name":"external_table_broad_fullroi_far_from_candidates","source":"external_main_table_broad_rescue","max_mask_fraction_near_any_candidate":0.01,"min_table_roi_area_fraction":0.9},{"name":"table_hough_pure_white_far_from_candidate_center","source":"table_hough","min_mask_white_fraction":0.965,"min_min_candidate_center_distance":14.8},{"name":"reticle_global_off_table","source":"reticle_global_0","max_mask_fraction_in_table_roi":0.0},{"name":"white_blob_far_from_candidates","source":"white_blob","max_mask_fraction_near_any_candidate":0.0,"min_min_abs_candidate_boundary_distance":50.0},{"name":"external_reticle_large_roi_failure","source":"external_main_reticle_rescue","min_table_roi_area_fraction":0.410009},{"name":"table_hough_far_from_candidate_boundary","source":"table_hough","min_min_abs_candidate_boundary_distance":52.0},{"name":"external_lowmid_far_from_candidate_boundary","source":"external_main_lowmid_rescue","min_min_abs_candidate_boundary_distance":430.0},{"name":"external_table_broad_far_from_candidate_boundary","source":"external_main_table_broad_rescue","min_min_abs_candidate_boundary_distance":410.0},{"name":"external_reticle_tiny_far_from_candidates","source":"external_main_reticle_rescue","max_pred_pixels":50,"max_mask_fraction_near_any_candidate":0.0,"min_min_abs_candidate_boundary_distance":40.0},{"name":"table_hough_tiny_far_from_candidates","source":"table_hough","max_pred_pixels":30,"max_mask_fraction_near_any_candidate":0.0,"min_min_abs_candidate_boundary_distance":20.0},{"name":"external_table_broad_oversized_negative_mask","source":"external_main_table_broad_rescue","min_pred_pixels":1000},{"name":"external_reticle_tiny_offtable_far_from_candidates","source":"external_main_reticle_rescue","max_pred_pixels":100,"max_mask_fraction_in_table_roi":0.0,"max_mask_fraction_near_any_candidate":0.0,"min_min_abs_candidate_boundary_distance":200.0},{"name":"external_reticle_borderline_large_roi_not_near_candidates","source":"external_main_reticle_rescue","max_pred_pixels":400,"min_table_roi_area_fraction":0.410008,"max_mask_fraction_near_any_candidate":0.0,"min_min_abs_candidate_boundary_distance":20.0},{"name":"external_main_fullroi_far_from_candidates","source":"external_main_rescue","min_table_roi_area_fraction":0.99,"max_mask_fraction_near_any_candidate":0.0,"min_min_abs_candidate_boundary_distance":70.0},{"name":"reticle_global_zero_near_candidate_pure_white","source":"reticle_global_0","max_mask_fraction_near_any_candidate":0.0,"min_mask_white_fraction":0.98,"min_min_abs_candidate_boundary_distance":10.0,"max_pred_pixels":450}]'
CANDIDATE_SELECTOR_RESCUE_RULES_JSON='[{"name":"rescue_large_disconnected_table_to_reticle_attached_candidate","current_source_group":"table_hough","current_required_flag":"selected_via_rescue","min_current_score":0.5,"max_current_score":1.0,"min_current_pred_pixels":150,"max_current_ball_fill_fraction":0.05,"candidate_source_group":"reticle_global","min_candidate_score":0.4,"min_candidate_heuristic_score":2.5,"min_candidate_pred_pixels":60,"min_candidate_ball_fill_fraction":0.8,"min_candidate_connected_to_ball":1.0,"selection_key":"pred_pixels"},{"name":"rescue_table_to_lowscore_longer_attached_table_candidate","current_source_group":"table_hough","current_required_flag":"selected_via_rescue","min_current_score":0.5,"max_current_score":0.9,"candidate_source_group":"table_hough","min_candidate_score":-0.2,"max_candidate_score":0.0,"min_candidate_heuristic_score":2.5,"min_candidate_pred_pixels":70,"min_candidate_ball_fill_fraction":0.25,"max_candidate_ball_fill_fraction":0.45,"min_candidate_connected_to_ball":1.0,"min_candidate_outward_extension":0.15,"selection_key":"pred_pixels"},{"name":"rescue_fallback_table_to_wide_attached_secondary_candidate","current_source_group":"table_hough","current_required_flag":"fallback","min_current_score":1.5,"max_current_score":2.0,"min_current_pred_pixels":150,"max_current_ball_fill_fraction":0.5,"max_current_outward_extension":0.12,"candidate_pool":"secondary_rescue","candidate_source_group":"table_hough","min_candidate_score":-0.25,"max_candidate_score":0.7,"min_candidate_heuristic_score":3.0,"min_candidate_pred_pixels":80,"max_candidate_pred_pixels":130,"min_candidate_ball_fill_fraction":0.7,"min_candidate_connected_to_ball":1.0,"max_candidate_outward_extension":0.08,"selection_key":"heuristic_score"},{"name":"micro_line_table_fallback_to_tiny_white_blob_candidate","current_source_group":"table_hough","current_required_flag":"final_fallback","min_current_score":-5.0,"max_current_score":0.1,"min_current_pred_pixels":40,"max_current_pred_pixels":90,"max_current_ball_fill_fraction":0.1,"candidate_pool":"micro_line_rescue","candidate_source":"white_blob","min_candidate_pred_pixels":24,"max_candidate_pred_pixels":40,"min_candidate_confidence":0.75,"min_candidate_line_likeness":0.8,"min_candidate_skeleton_length":12,"max_candidate_outward_extension":0.25,"max_candidate_ball_fill_fraction":0.05,"selection_key":"line_likeness"},{"name":"rescue_table_wrong_line_to_colored_object_ball_candidate","current_source_group":"table_hough","current_required_flag":"rescue","min_current_score":0.5,"max_current_score":0.9,"min_current_pred_pixels":150,"max_current_pred_pixels":230,"max_current_ball_fill_fraction":0.05,"max_current_outward_extension":0.05,"candidate_pool":"colored_blob_rescue","candidate_source":"colored_blob","min_candidate_heuristic_score":3.0,"min_candidate_pred_pixels":250,"max_candidate_pred_pixels":380,"min_candidate_confidence":0.8,"min_candidate_line_likeness":0.9,"min_candidate_ball_fill_fraction":0.15,"min_candidate_outward_extension":0.3,"selection_key":"heuristic_score"},{"name":"v11c_midscore_reticle0_to_filled_white_blob_candidate","current_source":"reticle_global_0","min_current_score":2.0,"max_current_score":3.0,"candidate_source":"white_blob","min_candidate_confidence":0.84,"min_candidate_line_likeness":0.9,"min_candidate_ball_fill_fraction":0.94,"max_candidate_outward_extension":0.012,"min_candidate_pred_pixels":150,"selection_key":"score"},{"name":"v11c_midscore_reticle0_to_attached_table_candidate","current_source":"reticle_global_0","min_current_score":2.0,"max_current_score":3.0,"candidate_source_group":"table_hough","min_candidate_confidence":0.86,"min_candidate_line_likeness":0.9,"min_candidate_pred_pixels":100,"max_candidate_pred_pixels":260,"min_candidate_ball_fill_fraction":0.25,"max_candidate_ball_fill_fraction":0.55,"min_candidate_outward_extension":0.12,"selection_key":"heuristic_score"},{"name":"v15b_tight_final_fallback_table_hough_to_colored_blob_candidate","current_source_group":"table_hough","current_required_flag":"final_fallback","min_current_score":-6,"max_current_score":0.1,"max_current_pred_pixels":140,"max_current_ball_fill_fraction":0.2,"candidate_pool":"colored_blob_rescue","candidate_source":"colored_blob","min_candidate_score":-2.5,"max_candidate_score":-1.5,"min_candidate_pred_pixels":300,"max_candidate_pred_pixels":500,"min_candidate_confidence":0.55,"min_candidate_line_likeness":0.6,"min_candidate_ball_fill_fraction":0.15,"max_candidate_ball_fill_fraction":0.35,"min_candidate_connected_to_ball":1,"min_candidate_outward_extension":0,"selection_key":"heuristic_score"},{"name":"v17_highscore_reticle0_to_reticle1_attached_short_candidate","current_source":"reticle_global_0","min_current_score":1.9,"max_current_score":2.3,"candidate_source":"reticle_global_1","min_candidate_score":0.7,"max_candidate_score":1.0,"min_candidate_heuristic_score":3.2,"max_candidate_pred_pixels":170,"min_candidate_ball_fill_fraction":0.6,"max_candidate_ball_fill_fraction":0.75,"min_candidate_connected_to_ball":1.0,"max_candidate_outward_extension":0.08,"selection_key":"heuristic_score"},{"name":"v17_highscore_reticle0_to_filled_white_blob_candidate","current_source":"reticle_global_0","min_current_score":3.3,"max_current_score":3.8,"candidate_source":"white_blob","min_candidate_score":2.0,"max_candidate_score":2.6,"min_candidate_heuristic_score":2.4,"min_candidate_pred_pixels":280,"max_candidate_pred_pixels":340,"min_candidate_confidence":0.84,"min_candidate_line_likeness":0.9,"min_candidate_ball_fill_fraction":0.9,"min_candidate_connected_to_ball":1.0,"max_candidate_outward_extension":0.02,"selection_key":"heuristic_score"},{"name":"v17_highscore_reticle0_to_attached_table_candidate","current_source":"reticle_global_0","min_current_score":3.3,"max_current_score":3.6,"candidate_source_group":"table_hough","min_candidate_score":0.7,"max_candidate_score":1.0,"min_candidate_heuristic_score":2.7,"min_candidate_pred_pixels":180,"max_candidate_pred_pixels":230,"min_candidate_confidence":0.84,"min_candidate_line_likeness":0.9,"min_candidate_ball_fill_fraction":0.4,"max_candidate_ball_fill_fraction":0.55,"min_candidate_connected_to_ball":1.0,"min_candidate_outward_extension":0.1,"max_candidate_outward_extension":0.18,"selection_key":"heuristic_score"}]'
MICRO_LINE_RESCUE_JSON='{"checkpoint":"C:\\My_Project\\Test_New_Approach\\runs\\supervised_train_v10_repair_lr2e-4_pw18\\guideline_unet_best.pt","current_source_group":"table_hough","current_required_flag":"final_fallback","min_current_score":-5.0,"max_current_score":0.1,"min_current_pred_pixels":40,"max_current_pred_pixels":90,"max_current_ball_fill_fraction":0.1,"prediction_threshold":0.2,"max_ball_candidates":10,"image_size":256,"use_reranker":false}'
COLORED_BLOB_RESCUE_JSON='[{"checkpoint":"C:\\My_Project\\Test_New_Approach\\runs\\supervised_train_v2\\guideline_unet_best.pt","reranker_checkpoint":"C:\\My_Project\\Test_New_Approach\\runs\\supervised_train_v2\\guideline_reranker_best.pt","current_source_group":"table_hough","current_required_flag":"rescue","min_current_score":0.5,"max_current_score":0.9,"min_current_pred_pixels":150,"max_current_pred_pixels":230,"max_current_ball_fill_fraction":0.05,"max_current_outward_extension":0.05,"prediction_threshold":0.35,"max_ball_candidates":40,"image_size":384,"use_reranker":true,"colored_blob":{"min_saturation":120,"min_value":120,"min_area":30,"max_area":2500,"min_radius":5,"max_radius":35,"min_output_radius":16,"min_score":0.55,"score_cap":0.7},"name":"v5_promoted_wrong_line_colored_blob_rescue"},{"checkpoint":"C:\\My_Project\\Test_New_Approach\\runs\\supervised_train_v2\\guideline_unet_best.pt","reranker_checkpoint":"C:\\My_Project\\Test_New_Approach\\runs\\supervised_train_v2\\guideline_reranker_best.pt","current_source_group":"table_hough","min_current_score":-6,"max_current_score":0.1,"max_current_pred_pixels":140,"max_current_ball_fill_fraction":0.2,"prediction_threshold":0.3,"max_ball_candidates":60,"image_size":384,"use_reranker":true,"colored_blob":{"min_saturation":120,"min_value":120,"min_area":30,"max_area":2500,"min_radius":5,"max_radius":35,"min_output_radius":16,"min_score":0.55,"score_cap":0.7},"name":"v15b_final_fallback_table_hough_colored_blob_rescue","current_required_flags":["final_fallback"]}]'
IMAGE_FINAL_VETO_JSON='[{"checkpoint":"/mnt/c/My_Project/Test_New_Approach/runs/final_veto_v3_image_focus/image_veto_global_s42.pt","threshold":0.95,"sources":["white_blob","table_hough","external_main_blob_rescue","external_main_rescue"],"include_global":true,"image_size":128,"source_min_pred_pixels":{"external_main_rescue":180}},{"checkpoint":"/mnt/c/My_Project/Test_New_Approach/runs/supplemental_v7_validation/image_veto_v4_external_reticle/image_veto_v4_global_s42.pt","threshold":0.05,"sources":["external_main_rescue","reticle_global_0","reticle_global_2","grid","external_main_table_broad_rescue","external_main_table_rescue","external_main_reticle_rescue","external_main_lowmid_rescue","external_main_mid_rescue"],"include_global":true,"image_size":128,"source_min_pred_pixels":{"external_main_rescue":180}}]'

if [[ ! -x "$PYTHON_BIN" ]]; then
  bash "$REPO_ROOT/tools/setup_supervised_train_wsl.sh"
fi

CANDIDATE_SELECTOR_RESCUE_RULES_JSON="$("$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(manifest.get("candidate_selector_rescue_rules", []), separators=(",", ":")))
PY
)"
SELECTOR_CANDIDATE_POOL_RESCUE_JSON="$("$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = manifest.get("selector_candidate_pool_rescue")
print("" if value is None else json.dumps(value, separators=(",", ":")))
PY
)"
POST_EXTERNAL_SELECTOR_CANDIDATE_POOL_RESCUE_JSON="$("$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = manifest.get("post_external_selector_candidate_pool_rescue")
print("" if value is None else json.dumps(value, separators=(",", ":")))
PY
)"
POST_EXTERNAL_CANDIDATE_SELECTOR_RESCUE_RULES_JSON="$("$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(manifest.get("post_external_candidate_selector_rescue_rules", []), separators=(",", ":")))
PY
)"
POST_FINAL_CANDIDATE_SELECTOR_RESCUE_RULES_JSON="$("$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(manifest.get("post_final_candidate_selector_rescue_rules", []), separators=(",", ":")))
PY
)"
IMAGE_FINAL_VETO_FROM_MANIFEST_JSON="$("$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = manifest.get("image_final_veto")
print("" if value is None else json.dumps(value, separators=(",", ":")))
PY
)"
if [[ -n "$IMAGE_FINAL_VETO_FROM_MANIFEST_JSON" ]]; then
  IMAGE_FINAL_VETO_JSON="$IMAGE_FINAL_VETO_FROM_MANIFEST_JSON"
fi
REJECT_WINNER_SOURCE_MIN_PRED_PIXELS_EXEMPTIONS_JSON="$("$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = manifest.get("reject_winner_source_min_pred_pixels_exemptions")
print("" if value is None else json.dumps(value, separators=(",", ":")))
PY
)"
SECONDARY_RESCUE_CURRENT_EXEMPTIONS_JSON="$("$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = manifest.get("secondary_rescue_current_exemptions")
print("" if value is None else json.dumps(value, separators=(",", ":")))
PY
)"
SELECTOR_CANDIDATE_POOL_RESCUE_ARGS=()
if [[ -n "$SELECTOR_CANDIDATE_POOL_RESCUE_JSON" ]]; then
  SELECTOR_CANDIDATE_POOL_RESCUE_ARGS=(
    --selector-candidate-pool-rescue-json "$SELECTOR_CANDIDATE_POOL_RESCUE_JSON"
  )
fi
POST_EXTERNAL_SELECTOR_CANDIDATE_POOL_RESCUE_ARGS=()
if [[ -n "$POST_EXTERNAL_SELECTOR_CANDIDATE_POOL_RESCUE_JSON" ]]; then
  POST_EXTERNAL_SELECTOR_CANDIDATE_POOL_RESCUE_ARGS=(
    --post-external-selector-candidate-pool-rescue-json "$POST_EXTERNAL_SELECTOR_CANDIDATE_POOL_RESCUE_JSON"
  )
fi
REJECT_WINNER_SOURCE_MIN_PRED_PIXELS_EXEMPTIONS_ARGS=()
if [[ -n "$REJECT_WINNER_SOURCE_MIN_PRED_PIXELS_EXEMPTIONS_JSON" ]]; then
  REJECT_WINNER_SOURCE_MIN_PRED_PIXELS_EXEMPTIONS_ARGS=(
    --reject-winner-source-min-pred-pixels-exemptions-json "$REJECT_WINNER_SOURCE_MIN_PRED_PIXELS_EXEMPTIONS_JSON"
  )
fi
SECONDARY_RESCUE_CURRENT_EXEMPTIONS_ARGS=()
if [[ -n "$SECONDARY_RESCUE_CURRENT_EXEMPTIONS_JSON" ]]; then
  SECONDARY_RESCUE_CURRENT_EXEMPTIONS_ARGS=(
    --secondary-rescue-current-exemptions-json "$SECONDARY_RESCUE_CURRENT_EXEMPTIONS_JSON"
  )
fi

"$PYTHON_BIN" -m src.supervised.infer \
  --checkpoint "$PRIMARY_CHECKPOINT" \
  --reranker-checkpoint "$PRIMARY_RERANKER" \
  --fallback-checkpoint "$FALLBACK_CHECKPOINT" \
  --fallback-reranker-checkpoint "$FALLBACK_RERANKER" \
  --fallback-score-threshold "$FALLBACK_SCORE_THRESHOLD" \
  --rescue-checkpoint "$RESCUE_CHECKPOINT" \
  --rescue-reranker-checkpoint "$RESCUE_RERANKER" \
  --rescue-score-threshold "$RESCUE_SCORE_THRESHOLD" \
  --rescue-max-ball-candidates "$RESCUE_MAX_BALL_CANDIDATES" \
  --final-fallback-score-threshold "$FINAL_FALLBACK_SCORE_THRESHOLD" \
  --secondary-rescue-checkpoint "$SECONDARY_RESCUE_CHECKPOINT" \
  --secondary-rescue-reranker-checkpoint "$SECONDARY_RESCUE_RERANKER" \
  --secondary-rescue-min-score "$SECONDARY_RESCUE_MIN_SCORE" \
  --secondary-rescue-max-score "$SECONDARY_RESCUE_MAX_SCORE" \
  --secondary-rescue-source "$SECONDARY_RESCUE_SOURCE" \
  --secondary-rescue-min-ball-fill-fraction "$SECONDARY_RESCUE_MIN_BALL_FILL_FRACTION" \
  --secondary-rescue-max-ball-candidates "$SECONDARY_RESCUE_MAX_BALL_CANDIDATES" \
  "${SECONDARY_RESCUE_CURRENT_EXEMPTIONS_ARGS[@]}" \
  --primary-recovery-min-score "$PRIMARY_RECOVERY_MIN_SCORE" \
  --primary-recovery-max-score "$PRIMARY_RECOVERY_MAX_SCORE" \
  --primary-recovery-source "$PRIMARY_RECOVERY_SOURCE" \
  --primary-recovery-candidate-source "$PRIMARY_RECOVERY_CANDIDATE_SOURCE" \
  --primary-recovery-min-ball-fill-fraction "$PRIMARY_RECOVERY_MIN_BALL_FILL_FRACTION" \
  --reticle-recovery-min-score "$RETICLE_RECOVERY_MIN_SCORE" \
  --reticle-recovery-max-score "$RETICLE_RECOVERY_MAX_SCORE" \
  --reticle-recovery-min-primary-score "$RETICLE_RECOVERY_MIN_PRIMARY_SCORE" \
  --fallback-rescue-checkpoint "$FALLBACK_RESCUE_CHECKPOINT" \
  --fallback-rescue-reranker-checkpoint "$FALLBACK_RESCUE_RERANKER" \
  --fallback-rescue-min-score "$FALLBACK_RESCUE_MIN_SCORE" \
  --fallback-rescue-max-score "$FALLBACK_RESCUE_MAX_SCORE" \
  --fallback-rescue-source-group "$FALLBACK_RESCUE_SOURCE_GROUP" \
  --fallback-rescue-max-ball-candidates "$FALLBACK_RESCUE_MAX_BALL_CANDIDATES" \
  --reticle-fill-rescue-checkpoint "$RETICLE_FILL_RESCUE_CHECKPOINT" \
  --reticle-fill-rescue-reranker-checkpoint "$RETICLE_FILL_RESCUE_RERANKER" \
  --reticle-fill-rescue-min-score "$RETICLE_FILL_RESCUE_MIN_SCORE" \
  --reticle-fill-rescue-max-score "$RETICLE_FILL_RESCUE_MAX_SCORE" \
  --reticle-fill-rescue-current-max-ball-fill-fraction "$RETICLE_FILL_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION" \
  --reticle-fill-rescue-target-min-ball-fill-fraction "$RETICLE_FILL_RESCUE_TARGET_MIN_BALL_FILL_FRACTION" \
  --reticle-fill-rescue-max-ball-candidates "$RETICLE_FILL_RESCUE_MAX_BALL_CANDIDATES" \
  --external-rescue-checkpoint "$EXTERNAL_RESCUE_CHECKPOINT" \
  --external-rescue-source-group "$EXTERNAL_RESCUE_SOURCE_GROUP" \
  --external-rescue-min-score "$EXTERNAL_RESCUE_MIN_SCORE" \
  --external-rescue-max-score "$EXTERNAL_RESCUE_MAX_SCORE" \
  --external-rescue-current-max-ball-fill-fraction "$EXTERNAL_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION" \
  --external-rescue-min-pred-area "$EXTERNAL_RESCUE_MIN_PRED_AREA" \
  --external-rescue-min-prob-mean "$EXTERNAL_RESCUE_MIN_PROB_MEAN" \
  --external-mid-rescue-checkpoint "$EXTERNAL_MID_RESCUE_CHECKPOINT" \
  --external-mid-rescue-source-group "$EXTERNAL_MID_RESCUE_SOURCE_GROUP" \
  --external-mid-rescue-min-score "$EXTERNAL_MID_RESCUE_MIN_SCORE" \
  --external-mid-rescue-max-score "$EXTERNAL_MID_RESCUE_MAX_SCORE" \
  --external-mid-rescue-current-max-ball-fill-fraction "$EXTERNAL_MID_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION" \
  --external-mid-rescue-min-pred-area "$EXTERNAL_MID_RESCUE_MIN_PRED_AREA" \
  --external-mid-rescue-min-prob-mean "$EXTERNAL_MID_RESCUE_MIN_PROB_MEAN" \
  --external-lowmid-rescue-checkpoint "$EXTERNAL_LOWMID_RESCUE_CHECKPOINT" \
  --external-lowmid-rescue-source-group "$EXTERNAL_LOWMID_RESCUE_SOURCE_GROUP" \
  --external-lowmid-rescue-min-score "$EXTERNAL_LOWMID_RESCUE_MIN_SCORE" \
  --external-lowmid-rescue-max-score "$EXTERNAL_LOWMID_RESCUE_MAX_SCORE" \
  --external-lowmid-rescue-current-max-ball-fill-fraction "$EXTERNAL_LOWMID_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION" \
  --external-lowmid-rescue-min-pred-area "$EXTERNAL_LOWMID_RESCUE_MIN_PRED_AREA" \
  --external-lowmid-rescue-min-prob-mean "$EXTERNAL_LOWMID_RESCUE_MIN_PROB_MEAN" \
  --external-reticle-rescue-checkpoint "$EXTERNAL_RETICLE_RESCUE_CHECKPOINT" \
  --external-reticle-rescue-source-group "$EXTERNAL_RETICLE_RESCUE_SOURCE_GROUP" \
  --external-reticle-rescue-min-score "$EXTERNAL_RETICLE_RESCUE_MIN_SCORE" \
  --external-reticle-rescue-max-score "$EXTERNAL_RETICLE_RESCUE_MAX_SCORE" \
  --external-reticle-rescue-current-max-ball-fill-fraction "$EXTERNAL_RETICLE_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION" \
  --external-reticle-rescue-min-pred-area "$EXTERNAL_RETICLE_RESCUE_MIN_PRED_AREA" \
  --external-reticle-rescue-min-prob-mean "$EXTERNAL_RETICLE_RESCUE_MIN_PROB_MEAN" \
  --external-tiny-reticle-rescue-checkpoint "$EXTERNAL_TINY_RETICLE_RESCUE_CHECKPOINT" \
  --external-tiny-reticle-rescue-source-group "$EXTERNAL_TINY_RETICLE_RESCUE_SOURCE_GROUP" \
  --external-tiny-reticle-rescue-min-score "$EXTERNAL_TINY_RETICLE_RESCUE_MIN_SCORE" \
  --external-tiny-reticle-rescue-max-score "$EXTERNAL_TINY_RETICLE_RESCUE_MAX_SCORE" \
  --external-tiny-reticle-rescue-current-max-ball-fill-fraction "$EXTERNAL_TINY_RETICLE_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION" \
  --external-tiny-reticle-rescue-min-pred-area "$EXTERNAL_TINY_RETICLE_RESCUE_MIN_PRED_AREA" \
  --external-tiny-reticle-rescue-min-prob-mean "$EXTERNAL_TINY_RETICLE_RESCUE_MIN_PROB_MEAN" \
  --external-reticle-large-rescue-checkpoint "$EXTERNAL_RETICLE_LARGE_RESCUE_CHECKPOINT" \
  --external-reticle-large-rescue-source-group "$EXTERNAL_RETICLE_LARGE_RESCUE_SOURCE_GROUP" \
  --external-reticle-large-rescue-min-score "$EXTERNAL_RETICLE_LARGE_RESCUE_MIN_SCORE" \
  --external-reticle-large-rescue-max-score "$EXTERNAL_RETICLE_LARGE_RESCUE_MAX_SCORE" \
  --external-reticle-large-rescue-current-max-ball-fill-fraction "$EXTERNAL_RETICLE_LARGE_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION" \
  --external-reticle-large-rescue-min-pred-area "$EXTERNAL_RETICLE_LARGE_RESCUE_MIN_PRED_AREA" \
  --external-reticle-large-rescue-min-prob-mean "$EXTERNAL_RETICLE_LARGE_RESCUE_MIN_PROB_MEAN" \
  --external-table-broad-rescue-checkpoint "$EXTERNAL_TABLE_BROAD_RESCUE_CHECKPOINT" \
  --external-table-broad-rescue-source-group "$EXTERNAL_TABLE_BROAD_RESCUE_SOURCE_GROUP" \
  --external-table-broad-rescue-min-score "$EXTERNAL_TABLE_BROAD_RESCUE_MIN_SCORE" \
  --external-table-broad-rescue-max-score "$EXTERNAL_TABLE_BROAD_RESCUE_MAX_SCORE" \
  --external-table-broad-rescue-current-max-ball-fill-fraction "$EXTERNAL_TABLE_BROAD_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION" \
  --external-table-broad-rescue-min-pred-area "$EXTERNAL_TABLE_BROAD_RESCUE_MIN_PRED_AREA" \
  --external-table-broad-rescue-min-prob-mean "$EXTERNAL_TABLE_BROAD_RESCUE_MIN_PROB_MEAN" \
  --external-table-rescue-checkpoint "$EXTERNAL_TABLE_RESCUE_CHECKPOINT" \
  --external-table-rescue-source-group "$EXTERNAL_TABLE_RESCUE_SOURCE_GROUP" \
  --external-table-rescue-min-score "$EXTERNAL_TABLE_RESCUE_MIN_SCORE" \
  --external-table-rescue-max-score "$EXTERNAL_TABLE_RESCUE_MAX_SCORE" \
  --external-table-rescue-current-max-ball-fill-fraction "$EXTERNAL_TABLE_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION" \
  --external-table-rescue-min-pred-area "$EXTERNAL_TABLE_RESCUE_MIN_PRED_AREA" \
  --external-table-rescue-min-prob-mean "$EXTERNAL_TABLE_RESCUE_MIN_PROB_MEAN" \
  --external-blob-rescue-checkpoint "$EXTERNAL_BLOB_RESCUE_CHECKPOINT" \
  --external-blob-rescue-source-group "$EXTERNAL_BLOB_RESCUE_SOURCE_GROUP" \
  --external-blob-rescue-min-score "$EXTERNAL_BLOB_RESCUE_MIN_SCORE" \
  --external-blob-rescue-max-score "$EXTERNAL_BLOB_RESCUE_MAX_SCORE" \
  --external-blob-rescue-current-max-ball-fill-fraction "$EXTERNAL_BLOB_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION" \
  --external-blob-rescue-min-pred-area "$EXTERNAL_BLOB_RESCUE_MIN_PRED_AREA" \
  --external-blob-rescue-min-prob-mean "$EXTERNAL_BLOB_RESCUE_MIN_PROB_MEAN" \
  --external-tiny-blob-rescue-checkpoint "$EXTERNAL_TINY_BLOB_RESCUE_CHECKPOINT" \
  --external-tiny-blob-rescue-source-group "$EXTERNAL_TINY_BLOB_RESCUE_SOURCE_GROUP" \
  --external-tiny-blob-rescue-min-score "$EXTERNAL_TINY_BLOB_RESCUE_MIN_SCORE" \
  --external-tiny-blob-rescue-max-score "$EXTERNAL_TINY_BLOB_RESCUE_MAX_SCORE" \
  --external-tiny-blob-rescue-current-max-ball-fill-fraction "$EXTERNAL_TINY_BLOB_RESCUE_CURRENT_MAX_BALL_FILL_FRACTION" \
  --external-tiny-blob-rescue-min-pred-area "$EXTERNAL_TINY_BLOB_RESCUE_MIN_PRED_AREA" \
  --external-tiny-blob-rescue-min-prob-mean "$EXTERNAL_TINY_BLOB_RESCUE_MIN_PROB_MEAN" \
  --reject-table-hough-final-fallback-min-pred-pixels "$REJECT_TABLE_HOUGH_FINAL_FALLBACK_MIN_PRED_PIXELS" \
  --reject-table-hough-final-fallback-max-score "$REJECT_TABLE_HOUGH_FINAL_FALLBACK_MAX_SCORE" \
  --reject-winner-source-min-pred-pixels-json "$REJECT_WINNER_SOURCE_MIN_PRED_PIXELS_JSON" \
  "${REJECT_WINNER_SOURCE_MIN_PRED_PIXELS_EXEMPTIONS_ARGS[@]}" \
  --reject-winner-source-area-score-json "$REJECT_WINNER_SOURCE_AREA_SCORE_JSON" \
  --reject-pre-final-fallback-max-score "$REJECT_PRE_FINAL_FALLBACK_MAX_SCORE" \
  --reject-final-context-rules-json "$REJECT_FINAL_CONTEXT_RULES_JSON" \
  --candidate-selector-rescue-rules-json "$CANDIDATE_SELECTOR_RESCUE_RULES_JSON" \
  "${SELECTOR_CANDIDATE_POOL_RESCUE_ARGS[@]}" \
  "${POST_EXTERNAL_SELECTOR_CANDIDATE_POOL_RESCUE_ARGS[@]}" \
  --post-external-candidate-selector-rescue-rules-json "$POST_EXTERNAL_CANDIDATE_SELECTOR_RESCUE_RULES_JSON" \
  --post-final-candidate-selector-rescue-rules-json "$POST_FINAL_CANDIDATE_SELECTOR_RESCUE_RULES_JSON" \
  --micro-line-rescue-json "$MICRO_LINE_RESCUE_JSON" \
  --colored-blob-rescue-json "$COLORED_BLOB_RESCUE_JSON" \
  --image-final-veto-json "$IMAGE_FINAL_VETO_JSON" \
  "$@"
