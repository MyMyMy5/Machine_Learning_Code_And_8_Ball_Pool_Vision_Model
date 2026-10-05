from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.supervised.infer import run_supervised_inference


def normalize_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path_str = str(path)
    if os.name != "nt" and path_str.startswith("C:\\"):
        return Path("/mnt/c/" + path_str[3:].replace("\\", "/"))
    if os.name != "nt" and path_str.startswith("C:/"):
        return Path("/mnt/c/" + path_str[3:])
    if os.name == "nt" and path_str.startswith("/mnt/c/"):
        return Path("C:/" + path_str[len("/mnt/c/"):])
    return Path(path_str)


def _normalize_nested_config_paths(value: dict[str, Any]) -> dict[str, Any]:
    normalized_value = dict(value)
    for nested_key in ("checkpoint", "checkpoint_path", "reranker_checkpoint", "reranker_checkpoint_path"):
        if nested_key in normalized_value and normalized_value[nested_key]:
            normalized_value[nested_key] = str(normalize_path(normalized_value[nested_key]))
    return normalized_value


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized: dict[str, Any] = {}
    for key, value in manifest.items():
        if value is None:
            normalized[key] = None
        elif key in {
            "micro_line_rescue",
            "colored_blob_rescue",
            "selector_candidate_pool_rescue",
            "post_external_selector_candidate_pool_rescue",
            "rescue_arbiter",
            "post_external_rescue_arbiter",
            "image_final_veto",
        } and isinstance(value, dict):
            normalized[key] = _normalize_nested_config_paths(value)
        elif key in {
            "colored_blob_rescue",
            "selector_candidate_pool_rescue",
            "post_external_selector_candidate_pool_rescue",
            "rescue_arbiter",
            "post_external_rescue_arbiter",
            "image_final_veto",
        } and isinstance(value, list):
            normalized[key] = [
                _normalize_nested_config_paths(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif key.endswith("_checkpoint") or key.endswith("_reranker"):
            normalized[key] = normalize_path(value)
        else:
            normalized[key] = value
    return normalized


def run_manifest_inference(
    *,
    manifest: dict[str, Any],
    input_path: Path,
    output_dir: Path,
    image_size: int = 384,
    crop_batch_size: int = 1,
    save_outputs: bool = True,
    save_intermediates: bool = False,
    save_report: bool = True,
    save_recolor_preview: bool = False,
    return_overlay: bool = True,
    amp_enabled: bool = True,
) -> dict[str, object]:
    return run_supervised_inference(
        checkpoint_path=manifest["primary_checkpoint"],
        input_path=input_path,
        output_dir=output_dir,
        image_size=image_size,
        reranker_checkpoint_path=manifest.get("primary_reranker"),
        fallback_checkpoint_path=manifest.get("fallback_checkpoint"),
        fallback_reranker_checkpoint_path=manifest.get("fallback_reranker"),
        fallback_score_threshold=manifest.get("fallback_score_threshold"),
        meta_selector_checkpoint_path=manifest.get("meta_selector_checkpoint"),
        rescue_checkpoint_path=manifest.get("rescue_checkpoint"),
        rescue_reranker_checkpoint_path=manifest.get("rescue_reranker"),
        rescue_score_threshold=manifest.get("rescue_score_threshold"),
        rescue_max_ball_candidates=int(manifest.get("rescue_max_ball_candidates", 20)),
        final_fallback_score_threshold=manifest.get("final_fallback_score_threshold"),
        secondary_rescue_checkpoint_path=manifest.get("secondary_rescue_checkpoint"),
        secondary_rescue_reranker_checkpoint_path=manifest.get("secondary_rescue_reranker"),
        secondary_rescue_min_score=manifest.get("secondary_rescue_min_score"),
        secondary_rescue_max_score=manifest.get("secondary_rescue_max_score"),
        secondary_rescue_source=manifest.get("secondary_rescue_source"),
        secondary_rescue_min_ball_fill_fraction=manifest.get("secondary_rescue_min_ball_fill_fraction"),
        secondary_rescue_max_ball_candidates=int(manifest.get("secondary_rescue_max_ball_candidates", 24)),
        secondary_rescue_current_exemptions=manifest.get("secondary_rescue_current_exemptions"),
        primary_recovery_min_score=manifest.get("primary_recovery_min_score"),
        primary_recovery_max_score=manifest.get("primary_recovery_max_score"),
        primary_recovery_source=manifest.get("primary_recovery_source"),
        primary_recovery_candidate_source=manifest.get("primary_recovery_candidate_source"),
        primary_recovery_current_group=manifest.get("primary_recovery_current_group"),
        primary_recovery_candidate_group=manifest.get("primary_recovery_candidate_group"),
        primary_recovery_min_primary_score=manifest.get("primary_recovery_min_primary_score"),
        primary_recovery_min_ball_fill_fraction=manifest.get("primary_recovery_min_ball_fill_fraction"),
        reticle_recovery_min_score=manifest.get("reticle_recovery_min_score"),
        reticle_recovery_max_score=manifest.get("reticle_recovery_max_score"),
        reticle_recovery_min_primary_score=manifest.get("reticle_recovery_min_primary_score"),
        fallback_rescue_checkpoint_path=manifest.get("fallback_rescue_checkpoint"),
        fallback_rescue_reranker_checkpoint_path=manifest.get("fallback_rescue_reranker"),
        fallback_rescue_min_score=manifest.get("fallback_rescue_min_score"),
        fallback_rescue_max_score=manifest.get("fallback_rescue_max_score"),
        fallback_rescue_source_group=manifest.get("fallback_rescue_source_group"),
        fallback_rescue_max_ball_candidates=int(manifest.get("fallback_rescue_max_ball_candidates", 20)),
        reticle_fill_rescue_checkpoint_path=manifest.get("reticle_fill_rescue_checkpoint"),
        reticle_fill_rescue_reranker_checkpoint_path=manifest.get("reticle_fill_rescue_reranker"),
        reticle_fill_rescue_min_score=manifest.get("reticle_fill_rescue_min_score"),
        reticle_fill_rescue_max_score=manifest.get("reticle_fill_rescue_max_score"),
        reticle_fill_rescue_current_max_ball_fill_fraction=manifest.get("reticle_fill_rescue_current_max_ball_fill_fraction"),
        reticle_fill_rescue_target_min_ball_fill_fraction=manifest.get("reticle_fill_rescue_target_min_ball_fill_fraction"),
        reticle_fill_rescue_max_ball_candidates=int(manifest.get("reticle_fill_rescue_max_ball_candidates", 20)),
        external_rescue_checkpoint_path=manifest.get("external_rescue_checkpoint"),
        external_rescue_source_group=manifest.get("external_rescue_source_group"),
        external_rescue_min_score=manifest.get("external_rescue_min_score"),
        external_rescue_max_score=manifest.get("external_rescue_max_score"),
        external_rescue_current_max_ball_fill_fraction=manifest.get("external_rescue_current_max_ball_fill_fraction"),
        external_rescue_min_pred_area=int(manifest.get("external_rescue_min_pred_area", 20)),
        external_rescue_min_prob_mean=float(manifest.get("external_rescue_min_prob_mean", 0.9)),
        external_mid_rescue_checkpoint_path=manifest.get("external_mid_rescue_checkpoint"),
        external_mid_rescue_source_group=manifest.get("external_mid_rescue_source_group"),
        external_mid_rescue_min_score=manifest.get("external_mid_rescue_min_score"),
        external_mid_rescue_max_score=manifest.get("external_mid_rescue_max_score"),
        external_mid_rescue_current_max_ball_fill_fraction=manifest.get("external_mid_rescue_current_max_ball_fill_fraction"),
        external_mid_rescue_min_pred_area=int(manifest.get("external_mid_rescue_min_pred_area", 20)),
        external_mid_rescue_min_prob_mean=float(manifest.get("external_mid_rescue_min_prob_mean", 0.88)),
        external_lowmid_rescue_checkpoint_path=manifest.get("external_lowmid_rescue_checkpoint"),
        external_lowmid_rescue_source_group=manifest.get("external_lowmid_rescue_source_group"),
        external_lowmid_rescue_min_score=manifest.get("external_lowmid_rescue_min_score"),
        external_lowmid_rescue_max_score=manifest.get("external_lowmid_rescue_max_score"),
        external_lowmid_rescue_current_max_ball_fill_fraction=manifest.get("external_lowmid_rescue_current_max_ball_fill_fraction"),
        external_lowmid_rescue_min_pred_area=int(manifest.get("external_lowmid_rescue_min_pred_area", 400)),
        external_lowmid_rescue_min_prob_mean=float(manifest.get("external_lowmid_rescue_min_prob_mean", 0.8)),
        external_reticle_rescue_checkpoint_path=manifest.get("external_reticle_rescue_checkpoint"),
        external_reticle_rescue_source_group=manifest.get("external_reticle_rescue_source_group"),
        external_reticle_rescue_min_score=manifest.get("external_reticle_rescue_min_score"),
        external_reticle_rescue_max_score=manifest.get("external_reticle_rescue_max_score"),
        external_reticle_rescue_current_max_ball_fill_fraction=manifest.get("external_reticle_rescue_current_max_ball_fill_fraction"),
        external_reticle_rescue_min_pred_area=int(manifest.get("external_reticle_rescue_min_pred_area", 20)),
        external_reticle_rescue_min_prob_mean=float(manifest.get("external_reticle_rescue_min_prob_mean", 0.8)),
        external_tiny_reticle_rescue_checkpoint_path=manifest.get("external_tiny_reticle_rescue_checkpoint"),
        external_tiny_reticle_rescue_source_group=manifest.get("external_tiny_reticle_rescue_source_group"),
        external_tiny_reticle_rescue_min_score=manifest.get("external_tiny_reticle_rescue_min_score"),
        external_tiny_reticle_rescue_max_score=manifest.get("external_tiny_reticle_rescue_max_score"),
        external_tiny_reticle_rescue_current_max_ball_fill_fraction=manifest.get("external_tiny_reticle_rescue_current_max_ball_fill_fraction"),
        external_tiny_reticle_rescue_min_pred_area=int(manifest.get("external_tiny_reticle_rescue_min_pred_area", 50)),
        external_tiny_reticle_rescue_min_prob_mean=float(manifest.get("external_tiny_reticle_rescue_min_prob_mean", 0.94)),
        external_reticle_large_rescue_checkpoint_path=manifest.get("external_reticle_large_rescue_checkpoint"),
        external_reticle_large_rescue_source_group=manifest.get("external_reticle_large_rescue_source_group"),
        external_reticle_large_rescue_min_score=manifest.get("external_reticle_large_rescue_min_score"),
        external_reticle_large_rescue_max_score=manifest.get("external_reticle_large_rescue_max_score"),
        external_reticle_large_rescue_current_max_ball_fill_fraction=manifest.get("external_reticle_large_rescue_current_max_ball_fill_fraction"),
        external_reticle_large_rescue_min_pred_area=int(manifest.get("external_reticle_large_rescue_min_pred_area", 400)),
        external_reticle_large_rescue_min_prob_mean=float(manifest.get("external_reticle_large_rescue_min_prob_mean", 0.86)),
        external_table_broad_rescue_checkpoint_path=manifest.get("external_table_broad_rescue_checkpoint"),
        external_table_broad_rescue_source_group=manifest.get("external_table_broad_rescue_source_group"),
        external_table_broad_rescue_min_score=manifest.get("external_table_broad_rescue_min_score"),
        external_table_broad_rescue_max_score=manifest.get("external_table_broad_rescue_max_score"),
        external_table_broad_rescue_current_max_ball_fill_fraction=manifest.get("external_table_broad_rescue_current_max_ball_fill_fraction"),
        external_table_broad_rescue_min_pred_area=int(manifest.get("external_table_broad_rescue_min_pred_area", 350)),
        external_table_broad_rescue_min_prob_mean=float(manifest.get("external_table_broad_rescue_min_prob_mean", 0.8)),
        external_blob_low_rescue_checkpoint_path=manifest.get("external_blob_low_rescue_checkpoint"),
        external_blob_low_rescue_source_group=manifest.get("external_blob_low_rescue_source_group"),
        external_blob_low_rescue_min_score=manifest.get("external_blob_low_rescue_min_score"),
        external_blob_low_rescue_max_score=manifest.get("external_blob_low_rescue_max_score"),
        external_blob_low_rescue_current_max_ball_fill_fraction=manifest.get("external_blob_low_rescue_current_max_ball_fill_fraction"),
        external_blob_low_rescue_min_pred_area=int(manifest.get("external_blob_low_rescue_min_pred_area", 20)),
        external_blob_low_rescue_min_prob_mean=float(manifest.get("external_blob_low_rescue_min_prob_mean", 0.75)),
        external_table_rescue_checkpoint_path=manifest.get("external_table_rescue_checkpoint"),
        external_table_rescue_source_group=manifest.get("external_table_rescue_source_group"),
        external_table_rescue_min_score=manifest.get("external_table_rescue_min_score"),
        external_table_rescue_max_score=manifest.get("external_table_rescue_max_score"),
        external_table_rescue_current_max_ball_fill_fraction=manifest.get("external_table_rescue_current_max_ball_fill_fraction"),
        external_table_rescue_min_pred_area=int(manifest.get("external_table_rescue_min_pred_area", 120)),
        external_table_rescue_min_prob_mean=float(manifest.get("external_table_rescue_min_prob_mean", 0.84)),
        external_blob_rescue_checkpoint_path=manifest.get("external_blob_rescue_checkpoint"),
        external_blob_rescue_source_group=manifest.get("external_blob_rescue_source_group"),
        external_blob_rescue_min_score=manifest.get("external_blob_rescue_min_score"),
        external_blob_rescue_max_score=manifest.get("external_blob_rescue_max_score"),
        external_blob_rescue_current_max_ball_fill_fraction=manifest.get("external_blob_rescue_current_max_ball_fill_fraction"),
        external_blob_rescue_min_pred_area=int(manifest.get("external_blob_rescue_min_pred_area", 300)),
        external_blob_rescue_min_prob_mean=float(manifest.get("external_blob_rescue_min_prob_mean", 0.95)),
        external_tiny_blob_rescue_checkpoint_path=manifest.get("external_tiny_blob_rescue_checkpoint"),
        external_tiny_blob_rescue_source_group=manifest.get("external_tiny_blob_rescue_source_group"),
        external_tiny_blob_rescue_min_score=manifest.get("external_tiny_blob_rescue_min_score"),
        external_tiny_blob_rescue_max_score=manifest.get("external_tiny_blob_rescue_max_score"),
        external_tiny_blob_rescue_current_max_ball_fill_fraction=manifest.get("external_tiny_blob_rescue_current_max_ball_fill_fraction"),
        external_tiny_blob_rescue_min_pred_area=int(manifest.get("external_tiny_blob_rescue_min_pred_area", 80)),
        external_tiny_blob_rescue_min_prob_mean=float(manifest.get("external_tiny_blob_rescue_min_prob_mean", 0.88)),
        micro_line_rescue_config=manifest.get("micro_line_rescue"),
        colored_blob_rescue_config=manifest.get("colored_blob_rescue"),
        selector_candidate_pool_rescue_config=manifest.get("selector_candidate_pool_rescue"),
        post_external_selector_candidate_pool_rescue_config=manifest.get(
            "post_external_selector_candidate_pool_rescue"
        ),
        reject_table_hough_final_fallback_min_pred_pixels=manifest.get(
            "reject_table_hough_final_fallback_min_pred_pixels"
        ),
        reject_table_hough_final_fallback_max_score=manifest.get(
            "reject_table_hough_final_fallback_max_score"
        ),
        reject_winner_source_min_pred_pixels=manifest.get("reject_winner_source_min_pred_pixels"),
        reject_winner_source_min_pred_pixels_exemptions=manifest.get(
            "reject_winner_source_min_pred_pixels_exemptions"
        ),
        reject_winner_source_area_score=manifest.get("reject_winner_source_area_score"),
        reject_pre_final_fallback_max_score=manifest.get("reject_pre_final_fallback_max_score"),
        reject_final_context_rules=manifest.get("reject_final_context_rules"),
        candidate_selector_rescue_rules=manifest.get("candidate_selector_rescue_rules"),
        rescue_arbiter_config=manifest.get("rescue_arbiter"),
        post_external_candidate_selector_rescue_rules=manifest.get(
            "post_external_candidate_selector_rescue_rules"
        ),
        post_external_rescue_arbiter_config=manifest.get("post_external_rescue_arbiter"),
        post_final_candidate_selector_rescue_rules=manifest.get(
            "post_final_candidate_selector_rescue_rules"
        ),
        final_refine_checkpoint_path=manifest.get("final_refine_checkpoint"),
        final_refine_image_size=manifest.get("final_refine_image_size"),
        image_final_veto_config=manifest.get("image_final_veto"),
        save_outputs=save_outputs,
        save_intermediates=save_intermediates,
        save_report=save_report,
        save_recolor_preview=save_recolor_preview,
        return_overlay=return_overlay,
        crop_batch_size=crop_batch_size,
        amp_enabled=amp_enabled,
    )
