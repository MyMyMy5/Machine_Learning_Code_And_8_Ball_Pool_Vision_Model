from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.crop_utils import remap_mask_to_image
from src.stages.propose_ball_crops import generate_ball_candidates
from src.supervised.infer import (
    _load_guideline_model,
    _load_reranker_cached,
    _rescue_config_should_run,
    _rescue_config_list,
    _score_candidates,
    run_supervised_inference,
)


def _normalize_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    if os.name != "nt" and path_str.startswith("C:\\"):
        return Path("/mnt/c/" + path_str[3:].replace("\\", "/"))
    if os.name != "nt" and path_str.startswith("C:/"):
        return Path("/mnt/c/" + path_str[3:])
    if os.name == "nt" and path_str.startswith("/mnt/c/"):
        return Path("C:/" + path_str[len("/mnt/c/"):])
    return Path(path_str)


def _normalize_rescue_config(
    manifest: dict[str, Any],
    key: str,
) -> dict[str, object] | list[dict[str, object]] | None:
    config = manifest.get(key)
    path_keys = ("checkpoint", "checkpoint_path", "reranker_checkpoint", "reranker_checkpoint_path")

    def normalize_one(value: dict[str, object]) -> dict[str, object]:
        normalized = dict(value)
        for path_key in path_keys:
            if path_key in normalized and normalized[path_key]:
                normalized[path_key] = str(_normalize_path(str(normalized[path_key])))
        return normalized

    if isinstance(config, dict):
        return normalize_one(config)
    if isinstance(config, list):
        return [normalize_one(item) for item in config if isinstance(item, dict)]
    return None


def _load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.uint8)
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def _compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float(np.logical_and(pred > 0, gt > 0).sum())
    union = float(np.logical_or(pred > 0, gt > 0).sum())
    return 0.0 if union == 0.0 else inter / union


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "array_shape": list(value.shape),
            "array_dtype": str(value.dtype),
            "nonzero": int(np.count_nonzero(value)),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def classify_failure(
    *,
    gt_pixels: int,
    final_iou: float,
    final_pred_pixels: int,
    best_candidate_gt_pixels: int,
    best_candidate_iou: float,
    candidate_recall_pixels: int = 12,
    success_iou: float = 0.5,
) -> str:
    if gt_pixels <= 0:
        return "true_negative" if final_pred_pixels <= 0 else "negative_false_positive"
    if final_iou >= success_iou:
        return "success"
    if best_candidate_gt_pixels < candidate_recall_pixels:
        return "candidate_generation_miss"
    if best_candidate_iou >= success_iou:
        return "selector_or_reranker_miss"
    return "segmentation_miss"


def _candidate_config(max_candidates: int) -> dict[str, Any]:
    return {
        "max_ball_candidates": int(max_candidates),
        "crop_scale": 4.25,
        "crop_padding_px": 24,
        "upscale_factor": 1.0,
        "hough": {
            "dp": 1.2,
            "min_dist_factor": 1.5,
            "param1": 110,
            "param2": 18,
            "min_radius": 7,
            "max_radius": 80,
        },
        "fallback": {"grid_candidates": 6, "white_ball_bias": True},
    }


def _candidate_diagnostic_summary(
    candidate: dict[str, Any],
    *,
    gt: np.ndarray,
    image_shape: tuple[int, ...],
) -> dict[str, object]:
    crop_meta = candidate.get("crop_meta")
    candidate_iou = 0.0
    candidate_gt_pixels = 0
    crop_box = None
    if crop_meta is not None:
        full_mask = remap_mask_to_image(candidate["mask"], crop_meta, image_shape)
        candidate_iou = _compute_iou(full_mask, gt)
        x0, y0, x1, y1 = [int(v) for v in crop_meta["crop_box"]]
        crop_box = [x0, y0, x1, y1]
        candidate_gt_pixels = int(gt[y0:y1, x0:x1].sum())
    return {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_source": candidate.get("candidate_source"),
        "selector_pool": candidate.get("selector_pool"),
        "selector_candidate_pool_rescue": _jsonable(candidate.get("selector_candidate_pool_rescue")),
        "selector_candidate_pool_rescue_name": _jsonable(candidate.get("selector_candidate_pool_rescue_name")),
        "post_external_selector_candidate_pool_rescue": _jsonable(
            candidate.get("post_external_selector_candidate_pool_rescue")
        ),
        "post_external_selector_candidate_pool_rescue_name": _jsonable(
            candidate.get("post_external_selector_candidate_pool_rescue_name")
        ),
        "score": _jsonable(candidate.get("score")),
        "heuristic_score": _jsonable(candidate.get("heuristic_score")),
        "reranker_score": _jsonable(candidate.get("reranker_score")),
        "raw_score": _jsonable(candidate.get("raw_score")),
        "candidate_score": _jsonable(candidate.get("candidate_score")),
        "pred_pixels": _jsonable(candidate.get("pred_pixels")),
        "confidence": _jsonable(candidate.get("confidence")),
        "validity_score": _jsonable(candidate.get("validity_score")),
        "candidate_center": _jsonable(candidate.get("candidate_center")),
        "candidate_radius": _jsonable(candidate.get("candidate_radius")),
        "reticle_distance": _jsonable(candidate.get("reticle_distance")),
        "features": _jsonable(candidate.get("features", {})),
        "crop_box": crop_box,
        "candidate_iou": candidate_iou,
        "candidate_gt_pixels": candidate_gt_pixels,
    }


def diagnose_item(
    *,
    item: dict[str, Any],
    manifest: dict[str, Any],
    output_root: Path,
    max_candidates: int,
    image_size: int,
    crop_batch_size: int,
    save_outputs: bool,
    device,
) -> dict[str, object]:
    image_path = _normalize_path(item["image_path"])
    if image_path is None:
        raise ValueError(f"gold-board item has no image_path: {item}")
    image = np.asarray(Image.open(image_path).convert("RGB"))
    gt = _load_mask(_normalize_path(item.get("mask_path")), image.shape[:2])
    item_output_dir = output_root / str(item["id"])
    final = run_supervised_inference(
        checkpoint_path=_normalize_path(manifest["primary_checkpoint"]),
        input_path=image_path,
        output_dir=item_output_dir,
        image_size=image_size,
        reranker_checkpoint_path=_normalize_path(manifest.get("primary_reranker")),
        fallback_checkpoint_path=_normalize_path(manifest.get("fallback_checkpoint")),
        fallback_reranker_checkpoint_path=_normalize_path(manifest.get("fallback_reranker")),
        fallback_score_threshold=manifest.get("fallback_score_threshold"),
        rescue_checkpoint_path=_normalize_path(manifest.get("rescue_checkpoint")),
        rescue_reranker_checkpoint_path=_normalize_path(manifest.get("rescue_reranker")),
        rescue_score_threshold=manifest.get("rescue_score_threshold"),
        rescue_max_ball_candidates=int(manifest.get("rescue_max_ball_candidates", 20)),
        final_fallback_score_threshold=manifest.get("final_fallback_score_threshold"),
        secondary_rescue_checkpoint_path=_normalize_path(manifest.get("secondary_rescue_checkpoint")),
        secondary_rescue_reranker_checkpoint_path=_normalize_path(manifest.get("secondary_rescue_reranker")),
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
        fallback_rescue_checkpoint_path=_normalize_path(manifest.get("fallback_rescue_checkpoint")),
        fallback_rescue_reranker_checkpoint_path=_normalize_path(manifest.get("fallback_rescue_reranker")),
        fallback_rescue_min_score=manifest.get("fallback_rescue_min_score"),
        fallback_rescue_max_score=manifest.get("fallback_rescue_max_score"),
        fallback_rescue_source_group=manifest.get("fallback_rescue_source_group"),
        fallback_rescue_max_ball_candidates=int(manifest.get("fallback_rescue_max_ball_candidates", 20)),
        reticle_fill_rescue_checkpoint_path=_normalize_path(manifest.get("reticle_fill_rescue_checkpoint")),
        reticle_fill_rescue_reranker_checkpoint_path=_normalize_path(manifest.get("reticle_fill_rescue_reranker")),
        reticle_fill_rescue_min_score=manifest.get("reticle_fill_rescue_min_score"),
        reticle_fill_rescue_max_score=manifest.get("reticle_fill_rescue_max_score"),
        reticle_fill_rescue_current_max_ball_fill_fraction=manifest.get("reticle_fill_rescue_current_max_ball_fill_fraction"),
        reticle_fill_rescue_target_min_ball_fill_fraction=manifest.get("reticle_fill_rescue_target_min_ball_fill_fraction"),
        reticle_fill_rescue_max_ball_candidates=int(manifest.get("reticle_fill_rescue_max_ball_candidates", 20)),
        external_rescue_checkpoint_path=_normalize_path(manifest.get("external_rescue_checkpoint")),
        external_rescue_source_group=manifest.get("external_rescue_source_group"),
        external_rescue_min_score=manifest.get("external_rescue_min_score"),
        external_rescue_max_score=manifest.get("external_rescue_max_score"),
        external_rescue_current_max_ball_fill_fraction=manifest.get("external_rescue_current_max_ball_fill_fraction"),
        external_rescue_min_pred_area=int(manifest.get("external_rescue_min_pred_area", 20)),
        external_rescue_min_prob_mean=float(manifest.get("external_rescue_min_prob_mean", 0.9)),
        external_mid_rescue_checkpoint_path=_normalize_path(manifest.get("external_mid_rescue_checkpoint")),
        external_mid_rescue_source_group=manifest.get("external_mid_rescue_source_group"),
        external_mid_rescue_min_score=manifest.get("external_mid_rescue_min_score"),
        external_mid_rescue_max_score=manifest.get("external_mid_rescue_max_score"),
        external_mid_rescue_current_max_ball_fill_fraction=manifest.get("external_mid_rescue_current_max_ball_fill_fraction"),
        external_mid_rescue_min_pred_area=int(manifest.get("external_mid_rescue_min_pred_area", 20)),
        external_mid_rescue_min_prob_mean=float(manifest.get("external_mid_rescue_min_prob_mean", 0.88)),
        external_lowmid_rescue_checkpoint_path=_normalize_path(manifest.get("external_lowmid_rescue_checkpoint")),
        external_lowmid_rescue_source_group=manifest.get("external_lowmid_rescue_source_group"),
        external_lowmid_rescue_min_score=manifest.get("external_lowmid_rescue_min_score"),
        external_lowmid_rescue_max_score=manifest.get("external_lowmid_rescue_max_score"),
        external_lowmid_rescue_current_max_ball_fill_fraction=manifest.get("external_lowmid_rescue_current_max_ball_fill_fraction"),
        external_lowmid_rescue_min_pred_area=int(manifest.get("external_lowmid_rescue_min_pred_area", 400)),
        external_lowmid_rescue_min_prob_mean=float(manifest.get("external_lowmid_rescue_min_prob_mean", 0.8)),
        external_reticle_rescue_checkpoint_path=_normalize_path(manifest.get("external_reticle_rescue_checkpoint")),
        external_reticle_rescue_source_group=manifest.get("external_reticle_rescue_source_group"),
        external_reticle_rescue_min_score=manifest.get("external_reticle_rescue_min_score"),
        external_reticle_rescue_max_score=manifest.get("external_reticle_rescue_max_score"),
        external_reticle_rescue_current_max_ball_fill_fraction=manifest.get("external_reticle_rescue_current_max_ball_fill_fraction"),
        external_reticle_rescue_min_pred_area=int(manifest.get("external_reticle_rescue_min_pred_area", 20)),
        external_reticle_rescue_min_prob_mean=float(manifest.get("external_reticle_rescue_min_prob_mean", 0.8)),
        external_tiny_reticle_rescue_checkpoint_path=_normalize_path(manifest.get("external_tiny_reticle_rescue_checkpoint")),
        external_tiny_reticle_rescue_source_group=manifest.get("external_tiny_reticle_rescue_source_group"),
        external_tiny_reticle_rescue_min_score=manifest.get("external_tiny_reticle_rescue_min_score"),
        external_tiny_reticle_rescue_max_score=manifest.get("external_tiny_reticle_rescue_max_score"),
        external_tiny_reticle_rescue_current_max_ball_fill_fraction=manifest.get("external_tiny_reticle_rescue_current_max_ball_fill_fraction"),
        external_tiny_reticle_rescue_min_pred_area=int(manifest.get("external_tiny_reticle_rescue_min_pred_area", 50)),
        external_tiny_reticle_rescue_min_prob_mean=float(manifest.get("external_tiny_reticle_rescue_min_prob_mean", 0.94)),
        external_reticle_large_rescue_checkpoint_path=_normalize_path(manifest.get("external_reticle_large_rescue_checkpoint")),
        external_reticle_large_rescue_source_group=manifest.get("external_reticle_large_rescue_source_group"),
        external_reticle_large_rescue_min_score=manifest.get("external_reticle_large_rescue_min_score"),
        external_reticle_large_rescue_max_score=manifest.get("external_reticle_large_rescue_max_score"),
        external_reticle_large_rescue_current_max_ball_fill_fraction=manifest.get("external_reticle_large_rescue_current_max_ball_fill_fraction"),
        external_reticle_large_rescue_min_pred_area=int(manifest.get("external_reticle_large_rescue_min_pred_area", 400)),
        external_reticle_large_rescue_min_prob_mean=float(manifest.get("external_reticle_large_rescue_min_prob_mean", 0.86)),
        external_table_broad_rescue_checkpoint_path=_normalize_path(manifest.get("external_table_broad_rescue_checkpoint")),
        external_table_broad_rescue_source_group=manifest.get("external_table_broad_rescue_source_group"),
        external_table_broad_rescue_min_score=manifest.get("external_table_broad_rescue_min_score"),
        external_table_broad_rescue_max_score=manifest.get("external_table_broad_rescue_max_score"),
        external_table_broad_rescue_current_max_ball_fill_fraction=manifest.get("external_table_broad_rescue_current_max_ball_fill_fraction"),
        external_table_broad_rescue_min_pred_area=int(manifest.get("external_table_broad_rescue_min_pred_area", 350)),
        external_table_broad_rescue_min_prob_mean=float(manifest.get("external_table_broad_rescue_min_prob_mean", 0.8)),
        external_blob_low_rescue_checkpoint_path=_normalize_path(manifest.get("external_blob_low_rescue_checkpoint")),
        external_blob_low_rescue_source_group=manifest.get("external_blob_low_rescue_source_group"),
        external_blob_low_rescue_min_score=manifest.get("external_blob_low_rescue_min_score"),
        external_blob_low_rescue_max_score=manifest.get("external_blob_low_rescue_max_score"),
        external_blob_low_rescue_current_max_ball_fill_fraction=manifest.get("external_blob_low_rescue_current_max_ball_fill_fraction"),
        external_blob_low_rescue_min_pred_area=int(manifest.get("external_blob_low_rescue_min_pred_area", 20)),
        external_blob_low_rescue_min_prob_mean=float(manifest.get("external_blob_low_rescue_min_prob_mean", 0.75)),
        external_table_rescue_checkpoint_path=_normalize_path(manifest.get("external_table_rescue_checkpoint")),
        external_table_rescue_source_group=manifest.get("external_table_rescue_source_group"),
        external_table_rescue_min_score=manifest.get("external_table_rescue_min_score"),
        external_table_rescue_max_score=manifest.get("external_table_rescue_max_score"),
        external_table_rescue_current_max_ball_fill_fraction=manifest.get("external_table_rescue_current_max_ball_fill_fraction"),
        external_table_rescue_min_pred_area=int(manifest.get("external_table_rescue_min_pred_area", 120)),
        external_table_rescue_min_prob_mean=float(manifest.get("external_table_rescue_min_prob_mean", 0.84)),
        external_blob_rescue_checkpoint_path=_normalize_path(manifest.get("external_blob_rescue_checkpoint")),
        external_blob_rescue_source_group=manifest.get("external_blob_rescue_source_group"),
        external_blob_rescue_min_score=manifest.get("external_blob_rescue_min_score"),
        external_blob_rescue_max_score=manifest.get("external_blob_rescue_max_score"),
        external_blob_rescue_current_max_ball_fill_fraction=manifest.get("external_blob_rescue_current_max_ball_fill_fraction"),
        external_blob_rescue_min_pred_area=int(manifest.get("external_blob_rescue_min_pred_area", 300)),
        external_blob_rescue_min_prob_mean=float(manifest.get("external_blob_rescue_min_prob_mean", 0.95)),
        external_tiny_blob_rescue_checkpoint_path=_normalize_path(manifest.get("external_tiny_blob_rescue_checkpoint")),
        external_tiny_blob_rescue_source_group=manifest.get("external_tiny_blob_rescue_source_group"),
        external_tiny_blob_rescue_min_score=manifest.get("external_tiny_blob_rescue_min_score"),
        external_tiny_blob_rescue_max_score=manifest.get("external_tiny_blob_rescue_max_score"),
        external_tiny_blob_rescue_current_max_ball_fill_fraction=manifest.get("external_tiny_blob_rescue_current_max_ball_fill_fraction"),
        external_tiny_blob_rescue_min_pred_area=int(manifest.get("external_tiny_blob_rescue_min_pred_area", 80)),
        external_tiny_blob_rescue_min_prob_mean=float(manifest.get("external_tiny_blob_rescue_min_prob_mean", 0.88)),
        micro_line_rescue_config=_normalize_rescue_config(manifest, "micro_line_rescue"),
        colored_blob_rescue_config=_normalize_rescue_config(manifest, "colored_blob_rescue"),
        selector_candidate_pool_rescue_config=_normalize_rescue_config(manifest, "selector_candidate_pool_rescue"),
        post_external_selector_candidate_pool_rescue_config=_normalize_rescue_config(
            manifest,
            "post_external_selector_candidate_pool_rescue",
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
        rescue_arbiter_config=_normalize_rescue_config(manifest, "rescue_arbiter"),
        post_external_candidate_selector_rescue_rules=manifest.get("post_external_candidate_selector_rescue_rules"),
        post_external_rescue_arbiter_config=_normalize_rescue_config(manifest, "post_external_rescue_arbiter"),
        post_final_candidate_selector_rescue_rules=manifest.get("post_final_candidate_selector_rescue_rules"),
        image_final_veto_config=_normalize_rescue_config(manifest, "image_final_veto"),
        save_outputs=save_outputs,
        save_intermediates=False,
        save_report=save_outputs,
        crop_batch_size=crop_batch_size,
    )
    final_mask = np.asarray(final["final_mask"], dtype=np.uint8)
    final_iou = _compute_iou(final_mask, gt)
    current_winner = final["winner"]
    model = _load_guideline_model(_normalize_path(manifest["primary_checkpoint"]), device)
    reranker = None
    if manifest.get("primary_reranker"):
        reranker = _load_reranker_cached(_normalize_path(manifest["primary_reranker"]), device)
    candidates = generate_ball_candidates(image, _candidate_config(max_candidates))
    _, _, candidate_items = _score_candidates(
        model=model,
        image=image,
        candidates=candidates,
        output_dir=output_root / str(item["id"]) / "primary_candidates",
        image_size=image_size,
        device=device,
        reranker=reranker,
        save_intermediates=False,
        crop_batch_size=crop_batch_size,
        return_items=True,
    )
    pool_config_groups = [
        ("selector_candidate_pool_rescue", "selector_pool_candidates", ""),
        (
            "post_external_selector_candidate_pool_rescue",
            "post_external_selector_pool_candidates",
            "post_external_",
        ),
    ]
    for config_key, output_name, stage_prefix in pool_config_groups:
        selector_pool_config = _normalize_rescue_config(manifest, config_key)
        for pool_index, pool_config in enumerate(_rescue_config_list(selector_pool_config)):
            if not _rescue_config_should_run(current_winner, pool_config):
                continue
            checkpoint_value = pool_config.get("checkpoint_path") or pool_config.get("checkpoint")
            if not checkpoint_value:
                continue
            pool_name = str(pool_config.get("pool", pool_config.get("name", f"{output_name}_{pool_index}")))
            pool_candidates = generate_ball_candidates(
                image,
                {
                    **_candidate_config(int(pool_config.get("max_ball_candidates", max_candidates))),
                    "crop_scale": float(pool_config.get("crop_scale", 4.25)),
                    "crop_padding_px": int(pool_config.get("crop_padding_px", 24)),
                    "line_endpoint_candidates": bool(pool_config.get("line_endpoint_candidates", False)),
                    "line_endpoint": pool_config.get("line_endpoint", {}),
                    "colored_blob_candidates": bool(pool_config.get("colored_blob_candidates", False)),
                    "colored_blob": pool_config.get("colored_blob", {}),
                },
            )
            pool_model = _load_guideline_model(_normalize_path(str(checkpoint_value)), device)
            pool_reranker = None
            if bool(pool_config.get("use_reranker", True)):
                reranker_value = pool_config.get("reranker_checkpoint_path") or pool_config.get("reranker_checkpoint")
                reranker_path = _normalize_path(str(reranker_value)) if reranker_value else _normalize_path(str(checkpoint_value)).with_name("guideline_reranker_best.pt")
                if reranker_path is not None and reranker_path.exists():
                    pool_reranker = _load_reranker_cached(reranker_path, device)
            _, _, pool_candidate_items = _score_candidates(
                model=pool_model,
                image=image,
                candidates=pool_candidates,
                output_dir=output_root / str(item["id"]) / f"{output_name}_{pool_index:02d}",
                image_size=int(pool_config.get("image_size", image_size)),
                device=device,
                reranker=pool_reranker,
                save_intermediates=False,
                crop_batch_size=crop_batch_size,
                prediction_threshold=float(pool_config.get("prediction_threshold", 0.35)),
                return_items=True,
            )
            for candidate_item in pool_candidate_items:
                candidate_item["selector_pool"] = pool_name
                candidate_item["selector_candidate_pool_rescue"] = True
                candidate_item["selector_candidate_pool_rescue_name"] = str(pool_config.get("name", pool_name))
                if stage_prefix:
                    candidate_item[f"{stage_prefix}selector_candidate_pool_rescue"] = True
                    candidate_item[f"{stage_prefix}selector_candidate_pool_rescue_name"] = str(
                        pool_config.get("name", pool_name)
                    )
            candidate_items.extend(pool_candidate_items)
    best_candidate_gt_pixels = 0
    best_candidate_iou = 0.0
    candidate_summaries: list[dict[str, object]] = []
    best_candidate_summary: dict[str, object] | None = None
    for candidate in candidate_items:
        summary = _candidate_diagnostic_summary(candidate, gt=gt, image_shape=image.shape)
        candidate_summaries.append(summary)
        candidate_iou = float(summary["candidate_iou"])
        candidate_gt_pixels = int(summary["candidate_gt_pixels"])
        if candidate_iou > best_candidate_iou:
            best_candidate_iou = candidate_iou
            best_candidate_summary = summary
        best_candidate_gt_pixels = max(best_candidate_gt_pixels, candidate_gt_pixels)
    candidate_summaries_by_iou = sorted(
        candidate_summaries,
        key=lambda row: (float(row["candidate_iou"]), float(row.get("score") or -1e9)),
        reverse=True,
    )
    candidate_summaries_by_score = sorted(
        candidate_summaries,
        key=lambda row: float(row.get("score") or -1e9),
        reverse=True,
    )
    label = classify_failure(
        gt_pixels=int(gt.sum()),
        final_iou=final_iou,
        final_pred_pixels=int(final_mask.sum()),
        best_candidate_gt_pixels=best_candidate_gt_pixels,
        best_candidate_iou=best_candidate_iou,
    )
    return {
        "id": item["id"],
        "split_source": item.get("source"),
        "kind": item.get("kind"),
        "image_path": str(image_path),
        "mask_path": str(_normalize_path(item.get("mask_path"))) if item.get("mask_path") else None,
        "failure_type": label,
        "final_iou": final_iou,
        "final_pred_pixels": int(final_mask.sum()),
        "gt_pixels": int(gt.sum()),
        "best_candidate_iou": best_candidate_iou,
        "best_candidate_gt_pixels": best_candidate_gt_pixels,
        "best_candidate": best_candidate_summary,
        "top_candidates_by_iou": candidate_summaries_by_iou[:8],
        "top_candidates_by_score": candidate_summaries_by_score[:8],
        "winner": _jsonable(final["winner"]),
        "output_dir": str(item_output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-board", default="runs/gold_board_v1/gold_board.json")
    parser.add_argument("--manifest", default="runs/supervised_best_manifest.json")
    parser.add_argument("--output-root", default="runs/gold_board_v1/current_best_diagnostics")
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-batch-size", type=int, default=32)
    parser.add_argument("--save-outputs", action="store_true")
    args = parser.parse_args()

    import torch

    board = json.loads(Path(args.gold_board).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    splits = args.split or list(board["splits"].keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for split in splits:
        for item in board["splits"].get(split, []):
            if args.limit is not None and len(rows) >= args.limit:
                break
            rows.append(
                diagnose_item(
                    item=item,
                    manifest=manifest,
                    output_root=output_root,
                    max_candidates=args.max_candidates,
                    image_size=args.image_size,
                    crop_batch_size=args.crop_batch_size,
                    save_outputs=args.save_outputs,
                    device=device,
                )
            )
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["failure_type"])] = counts.get(str(row["failure_type"]), 0) + 1
    payload = {"count": len(rows), "failure_counts": counts, "rows": rows}
    (output_root / "diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(rows), "failure_counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
