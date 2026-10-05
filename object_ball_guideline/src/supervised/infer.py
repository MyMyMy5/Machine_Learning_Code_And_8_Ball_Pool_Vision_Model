from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.pipeline.color_recolor import recolor_masked_region
from src.pipeline.crop_utils import extract_crop, local_ball_mask, remap_mask_to_image
from src.pipeline.debug_viz import draw_ball_candidates, overlay_mask
from src.pipeline.io_utils import ensure_dir, load_rgb_image, save_json, save_mask_png, save_rgb_image
from src.pipeline.mask_postprocess import cleanup_mask, soft_alpha_mask
from src.pipeline.metrics import compute_mask_features
from src.supervised.conditioned import ConditionedGuidelineUNet
from src.supervised.conditioned_data import build_candidate_heatmap
from src.supervised.image_final_veto import build_veto_tensor, load_image_final_veto_checkpoint
from src.supervised.meta_selector import load_meta_selector_checkpoint
from src.supervised.reranker import build_reranker_features, load_reranker_checkpoint
from src.supervised.rescue_arbiter import LoadedRescueArbiter, load_rescue_arbiter_checkpoint
from src.stages.propose_ball_crops import _detect_table_roi, generate_ball_candidates
from src.supervised.model import GuidelineUNet, build_guideline_model

_EXTERNAL_GUIDELINE_RUNTIME_CACHE: dict[str, dict[str, object]] = {}
_GUIDELINE_MODEL_CACHE: dict[str, torch.nn.Module] = {}
_RERANKER_CACHE: dict[str, object] = {}
_META_SELECTOR_CACHE: dict[str, object] = {}
_IMAGE_FINAL_VETO_CACHE: dict[str, tuple[torch.nn.Module, dict[str, object]]] = {}
_RESCUE_ARBITER_CACHE: dict[str, LoadedRescueArbiter] = {}


def _resize_image(image: np.ndarray, image_size: int) -> torch.Tensor:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB").resize((image_size, image_size))
    array = np.asarray(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1))


def _model_uses_conditioning(model: torch.nn.Module) -> bool:
    return isinstance(model, ConditionedGuidelineUNet)


def _heatmap_from_crop_meta(crop_meta: dict[str, object], crop_shape: tuple[int, int], image_size: int) -> torch.Tensor:
    height, width = crop_shape
    local_center = crop_meta.get("local_center")
    local_radius = crop_meta.get("local_radius")
    heatmap = build_candidate_heatmap(
        crop_box=[0, 0, int(width), int(height)],
        candidate_center=[int(local_center[0]), int(local_center[1])] if local_center is not None else None,
        candidate_radius=int(local_radius) if local_radius is not None else None,
        output_size=image_size,
    )
    return torch.from_numpy(heatmap[None, ...])


def _predict_crop_outputs(
    model: torch.nn.Module,
    crops: list[np.ndarray],
    image_size: int,
    device: torch.device,
    *,
    crop_metas: list[dict[str, object]] | None = None,
    crop_batch_size: int,
    amp_enabled: bool,
) -> tuple[list[np.ndarray], list[float | None]]:
    if not crops:
        return [], []

    batch_size = max(1, int(crop_batch_size))
    outputs: list[np.ndarray] = []
    validity_scores: list[float | None] = []
    for start in range(0, len(crops), batch_size):
        batch_crops = crops[start : start + batch_size]
        rgb_tensors = [_resize_image(crop, image_size) for crop in batch_crops]
        if _model_uses_conditioning(model):
            if crop_metas is None:
                raise ValueError("Conditioned model requires crop metadata")
            batch_metas = crop_metas[start : start + batch_size]
            heatmaps = [
                _heatmap_from_crop_meta(meta, crop.shape[:2], image_size)
                for crop, meta in zip(batch_crops, batch_metas, strict=True)
            ]
            batch_tensor = torch.stack(
                [torch.cat([rgb, heatmap], dim=0) for rgb, heatmap in zip(rgb_tensors, heatmaps, strict=True)],
                dim=0,
            ).to(device)
        else:
            batch_tensor = torch.stack(rgb_tensors, dim=0).to(device)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda" and amp_enabled
            else nullcontext()
        )
        with torch.inference_mode():
            with autocast_context:
                model_output = model(batch_tensor)
            if isinstance(model_output, dict):
                logits = model_output["mask_logits"]
                validity = torch.sigmoid(model_output["validity_logits"]).detach().cpu().flatten().tolist()
            else:
                logits = model_output
                validity = [None] * len(batch_crops)
            probs = torch.sigmoid(logits[:, 0]).detach().cpu().numpy()
        for crop, prob in zip(batch_crops, probs, strict=True):
            pil = Image.fromarray((prob * 255).astype(np.uint8), mode="L").resize((crop.shape[1], crop.shape[0]))
            outputs.append(np.asarray(pil).astype(np.float32) / 255.0)
        validity_scores.extend(float(item) if item is not None else None for item in validity)
    return outputs, validity_scores


def _predict_crops(
    model: torch.nn.Module,
    crops: list[np.ndarray],
    image_size: int,
    device: torch.device,
    *,
    crop_metas: list[dict[str, object]] | None = None,
    crop_batch_size: int,
    amp_enabled: bool,
) -> list[np.ndarray]:
    probs, _ = _predict_crop_outputs(
        model,
        crops,
        image_size,
        device,
        crop_metas=crop_metas,
        crop_batch_size=crop_batch_size,
        amp_enabled=amp_enabled,
    )
    return probs


def _predict_crop(
    model: GuidelineUNet,
    crop: np.ndarray,
    image_size: int,
    device: torch.device,
    *,
    amp_enabled: bool = True,
) -> np.ndarray:
    return _predict_crops(
        model,
        [crop],
        image_size,
        device,
        crop_batch_size=1,
        amp_enabled=amp_enabled,
    )[0]


def _refine_crop_winner_mask(
    *,
    best: dict[str, object],
    image: np.ndarray,
    model: torch.nn.Module,
    image_size: int,
    device: torch.device,
    amp_enabled: bool,
) -> bool:
    if "full_frame_mask" in best or "crop_meta" not in best:
        return False
    crop_meta = best["crop_meta"]
    if not isinstance(crop_meta, dict) or "crop_box" not in crop_meta:
        return False
    x0, y0, x1, y1 = [int(value) for value in crop_meta["crop_box"]]
    crop_image = image[y0:y1, x0:x1]
    if crop_image.size == 0:
        return False

    ball_mask = local_ball_mask(crop_meta)
    prob = _predict_crop(model, crop_image, image_size, device, amp_enabled=amp_enabled)
    raw_score, binary = _score_prediction(prob, ball_mask)
    cleaned = cleanup_mask(
        binary,
        {
            "morphology_open_radius": 0,
            "morphology_close_radius": 1,
            "min_component_area": 4,
            "keep_only_best_component": True,
        },
        ball_mask=ball_mask,
    )
    refined_pixels = int(np.count_nonzero(cleaned))
    best["selected_via_final_refine"] = True
    best["final_refine_raw_score"] = float(raw_score)
    best["final_refine_pred_pixels"] = refined_pixels
    best["pre_final_refine_pred_pixels"] = int(np.count_nonzero(best.get("mask", np.zeros(0, dtype=np.uint8))))
    if refined_pixels <= 0:
        best["selected_via_final_refine_empty_keep_original"] = True
        return False

    features = compute_mask_features(
        cleaned,
        ball_mask,
        tuple(crop_meta["local_center"]),
        int(crop_meta["local_radius"]),
    )
    best["pre_final_refine_features"] = dict(best.get("features", {}))
    best["mask"] = cleaned
    best["features"] = features
    return True


def _score_prediction(
    prob: np.ndarray,
    ball_mask: np.ndarray,
    threshold: float = 0.35,
) -> tuple[float, np.ndarray]:
    binary = (prob >= float(threshold)).astype(np.uint8)
    if binary.sum() == 0:
        return 0.0, binary
    overlap = float(np.count_nonzero(binary & ball_mask) / max(1, np.count_nonzero(ball_mask)))
    score = float(prob[binary > 0].mean()) + 0.8 * overlap + 0.002 * np.count_nonzero(binary)
    return score, binary


def _rank_prediction(raw_score: float, candidate_score: float, features: dict[str, float]) -> float:
    connection = (
        features.get("ball_boundary_touch", 0.0)
        + features.get("connected_to_ball", 0.0)
        + features.get("ball_core_overlap", 0.0)
    )
    return (
        float(np.log1p(max(raw_score, 0.0)))
        + 0.7 * float(candidate_score)
        + 0.2 * float(features.get("continuity", 0.0))
        + 1.2 * float(connection)
        + 0.1 * float(features.get("outward_extension", 0.0))
    )


def _candidate_reticle_distance(candidate) -> float | None:
    reticle_center = candidate.metadata.get("reticle_center")
    reticle_radius = candidate.metadata.get("reticle_radius")
    if not reticle_center or not reticle_radius:
        return None
    return float(
        math.hypot(
            candidate.center_x - reticle_center[0],
            candidate.center_y - reticle_center[1],
        )
        / max(float(reticle_radius), 1.0)
    )


def _secondary_rescue_is_eligible(
    item: dict[str, object],
    min_ball_fill_fraction: float | None,
) -> bool:
    if min_ball_fill_fraction is None:
        return True
    features = item.get("features")
    if not isinstance(features, dict):
        return False
    ball_fill_fraction = float(features.get("ball_fill_fraction", 0.0))
    return ball_fill_fraction >= float(min_ball_fill_fraction)


def _rescue_config_should_run(
    current_item: dict[str, object],
    config: dict[str, object] | None,
) -> bool:
    if not config:
        return False
    source = current_item.get("candidate_source")
    if not isinstance(source, str):
        return False
    current_source_rule = config.get("current_source")
    current_source_group_rule = config.get("current_source_group")
    if (
        (current_source_rule is not None or current_source_group_rule is not None)
        and not _source_matches(
            source,
            exact=current_source_rule,
            group=current_source_group_rule,
        )
    ):
        return False
    required_flags = config.get("current_required_flags", config.get("current_required_flag"))
    if isinstance(required_flags, list):
        if not all(_required_flag_present(current_item, flag) for flag in required_flags):
            return False
    elif required_flags is not None and not _required_flag_present(current_item, required_flags):
        return False
    return _thresholds_match_item(current_item, config, prefix="current")


def _rescue_config_list(config: object) -> list[dict[str, object]]:
    if isinstance(config, dict):
        return [config]
    if isinstance(config, list):
        return [item for item in config if isinstance(item, dict)]
    return []


def _current_item_exemption_matches(
    item: dict[str, object],
    exemptions: object,
) -> tuple[bool, dict[str, object] | None]:
    for rule in _rescue_config_list(exemptions):
        source = rule.get("source", rule.get("current_source"))
        source_group = rule.get("source_group", rule.get("current_source_group"))
        if (source is not None or source_group is not None) and not _source_matches(
            item.get("candidate_source"),
            exact=source,
            group=source_group,
        ):
            continue
        required_flags = rule.get("required_flags", rule.get("current_required_flags", rule.get("current_required_flag")))
        if isinstance(required_flags, list):
            if not all(_required_flag_present(item, flag) for flag in required_flags):
                continue
        elif required_flags is not None and not _required_flag_present(item, required_flags):
            continue
        if _thresholds_match_unprefixed_item(item, rule):
            return True, rule
    return False, None


def _micro_line_rescue_should_run(
    current_item: dict[str, object],
    config: dict[str, object] | None,
) -> bool:
    return _rescue_config_should_run(current_item, config)


def _candidate_source_group(source: object) -> str:
    source_str = str(source or "")
    if source_str.startswith("reticle_global"):
        return "reticle_global"
    return source_str


def _external_rescue_is_eligible(
    current_item: dict[str, object],
    *,
    source_group: str | None,
    min_score: float | None,
    max_score: float | None,
    current_max_ball_fill_fraction: float | None,
) -> bool:
    if source_group is not None and _candidate_source_group(current_item.get("candidate_source")) != source_group:
        return False
    if min_score is not None and float(current_item["score"]) < float(min_score):
        return False
    if max_score is not None and float(current_item["score"]) >= float(max_score):
        return False
    if (
        current_max_ball_fill_fraction is not None
        and float(current_item.get("features", {}).get("ball_fill_fraction", 0.0))
        > float(current_max_ball_fill_fraction)
    ):
        return False
    return True


def _table_hough_final_fallback_reject_is_eligible(
    current_item: dict[str, object],
    *,
    pred_pixels: int,
    min_pred_pixels: int | None,
    max_score: float | None = None,
) -> bool:
    if _candidate_source_group(current_item.get("candidate_source")) != "table_hough":
        return False
    if not bool(current_item.get("selected_via_final_fallback", False)):
        return False
    large_mask_match = min_pred_pixels is not None and int(pred_pixels) >= int(min_pred_pixels)
    low_score_match = max_score is not None and float(current_item["score"]) < float(max_score)
    return large_mask_match or low_score_match


def _winner_source_area_reject_is_eligible(
    current_item: dict[str, object],
    *,
    pred_pixels: int,
    source_min_pred_pixels: dict[str, int] | None,
    source_exemptions: dict[str, list[dict[str, object]]] | None = None,
) -> bool:
    if not source_min_pred_pixels:
        return False
    source = str(current_item.get("candidate_source") or "")
    min_pred_pixels = source_min_pred_pixels.get(source)
    if min_pred_pixels is None:
        min_pred_pixels = source_min_pred_pixels.get(_candidate_source_group(source))
    if min_pred_pixels is None or int(pred_pixels) < int(min_pred_pixels):
        return False
    exemptions: list[dict[str, object]] = []
    if source_exemptions:
        exact_exemptions = source_exemptions.get(source)
        group_exemptions = source_exemptions.get(_candidate_source_group(source))
        if isinstance(exact_exemptions, list):
            exemptions.extend(exact_exemptions)
        if isinstance(group_exemptions, list) and group_exemptions is not exact_exemptions:
            exemptions.extend(group_exemptions)
    if exemptions:
        item_with_pred_pixels = dict(current_item)
        item_with_pred_pixels["pred_pixels"] = int(pred_pixels)
        for exemption in exemptions:
            if isinstance(exemption, dict) and _thresholds_match_unprefixed_item(
                item_with_pred_pixels,
                exemption,
            ):
                return False
    return True


def _winner_source_area_score_reject_is_eligible(
    current_item: dict[str, object],
    *,
    pred_pixels: int,
    source_rules: dict[str, dict[str, object]] | None,
) -> bool:
    if not source_rules:
        return False
    source = str(current_item.get("candidate_source") or "")
    rule = source_rules.get(source)
    if rule is None:
        rule = source_rules.get(_candidate_source_group(source))
    if rule is None:
        return False
    min_pred_pixels = rule.get("min_pred_pixels")
    max_score = rule.get("max_score")
    if min_pred_pixels is None or max_score is None:
        return False
    if int(pred_pixels) < int(min_pred_pixels) or float(current_item["score"]) >= float(max_score):
        return False
    exemptions = rule.get("exemptions")
    if isinstance(exemptions, list):
        for exemption in exemptions:
            if isinstance(exemption, dict) and _thresholds_match_unprefixed_item(current_item, exemption):
                return False
    return True


def _image_final_veto_min_pred_pixels_for_source(config: dict[str, object], source: str) -> int | None:
    source_min_pred_pixels = config.get("source_min_pred_pixels")
    if isinstance(source_min_pred_pixels, dict):
        min_pred_pixels = source_min_pred_pixels.get(source)
        if min_pred_pixels is None:
            min_pred_pixels = source_min_pred_pixels.get(_candidate_source_group(source))
        if isinstance(min_pred_pixels, (int, float)):
            return int(min_pred_pixels)
    min_pred_pixels = config.get("min_pred_pixels")
    if isinstance(min_pred_pixels, (int, float)):
        return int(min_pred_pixels)
    return None


def _pre_final_fallback_score_reject_is_eligible(
    current_item: dict[str, object],
    *,
    pred_pixels: int,
    max_score: float | None,
) -> bool:
    if max_score is None or int(pred_pixels) <= 0:
        return False
    score = current_item.get("pre_final_fallback_score")
    return isinstance(score, (int, float)) and float(score) <= float(max_score)


def _candidate_circles_from_summaries(
    summaries: list[dict[str, object]],
    current_item: dict[str, object],
) -> list[tuple[float, float, float]]:
    circles: list[tuple[float, float, float]] = []
    for item in [*summaries, current_item]:
        center = item.get("candidate_center")
        radius = item.get("candidate_radius")
        if (
            isinstance(center, list)
            and len(center) == 2
            and isinstance(center[0], (int, float))
            and isinstance(center[1], (int, float))
            and isinstance(radius, (int, float))
        ):
            circles.append((float(center[0]), float(center[1]), max(1.0, float(radius))))
    return circles


def _compute_final_context_features(
    *,
    image: np.ndarray,
    final_mask: np.ndarray,
    summaries: list[dict[str, object]],
    current_item: dict[str, object],
) -> dict[str, float]:
    mask = final_mask > 0
    pred_pixels = int(np.count_nonzero(mask))
    features: dict[str, float] = {"pred_pixels": float(pred_pixels)}
    if pred_pixels <= 0:
        return features

    ys, xs = np.where(mask)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    white = (hsv[..., 1] < 80) & (hsv[..., 2] > 150)
    features["mask_white_fraction"] = float(np.count_nonzero(white[mask]) / max(pred_pixels, 1))

    try:
        x0, y0, x1, y1 = _detect_table_roi(image)
    except Exception:
        height, width = image.shape[:2]
        x0, y0, x1, y1 = (0, 0, width, height)
    height, width = image.shape[:2]
    in_table = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    features["table_roi_area_fraction"] = float(((x1 - x0) * (y1 - y0)) / max(width * height, 1))
    features["mask_fraction_in_table_roi"] = float(np.count_nonzero(in_table) / max(pred_pixels, 1))

    circles = _candidate_circles_from_summaries(summaries, current_item)
    features["candidate_count"] = float(len(circles))
    if not circles:
        features["min_candidate_center_distance"] = float("inf")
        features["min_abs_candidate_boundary_distance"] = float("inf")
        features["mask_fraction_near_any_candidate"] = 0.0
        return features

    point_x = xs.astype(np.float32)
    point_y = ys.astype(np.float32)
    near_any = np.zeros(pred_pixels, dtype=bool)
    min_center_distance = float("inf")
    min_abs_boundary_distance = float("inf")
    for cx, cy, radius in circles:
        distances = np.sqrt((point_x - cx) ** 2 + (point_y - cy) ** 2)
        candidate_min_center = float(np.min(distances))
        min_center_distance = min(min_center_distance, candidate_min_center)
        min_abs_boundary_distance = min(min_abs_boundary_distance, abs(candidate_min_center - radius))
        near_radius = radius + max(6.0, radius * 0.45)
        near_any |= distances <= near_radius

    features["min_candidate_center_distance"] = float(min_center_distance)
    features["min_abs_candidate_boundary_distance"] = float(min_abs_boundary_distance)
    features["mask_fraction_near_any_candidate"] = float(np.count_nonzero(near_any) / max(pred_pixels, 1))
    return features


def _source_matches(source: str, *, exact: object = None, group: object = None) -> bool:
    if isinstance(exact, str) and source == exact:
        return True
    if isinstance(group, str) and (source == group or source.startswith(f"{group}_")):
        return True
    return False


def _final_context_reject_is_eligible(
    current_item: dict[str, object],
    *,
    context_features: dict[str, float],
    source_rules: list[dict[str, object]] | None,
) -> tuple[bool, dict[str, object] | None]:
    if not source_rules:
        return False, None
    source = current_item.get("candidate_source")
    if not isinstance(source, str):
        return False, None
    for rule in source_rules:
        if not isinstance(rule, dict):
            continue
        if not _source_matches(source, exact=rule.get("source"), group=rule.get("source_group")):
            continue
        matched = True
        for key, threshold in rule.items():
            if key in {"name", "source", "source_group"} or threshold is None:
                continue
            if key.startswith("min_"):
                feature_key = key.removeprefix("min_")
                value = context_features.get(feature_key)
                matched = isinstance(value, (int, float)) and float(value) >= float(threshold)
            elif key.startswith("max_"):
                feature_key = key.removeprefix("max_")
                value = context_features.get(feature_key)
                matched = isinstance(value, (int, float)) and float(value) <= float(threshold)
            else:
                raise ValueError(f"Unsupported final context reject rule key: {key}")
            if not matched:
                break
        if matched:
            return True, rule
    return False, None


def _numeric_item_value(item: dict[str, object], key: str) -> float | None:
    value = item.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    features = item.get("features")
    if isinstance(features, dict):
        feature_value = features.get(key)
        if isinstance(feature_value, (int, float)):
            return float(feature_value)
    return None


def _thresholds_match_item(
    item: dict[str, object],
    rule: dict[str, object],
    *,
    prefix: str,
) -> bool:
    min_prefix = f"min_{prefix}_"
    max_prefix = f"max_{prefix}_"
    for key, threshold in rule.items():
        if threshold is None:
            continue
        if key.startswith(min_prefix):
            value = _numeric_item_value(item, key.removeprefix(min_prefix))
            if not isinstance(value, (int, float)) or float(value) < float(threshold):
                return False
        elif key.startswith(max_prefix):
            value = _numeric_item_value(item, key.removeprefix(max_prefix))
            if not isinstance(value, (int, float)) or float(value) > float(threshold):
                return False
    return True


def _thresholds_match_unprefixed_item(item: dict[str, object], rule: dict[str, object]) -> bool:
    for key, threshold in rule.items():
        if threshold is None or key in {"name", "description"}:
            continue
        if key.startswith("min_"):
            value = _numeric_item_value(item, key.removeprefix("min_"))
            if not isinstance(value, (int, float)) or float(value) < float(threshold):
                return False
        elif key.startswith("max_"):
            value = _numeric_item_value(item, key.removeprefix("max_"))
            if not isinstance(value, (int, float)) or float(value) > float(threshold):
                return False
    return True


def _required_flag_present(item: dict[str, object], flag: object) -> bool:
    if not isinstance(flag, str) or not flag:
        return True
    flag_key = flag if flag.startswith("selected_via_") else f"selected_via_{flag}"
    return bool(item.get(flag_key, False))


def _pool_matches(item: dict[str, object], expected: object) -> bool:
    if expected is None:
        return True
    pool = item.get("selector_pool")
    if not isinstance(pool, str):
        return False
    if isinstance(expected, str):
        return pool == expected
    if isinstance(expected, list):
        return pool in {str(value) for value in expected}
    return False


def _uses_selector_pool(rule: dict[str, object]) -> bool:
    return "candidate_pool" in rule or "candidate_pools" in rule


def _candidate_selector_rescue_is_eligible(
    current_item: dict[str, object],
    candidate_item: dict[str, object],
    rule: dict[str, object],
) -> bool:
    if (
        candidate_item.get("candidate_id") == current_item.get("candidate_id")
        and candidate_item.get("selector_pool") == current_item.get("selector_pool")
    ):
        return False
    current_source = current_item.get("candidate_source")
    candidate_source = candidate_item.get("candidate_source")
    if not isinstance(current_source, str) or not isinstance(candidate_source, str):
        return False
    current_source_rule = rule.get("current_source")
    current_source_group_rule = rule.get("current_source_group")
    if (
        (current_source_rule is not None or current_source_group_rule is not None)
        and not _source_matches(
            current_source,
            exact=current_source_rule,
            group=current_source_group_rule,
        )
    ):
        return False
    if not _source_matches(
        candidate_source,
        exact=rule.get("candidate_source"),
        group=rule.get("candidate_source_group"),
    ):
        return False
    if not _pool_matches(candidate_item, rule.get("candidate_pools", rule.get("candidate_pool"))):
        return False
    required_flags = rule.get("current_required_flags", rule.get("current_required_flag"))
    if isinstance(required_flags, list):
        if not all(_required_flag_present(current_item, flag) for flag in required_flags):
            return False
    elif required_flags is not None and not _required_flag_present(current_item, required_flags):
        return False
    return _thresholds_match_item(current_item, rule, prefix="current") and _thresholds_match_item(
        candidate_item,
        rule,
        prefix="candidate",
    )


def _select_candidate_selector_rescue(
    current_item: dict[str, object],
    candidate_items: list[dict[str, object]],
    rules: list[dict[str, object]] | None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if not rules:
        return None, None
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        eligible = [
            item
            for item in candidate_items
            if _candidate_selector_rescue_is_eligible(current_item, item, rule)
        ]
        if not eligible:
            continue
        selection_key = str(rule.get("selection_key", "score"))
        reverse = str(rule.get("selection_direction", "max")) != "min"

        def sort_key(item: dict[str, object]) -> float:
            value = _numeric_item_value(item, selection_key)
            if value is None:
                return float("-inf") if reverse else float("inf")
            return float(value)

        return sorted(eligible, key=sort_key, reverse=reverse)[0], rule
    return None, None


def _split_candidate_selector_rules(
    rules: list[dict[str, object]] | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not rules:
        return [], []
    active_rules: list[dict[str, object]] = []
    pooled_rules: list[dict[str, object]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if _uses_selector_pool(rule):
            pooled_rules.append(rule)
        else:
            active_rules.append(rule)
    return active_rules, pooled_rules


def _apply_candidate_selector_rescue(
    current_item: dict[str, object],
    active_candidate_items: list[dict[str, object]],
    selector_candidate_pool_items: list[dict[str, object]],
    rules: list[dict[str, object]] | None,
    *,
    stage_name: str = "",
) -> dict[str, object]:
    active_selector_rules, pooled_selector_rules = _split_candidate_selector_rules(rules)
    selector_rescue_best, selector_rescue_rule = _select_candidate_selector_rescue(
        current_item,
        active_candidate_items,
        active_selector_rules,
    )
    pooled = False
    if selector_rescue_best is None:
        selector_rescue_best, selector_rescue_rule = _select_candidate_selector_rescue(
            current_item,
            selector_candidate_pool_items,
            pooled_selector_rules,
        )
        pooled = selector_rescue_best is not None
    if selector_rescue_best is None:
        return current_item

    selected = dict(selector_rescue_best)
    field_prefix = f"{stage_name}_" if stage_name else ""
    flag_prefix = f"{stage_name}_" if stage_name else ""
    if pooled:
        selected[f"selected_via_pooled_{flag_prefix}candidate_selector_rescue"] = True
    selected[f"selected_via_{flag_prefix}candidate_selector_rescue"] = True
    selected[f"pre_{field_prefix}candidate_selector_rescue_candidate_id"] = current_item.get("candidate_id")
    selected[f"pre_{field_prefix}candidate_selector_rescue_source"] = current_item.get("candidate_source")
    selected[f"pre_{field_prefix}candidate_selector_rescue_score"] = float(current_item["score"])
    selected[f"{field_prefix}candidate_selector_rescue_rule"] = selector_rescue_rule
    return selected


def _apply_post_final_candidate_selector_rescue(
    current_item: dict[str, object],
    active_candidate_items: list[dict[str, object]],
    selector_candidate_pool_items: list[dict[str, object]],
    rules: list[dict[str, object]] | None,
    *,
    final_pred_pixels: int,
) -> dict[str, object]:
    if not rules:
        return current_item
    usable_rules: list[dict[str, object]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if final_pred_pixels > 0 and not bool(rule.get("allow_nonzero_final_mask", False)):
            continue
        usable_rules.append(rule)
    if not usable_rules:
        return current_item

    current_with_final_state = dict(current_item)
    current_with_final_state["final_pred_pixels"] = int(final_pred_pixels)
    selected = _apply_candidate_selector_rescue(
        current_with_final_state,
        active_candidate_items,
        selector_candidate_pool_items,
        usable_rules,
        stage_name="post_final",
    )
    if selected.get("selected_via_post_final_candidate_selector_rescue"):
        selected["pre_post_final_candidate_selector_rescue_final_pred_pixels"] = int(final_pred_pixels)
    return selected


def _full_frame_mask_from_candidate_item(
    item: dict[str, object],
    image_shape: tuple[int, ...],
) -> np.ndarray | None:
    if "full_frame_mask" in item:
        return np.asarray(item["full_frame_mask"], dtype=np.uint8)
    mask = item.get("mask")
    crop_meta = item.get("crop_meta")
    if mask is None or not isinstance(crop_meta, dict):
        return None
    return remap_mask_to_image(np.asarray(mask, dtype=np.uint8), crop_meta, image_shape)


def _rescue_arbiter_config_without_checkpoint(config: dict[str, object]) -> dict[str, object]:
    return {
        str(key): value
        for key, value in config.items()
        if key not in {"checkpoint", "checkpoint_path"}
    }


def _rescue_arbiter_candidate_is_eligible(
    current_item: dict[str, object],
    candidate_item: dict[str, object],
    config: dict[str, object],
) -> bool:
    if (
        candidate_item.get("candidate_id") == current_item.get("candidate_id")
        and candidate_item.get("selector_pool") == current_item.get("selector_pool")
    ):
        return False
    candidate_source = candidate_item.get("candidate_source")
    if not isinstance(candidate_source, str):
        return False
    candidate_source_rule = config.get("candidate_source")
    candidate_source_group_rule = config.get("candidate_source_group")
    if (
        (candidate_source_rule is not None or candidate_source_group_rule is not None)
        and not _source_matches(
            candidate_source,
            exact=candidate_source_rule,
            group=candidate_source_group_rule,
        )
    ):
        return False
    if not _pool_matches(candidate_item, config.get("candidate_pools", config.get("candidate_pool"))):
        return False
    return _thresholds_match_item(candidate_item, config, prefix="candidate")


def _apply_rescue_arbiter(
    current_item: dict[str, object],
    candidate_items: list[dict[str, object]],
    config: object,
    *,
    device: torch.device,
    stage_name: str = "",
) -> dict[str, object]:
    for current_config in _rescue_config_list(config):
        checkpoint_value = current_config.get("checkpoint_path") or current_config.get("checkpoint")
        if not checkpoint_value:
            continue
        if not _rescue_config_should_run(current_item, current_config):
            continue
        eligible = [
            item
            for item in candidate_items
            if _rescue_arbiter_candidate_is_eligible(current_item, item, current_config)
        ]
        if not eligible:
            continue
        arbiter = _load_rescue_arbiter_cached(Path(str(checkpoint_value)), device)
        scored = [
            (
                arbiter.candidate_probability(current_item, item),
                item,
            )
            for item in eligible
        ]
        probability, candidate = max(scored, key=lambda pair: pair[0])
        threshold = float(current_config.get("threshold", arbiter.threshold))
        if float(probability) < threshold:
            continue

        selected = dict(candidate)
        field_prefix = f"{stage_name}_" if stage_name else ""
        flag_prefix = f"{stage_name}_" if stage_name else ""
        selected[f"selected_via_{flag_prefix}rescue_arbiter"] = True
        selected[f"pre_{field_prefix}rescue_arbiter_candidate_id"] = current_item.get("candidate_id")
        selected[f"pre_{field_prefix}rescue_arbiter_source"] = current_item.get("candidate_source")
        selected[f"pre_{field_prefix}rescue_arbiter_score"] = (
            float(current_item["score"]) if isinstance(current_item.get("score"), (int, float)) else None
        )
        selected[f"{field_prefix}rescue_arbiter_probability"] = float(probability)
        selected[f"{field_prefix}rescue_arbiter_threshold"] = float(threshold)
        selected[f"{field_prefix}rescue_arbiter_config"] = _rescue_arbiter_config_without_checkpoint(
            current_config
        )
        return selected
    return current_item


def _tag_selector_pool(items: list[dict[str, object]], pool: str) -> None:
    for item in items:
        item["selector_pool"] = pool


def _selector_pool_candidate_config(config: dict[str, object]) -> dict[str, object]:
    return {
        "max_ball_candidates": int(config.get("max_ball_candidates", 32)),
        "crop_scale": float(config.get("crop_scale", 4.25)),
        "crop_padding_px": int(config.get("crop_padding_px", 24)),
        "upscale_factor": float(config.get("upscale_factor", 1.0)),
        "line_endpoint_candidates": bool(config.get("line_endpoint_candidates", False)),
        "line_endpoint": config.get("line_endpoint", {}),
        "colored_blob_candidates": bool(config.get("colored_blob_candidates", False)),
        "colored_blob": config.get("colored_blob", {}),
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


def _run_selector_candidate_pool_rescue_configs(
    *,
    image: np.ndarray,
    output_dir: Path,
    current_item: dict[str, object],
    configs: object,
    image_size: int,
    device: torch.device,
    save_intermediates: bool,
    crop_batch_size: int,
    amp_enabled: bool,
    stage_name: str = "",
) -> list[dict[str, object]]:
    candidate_items: list[dict[str, object]] = []
    stage_prefix = f"{stage_name}_" if stage_name else ""
    dir_prefix = f"{stage_prefix}selector_candidate_pool_rescue"
    for pool_rescue_index, current_pool_rescue_config in enumerate(_rescue_config_list(configs)):
        if not _rescue_config_should_run(current_item, current_pool_rescue_config):
            continue
        checkpoint_value = current_pool_rescue_config.get("checkpoint_path") or current_pool_rescue_config.get(
            "checkpoint"
        )
        if not checkpoint_value:
            continue
        pool_checkpoint_path = Path(str(checkpoint_value))
        pool_name = str(
            current_pool_rescue_config.get(
                "pool",
                current_pool_rescue_config.get("name", f"{dir_prefix}_{pool_rescue_index}"),
            )
        )
        pool_candidates = generate_ball_candidates(
            image,
            _selector_pool_candidate_config(current_pool_rescue_config),
            output_dir=(
                output_dir / f"{dir_prefix}_intermediates_{pool_rescue_index:02d}"
                if save_intermediates
                else None
            ),
        )
        pool_model = _load_guideline_model(pool_checkpoint_path, device)
        pool_reranker = None
        if bool(current_pool_rescue_config.get("use_reranker", True)):
            reranker_value = current_pool_rescue_config.get("reranker_checkpoint_path") or current_pool_rescue_config.get(
                "reranker_checkpoint"
            )
            pool_reranker_path = (
                Path(str(reranker_value))
                if reranker_value
                else pool_checkpoint_path.with_name("guideline_reranker_best.pt")
            )
            pool_reranker = _load_reranker_cached(pool_reranker_path, device) if pool_reranker_path.exists() else None
        pool_dir = ensure_dir(output_dir / f"{dir_prefix}_model_{pool_rescue_index:02d}")
        _, _, pool_candidate_items = _score_candidates(
            model=pool_model,
            image=image,
            candidates=pool_candidates,
            output_dir=pool_dir,
            image_size=int(current_pool_rescue_config.get("image_size", image_size)),
            device=device,
            reranker=pool_reranker,
            save_intermediates=save_intermediates,
            crop_batch_size=crop_batch_size,
            amp_enabled=amp_enabled,
            prediction_threshold=float(current_pool_rescue_config.get("prediction_threshold", 0.35)),
            return_items=True,
        )
        _tag_selector_pool(pool_candidate_items, pool_name)
        for item in pool_candidate_items:
            item["selector_candidate_pool_rescue"] = True
            item["selector_candidate_pool_rescue_name"] = str(current_pool_rescue_config.get("name", pool_name))
            item["selector_candidate_pool_rescue_index"] = int(pool_rescue_index)
            item["selector_candidate_pool_rescue_threshold"] = float(
                current_pool_rescue_config.get("prediction_threshold", 0.35)
            )
            item["pre_selector_candidate_pool_rescue_score"] = float(current_item["score"])
            if stage_name:
                item[f"{stage_prefix}selector_candidate_pool_rescue"] = True
                item[f"{stage_prefix}selector_candidate_pool_rescue_name"] = str(
                    current_pool_rescue_config.get("name", pool_name)
                )
                item[f"{stage_prefix}selector_candidate_pool_rescue_index"] = int(pool_rescue_index)
                item[f"{stage_prefix}selector_candidate_pool_rescue_threshold"] = float(
                    current_pool_rescue_config.get("prediction_threshold", 0.35)
                )
                item[f"pre_{stage_prefix}selector_candidate_pool_rescue_source"] = current_item.get("candidate_source")
                item[f"pre_{stage_prefix}selector_candidate_pool_rescue_score"] = float(current_item["score"])
        candidate_items.extend(pool_candidate_items)
    return candidate_items


def _load_external_guideline_runtime(checkpoint_path: Path) -> dict[str, object]:
    cache_key = str(checkpoint_path.resolve())
    cached = _EXTERNAL_GUIDELINE_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    external_root = checkpoint_path.parents[2]
    if not external_root.exists():
        raise FileNotFoundError(f"External guideline root not found for checkpoint: {checkpoint_path}")
    if str(external_root) not in sys.path:
        sys.path.insert(0, str(external_root))
    from cv_guideline_common import (  # type: ignore
        build_guideline_postprocess_context,
        load_checkpoint_model,
        postprocess_guideline_mask,
        predict_prob_multi_tta,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint_model(checkpoint_path, device=device)
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    tile_cfg = ckpt.get("tile_config", {}) if isinstance(ckpt, dict) else {}
    tta_cfg = ckpt.get("tta_config", {}) if isinstance(ckpt, dict) else {}
    runtime = {
        "model": model,
        "device": device,
        "threshold": float(ckpt.get("best_threshold", 0.5)),
        "tile_sizes": [int(x) for x in tile_cfg.get("tile_sizes", [640, 896])],
        "tile_overlaps": [int(x) for x in tile_cfg.get("tile_overlaps", [160, 224])],
        "tta_scales": [float(x) for x in tta_cfg.get("tta_scales", [1.0, 1.15])],
        "tta_hflip": bool(tta_cfg.get("tta_hflip", True)),
        "axis_complete": bool(args.get("axis_complete", True)),
        "axis_complete_max_gap": int(args.get("axis_complete_max_gap", 24)),
        "axis_complete_max_extension": int(args.get("axis_complete_max_extension", 160)),
        "axis_complete_min_white_ratio": float(args.get("axis_complete_min_white_ratio", 0.42)),
        "axis_complete_min_dark_border_ratio": float(args.get("axis_complete_min_dark_border_ratio", 0.05)),
        "use_amp": bool(args.get("amp", True)),
        "predict_prob_multi_tta": predict_prob_multi_tta,
        "build_guideline_postprocess_context": build_guideline_postprocess_context,
        "postprocess_guideline_mask": postprocess_guideline_mask,
    }
    _EXTERNAL_GUIDELINE_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _run_external_guideline_rescue(image_rgb: np.ndarray, checkpoint_path: Path) -> tuple[np.ndarray, dict[str, float]]:
    runtime = _load_external_guideline_runtime(checkpoint_path)
    image_bgr = image_rgb[:, :, ::-1].copy()
    prob_np = runtime["predict_prob_multi_tta"](
        model=runtime["model"],
        frame_bgr=image_bgr,
        device=runtime["device"],
        tile_sizes=runtime["tile_sizes"],
        tile_overlaps=runtime["tile_overlaps"],
        tta_scales=runtime["tta_scales"],
        tta_hflip=runtime["tta_hflip"],
        use_amp=runtime["use_amp"],
    )
    context = runtime["build_guideline_postprocess_context"](image_bgr)
    pred = runtime["postprocess_guideline_mask"](
        image_bgr,
        prob_np,
        threshold=runtime["threshold"],
        axis_complete=runtime["axis_complete"],
        axis_complete_max_gap=runtime["axis_complete_max_gap"],
        axis_complete_max_extension=runtime["axis_complete_max_extension"],
        axis_complete_min_white_ratio=runtime["axis_complete_min_white_ratio"],
        axis_complete_min_dark_border_ratio=runtime["axis_complete_min_dark_border_ratio"],
        context=context,
    )
    pred_u8 = (pred > 0).astype(np.uint8)
    pred_area = int(pred_u8.sum())
    prob_mean = float(prob_np[pred_u8 > 0].mean()) if pred_area > 0 else 0.0
    return pred_u8, {
        "threshold": float(runtime["threshold"]),
        "pred_area": float(pred_area),
        "prob_mean_on_pred": float(prob_mean),
    }


def _load_image_final_veto_runtime(
    config: dict[str, object],
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint_value = config.get("checkpoint", config.get("checkpoint_path"))
    if checkpoint_value is None:
        raise ValueError("image_final_veto config requires checkpoint")
    checkpoint_path = Path(str(checkpoint_value))
    cache_key = _checkpoint_cache_key(checkpoint_path, device)
    cached = _IMAGE_FINAL_VETO_CACHE.get(cache_key)
    if cached is not None:
        return cached
    model, checkpoint = load_image_final_veto_checkpoint(checkpoint_path, device=device)
    runtime = (model, checkpoint)
    _IMAGE_FINAL_VETO_CACHE[cache_key] = runtime
    return runtime


def _image_final_veto_should_reject(
    *,
    image: np.ndarray,
    final_mask: np.ndarray,
    current_item: dict[str, object],
    config: dict[str, object] | None,
    device: torch.device,
) -> tuple[bool, float | None]:
    if not config or int(np.count_nonzero(final_mask)) <= 0:
        return False, None
    source = str(current_item.get("candidate_source", ""))
    sources = config.get("sources")
    if isinstance(sources, list) and source not in {str(item) for item in sources}:
        return False, None
    min_pred_pixels = _image_final_veto_min_pred_pixels_for_source(config, source)
    pred_pixels = int(np.count_nonzero(final_mask))
    if min_pred_pixels is not None and pred_pixels < min_pred_pixels:
        return False, None
    source_exemptions = config.get("source_exemptions")
    if isinstance(source_exemptions, dict):
        exemptions: list[dict[str, object]] = []
        exact_exemptions = source_exemptions.get(source)
        group_exemptions = source_exemptions.get(_candidate_source_group(source))
        if isinstance(exact_exemptions, list):
            exemptions.extend(exact_exemptions)
        if isinstance(group_exemptions, list) and group_exemptions is not exact_exemptions:
            exemptions.extend(group_exemptions)
        if exemptions:
            item_with_pred_pixels = dict(current_item)
            item_with_pred_pixels["pred_pixels"] = pred_pixels
            if any(
                isinstance(exemption, dict)
                and _thresholds_match_unprefixed_item(item_with_pred_pixels, exemption)
                for exemption in exemptions
            ):
                return False, None
    model, checkpoint = _load_image_final_veto_runtime(config, device=device)
    checkpoint_config = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    image_size = int(config.get("image_size", checkpoint_config.get("image_size", 128)))
    include_global = bool(config.get("include_global", checkpoint_config.get("include_global", False)))
    threshold = float(config.get("threshold", checkpoint.get("threshold", 0.5)))
    tensor = build_veto_tensor(
        image,
        final_mask,
        image_size=image_size,
        include_global=include_global,
    )[None, ...].to(device)
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(tensor)).detach().cpu().item())
    return probability >= threshold, probability


def _image_final_veto_configs(config: object) -> list[dict[str, object]]:
    if not config:
        return []
    if isinstance(config, dict):
        return [config]
    if isinstance(config, list):
        return [item for item in config if isinstance(item, dict)]
    return []


def _image_final_veto_config_without_checkpoint(config: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in config.items() if key not in {"checkpoint", "checkpoint_path"}}


def _image_final_veto_should_reject_any(
    *,
    image: np.ndarray,
    final_mask: np.ndarray,
    current_item: dict[str, object],
    config: object,
    device: torch.device,
) -> tuple[bool, float | None, dict[str, object] | None]:
    probabilities: list[dict[str, object]] = []
    for item_config in _image_final_veto_configs(config):
        should_reject, probability = _image_final_veto_should_reject(
            image=image,
            final_mask=final_mask,
            current_item=current_item,
            config=item_config,
            device=device,
        )
        if probability is None:
            continue
        probabilities.append(
            {
                "probability": float(probability),
                "config": _image_final_veto_config_without_checkpoint(item_config),
            }
        )
        if should_reject:
            current_item["image_final_veto_probabilities"] = probabilities
            return True, probability, item_config
    if probabilities:
        current_item["image_final_veto_probabilities"] = probabilities
        return False, float(probabilities[-1]["probability"]), None
    return False, None, None


def _checkpoint_cache_key(checkpoint_path: Path, device: torch.device) -> str:
    device_index = device.index if device.index is not None else -1
    return f"{checkpoint_path.resolve()}::{device.type}:{device_index}"


def _primary_recovery_is_eligible(
    current_item: dict[str, object],
    primary_item: dict[str, object],
    current_source: str | None,
    current_source_group: str | None,
    min_score: float | None,
    max_score: float | None,
    primary_source: str | None,
    primary_source_group: str | None,
    primary_min_score: float | None,
    primary_min_ball_fill_fraction: float | None,
) -> bool:
    if current_source is not None and current_item.get("candidate_source") != current_source:
        return False
    if current_source_group is not None and _candidate_source_group(current_item.get("candidate_source")) != current_source_group:
        return False
    current_score = float(current_item["score"])
    if min_score is not None and current_score < float(min_score):
        return False
    if max_score is not None and current_score >= float(max_score):
        return False
    if primary_source is not None and primary_item.get("candidate_source") != primary_source:
        return False
    if primary_source_group is not None and _candidate_source_group(primary_item.get("candidate_source")) != primary_source_group:
        return False
    if primary_min_score is not None and float(primary_item["score"]) < float(primary_min_score):
        return False
    return _secondary_rescue_is_eligible(primary_item, primary_min_ball_fill_fraction)


def _load_guideline_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    cache_key = _checkpoint_cache_key(checkpoint_path, device)
    cached = _GUIDELINE_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    state = torch.load(checkpoint_path, map_location=device)
    model_config = state.get("model_config") if isinstance(state, dict) else None
    base_channels = None
    in_channels = 3
    model_type = "guideline_unet"
    if isinstance(model_config, dict):
        if "base_channels" in model_config:
            base_channels = int(model_config["base_channels"])
        in_channels = int(model_config.get("in_channels", 3))
        model_type = str(model_config.get("model_type", "guideline_unet"))
    elif isinstance(state, dict) and isinstance(state.get("model"), dict):
        stem_weight = state["model"].get("stem.block.0.weight")
        if isinstance(stem_weight, torch.Tensor):
            base_channels = int(stem_weight.shape[0])
            in_channels = int(stem_weight.shape[1])
        if any(str(key).startswith("validity_head.") for key in state["model"]):
            model_type = "conditioned_guideline_unet"
    if base_channels is None:
        base_channels = 32
    if model_type == "conditioned_guideline_unet":
        model = ConditionedGuidelineUNet(base_channels=base_channels, in_channels=in_channels).to(device)
    else:
        model = build_guideline_model(
            model_type=model_type,
            base_channels=base_channels,
            in_channels=in_channels,
            pretrained_encoder=False,
        ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    _GUIDELINE_MODEL_CACHE[cache_key] = model
    return model


def _load_reranker_cached(checkpoint_path: Path, device: torch.device):
    cache_key = _checkpoint_cache_key(checkpoint_path, device)
    cached = _RERANKER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    reranker = load_reranker_checkpoint(checkpoint_path, device=device)
    _RERANKER_CACHE[cache_key] = reranker
    return reranker


def _load_rescue_arbiter_cached(checkpoint_path: Path, device: torch.device) -> LoadedRescueArbiter:
    cache_key = _checkpoint_cache_key(checkpoint_path, device)
    cached = _RESCUE_ARBITER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    arbiter = load_rescue_arbiter_checkpoint(checkpoint_path, device=device)
    _RESCUE_ARBITER_CACHE[cache_key] = arbiter
    return arbiter


def _load_meta_selector_cached(checkpoint_path: Path, device: torch.device):
    cache_key = _checkpoint_cache_key(checkpoint_path, device)
    cached = _META_SELECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    selector = load_meta_selector_checkpoint(checkpoint_path, device=device)
    _META_SELECTOR_CACHE[cache_key] = selector
    return selector


def _score_candidates(
    *,
    model: torch.nn.Module,
    image: np.ndarray,
    candidates: list,
    output_dir: Path,
    image_size: int,
    device: torch.device,
    reranker,
    save_intermediates: bool = True,
    crop_batch_size: int = 1,
    amp_enabled: bool = True,
    prediction_threshold: float = 0.35,
    return_items: bool = False,
) -> (
    tuple[dict[str, object], list[dict[str, object]]]
    | tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]
):
    best = None
    candidate_items = []
    summaries = []
    prepared_candidates: list[tuple[object, np.ndarray, dict[str, object], np.ndarray, Path | None]] = []
    crop_images: list[np.ndarray] = []
    intermediates_root = ensure_dir(output_dir / "intermediates") if save_intermediates else None
    for candidate in candidates:
        crop_dir = None
        if intermediates_root is not None:
            crop_dir = ensure_dir(intermediates_root / candidate.candidate_id)
        crop_image, crop_meta = extract_crop(image, candidate)
        if crop_dir is not None:
            save_rgb_image(crop_dir / "crop.png", crop_image)
        ball_mask = local_ball_mask(crop_meta)
        crop_images.append(crop_image)
        prepared_candidates.append((candidate, crop_image, crop_meta, ball_mask, crop_dir))
    probs, validity_scores = _predict_crop_outputs(
        model,
        crop_images,
        image_size,
        device,
        crop_metas=[item[2] for item in prepared_candidates],
        crop_batch_size=crop_batch_size,
        amp_enabled=amp_enabled,
    )
    for (candidate, crop_image, crop_meta, ball_mask, crop_dir), prob, validity_score in zip(
        prepared_candidates,
        probs,
        validity_scores,
        strict=True,
    ):
        raw_score, binary = _score_prediction(prob, ball_mask, threshold=prediction_threshold)
        cleaned = cleanup_mask(binary, {"morphology_open_radius": 0, "morphology_close_radius": 1, "min_component_area": 4, "keep_only_best_component": True}, ball_mask=ball_mask)
        features = compute_mask_features(
            cleaned,
            ball_mask,
            tuple(crop_meta["local_center"]),
            int(crop_meta["local_radius"]),
        )
        score = _rank_prediction(raw_score, candidate.score, features)
        if validity_score is not None:
            score += 0.5 * float(validity_score)
        reranker_features = build_reranker_features(
            candidate_source=candidate.source,
            candidate_score=float(candidate.score),
            raw_score=float(raw_score),
            confidence=float(prob[cleaned > 0].mean()) if np.count_nonzero(cleaned) else 0.0,
            pred_pixels=int(np.count_nonzero(cleaned)),
            reticle_distance=_candidate_reticle_distance(candidate),
            mask_features=features,
        )
        if crop_dir is not None:
            save_mask_png(crop_dir / "pred_mask.png", cleaned)
            save_rgb_image(crop_dir / "pred_overlay.png", overlay_mask(crop_image, cleaned))
        item = {
            "candidate_id": candidate.candidate_id,
            "score": score,
            "raw_score": raw_score,
            "candidate_score": candidate.score,
            "candidate_source": candidate.source,
            "candidate_center": [candidate.center_x, candidate.center_y],
            "candidate_radius": candidate.radius,
            "confidence": float(prob[cleaned > 0].mean()) if np.count_nonzero(cleaned) else 0.0,
            "validity_score": float(validity_score) if validity_score is not None else None,
            "pred_pixels": int(np.count_nonzero(cleaned)),
            "reticle_distance": _candidate_reticle_distance(candidate),
            "features": features,
            "reranker_features": reranker_features,
            "mask": cleaned,
            "crop_meta": crop_meta,
        }
        candidate_items.append(item)
        summaries.append(
            {
                k: v
                for k, v in item.items()
                if k not in {"mask", "crop_meta", "reranker_features"}
            }
        )
        if best is None or score > best["score"]:
            best = item

    if reranker is not None and candidate_items:
        feature_matrix = np.stack([item["reranker_features"] for item in candidate_items]).astype(np.float32)
        reranker_scores = reranker.score(feature_matrix)
        for item, summary, reranker_score in zip(candidate_items, summaries, reranker_scores, strict=True):
            item["heuristic_score"] = item["score"]
            item["reranker_score"] = float(reranker_score)
            item["score"] = float(reranker_score)
            summary["heuristic_score"] = summary["score"]
            summary["reranker_score"] = float(reranker_score)
            summary["score"] = float(reranker_score)
        best = max(candidate_items, key=lambda item: float(item["score"]))

    assert best is not None
    if return_items:
        return best, summaries, candidate_items
    return best, summaries


def run_supervised_inference(
    checkpoint_path: Path,
    input_path: Path,
    output_dir: Path,
    image_size: int = 384,
    reranker_checkpoint_path: Path | None = None,
    fallback_checkpoint_path: Path | None = None,
    fallback_reranker_checkpoint_path: Path | None = None,
    fallback_score_threshold: float | None = None,
    meta_selector_checkpoint_path: Path | None = None,
    rescue_checkpoint_path: Path | None = None,
    rescue_reranker_checkpoint_path: Path | None = None,
    rescue_score_threshold: float | None = None,
    rescue_max_ball_candidates: int = 20,
    final_fallback_score_threshold: float | None = None,
    secondary_rescue_checkpoint_path: Path | None = None,
    secondary_rescue_reranker_checkpoint_path: Path | None = None,
    secondary_rescue_min_score: float | None = None,
    secondary_rescue_max_score: float | None = None,
    secondary_rescue_source: str | None = None,
    secondary_rescue_min_ball_fill_fraction: float | None = None,
    secondary_rescue_max_ball_candidates: int = 24,
    secondary_rescue_current_exemptions: object = None,
    primary_recovery_min_score: float | None = None,
    primary_recovery_max_score: float | None = None,
    primary_recovery_source: str | None = None,
    primary_recovery_candidate_source: str | None = None,
    primary_recovery_current_group: str | None = None,
    primary_recovery_candidate_group: str | None = None,
    primary_recovery_min_primary_score: float | None = None,
    primary_recovery_min_ball_fill_fraction: float | None = None,
    reticle_recovery_min_score: float | None = None,
    reticle_recovery_max_score: float | None = None,
    reticle_recovery_min_primary_score: float | None = None,
    fallback_rescue_checkpoint_path: Path | None = None,
    fallback_rescue_reranker_checkpoint_path: Path | None = None,
    fallback_rescue_min_score: float | None = None,
    fallback_rescue_max_score: float | None = None,
    fallback_rescue_source_group: str | None = None,
    fallback_rescue_max_ball_candidates: int = 20,
    reticle_fill_rescue_checkpoint_path: Path | None = None,
    reticle_fill_rescue_reranker_checkpoint_path: Path | None = None,
    reticle_fill_rescue_min_score: float | None = None,
    reticle_fill_rescue_max_score: float | None = None,
    reticle_fill_rescue_current_max_ball_fill_fraction: float | None = None,
    reticle_fill_rescue_target_min_ball_fill_fraction: float | None = None,
    reticle_fill_rescue_max_ball_candidates: int = 20,
    external_rescue_checkpoint_path: Path | None = None,
    external_rescue_source_group: str | None = None,
    external_rescue_min_score: float | None = None,
    external_rescue_max_score: float | None = None,
    external_rescue_current_max_ball_fill_fraction: float | None = None,
    external_rescue_min_pred_area: int = 20,
    external_rescue_min_prob_mean: float = 0.9,
    external_mid_rescue_checkpoint_path: Path | None = None,
    external_mid_rescue_source_group: str | None = None,
    external_mid_rescue_min_score: float | None = None,
    external_mid_rescue_max_score: float | None = None,
    external_mid_rescue_current_max_ball_fill_fraction: float | None = None,
    external_mid_rescue_min_pred_area: int = 20,
    external_mid_rescue_min_prob_mean: float = 0.88,
    external_lowmid_rescue_checkpoint_path: Path | None = None,
    external_lowmid_rescue_source_group: str | None = None,
    external_lowmid_rescue_min_score: float | None = None,
    external_lowmid_rescue_max_score: float | None = None,
    external_lowmid_rescue_current_max_ball_fill_fraction: float | None = None,
    external_lowmid_rescue_min_pred_area: int = 400,
    external_lowmid_rescue_min_prob_mean: float = 0.8,
    external_reticle_rescue_checkpoint_path: Path | None = None,
    external_reticle_rescue_source_group: str | None = None,
    external_reticle_rescue_min_score: float | None = None,
    external_reticle_rescue_max_score: float | None = None,
    external_reticle_rescue_current_max_ball_fill_fraction: float | None = None,
    external_reticle_rescue_min_pred_area: int = 20,
    external_reticle_rescue_min_prob_mean: float = 0.8,
    external_tiny_reticle_rescue_checkpoint_path: Path | None = None,
    external_tiny_reticle_rescue_source_group: str | None = None,
    external_tiny_reticle_rescue_min_score: float | None = None,
    external_tiny_reticle_rescue_max_score: float | None = None,
    external_tiny_reticle_rescue_current_max_ball_fill_fraction: float | None = None,
    external_tiny_reticle_rescue_min_pred_area: int = 50,
    external_tiny_reticle_rescue_min_prob_mean: float = 0.94,
    external_reticle_large_rescue_checkpoint_path: Path | None = None,
    external_reticle_large_rescue_source_group: str | None = None,
    external_reticle_large_rescue_min_score: float | None = None,
    external_reticle_large_rescue_max_score: float | None = None,
    external_reticle_large_rescue_current_max_ball_fill_fraction: float | None = None,
    external_reticle_large_rescue_min_pred_area: int = 400,
    external_reticle_large_rescue_min_prob_mean: float = 0.86,
    external_table_broad_rescue_checkpoint_path: Path | None = None,
    external_table_broad_rescue_source_group: str | None = None,
    external_table_broad_rescue_min_score: float | None = None,
    external_table_broad_rescue_max_score: float | None = None,
    external_table_broad_rescue_current_max_ball_fill_fraction: float | None = None,
    external_table_broad_rescue_min_pred_area: int = 350,
    external_table_broad_rescue_min_prob_mean: float = 0.8,
    external_blob_low_rescue_checkpoint_path: Path | None = None,
    external_blob_low_rescue_source_group: str | None = None,
    external_blob_low_rescue_min_score: float | None = None,
    external_blob_low_rescue_max_score: float | None = None,
    external_blob_low_rescue_current_max_ball_fill_fraction: float | None = None,
    external_blob_low_rescue_min_pred_area: int = 20,
    external_blob_low_rescue_min_prob_mean: float = 0.75,
    external_table_rescue_checkpoint_path: Path | None = None,
    external_table_rescue_source_group: str | None = None,
    external_table_rescue_min_score: float | None = None,
    external_table_rescue_max_score: float | None = None,
    external_table_rescue_current_max_ball_fill_fraction: float | None = None,
    external_table_rescue_min_pred_area: int = 120,
    external_table_rescue_min_prob_mean: float = 0.84,
    external_blob_rescue_checkpoint_path: Path | None = None,
    external_blob_rescue_source_group: str | None = None,
    external_blob_rescue_min_score: float | None = None,
    external_blob_rescue_max_score: float | None = None,
    external_blob_rescue_current_max_ball_fill_fraction: float | None = None,
    external_blob_rescue_min_pred_area: int = 300,
    external_blob_rescue_min_prob_mean: float = 0.95,
    external_tiny_blob_rescue_checkpoint_path: Path | None = None,
    external_tiny_blob_rescue_source_group: str | None = None,
    external_tiny_blob_rescue_min_score: float | None = None,
    external_tiny_blob_rescue_max_score: float | None = None,
    external_tiny_blob_rescue_current_max_ball_fill_fraction: float | None = None,
    external_tiny_blob_rescue_min_pred_area: int = 80,
    external_tiny_blob_rescue_min_prob_mean: float = 0.88,
    micro_line_rescue_config: dict[str, object] | None = None,
    colored_blob_rescue_config: object = None,
    selector_candidate_pool_rescue_config: object = None,
    post_external_selector_candidate_pool_rescue_config: object = None,
    reject_table_hough_final_fallback_min_pred_pixels: int | None = None,
    reject_table_hough_final_fallback_max_score: float | None = None,
    reject_winner_source_min_pred_pixels: dict[str, int] | None = None,
    reject_winner_source_min_pred_pixels_exemptions: dict[str, list[dict[str, object]]] | None = None,
    reject_winner_source_area_score: dict[str, dict[str, object]] | None = None,
    reject_pre_final_fallback_max_score: float | None = None,
    reject_final_context_rules: list[dict[str, object]] | None = None,
    candidate_selector_rescue_rules: list[dict[str, object]] | None = None,
    rescue_arbiter_config: object = None,
    post_external_candidate_selector_rescue_rules: list[dict[str, object]] | None = None,
    post_external_rescue_arbiter_config: object = None,
    post_final_candidate_selector_rescue_rules: list[dict[str, object]] | None = None,
    final_refine_checkpoint_path: Path | None = None,
    final_refine_image_size: int | None = None,
    image_final_veto_config: object = None,
    save_outputs: bool = True,
    save_intermediates: bool = True,
    save_report: bool = True,
    save_recolor_preview: bool = True,
    return_overlay: bool = True,
    crop_batch_size: int = 1,
    amp_enabled: bool = True,
) -> dict[str, object]:
    if save_outputs or save_intermediates or save_report or save_recolor_preview:
        output_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb_image(input_path)
    candidates = generate_ball_candidates(
        image,
        {"max_ball_candidates": 10, "crop_scale": 4.25, "crop_padding_px": 24, "upscale_factor": 1.0, "hough": {"dp": 1.2, "min_dist_factor": 1.5, "param1": 110, "param2": 18, "min_radius": 7, "max_radius": 80}, "fallback": {"grid_candidates": 6, "white_ball_bias": True}},
        output_dir=output_dir / "intermediates" if save_intermediates else None,
    )
    if save_intermediates:
        save_rgb_image(output_dir / "intermediates" / "ball_candidate_overview.png", draw_ball_candidates(image, candidates))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = _load_guideline_model(checkpoint_path, device)
    reranker_path = reranker_checkpoint_path or checkpoint_path.with_name("guideline_reranker_best.pt")
    reranker = _load_reranker_cached(reranker_path, device) if reranker_path.exists() else None
    meta_selector = (
        _load_meta_selector_cached(meta_selector_checkpoint_path, device)
        if meta_selector_checkpoint_path is not None and meta_selector_checkpoint_path.exists()
        else None
    )
    fallback_best = None
    fallback_summaries = None
    fallback_candidate_items = None
    best, summaries, active_candidate_items = _score_candidates(
        model=model,
        image=image,
        candidates=candidates,
        output_dir=output_dir,
        image_size=image_size,
        device=device,
        reranker=reranker,
        save_intermediates=save_intermediates,
        crop_batch_size=crop_batch_size,
        amp_enabled=amp_enabled,
        return_items=True,
    )
    _tag_selector_pool(active_candidate_items, "primary")
    selector_candidate_pool_items = list(active_candidate_items)
    primary_best = dict(best)
    primary_summaries = [dict(item) for item in summaries]
    primary_candidate_items = list(active_candidate_items)

    should_run_fallback = False
    if fallback_checkpoint_path is not None:
        if meta_selector is not None:
            should_run_fallback = True
        elif fallback_score_threshold is not None and float(best["score"]) < float(fallback_score_threshold):
            should_run_fallback = True

    if should_run_fallback and fallback_checkpoint_path is not None:
        fallback_model = _load_guideline_model(fallback_checkpoint_path, device)
        fallback_reranker_path = (
            fallback_reranker_checkpoint_path
            or fallback_checkpoint_path.with_name("guideline_reranker_best.pt")
        )
        fallback_reranker = (
            _load_reranker_cached(fallback_reranker_path, device)
            if fallback_reranker_path.exists()
            else None
        )
        fallback_dir = ensure_dir(output_dir / "fallback_model")
        fallback_best, fallback_summaries, fallback_candidate_items = _score_candidates(
            model=fallback_model,
            image=image,
            candidates=candidates,
            output_dir=fallback_dir,
            image_size=image_size,
            device=device,
            reranker=fallback_reranker,
            save_intermediates=save_intermediates,
            crop_batch_size=crop_batch_size,
            amp_enabled=amp_enabled,
            return_items=True,
        )
        _tag_selector_pool(fallback_candidate_items, "fallback")
        selector_candidate_pool_items.extend(fallback_candidate_items)
        choose_fallback = False
        if meta_selector is not None:
            choose_fallback, meta_logit = meta_selector.choose_fallback(best, fallback_best)
            best["meta_selector_logit"] = float(meta_logit)
            fallback_best["meta_selector_logit"] = float(meta_logit)
            if choose_fallback:
                fallback_best["selected_via_fallback"] = True
                fallback_best["selected_via_meta_selector"] = True
                best = fallback_best
                summaries = fallback_summaries
                active_candidate_items = fallback_candidate_items
            else:
                best["selected_via_meta_selector"] = False
        elif fallback_score_threshold is not None and float(best["score"]) < float(fallback_score_threshold):
            fallback_best["selected_via_fallback"] = True
            fallback_best["primary_score"] = float(best["score"])
            fallback_best["fallback_threshold"] = float(fallback_score_threshold)
            best = fallback_best
            summaries = fallback_summaries
            active_candidate_items = fallback_candidate_items

    if (
        rescue_checkpoint_path is not None
        and rescue_score_threshold is not None
        and float(best["score"]) < float(rescue_score_threshold)
    ):
        rescue_candidates = generate_ball_candidates(
            image,
            {
                "max_ball_candidates": int(rescue_max_ball_candidates),
                "crop_scale": 4.25,
                "crop_padding_px": 24,
                "upscale_factor": 1.0,
                "hough": {"dp": 1.2, "min_dist_factor": 1.5, "param1": 110, "param2": 18, "min_radius": 7, "max_radius": 80},
                "fallback": {"grid_candidates": 6, "white_ball_bias": True},
            },
            output_dir=output_dir / "rescue_intermediates" if save_intermediates else None,
        )
        rescue_model = _load_guideline_model(rescue_checkpoint_path, device)
        rescue_reranker_path = (
            rescue_reranker_checkpoint_path
            or rescue_checkpoint_path.with_name("guideline_reranker_best.pt")
        )
        rescue_reranker = (
            _load_reranker_cached(rescue_reranker_path, device)
            if rescue_reranker_path.exists()
            else None
        )
        rescue_dir = ensure_dir(output_dir / "rescue_model")
        rescue_best, rescue_summaries, rescue_candidate_items = _score_candidates(
            model=rescue_model,
            image=image,
            candidates=rescue_candidates,
            output_dir=rescue_dir,
            image_size=image_size,
            device=device,
            reranker=rescue_reranker,
            save_intermediates=save_intermediates,
            crop_batch_size=crop_batch_size,
            amp_enabled=amp_enabled,
            return_items=True,
        )
        _tag_selector_pool(rescue_candidate_items, "rescue")
        selector_candidate_pool_items.extend(rescue_candidate_items)
        rescue_best["selected_via_rescue"] = True
        rescue_best["pre_rescue_score"] = float(best["score"])
        rescue_best["rescue_threshold"] = float(rescue_score_threshold)
        rescue_best["rescue_max_ball_candidates"] = int(rescue_max_ball_candidates)
        best = rescue_best
        summaries = rescue_summaries
        active_candidate_items = rescue_candidate_items

    if (
        final_fallback_score_threshold is not None
        and fallback_best is not None
        and float(best["score"]) < float(final_fallback_score_threshold)
    ):
        fallback_best["selected_via_final_fallback"] = True
        fallback_best["pre_final_fallback_score"] = float(best["score"])
        fallback_best["final_fallback_score_threshold"] = float(final_fallback_score_threshold)
        best = fallback_best
        if fallback_summaries is not None:
            summaries = fallback_summaries
        if fallback_candidate_items is not None:
            active_candidate_items = fallback_candidate_items

    secondary_rescue_current_exempted, secondary_rescue_current_exemption = _current_item_exemption_matches(
        best,
        secondary_rescue_current_exemptions,
    )
    if secondary_rescue_current_exempted:
        best["selected_via_secondary_rescue_current_exemption"] = True
        best["secondary_rescue_current_exemption_rule"] = secondary_rescue_current_exemption

    if (
        secondary_rescue_checkpoint_path is not None
        and secondary_rescue_min_score is not None
        and secondary_rescue_max_score is not None
        and float(best["score"]) >= float(secondary_rescue_min_score)
        and float(best["score"]) < float(secondary_rescue_max_score)
        and (secondary_rescue_source is None or best.get("candidate_source") == secondary_rescue_source)
        and not secondary_rescue_current_exempted
    ):
        secondary_candidates = generate_ball_candidates(
            image,
            {
                "max_ball_candidates": int(secondary_rescue_max_ball_candidates),
                "crop_scale": 4.25,
                "crop_padding_px": 24,
                "upscale_factor": 1.0,
                "hough": {"dp": 1.2, "min_dist_factor": 1.5, "param1": 110, "param2": 18, "min_radius": 7, "max_radius": 80},
                "fallback": {"grid_candidates": 6, "white_ball_bias": True},
            },
            output_dir=output_dir / "secondary_rescue_intermediates" if save_intermediates else None,
        )
        secondary_model = _load_guideline_model(secondary_rescue_checkpoint_path, device)
        secondary_reranker_path = (
            secondary_rescue_reranker_checkpoint_path
            or secondary_rescue_checkpoint_path.with_name("guideline_reranker_best.pt")
        )
        secondary_reranker = (
            _load_reranker_cached(secondary_reranker_path, device)
            if secondary_reranker_path.exists()
            else None
        )
        secondary_dir = ensure_dir(output_dir / "secondary_rescue_model")
        secondary_best, secondary_summaries, secondary_candidate_items = _score_candidates(
            model=secondary_model,
            image=image,
            candidates=secondary_candidates,
            output_dir=secondary_dir,
            image_size=image_size,
            device=device,
            reranker=secondary_reranker,
            save_intermediates=save_intermediates,
            crop_batch_size=crop_batch_size,
            amp_enabled=amp_enabled,
            return_items=True,
        )
        _tag_selector_pool(secondary_candidate_items, "secondary_rescue")
        selector_candidate_pool_items.extend(secondary_candidate_items)
        if _secondary_rescue_is_eligible(secondary_best, secondary_rescue_min_ball_fill_fraction):
            secondary_best["selected_via_secondary_rescue"] = True
            secondary_best["pre_secondary_rescue_score"] = float(best["score"])
            secondary_best["secondary_rescue_min_score"] = float(secondary_rescue_min_score)
            secondary_best["secondary_rescue_max_score"] = float(secondary_rescue_max_score)
            secondary_best["secondary_rescue_source"] = secondary_rescue_source
            secondary_best["secondary_rescue_min_ball_fill_fraction"] = (
                float(secondary_rescue_min_ball_fill_fraction)
                if secondary_rescue_min_ball_fill_fraction is not None
                else None
            )
            secondary_best["secondary_rescue_max_ball_candidates"] = int(secondary_rescue_max_ball_candidates)
            best = secondary_best
            summaries = secondary_summaries
            active_candidate_items = secondary_candidate_items

    if _primary_recovery_is_eligible(
        best,
        primary_best,
        primary_recovery_source,
        primary_recovery_current_group,
        primary_recovery_min_score,
        primary_recovery_max_score,
        primary_recovery_candidate_source,
        primary_recovery_candidate_group,
        primary_recovery_min_primary_score,
        primary_recovery_min_ball_fill_fraction,
    ):
        primary_best["selected_via_primary_recovery"] = True
        primary_best["pre_primary_recovery_score"] = float(best["score"])
        primary_best["primary_recovery_min_score"] = (
            float(primary_recovery_min_score) if primary_recovery_min_score is not None else None
        )
        primary_best["primary_recovery_max_score"] = (
            float(primary_recovery_max_score) if primary_recovery_max_score is not None else None
        )
        primary_best["primary_recovery_source"] = primary_recovery_source
        primary_best["primary_recovery_candidate_source"] = primary_recovery_candidate_source
        primary_best["primary_recovery_current_group"] = primary_recovery_current_group
        primary_best["primary_recovery_candidate_group"] = primary_recovery_candidate_group
        primary_best["primary_recovery_min_primary_score"] = (
            float(primary_recovery_min_primary_score)
            if primary_recovery_min_primary_score is not None
            else None
        )
        primary_best["primary_recovery_min_ball_fill_fraction"] = (
            float(primary_recovery_min_ball_fill_fraction)
            if primary_recovery_min_ball_fill_fraction is not None
            else None
        )
        best = primary_best
        summaries = primary_summaries
        active_candidate_items = primary_candidate_items

    if _primary_recovery_is_eligible(
        best,
        primary_best,
        None,
        "reticle_global",
        reticle_recovery_min_score,
        reticle_recovery_max_score,
        None,
        "reticle_global",
        reticle_recovery_min_primary_score,
        None,
    ):
        primary_best["selected_via_reticle_recovery"] = True
        primary_best["pre_reticle_recovery_score"] = float(best["score"])
        primary_best["reticle_recovery_min_score"] = (
            float(reticle_recovery_min_score) if reticle_recovery_min_score is not None else None
        )
        primary_best["reticle_recovery_max_score"] = (
            float(reticle_recovery_max_score) if reticle_recovery_max_score is not None else None
        )
        primary_best["reticle_recovery_min_primary_score"] = (
            float(reticle_recovery_min_primary_score)
            if reticle_recovery_min_primary_score is not None
            else None
        )
        best = primary_best
        summaries = primary_summaries
        active_candidate_items = primary_candidate_items

    if (
        fallback_rescue_checkpoint_path is not None
        and fallback_rescue_min_score is not None
        and fallback_rescue_max_score is not None
        and bool(best.get("selected_via_fallback", False))
        and float(best["score"]) >= float(fallback_rescue_min_score)
        and float(best["score"]) < float(fallback_rescue_max_score)
        and (
            fallback_rescue_source_group is None
            or _candidate_source_group(best.get("candidate_source")) == fallback_rescue_source_group
        )
    ):
        fallback_rescue_candidates = generate_ball_candidates(
            image,
            {
                "max_ball_candidates": int(fallback_rescue_max_ball_candidates),
                "crop_scale": 4.25,
                "crop_padding_px": 24,
                "upscale_factor": 1.0,
                "hough": {"dp": 1.2, "min_dist_factor": 1.5, "param1": 110, "param2": 18, "min_radius": 7, "max_radius": 80},
                "fallback": {"grid_candidates": 6, "white_ball_bias": True},
            },
            output_dir=output_dir / "fallback_rescue_intermediates" if save_intermediates else None,
        )
        fallback_rescue_model = _load_guideline_model(fallback_rescue_checkpoint_path, device)
        fallback_rescue_reranker_path = (
            fallback_rescue_reranker_checkpoint_path
            or fallback_rescue_checkpoint_path.with_name("guideline_reranker_best.pt")
        )
        fallback_rescue_reranker = (
            _load_reranker_cached(fallback_rescue_reranker_path, device)
            if fallback_rescue_reranker_path.exists()
            else None
        )
        fallback_rescue_dir = ensure_dir(output_dir / "fallback_rescue_model")
        fallback_rescue_best, fallback_rescue_summaries, fallback_rescue_candidate_items = _score_candidates(
            model=fallback_rescue_model,
            image=image,
            candidates=fallback_rescue_candidates,
            output_dir=fallback_rescue_dir,
            image_size=image_size,
            device=device,
            reranker=fallback_rescue_reranker,
            save_intermediates=save_intermediates,
            crop_batch_size=crop_batch_size,
            amp_enabled=amp_enabled,
            return_items=True,
        )
        _tag_selector_pool(fallback_rescue_candidate_items, "fallback_rescue")
        selector_candidate_pool_items.extend(fallback_rescue_candidate_items)
        if float(fallback_rescue_best["score"]) > float(best["score"]):
            fallback_rescue_best["selected_via_fallback_rescue"] = True
            fallback_rescue_best["pre_fallback_rescue_score"] = float(best["score"])
            fallback_rescue_best["fallback_rescue_min_score"] = float(fallback_rescue_min_score)
            fallback_rescue_best["fallback_rescue_max_score"] = float(fallback_rescue_max_score)
            fallback_rescue_best["fallback_rescue_source_group"] = fallback_rescue_source_group
            fallback_rescue_best["fallback_rescue_max_ball_candidates"] = int(fallback_rescue_max_ball_candidates)
            best = fallback_rescue_best
            summaries = fallback_rescue_summaries
            active_candidate_items = fallback_rescue_candidate_items

    if (
        reticle_fill_rescue_checkpoint_path is not None
        and reticle_fill_rescue_min_score is not None
        and reticle_fill_rescue_max_score is not None
        and bool(best.get("selected_via_reticle_recovery", False))
        and _candidate_source_group(best.get("candidate_source")) == "reticle_global"
        and float(best["score"]) >= float(reticle_fill_rescue_min_score)
        and float(best["score"]) < float(reticle_fill_rescue_max_score)
        and (
            reticle_fill_rescue_current_max_ball_fill_fraction is None
            or float(best.get("features", {}).get("ball_fill_fraction", 0.0))
            < float(reticle_fill_rescue_current_max_ball_fill_fraction)
        )
    ):
        reticle_fill_rescue_candidates = generate_ball_candidates(
            image,
            {
                "max_ball_candidates": int(reticle_fill_rescue_max_ball_candidates),
                "crop_scale": 4.25,
                "crop_padding_px": 24,
                "upscale_factor": 1.0,
                "hough": {"dp": 1.2, "min_dist_factor": 1.5, "param1": 110, "param2": 18, "min_radius": 7, "max_radius": 80},
                "fallback": {"grid_candidates": 6, "white_ball_bias": True},
            },
            output_dir=output_dir / "reticle_fill_rescue_intermediates" if save_intermediates else None,
        )
        reticle_fill_rescue_model = _load_guideline_model(reticle_fill_rescue_checkpoint_path, device)
        reticle_fill_rescue_reranker_path = (
            reticle_fill_rescue_reranker_checkpoint_path
            or reticle_fill_rescue_checkpoint_path.with_name("guideline_reranker_best.pt")
        )
        reticle_fill_rescue_reranker = (
            _load_reranker_cached(reticle_fill_rescue_reranker_path, device)
            if reticle_fill_rescue_reranker_path.exists()
            else None
        )
        reticle_fill_rescue_dir = ensure_dir(output_dir / "reticle_fill_rescue_model")
        reticle_fill_rescue_best, reticle_fill_rescue_summaries, reticle_fill_rescue_candidate_items = _score_candidates(
            model=reticle_fill_rescue_model,
            image=image,
            candidates=reticle_fill_rescue_candidates,
            output_dir=reticle_fill_rescue_dir,
            image_size=image_size,
            device=device,
            reranker=reticle_fill_rescue_reranker,
            save_intermediates=save_intermediates,
            crop_batch_size=crop_batch_size,
            amp_enabled=amp_enabled,
            return_items=True,
        )
        _tag_selector_pool(reticle_fill_rescue_candidate_items, "reticle_fill_rescue")
        selector_candidate_pool_items.extend(reticle_fill_rescue_candidate_items)
        if (
            _candidate_source_group(reticle_fill_rescue_best.get("candidate_source")) == "reticle_global"
            and (
                reticle_fill_rescue_target_min_ball_fill_fraction is None
                or float(reticle_fill_rescue_best.get("features", {}).get("ball_fill_fraction", 0.0))
                >= float(reticle_fill_rescue_target_min_ball_fill_fraction)
            )
        ):
            reticle_fill_rescue_best["selected_via_reticle_fill_rescue"] = True
            reticle_fill_rescue_best["pre_reticle_fill_rescue_score"] = float(best["score"])
            reticle_fill_rescue_best["reticle_fill_rescue_min_score"] = float(reticle_fill_rescue_min_score)
            reticle_fill_rescue_best["reticle_fill_rescue_max_score"] = float(reticle_fill_rescue_max_score)
            reticle_fill_rescue_best["reticle_fill_rescue_current_max_ball_fill_fraction"] = (
                float(reticle_fill_rescue_current_max_ball_fill_fraction)
                if reticle_fill_rescue_current_max_ball_fill_fraction is not None
                else None
            )
            reticle_fill_rescue_best["reticle_fill_rescue_target_min_ball_fill_fraction"] = (
                float(reticle_fill_rescue_target_min_ball_fill_fraction)
                if reticle_fill_rescue_target_min_ball_fill_fraction is not None
                else None
            )
            reticle_fill_rescue_best["reticle_fill_rescue_max_ball_candidates"] = int(reticle_fill_rescue_max_ball_candidates)
            best = reticle_fill_rescue_best
            summaries = reticle_fill_rescue_summaries
            active_candidate_items = reticle_fill_rescue_candidate_items

    if _micro_line_rescue_should_run(best, micro_line_rescue_config):
        assert micro_line_rescue_config is not None
        checkpoint_value = micro_line_rescue_config.get("checkpoint_path") or micro_line_rescue_config.get("checkpoint")
        if checkpoint_value:
            micro_line_checkpoint_path = Path(str(checkpoint_value))
            micro_line_candidates = generate_ball_candidates(
                image,
                {
                    "max_ball_candidates": int(micro_line_rescue_config.get("max_ball_candidates", 10)),
                    "crop_scale": float(micro_line_rescue_config.get("crop_scale", 4.25)),
                    "crop_padding_px": int(micro_line_rescue_config.get("crop_padding_px", 24)),
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
                },
                output_dir=output_dir / "micro_line_rescue_intermediates" if save_intermediates else None,
            )
            micro_line_model = _load_guideline_model(micro_line_checkpoint_path, device)
            micro_line_reranker = None
            if bool(micro_line_rescue_config.get("use_reranker", False)):
                reranker_value = micro_line_rescue_config.get("reranker_checkpoint_path") or micro_line_rescue_config.get(
                    "reranker_checkpoint"
                )
                micro_line_reranker_path = (
                    Path(str(reranker_value))
                    if reranker_value
                    else micro_line_checkpoint_path.with_name("guideline_reranker_best.pt")
                )
                micro_line_reranker = (
                    _load_reranker_cached(micro_line_reranker_path, device)
                    if micro_line_reranker_path.exists()
                    else None
                )
            micro_line_dir = ensure_dir(output_dir / "micro_line_rescue_model")
            _, _, micro_line_candidate_items = _score_candidates(
                model=micro_line_model,
                image=image,
                candidates=micro_line_candidates,
                output_dir=micro_line_dir,
                image_size=int(micro_line_rescue_config.get("image_size", image_size)),
                device=device,
                reranker=micro_line_reranker,
                save_intermediates=save_intermediates,
                crop_batch_size=crop_batch_size,
                amp_enabled=amp_enabled,
                prediction_threshold=float(micro_line_rescue_config.get("prediction_threshold", 0.2)),
                return_items=True,
            )
            _tag_selector_pool(micro_line_candidate_items, "micro_line_rescue")
            for item in micro_line_candidate_items:
                item["micro_line_rescue_pool"] = True
                item["micro_line_rescue_threshold"] = float(
                    micro_line_rescue_config.get("prediction_threshold", 0.2)
                )
                item["pre_micro_line_rescue_score"] = float(best["score"])
            selector_candidate_pool_items.extend(micro_line_candidate_items)

    colored_blob_rescue_configs = _rescue_config_list(colored_blob_rescue_config)
    for colored_blob_rescue_index, current_colored_blob_rescue_config in enumerate(colored_blob_rescue_configs):
        if not _rescue_config_should_run(best, current_colored_blob_rescue_config):
            continue
        checkpoint_value = current_colored_blob_rescue_config.get("checkpoint_path") or current_colored_blob_rescue_config.get("checkpoint")
        if checkpoint_value:
            colored_blob_checkpoint_path = Path(str(checkpoint_value))
            rescue_name = str(
                current_colored_blob_rescue_config.get(
                    "name",
                    f"colored_blob_rescue_{colored_blob_rescue_index}",
                )
            )
            rescue_dir_suffix = "" if len(colored_blob_rescue_configs) == 1 else f"_{colored_blob_rescue_index:02d}"
            colored_blob_candidates = generate_ball_candidates(
                image,
                {
                    "max_ball_candidates": int(current_colored_blob_rescue_config.get("max_ball_candidates", 40)),
                    "crop_scale": float(current_colored_blob_rescue_config.get("crop_scale", 4.25)),
                    "crop_padding_px": int(current_colored_blob_rescue_config.get("crop_padding_px", 24)),
                    "upscale_factor": 1.0,
                    "colored_blob_candidates": True,
                    "colored_blob": current_colored_blob_rescue_config.get("colored_blob", {}),
                    "hough": {
                        "dp": 1.2,
                        "min_dist_factor": 1.5,
                        "param1": 110,
                        "param2": 18,
                        "min_radius": 7,
                        "max_radius": 80,
                    },
                    "fallback": {"grid_candidates": 6, "white_ball_bias": True},
                },
                output_dir=(
                    output_dir / f"colored_blob_rescue_intermediates{rescue_dir_suffix}"
                    if save_intermediates
                    else None
                ),
            )
            colored_blob_model = _load_guideline_model(colored_blob_checkpoint_path, device)
            colored_blob_reranker = None
            if bool(current_colored_blob_rescue_config.get("use_reranker", True)):
                reranker_value = current_colored_blob_rescue_config.get("reranker_checkpoint_path") or current_colored_blob_rescue_config.get(
                    "reranker_checkpoint"
                )
                colored_blob_reranker_path = (
                    Path(str(reranker_value))
                    if reranker_value
                    else colored_blob_checkpoint_path.with_name("guideline_reranker_best.pt")
                )
                colored_blob_reranker = (
                    _load_reranker_cached(colored_blob_reranker_path, device)
                    if colored_blob_reranker_path.exists()
                    else None
                )
            colored_blob_dir = ensure_dir(output_dir / f"colored_blob_rescue_model{rescue_dir_suffix}")
            _, _, colored_blob_candidate_items = _score_candidates(
                model=colored_blob_model,
                image=image,
                candidates=colored_blob_candidates,
                output_dir=colored_blob_dir,
                image_size=int(current_colored_blob_rescue_config.get("image_size", image_size)),
                device=device,
                reranker=colored_blob_reranker,
                save_intermediates=save_intermediates,
                crop_batch_size=crop_batch_size,
                amp_enabled=amp_enabled,
                prediction_threshold=float(current_colored_blob_rescue_config.get("prediction_threshold", 0.35)),
                return_items=True,
            )
            _tag_selector_pool(colored_blob_candidate_items, "colored_blob_rescue")
            for item in colored_blob_candidate_items:
                item["colored_blob_rescue_pool"] = True
                item["colored_blob_rescue_name"] = rescue_name
                item["colored_blob_rescue_index"] = int(colored_blob_rescue_index)
                item["colored_blob_rescue_threshold"] = float(
                    current_colored_blob_rescue_config.get("prediction_threshold", 0.35)
                )
                item["pre_colored_blob_rescue_score"] = float(best["score"])
            selector_candidate_pool_items.extend(colored_blob_candidate_items)

    selector_candidate_pool_items.extend(
        _run_selector_candidate_pool_rescue_configs(
            image=image,
            output_dir=output_dir,
            current_item=best,
            configs=selector_candidate_pool_rescue_config,
            image_size=image_size,
            device=device,
            save_intermediates=save_intermediates,
            crop_batch_size=crop_batch_size,
            amp_enabled=amp_enabled,
        )
    )

    best = _apply_candidate_selector_rescue(
        best,
        active_candidate_items,
        selector_candidate_pool_items,
        candidate_selector_rescue_rules,
    )
    best = _apply_rescue_arbiter(
        best,
        selector_candidate_pool_items,
        rescue_arbiter_config,
        device=device,
    )

    if (
        external_rescue_checkpoint_path is not None
        and _external_rescue_is_eligible(
            best,
            source_group=external_rescue_source_group,
            min_score=external_rescue_min_score,
            max_score=external_rescue_max_score,
            current_max_ball_fill_fraction=external_rescue_current_max_ball_fill_fraction,
        )
    ):
        external_mask, external_stats = _run_external_guideline_rescue(image, external_rescue_checkpoint_path)
        if (
            int(external_stats["pred_area"]) >= int(external_rescue_min_pred_area)
            and float(external_stats["prob_mean_on_pred"]) >= float(external_rescue_min_prob_mean)
        ):
            best = {
                "candidate_id": "external_main_rescue",
                "candidate_source": "external_main_rescue",
                "score": float(external_stats["prob_mean_on_pred"]),
                "full_frame_mask": external_mask,
                "selected_via_external_rescue": True,
                "pre_external_rescue_score": float(best["score"]),
                "external_rescue_source_group": external_rescue_source_group,
                "external_rescue_min_score": float(external_rescue_min_score) if external_rescue_min_score is not None else None,
                "external_rescue_max_score": float(external_rescue_max_score) if external_rescue_max_score is not None else None,
                "external_rescue_current_max_ball_fill_fraction": (
                    float(external_rescue_current_max_ball_fill_fraction)
                    if external_rescue_current_max_ball_fill_fraction is not None
                    else None
                ),
                "external_rescue_min_pred_area": int(external_rescue_min_pred_area),
                "external_rescue_min_prob_mean": float(external_rescue_min_prob_mean),
                "external_pred_area": int(external_stats["pred_area"]),
                "external_prob_mean_on_pred": float(external_stats["prob_mean_on_pred"]),
                "external_threshold": float(external_stats["threshold"]),
            }

    if (
        external_mid_rescue_checkpoint_path is not None
        and _external_rescue_is_eligible(
            best,
            source_group=external_mid_rescue_source_group,
            min_score=external_mid_rescue_min_score,
            max_score=external_mid_rescue_max_score,
            current_max_ball_fill_fraction=external_mid_rescue_current_max_ball_fill_fraction,
        )
    ):
        external_mid_mask, external_mid_stats = _run_external_guideline_rescue(image, external_mid_rescue_checkpoint_path)
        if (
            int(external_mid_stats["pred_area"]) >= int(external_mid_rescue_min_pred_area)
            and float(external_mid_stats["prob_mean_on_pred"]) >= float(external_mid_rescue_min_prob_mean)
        ):
            best = {
                "candidate_id": "external_main_mid_rescue",
                "candidate_source": "external_main_mid_rescue",
                "score": float(external_mid_stats["prob_mean_on_pred"]),
                "full_frame_mask": external_mid_mask,
                "selected_via_external_mid_rescue": True,
                "pre_external_mid_rescue_score": float(best["score"]),
                "external_mid_rescue_source_group": external_mid_rescue_source_group,
                "external_mid_rescue_min_score": float(external_mid_rescue_min_score) if external_mid_rescue_min_score is not None else None,
                "external_mid_rescue_max_score": float(external_mid_rescue_max_score) if external_mid_rescue_max_score is not None else None,
                "external_mid_rescue_current_max_ball_fill_fraction": (
                    float(external_mid_rescue_current_max_ball_fill_fraction)
                    if external_mid_rescue_current_max_ball_fill_fraction is not None
                    else None
                ),
                "external_mid_rescue_min_pred_area": int(external_mid_rescue_min_pred_area),
                "external_mid_rescue_min_prob_mean": float(external_mid_rescue_min_prob_mean),
                "external_mid_pred_area": int(external_mid_stats["pred_area"]),
                "external_mid_prob_mean_on_pred": float(external_mid_stats["prob_mean_on_pred"]),
                "external_mid_threshold": float(external_mid_stats["threshold"]),
            }

    if (
        external_lowmid_rescue_checkpoint_path is not None
        and _external_rescue_is_eligible(
            best,
            source_group=external_lowmid_rescue_source_group,
            min_score=external_lowmid_rescue_min_score,
            max_score=external_lowmid_rescue_max_score,
            current_max_ball_fill_fraction=external_lowmid_rescue_current_max_ball_fill_fraction,
        )
    ):
        external_lowmid_mask, external_lowmid_stats = _run_external_guideline_rescue(image, external_lowmid_rescue_checkpoint_path)
        if (
            int(external_lowmid_stats["pred_area"]) >= int(external_lowmid_rescue_min_pred_area)
            and float(external_lowmid_stats["prob_mean_on_pred"]) >= float(external_lowmid_rescue_min_prob_mean)
        ):
            best = {
                "candidate_id": "external_main_lowmid_rescue",
                "candidate_source": "external_main_lowmid_rescue",
                "score": float(external_lowmid_stats["prob_mean_on_pred"]),
                "full_frame_mask": external_lowmid_mask,
                "selected_via_external_lowmid_rescue": True,
                "pre_external_lowmid_rescue_score": float(best["score"]),
                "external_lowmid_rescue_source_group": external_lowmid_rescue_source_group,
                "external_lowmid_rescue_min_score": float(external_lowmid_rescue_min_score) if external_lowmid_rescue_min_score is not None else None,
                "external_lowmid_rescue_max_score": float(external_lowmid_rescue_max_score) if external_lowmid_rescue_max_score is not None else None,
                "external_lowmid_rescue_current_max_ball_fill_fraction": (
                    float(external_lowmid_rescue_current_max_ball_fill_fraction)
                    if external_lowmid_rescue_current_max_ball_fill_fraction is not None
                    else None
                ),
                "external_lowmid_rescue_min_pred_area": int(external_lowmid_rescue_min_pred_area),
                "external_lowmid_rescue_min_prob_mean": float(external_lowmid_rescue_min_prob_mean),
                "external_lowmid_pred_area": int(external_lowmid_stats["pred_area"]),
                "external_lowmid_prob_mean_on_pred": float(external_lowmid_stats["prob_mean_on_pred"]),
                "external_lowmid_threshold": float(external_lowmid_stats["threshold"]),
            }

    if (
        external_reticle_rescue_checkpoint_path is not None
        and _external_rescue_is_eligible(
            best,
            source_group=external_reticle_rescue_source_group,
            min_score=external_reticle_rescue_min_score,
            max_score=external_reticle_rescue_max_score,
            current_max_ball_fill_fraction=external_reticle_rescue_current_max_ball_fill_fraction,
        )
    ):
        external_reticle_mask, external_reticle_stats = _run_external_guideline_rescue(
            image, external_reticle_rescue_checkpoint_path
        )
        if (
            int(external_reticle_stats["pred_area"]) >= int(external_reticle_rescue_min_pred_area)
            and float(external_reticle_stats["prob_mean_on_pred"]) >= float(external_reticle_rescue_min_prob_mean)
        ):
            best = {
                "candidate_id": "external_main_reticle_rescue",
                "candidate_source": "external_main_reticle_rescue",
                "score": float(external_reticle_stats["prob_mean_on_pred"]),
                "full_frame_mask": external_reticle_mask,
                "selected_via_external_reticle_rescue": True,
                "pre_external_reticle_rescue_score": float(best["score"]),
                "external_reticle_rescue_source_group": external_reticle_rescue_source_group,
                "external_reticle_rescue_min_score": (
                    float(external_reticle_rescue_min_score) if external_reticle_rescue_min_score is not None else None
                ),
                "external_reticle_rescue_max_score": (
                    float(external_reticle_rescue_max_score) if external_reticle_rescue_max_score is not None else None
                ),
                "external_reticle_rescue_current_max_ball_fill_fraction": (
                    float(external_reticle_rescue_current_max_ball_fill_fraction)
                    if external_reticle_rescue_current_max_ball_fill_fraction is not None
                    else None
                ),
                "external_reticle_rescue_min_pred_area": int(external_reticle_rescue_min_pred_area),
                "external_reticle_rescue_min_prob_mean": float(external_reticle_rescue_min_prob_mean),
                "external_reticle_pred_area": int(external_reticle_stats["pred_area"]),
                "external_reticle_prob_mean_on_pred": float(external_reticle_stats["prob_mean_on_pred"]),
                "external_reticle_threshold": float(external_reticle_stats["threshold"]),
            }

    if (
        external_tiny_reticle_rescue_checkpoint_path is not None
        and _external_rescue_is_eligible(
            best,
            source_group=external_tiny_reticle_rescue_source_group,
            min_score=external_tiny_reticle_rescue_min_score,
            max_score=external_tiny_reticle_rescue_max_score,
            current_max_ball_fill_fraction=external_tiny_reticle_rescue_current_max_ball_fill_fraction,
        )
    ):
        external_tiny_reticle_mask, external_tiny_reticle_stats = _run_external_guideline_rescue(
            image, external_tiny_reticle_rescue_checkpoint_path
        )
        if (
            int(external_tiny_reticle_stats["pred_area"]) >= int(external_tiny_reticle_rescue_min_pred_area)
            and float(external_tiny_reticle_stats["prob_mean_on_pred"]) >= float(external_tiny_reticle_rescue_min_prob_mean)
        ):
            best = {
                "candidate_id": "external_tiny_reticle_rescue",
                "candidate_source": "external_tiny_reticle_rescue",
                "score": float(external_tiny_reticle_stats["prob_mean_on_pred"]),
                "full_frame_mask": external_tiny_reticle_mask,
                "selected_via_external_tiny_reticle_rescue": True,
                "pre_external_tiny_reticle_rescue_score": float(best["score"]),
                "external_tiny_reticle_rescue_source_group": external_tiny_reticle_rescue_source_group,
                "external_tiny_reticle_rescue_min_score": (
                    float(external_tiny_reticle_rescue_min_score)
                    if external_tiny_reticle_rescue_min_score is not None
                    else None
                ),
                "external_tiny_reticle_rescue_max_score": (
                    float(external_tiny_reticle_rescue_max_score)
                    if external_tiny_reticle_rescue_max_score is not None
                    else None
                ),
                "external_tiny_reticle_rescue_current_max_ball_fill_fraction": (
                    float(external_tiny_reticle_rescue_current_max_ball_fill_fraction)
                    if external_tiny_reticle_rescue_current_max_ball_fill_fraction is not None
                    else None
                ),
                "external_tiny_reticle_rescue_min_pred_area": int(external_tiny_reticle_rescue_min_pred_area),
                "external_tiny_reticle_rescue_min_prob_mean": float(external_tiny_reticle_rescue_min_prob_mean),
                "external_tiny_reticle_pred_area": int(external_tiny_reticle_stats["pred_area"]),
                "external_tiny_reticle_prob_mean_on_pred": float(external_tiny_reticle_stats["prob_mean_on_pred"]),
                "external_tiny_reticle_threshold": float(external_tiny_reticle_stats["threshold"]),
            }

    if (
        external_reticle_large_rescue_checkpoint_path is not None
        and _external_rescue_is_eligible(
            best,
            source_group=external_reticle_large_rescue_source_group,
            min_score=external_reticle_large_rescue_min_score,
            max_score=external_reticle_large_rescue_max_score,
            current_max_ball_fill_fraction=external_reticle_large_rescue_current_max_ball_fill_fraction,
        )
    ):
        external_reticle_large_mask, external_reticle_large_stats = _run_external_guideline_rescue(
            image, external_reticle_large_rescue_checkpoint_path
        )
        if (
            int(external_reticle_large_stats["pred_area"]) >= int(external_reticle_large_rescue_min_pred_area)
            and float(external_reticle_large_stats["prob_mean_on_pred"]) >= float(external_reticle_large_rescue_min_prob_mean)
        ):
            best = {
                "candidate_id": "external_main_reticle_large_rescue",
                "candidate_source": "external_main_reticle_large_rescue",
                "score": float(external_reticle_large_stats["prob_mean_on_pred"]),
                "full_frame_mask": external_reticle_large_mask,
                "selected_via_external_reticle_large_rescue": True,
                "pre_external_reticle_large_rescue_score": float(best["score"]),
                "external_reticle_large_rescue_source_group": external_reticle_large_rescue_source_group,
                "external_reticle_large_rescue_min_score": (
                    float(external_reticle_large_rescue_min_score)
                    if external_reticle_large_rescue_min_score is not None
                    else None
                ),
                "external_reticle_large_rescue_max_score": (
                    float(external_reticle_large_rescue_max_score)
                    if external_reticle_large_rescue_max_score is not None
                    else None
                ),
                "external_reticle_large_rescue_current_max_ball_fill_fraction": (
                    float(external_reticle_large_rescue_current_max_ball_fill_fraction)
                    if external_reticle_large_rescue_current_max_ball_fill_fraction is not None
                    else None
                ),
                "external_reticle_large_rescue_min_pred_area": int(external_reticle_large_rescue_min_pred_area),
                "external_reticle_large_rescue_min_prob_mean": float(external_reticle_large_rescue_min_prob_mean),
                "external_reticle_large_pred_area": int(external_reticle_large_stats["pred_area"]),
                "external_reticle_large_prob_mean_on_pred": float(external_reticle_large_stats["prob_mean_on_pred"]),
                "external_reticle_large_threshold": float(external_reticle_large_stats["threshold"]),
            }

    if (
        external_table_broad_rescue_checkpoint_path is not None
        and _external_rescue_is_eligible(
            best,
            source_group=external_table_broad_rescue_source_group,
            min_score=external_table_broad_rescue_min_score,
            max_score=external_table_broad_rescue_max_score,
            current_max_ball_fill_fraction=external_table_broad_rescue_current_max_ball_fill_fraction,
        )
    ):
        external_table_broad_mask, external_table_broad_stats = _run_external_guideline_rescue(
            image, external_table_broad_rescue_checkpoint_path
        )
        if (
            int(external_table_broad_stats["pred_area"]) >= int(external_table_broad_rescue_min_pred_area)
            and float(external_table_broad_stats["prob_mean_on_pred"]) >= float(external_table_broad_rescue_min_prob_mean)
        ):
            best = {
                "candidate_id": "external_main_table_broad_rescue",
                "candidate_source": "external_main_table_broad_rescue",
                "score": float(external_table_broad_stats["prob_mean_on_pred"]),
                "full_frame_mask": external_table_broad_mask,
                "selected_via_external_table_broad_rescue": True,
                "pre_external_table_broad_rescue_score": float(best["score"]),
                "external_table_broad_rescue_source_group": external_table_broad_rescue_source_group,
                "external_table_broad_rescue_min_score": (
                    float(external_table_broad_rescue_min_score)
                    if external_table_broad_rescue_min_score is not None
                    else None
                ),
                "external_table_broad_rescue_max_score": (
                    float(external_table_broad_rescue_max_score)
                    if external_table_broad_rescue_max_score is not None
                    else None
                ),
                "external_table_broad_rescue_current_max_ball_fill_fraction": (
                    float(external_table_broad_rescue_current_max_ball_fill_fraction)
                    if external_table_broad_rescue_current_max_ball_fill_fraction is not None
                    else None
                ),
                "external_table_broad_rescue_min_pred_area": int(external_table_broad_rescue_min_pred_area),
                "external_table_broad_rescue_min_prob_mean": float(external_table_broad_rescue_min_prob_mean),
                "external_table_broad_pred_area": int(external_table_broad_stats["pred_area"]),
                "external_table_broad_prob_mean_on_pred": float(external_table_broad_stats["prob_mean_on_pred"]),
                "external_table_broad_threshold": float(external_table_broad_stats["threshold"]),
            }

    if (
        external_blob_low_rescue_checkpoint_path is not None
        and _external_rescue_is_eligible(
            best,
            source_group=external_blob_low_rescue_source_group,
            min_score=external_blob_low_rescue_min_score,
            max_score=external_blob_low_rescue_max_score,
            current_max_ball_fill_fraction=external_blob_low_rescue_current_max_ball_fill_fraction,
        )
    ):
        external_blob_low_mask, external_blob_low_stats = _run_external_guideline_rescue(
            image, external_blob_low_rescue_checkpoint_path
        )
        if (
            int(external_blob_low_stats["pred_area"]) >= int(external_blob_low_rescue_min_pred_area)
            and float(external_blob_low_stats["prob_mean_on_pred"]) >= float(external_blob_low_rescue_min_prob_mean)
        ):
            best = {
                "candidate_id": "external_main_blob_low_rescue",
                "candidate_source": "external_main_blob_low_rescue",
                "score": float(external_blob_low_stats["prob_mean_on_pred"]),
                "full_frame_mask": external_blob_low_mask,
                "selected_via_external_blob_low_rescue": True,
                "pre_external_blob_low_rescue_score": float(best["score"]),
                "external_blob_low_rescue_source_group": external_blob_low_rescue_source_group,
                "external_blob_low_rescue_min_score": (
                    float(external_blob_low_rescue_min_score) if external_blob_low_rescue_min_score is not None else None
                ),
                "external_blob_low_rescue_max_score": (
                    float(external_blob_low_rescue_max_score) if external_blob_low_rescue_max_score is not None else None
                ),
                "external_blob_low_rescue_current_max_ball_fill_fraction": (
                    float(external_blob_low_rescue_current_max_ball_fill_fraction)
                    if external_blob_low_rescue_current_max_ball_fill_fraction is not None
                    else None
                ),
                "external_blob_low_rescue_min_pred_area": int(external_blob_low_rescue_min_pred_area),
                "external_blob_low_rescue_min_prob_mean": float(external_blob_low_rescue_min_prob_mean),
                "external_blob_low_pred_area": int(external_blob_low_stats["pred_area"]),
                "external_blob_low_prob_mean_on_pred": float(external_blob_low_stats["prob_mean_on_pred"]),
                "external_blob_low_threshold": float(external_blob_low_stats["threshold"]),
            }

    if (
        external_table_rescue_checkpoint_path is not None
        and _external_rescue_is_eligible(
            best,
            source_group=external_table_rescue_source_group,
            min_score=external_table_rescue_min_score,
            max_score=external_table_rescue_max_score,
            current_max_ball_fill_fraction=external_table_rescue_current_max_ball_fill_fraction,
        )
    ):
        external_table_mask, external_table_stats = _run_external_guideline_rescue(image, external_table_rescue_checkpoint_path)
        if (
            int(external_table_stats["pred_area"]) >= int(external_table_rescue_min_pred_area)
            and float(external_table_stats["prob_mean_on_pred"]) >= float(external_table_rescue_min_prob_mean)
        ):
            best = {
                "candidate_id": "external_main_table_rescue",
                "candidate_source": "external_main_table_rescue",
                "score": float(external_table_stats["prob_mean_on_pred"]),
                "full_frame_mask": external_table_mask,
                "selected_via_external_table_rescue": True,
                "pre_external_table_rescue_score": float(best["score"]),
                "external_table_rescue_source_group": external_table_rescue_source_group,
                "external_table_rescue_min_score": (
                    float(external_table_rescue_min_score) if external_table_rescue_min_score is not None else None
                ),
                "external_table_rescue_max_score": (
                    float(external_table_rescue_max_score) if external_table_rescue_max_score is not None else None
                ),
                "external_table_rescue_current_max_ball_fill_fraction": (
                    float(external_table_rescue_current_max_ball_fill_fraction)
                    if external_table_rescue_current_max_ball_fill_fraction is not None
                    else None
                ),
                "external_table_rescue_min_pred_area": int(external_table_rescue_min_pred_area),
                "external_table_rescue_min_prob_mean": float(external_table_rescue_min_prob_mean),
                "external_table_pred_area": int(external_table_stats["pred_area"]),
                "external_table_prob_mean_on_pred": float(external_table_stats["prob_mean_on_pred"]),
                "external_table_threshold": float(external_table_stats["threshold"]),
            }

    if (
        external_blob_rescue_checkpoint_path is not None
        and _external_rescue_is_eligible(
            best,
            source_group=external_blob_rescue_source_group,
            min_score=external_blob_rescue_min_score,
            max_score=external_blob_rescue_max_score,
            current_max_ball_fill_fraction=external_blob_rescue_current_max_ball_fill_fraction,
        )
    ):
        external_blob_mask, external_blob_stats = _run_external_guideline_rescue(image, external_blob_rescue_checkpoint_path)
        if (
            int(external_blob_stats["pred_area"]) >= int(external_blob_rescue_min_pred_area)
            and float(external_blob_stats["prob_mean_on_pred"]) >= float(external_blob_rescue_min_prob_mean)
        ):
            best = {
                "candidate_id": "external_main_blob_rescue",
                "candidate_source": "external_main_blob_rescue",
                "score": float(external_blob_stats["prob_mean_on_pred"]),
                "full_frame_mask": external_blob_mask,
                "selected_via_external_blob_rescue": True,
                "pre_external_blob_rescue_score": float(best["score"]),
                "external_blob_rescue_source_group": external_blob_rescue_source_group,
                "external_blob_rescue_min_score": float(external_blob_rescue_min_score) if external_blob_rescue_min_score is not None else None,
                "external_blob_rescue_max_score": float(external_blob_rescue_max_score) if external_blob_rescue_max_score is not None else None,
                "external_blob_rescue_current_max_ball_fill_fraction": (
                    float(external_blob_rescue_current_max_ball_fill_fraction)
                    if external_blob_rescue_current_max_ball_fill_fraction is not None
                    else None
                ),
                "external_blob_rescue_min_pred_area": int(external_blob_rescue_min_pred_area),
                "external_blob_rescue_min_prob_mean": float(external_blob_rescue_min_prob_mean),
                "external_blob_pred_area": int(external_blob_stats["pred_area"]),
                "external_blob_prob_mean_on_pred": float(external_blob_stats["prob_mean_on_pred"]),
                "external_blob_threshold": float(external_blob_stats["threshold"]),
            }

    if (
        external_tiny_blob_rescue_checkpoint_path is not None
        and _external_rescue_is_eligible(
            best,
            source_group=external_tiny_blob_rescue_source_group,
            min_score=external_tiny_blob_rescue_min_score,
            max_score=external_tiny_blob_rescue_max_score,
            current_max_ball_fill_fraction=external_tiny_blob_rescue_current_max_ball_fill_fraction,
        )
    ):
        external_tiny_blob_mask, external_tiny_blob_stats = _run_external_guideline_rescue(
            image, external_tiny_blob_rescue_checkpoint_path
        )
        if (
            int(external_tiny_blob_stats["pred_area"]) >= int(external_tiny_blob_rescue_min_pred_area)
            and float(external_tiny_blob_stats["prob_mean_on_pred"]) >= float(external_tiny_blob_rescue_min_prob_mean)
        ):
            best = {
                "candidate_id": "external_tiny_blob_rescue",
                "candidate_source": "external_tiny_blob_rescue",
                "score": float(external_tiny_blob_stats["prob_mean_on_pred"]),
                "full_frame_mask": external_tiny_blob_mask,
                "selected_via_external_tiny_blob_rescue": True,
                "pre_external_tiny_blob_rescue_score": float(best["score"]),
                "external_tiny_blob_rescue_source_group": external_tiny_blob_rescue_source_group,
                "external_tiny_blob_rescue_min_score": float(external_tiny_blob_rescue_min_score) if external_tiny_blob_rescue_min_score is not None else None,
                "external_tiny_blob_rescue_max_score": float(external_tiny_blob_rescue_max_score) if external_tiny_blob_rescue_max_score is not None else None,
                "external_tiny_blob_rescue_current_max_ball_fill_fraction": (
                    float(external_tiny_blob_rescue_current_max_ball_fill_fraction)
                    if external_tiny_blob_rescue_current_max_ball_fill_fraction is not None
                    else None
                ),
                "external_tiny_blob_rescue_min_pred_area": int(external_tiny_blob_rescue_min_pred_area),
                "external_tiny_blob_rescue_min_prob_mean": float(external_tiny_blob_rescue_min_prob_mean),
                "external_tiny_blob_pred_area": int(external_tiny_blob_stats["pred_area"]),
                "external_tiny_blob_prob_mean_on_pred": float(external_tiny_blob_stats["prob_mean_on_pred"]),
                "external_tiny_blob_threshold": float(external_tiny_blob_stats["threshold"]),
            }

    selector_candidate_pool_items.extend(
        _run_selector_candidate_pool_rescue_configs(
            image=image,
            output_dir=output_dir,
            current_item=best,
            configs=post_external_selector_candidate_pool_rescue_config,
            image_size=image_size,
            device=device,
            save_intermediates=save_intermediates,
            crop_batch_size=crop_batch_size,
            amp_enabled=amp_enabled,
            stage_name="post_external",
        )
    )

    best = _apply_candidate_selector_rescue(
        best,
        active_candidate_items,
        selector_candidate_pool_items,
        post_external_candidate_selector_rescue_rules,
        stage_name="post_external",
    )
    best = _apply_rescue_arbiter(
        best,
        selector_candidate_pool_items,
        post_external_rescue_arbiter_config,
        device=device,
        stage_name="post_external",
    )

    if final_refine_checkpoint_path is not None and "full_frame_mask" not in best:
        final_refine_model = _load_guideline_model(final_refine_checkpoint_path, device)
        if _refine_crop_winner_mask(
            best=best,
            image=image,
            model=final_refine_model,
            image_size=int(final_refine_image_size or image_size),
            device=device,
            amp_enabled=amp_enabled,
        ):
            best["final_refine_checkpoint"] = str(final_refine_checkpoint_path)

    if "full_frame_mask" in best:
        final_mask = np.asarray(best["full_frame_mask"], dtype=np.uint8)
    else:
        final_mask = remap_mask_to_image(best["mask"], best["crop_meta"], image.shape)
    final_pred_pixels = int(np.count_nonzero(final_mask))
    if _table_hough_final_fallback_reject_is_eligible(
        best,
        pred_pixels=final_pred_pixels,
        min_pred_pixels=reject_table_hough_final_fallback_min_pred_pixels,
        max_score=reject_table_hough_final_fallback_max_score,
    ):
        best["selected_via_table_hough_final_fallback_reject"] = True
        best["pre_reject_pred_pixels"] = final_pred_pixels
        best["pre_reject_score"] = float(best["score"])
        best["reject_table_hough_final_fallback_min_pred_pixels"] = (
            int(reject_table_hough_final_fallback_min_pred_pixels)
            if reject_table_hough_final_fallback_min_pred_pixels is not None
            else None
        )
        best["reject_table_hough_final_fallback_max_score"] = (
            float(reject_table_hough_final_fallback_max_score)
            if reject_table_hough_final_fallback_max_score is not None
            else None
        )
        final_mask = np.zeros_like(final_mask)
    if _winner_source_area_reject_is_eligible(
        best,
        pred_pixels=final_pred_pixels,
        source_min_pred_pixels=reject_winner_source_min_pred_pixels,
        source_exemptions=reject_winner_source_min_pred_pixels_exemptions,
    ):
        best["selected_via_winner_source_area_reject"] = True
        best["pre_winner_source_area_reject_pred_pixels"] = final_pred_pixels
        best["pre_winner_source_area_reject_score"] = float(best["score"])
        best["reject_winner_source_min_pred_pixels"] = dict(
            sorted((str(key), int(value)) for key, value in reject_winner_source_min_pred_pixels.items())
        )
        final_mask = np.zeros_like(final_mask)
    if _winner_source_area_score_reject_is_eligible(
        best,
        pred_pixels=final_pred_pixels,
        source_rules=reject_winner_source_area_score,
    ):
        best["selected_via_winner_source_area_score_reject"] = True
        best["pre_winner_source_area_score_reject_pred_pixels"] = final_pred_pixels
        best["pre_winner_source_area_score_reject_score"] = float(best["score"])
        best["reject_winner_source_area_score"] = reject_winner_source_area_score
        final_mask = np.zeros_like(final_mask)
    current_final_pred_pixels = int(np.count_nonzero(final_mask))
    if _pre_final_fallback_score_reject_is_eligible(
        best,
        pred_pixels=current_final_pred_pixels,
        max_score=reject_pre_final_fallback_max_score,
    ):
        best["selected_via_pre_final_fallback_score_reject"] = True
        best["pre_pre_final_fallback_score_reject_pred_pixels"] = current_final_pred_pixels
        best["reject_pre_final_fallback_max_score"] = float(reject_pre_final_fallback_max_score)
        final_mask = np.zeros_like(final_mask)
    current_final_pred_pixels = int(np.count_nonzero(final_mask))
    if reject_final_context_rules is not None and current_final_pred_pixels > 0:
        final_context_features = _compute_final_context_features(
            image=image,
            final_mask=final_mask,
            summaries=summaries,
            current_item=best,
        )
        best["final_context_features"] = final_context_features
        should_reject, matched_rule = _final_context_reject_is_eligible(
            best,
            context_features=final_context_features,
            source_rules=reject_final_context_rules,
        )
        if should_reject:
            best["selected_via_final_context_reject"] = True
            best["pre_final_context_reject_pred_pixels"] = current_final_pred_pixels
            best["final_context_reject_rule"] = matched_rule
            final_mask = np.zeros_like(final_mask)
    current_final_pred_pixels = int(np.count_nonzero(final_mask))
    should_image_veto, image_veto_probability, image_veto_config = _image_final_veto_should_reject_any(
        image=image,
        final_mask=final_mask,
        current_item=best,
        config=image_final_veto_config,
        device=device,
    )
    if image_veto_probability is not None:
        best["image_final_veto_probability"] = float(image_veto_probability)
    if should_image_veto:
        best["selected_via_image_final_veto"] = True
        best["pre_image_final_veto_pred_pixels"] = current_final_pred_pixels
        if image_veto_config is not None:
            best["image_final_veto_config"] = _image_final_veto_config_without_checkpoint(image_veto_config)
        final_mask = np.zeros_like(final_mask)
    current_final_pred_pixels = int(np.count_nonzero(final_mask))
    post_final_best = _apply_post_final_candidate_selector_rescue(
        best,
        active_candidate_items,
        selector_candidate_pool_items,
        post_final_candidate_selector_rescue_rules,
        final_pred_pixels=current_final_pred_pixels,
    )
    if post_final_best.get("selected_via_post_final_candidate_selector_rescue"):
        post_final_mask = _full_frame_mask_from_candidate_item(post_final_best, image.shape)
        if post_final_mask is not None:
            best = post_final_best
            final_mask = post_final_mask
    final_overlay = overlay_mask(image, final_mask) if save_outputs or return_overlay else None
    report_payload = {
        "winner": {k: v for k, v in best.items() if k not in {"mask", "crop_meta"}},
        "candidates": summaries,
    }
    if save_outputs:
        save_mask_png(output_dir / "mask_final.png", final_mask)
        if final_overlay is None:
            final_overlay = overlay_mask(image, final_mask)
        save_rgb_image(output_dir / "overlay_final.png", final_overlay)
    if save_recolor_preview:
        recolor = recolor_masked_region(image, soft_alpha_mask(final_mask, 3), "#00FF00", preserve_luminance=True)
        save_rgb_image(output_dir / "recolor_preview.png", recolor)
    if save_report:
        save_json(output_dir / "report.json", report_payload)
    return {
        "winner": report_payload["winner"],
        "candidates": summaries,
        "final_mask": final_mask,
        "final_overlay": final_overlay,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reranker-checkpoint", default=None)
    parser.add_argument("--fallback-checkpoint", default=None)
    parser.add_argument("--fallback-reranker-checkpoint", default=None)
    parser.add_argument("--fallback-score-threshold", type=float, default=None)
    parser.add_argument("--meta-selector-checkpoint", default=None)
    parser.add_argument("--rescue-checkpoint", default=None)
    parser.add_argument("--rescue-reranker-checkpoint", default=None)
    parser.add_argument("--rescue-score-threshold", type=float, default=None)
    parser.add_argument("--rescue-max-ball-candidates", type=int, default=20)
    parser.add_argument("--final-fallback-score-threshold", type=float, default=None)
    parser.add_argument("--secondary-rescue-checkpoint", default=None)
    parser.add_argument("--secondary-rescue-reranker-checkpoint", default=None)
    parser.add_argument("--secondary-rescue-min-score", type=float, default=None)
    parser.add_argument("--secondary-rescue-max-score", type=float, default=None)
    parser.add_argument("--secondary-rescue-source", default=None)
    parser.add_argument("--secondary-rescue-min-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--secondary-rescue-max-ball-candidates", type=int, default=24)
    parser.add_argument("--secondary-rescue-current-exemptions-json", default=None)
    parser.add_argument("--primary-recovery-min-score", type=float, default=None)
    parser.add_argument("--primary-recovery-max-score", type=float, default=None)
    parser.add_argument("--primary-recovery-source", default=None)
    parser.add_argument("--primary-recovery-candidate-source", default=None)
    parser.add_argument("--primary-recovery-current-group", default=None)
    parser.add_argument("--primary-recovery-candidate-group", default=None)
    parser.add_argument("--primary-recovery-min-primary-score", type=float, default=None)
    parser.add_argument("--primary-recovery-min-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--reticle-recovery-min-score", type=float, default=None)
    parser.add_argument("--reticle-recovery-max-score", type=float, default=None)
    parser.add_argument("--reticle-recovery-min-primary-score", type=float, default=None)
    parser.add_argument("--fallback-rescue-checkpoint", default=None)
    parser.add_argument("--fallback-rescue-reranker-checkpoint", default=None)
    parser.add_argument("--fallback-rescue-min-score", type=float, default=None)
    parser.add_argument("--fallback-rescue-max-score", type=float, default=None)
    parser.add_argument("--fallback-rescue-source-group", default=None)
    parser.add_argument("--fallback-rescue-max-ball-candidates", type=int, default=20)
    parser.add_argument("--reticle-fill-rescue-checkpoint", default=None)
    parser.add_argument("--reticle-fill-rescue-reranker-checkpoint", default=None)
    parser.add_argument("--reticle-fill-rescue-min-score", type=float, default=None)
    parser.add_argument("--reticle-fill-rescue-max-score", type=float, default=None)
    parser.add_argument("--reticle-fill-rescue-current-max-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--reticle-fill-rescue-target-min-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--reticle-fill-rescue-max-ball-candidates", type=int, default=20)
    parser.add_argument("--external-rescue-checkpoint", default=None)
    parser.add_argument("--external-rescue-source-group", default=None)
    parser.add_argument("--external-rescue-min-score", type=float, default=None)
    parser.add_argument("--external-rescue-max-score", type=float, default=None)
    parser.add_argument("--external-rescue-current-max-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--external-rescue-min-pred-area", type=int, default=20)
    parser.add_argument("--external-rescue-min-prob-mean", type=float, default=0.9)
    parser.add_argument("--external-mid-rescue-checkpoint", default=None)
    parser.add_argument("--external-mid-rescue-source-group", default=None)
    parser.add_argument("--external-mid-rescue-min-score", type=float, default=None)
    parser.add_argument("--external-mid-rescue-max-score", type=float, default=None)
    parser.add_argument("--external-mid-rescue-current-max-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--external-mid-rescue-min-pred-area", type=int, default=20)
    parser.add_argument("--external-mid-rescue-min-prob-mean", type=float, default=0.88)
    parser.add_argument("--external-lowmid-rescue-checkpoint", default=None)
    parser.add_argument("--external-lowmid-rescue-source-group", default=None)
    parser.add_argument("--external-lowmid-rescue-min-score", type=float, default=None)
    parser.add_argument("--external-lowmid-rescue-max-score", type=float, default=None)
    parser.add_argument("--external-lowmid-rescue-current-max-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--external-lowmid-rescue-min-pred-area", type=int, default=400)
    parser.add_argument("--external-lowmid-rescue-min-prob-mean", type=float, default=0.8)
    parser.add_argument("--external-reticle-rescue-checkpoint", default=None)
    parser.add_argument("--external-reticle-rescue-source-group", default=None)
    parser.add_argument("--external-reticle-rescue-min-score", type=float, default=None)
    parser.add_argument("--external-reticle-rescue-max-score", type=float, default=None)
    parser.add_argument("--external-reticle-rescue-current-max-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--external-reticle-rescue-min-pred-area", type=int, default=20)
    parser.add_argument("--external-reticle-rescue-min-prob-mean", type=float, default=0.8)
    parser.add_argument("--external-tiny-reticle-rescue-checkpoint", default=None)
    parser.add_argument("--external-tiny-reticle-rescue-source-group", default=None)
    parser.add_argument("--external-tiny-reticle-rescue-min-score", type=float, default=None)
    parser.add_argument("--external-tiny-reticle-rescue-max-score", type=float, default=None)
    parser.add_argument("--external-tiny-reticle-rescue-current-max-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--external-tiny-reticle-rescue-min-pred-area", type=int, default=50)
    parser.add_argument("--external-tiny-reticle-rescue-min-prob-mean", type=float, default=0.94)
    parser.add_argument("--external-reticle-large-rescue-checkpoint", default=None)
    parser.add_argument("--external-reticle-large-rescue-source-group", default=None)
    parser.add_argument("--external-reticle-large-rescue-min-score", type=float, default=None)
    parser.add_argument("--external-reticle-large-rescue-max-score", type=float, default=None)
    parser.add_argument("--external-reticle-large-rescue-current-max-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--external-reticle-large-rescue-min-pred-area", type=int, default=400)
    parser.add_argument("--external-reticle-large-rescue-min-prob-mean", type=float, default=0.86)
    parser.add_argument("--external-table-broad-rescue-checkpoint", default=None)
    parser.add_argument("--external-table-broad-rescue-source-group", default=None)
    parser.add_argument("--external-table-broad-rescue-min-score", type=float, default=None)
    parser.add_argument("--external-table-broad-rescue-max-score", type=float, default=None)
    parser.add_argument("--external-table-broad-rescue-current-max-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--external-table-broad-rescue-min-pred-area", type=int, default=350)
    parser.add_argument("--external-table-broad-rescue-min-prob-mean", type=float, default=0.8)
    parser.add_argument("--external-blob-low-rescue-checkpoint", default=None)
    parser.add_argument("--external-blob-low-rescue-source-group", default=None)
    parser.add_argument("--external-blob-low-rescue-min-score", type=float, default=None)
    parser.add_argument("--external-blob-low-rescue-max-score", type=float, default=None)
    parser.add_argument("--external-blob-low-rescue-current-max-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--external-blob-low-rescue-min-pred-area", type=int, default=20)
    parser.add_argument("--external-blob-low-rescue-min-prob-mean", type=float, default=0.75)
    parser.add_argument("--external-table-rescue-checkpoint", default=None)
    parser.add_argument("--external-table-rescue-source-group", default=None)
    parser.add_argument("--external-table-rescue-min-score", type=float, default=None)
    parser.add_argument("--external-table-rescue-max-score", type=float, default=None)
    parser.add_argument("--external-table-rescue-current-max-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--external-table-rescue-min-pred-area", type=int, default=120)
    parser.add_argument("--external-table-rescue-min-prob-mean", type=float, default=0.84)
    parser.add_argument("--external-blob-rescue-checkpoint", default=None)
    parser.add_argument("--external-blob-rescue-source-group", default=None)
    parser.add_argument("--external-blob-rescue-min-score", type=float, default=None)
    parser.add_argument("--external-blob-rescue-max-score", type=float, default=None)
    parser.add_argument("--external-blob-rescue-current-max-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--external-blob-rescue-min-pred-area", type=int, default=300)
    parser.add_argument("--external-blob-rescue-min-prob-mean", type=float, default=0.95)
    parser.add_argument("--external-tiny-blob-rescue-checkpoint", default=None)
    parser.add_argument("--external-tiny-blob-rescue-source-group", default=None)
    parser.add_argument("--external-tiny-blob-rescue-min-score", type=float, default=None)
    parser.add_argument("--external-tiny-blob-rescue-max-score", type=float, default=None)
    parser.add_argument("--external-tiny-blob-rescue-current-max-ball-fill-fraction", type=float, default=None)
    parser.add_argument("--external-tiny-blob-rescue-min-pred-area", type=int, default=80)
    parser.add_argument("--external-tiny-blob-rescue-min-prob-mean", type=float, default=0.88)
    parser.add_argument("--micro-line-rescue-json", default=None)
    parser.add_argument("--colored-blob-rescue-json", default=None)
    parser.add_argument("--selector-candidate-pool-rescue-json", default=None)
    parser.add_argument("--post-external-selector-candidate-pool-rescue-json", default=None)
    parser.add_argument("--reject-table-hough-final-fallback-min-pred-pixels", type=int, default=None)
    parser.add_argument("--reject-table-hough-final-fallback-max-score", type=float, default=None)
    parser.add_argument("--reject-winner-source-min-pred-pixels-json", default=None)
    parser.add_argument("--reject-winner-source-min-pred-pixels-exemptions-json", default=None)
    parser.add_argument("--reject-winner-source-area-score-json", default=None)
    parser.add_argument("--reject-pre-final-fallback-max-score", type=float, default=None)
    parser.add_argument("--reject-final-context-rules-json", default=None)
    parser.add_argument("--candidate-selector-rescue-rules-json", default=None)
    parser.add_argument("--rescue-arbiter-json", default=None)
    parser.add_argument("--post-external-candidate-selector-rescue-rules-json", default=None)
    parser.add_argument("--post-external-rescue-arbiter-json", default=None)
    parser.add_argument("--post-final-candidate-selector-rescue-rules-json", default=None)
    parser.add_argument("--final-refine-checkpoint", default=None)
    parser.add_argument("--final-refine-image-size", type=int, default=None)
    parser.add_argument("--image-final-veto-json", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()
    reject_winner_source_min_pred_pixels = (
        json.loads(args.reject_winner_source_min_pred_pixels_json)
        if args.reject_winner_source_min_pred_pixels_json
        else None
    )
    reject_winner_source_min_pred_pixels_exemptions = (
        json.loads(args.reject_winner_source_min_pred_pixels_exemptions_json)
        if args.reject_winner_source_min_pred_pixels_exemptions_json
        else None
    )
    reject_winner_source_area_score = (
        json.loads(args.reject_winner_source_area_score_json)
        if args.reject_winner_source_area_score_json
        else None
    )
    reject_final_context_rules = (
        json.loads(args.reject_final_context_rules_json)
        if args.reject_final_context_rules_json
        else None
    )
    candidate_selector_rescue_rules = (
        json.loads(args.candidate_selector_rescue_rules_json)
        if args.candidate_selector_rescue_rules_json
        else None
    )
    rescue_arbiter_config = json.loads(args.rescue_arbiter_json) if args.rescue_arbiter_json else None
    post_external_candidate_selector_rescue_rules = (
        json.loads(args.post_external_candidate_selector_rescue_rules_json)
        if args.post_external_candidate_selector_rescue_rules_json
        else None
    )
    post_external_rescue_arbiter_config = (
        json.loads(args.post_external_rescue_arbiter_json)
        if args.post_external_rescue_arbiter_json
        else None
    )
    post_final_candidate_selector_rescue_rules = (
        json.loads(args.post_final_candidate_selector_rescue_rules_json)
        if args.post_final_candidate_selector_rescue_rules_json
        else None
    )
    micro_line_rescue_config = json.loads(args.micro_line_rescue_json) if args.micro_line_rescue_json else None
    colored_blob_rescue_config = json.loads(args.colored_blob_rescue_json) if args.colored_blob_rescue_json else None
    selector_candidate_pool_rescue_config = (
        json.loads(args.selector_candidate_pool_rescue_json)
        if args.selector_candidate_pool_rescue_json
        else None
    )
    post_external_selector_candidate_pool_rescue_config = (
        json.loads(args.post_external_selector_candidate_pool_rescue_json)
        if args.post_external_selector_candidate_pool_rescue_json
        else None
    )
    secondary_rescue_current_exemptions = (
        json.loads(args.secondary_rescue_current_exemptions_json)
        if args.secondary_rescue_current_exemptions_json
        else None
    )
    image_final_veto_config = json.loads(args.image_final_veto_json) if args.image_final_veto_json else None
    run_supervised_inference(
        Path(args.checkpoint),
        Path(args.input),
        Path(args.output_dir),
        args.image_size,
        Path(args.reranker_checkpoint) if args.reranker_checkpoint else None,
        Path(args.fallback_checkpoint) if args.fallback_checkpoint else None,
        Path(args.fallback_reranker_checkpoint) if args.fallback_reranker_checkpoint else None,
        args.fallback_score_threshold,
        Path(args.meta_selector_checkpoint) if args.meta_selector_checkpoint else None,
        Path(args.rescue_checkpoint) if args.rescue_checkpoint else None,
        Path(args.rescue_reranker_checkpoint) if args.rescue_reranker_checkpoint else None,
        args.rescue_score_threshold,
        args.rescue_max_ball_candidates,
        args.final_fallback_score_threshold,
        Path(args.secondary_rescue_checkpoint) if args.secondary_rescue_checkpoint else None,
        Path(args.secondary_rescue_reranker_checkpoint) if args.secondary_rescue_reranker_checkpoint else None,
        args.secondary_rescue_min_score,
        args.secondary_rescue_max_score,
        args.secondary_rescue_source,
        args.secondary_rescue_min_ball_fill_fraction,
        args.secondary_rescue_max_ball_candidates,
        secondary_rescue_current_exemptions,
        args.primary_recovery_min_score,
        args.primary_recovery_max_score,
        args.primary_recovery_source,
        args.primary_recovery_candidate_source,
        args.primary_recovery_current_group,
        args.primary_recovery_candidate_group,
        args.primary_recovery_min_primary_score,
        args.primary_recovery_min_ball_fill_fraction,
        args.reticle_recovery_min_score,
        args.reticle_recovery_max_score,
        args.reticle_recovery_min_primary_score,
        Path(args.fallback_rescue_checkpoint) if args.fallback_rescue_checkpoint else None,
        Path(args.fallback_rescue_reranker_checkpoint) if args.fallback_rescue_reranker_checkpoint else None,
        args.fallback_rescue_min_score,
        args.fallback_rescue_max_score,
        args.fallback_rescue_source_group,
        args.fallback_rescue_max_ball_candidates,
        Path(args.reticle_fill_rescue_checkpoint) if args.reticle_fill_rescue_checkpoint else None,
        Path(args.reticle_fill_rescue_reranker_checkpoint) if args.reticle_fill_rescue_reranker_checkpoint else None,
        args.reticle_fill_rescue_min_score,
        args.reticle_fill_rescue_max_score,
        args.reticle_fill_rescue_current_max_ball_fill_fraction,
        args.reticle_fill_rescue_target_min_ball_fill_fraction,
        args.reticle_fill_rescue_max_ball_candidates,
        Path(args.external_rescue_checkpoint) if args.external_rescue_checkpoint else None,
        args.external_rescue_source_group,
        args.external_rescue_min_score,
        args.external_rescue_max_score,
        args.external_rescue_current_max_ball_fill_fraction,
        args.external_rescue_min_pred_area,
        args.external_rescue_min_prob_mean,
        Path(args.external_mid_rescue_checkpoint) if args.external_mid_rescue_checkpoint else None,
        args.external_mid_rescue_source_group,
        args.external_mid_rescue_min_score,
        args.external_mid_rescue_max_score,
        args.external_mid_rescue_current_max_ball_fill_fraction,
        args.external_mid_rescue_min_pred_area,
        args.external_mid_rescue_min_prob_mean,
        Path(args.external_lowmid_rescue_checkpoint) if args.external_lowmid_rescue_checkpoint else None,
        args.external_lowmid_rescue_source_group,
        args.external_lowmid_rescue_min_score,
        args.external_lowmid_rescue_max_score,
        args.external_lowmid_rescue_current_max_ball_fill_fraction,
        args.external_lowmid_rescue_min_pred_area,
        args.external_lowmid_rescue_min_prob_mean,
        Path(args.external_reticle_rescue_checkpoint) if args.external_reticle_rescue_checkpoint else None,
        args.external_reticle_rescue_source_group,
        args.external_reticle_rescue_min_score,
        args.external_reticle_rescue_max_score,
        args.external_reticle_rescue_current_max_ball_fill_fraction,
        args.external_reticle_rescue_min_pred_area,
        args.external_reticle_rescue_min_prob_mean,
        Path(args.external_tiny_reticle_rescue_checkpoint) if args.external_tiny_reticle_rescue_checkpoint else None,
        args.external_tiny_reticle_rescue_source_group,
        args.external_tiny_reticle_rescue_min_score,
        args.external_tiny_reticle_rescue_max_score,
        args.external_tiny_reticle_rescue_current_max_ball_fill_fraction,
        args.external_tiny_reticle_rescue_min_pred_area,
        args.external_tiny_reticle_rescue_min_prob_mean,
        Path(args.external_reticle_large_rescue_checkpoint) if args.external_reticle_large_rescue_checkpoint else None,
        args.external_reticle_large_rescue_source_group,
        args.external_reticle_large_rescue_min_score,
        args.external_reticle_large_rescue_max_score,
        args.external_reticle_large_rescue_current_max_ball_fill_fraction,
        args.external_reticle_large_rescue_min_pred_area,
        args.external_reticle_large_rescue_min_prob_mean,
        Path(args.external_table_broad_rescue_checkpoint) if args.external_table_broad_rescue_checkpoint else None,
        args.external_table_broad_rescue_source_group,
        args.external_table_broad_rescue_min_score,
        args.external_table_broad_rescue_max_score,
        args.external_table_broad_rescue_current_max_ball_fill_fraction,
        args.external_table_broad_rescue_min_pred_area,
        args.external_table_broad_rescue_min_prob_mean,
        Path(args.external_blob_low_rescue_checkpoint) if args.external_blob_low_rescue_checkpoint else None,
        args.external_blob_low_rescue_source_group,
        args.external_blob_low_rescue_min_score,
        args.external_blob_low_rescue_max_score,
        args.external_blob_low_rescue_current_max_ball_fill_fraction,
        args.external_blob_low_rescue_min_pred_area,
        args.external_blob_low_rescue_min_prob_mean,
        Path(args.external_table_rescue_checkpoint) if args.external_table_rescue_checkpoint else None,
        args.external_table_rescue_source_group,
        args.external_table_rescue_min_score,
        args.external_table_rescue_max_score,
        args.external_table_rescue_current_max_ball_fill_fraction,
        args.external_table_rescue_min_pred_area,
        args.external_table_rescue_min_prob_mean,
        Path(args.external_blob_rescue_checkpoint) if args.external_blob_rescue_checkpoint else None,
        args.external_blob_rescue_source_group,
        args.external_blob_rescue_min_score,
        args.external_blob_rescue_max_score,
        args.external_blob_rescue_current_max_ball_fill_fraction,
        args.external_blob_rescue_min_pred_area,
        args.external_blob_rescue_min_prob_mean,
        Path(args.external_tiny_blob_rescue_checkpoint) if args.external_tiny_blob_rescue_checkpoint else None,
        args.external_tiny_blob_rescue_source_group,
        args.external_tiny_blob_rescue_min_score,
        args.external_tiny_blob_rescue_max_score,
        args.external_tiny_blob_rescue_current_max_ball_fill_fraction,
        args.external_tiny_blob_rescue_min_pred_area,
        args.external_tiny_blob_rescue_min_prob_mean,
        micro_line_rescue_config,
        colored_blob_rescue_config,
        selector_candidate_pool_rescue_config,
        post_external_selector_candidate_pool_rescue_config,
        args.reject_table_hough_final_fallback_min_pred_pixels,
        args.reject_table_hough_final_fallback_max_score,
        reject_winner_source_min_pred_pixels,
        reject_winner_source_min_pred_pixels_exemptions,
        reject_winner_source_area_score,
        args.reject_pre_final_fallback_max_score,
        reject_final_context_rules,
        candidate_selector_rescue_rules,
        rescue_arbiter_config,
        post_external_candidate_selector_rescue_rules,
        post_external_rescue_arbiter_config,
        post_final_candidate_selector_rescue_rules,
        Path(args.final_refine_checkpoint) if args.final_refine_checkpoint else None,
        args.final_refine_image_size,
        image_final_veto_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
