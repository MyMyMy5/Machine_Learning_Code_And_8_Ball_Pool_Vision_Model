from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from cv_guideline_common import (
    build_guideline_postprocess_context,
    load_checkpoint_model,
    parse_csv_floats,
    parse_csv_ints,
    postprocess_guideline_mask,
    predict_prob_multi_tta,
)
from base_sam_zoom_probe import (
    NEG_LABEL,
    POS_LABEL,
    build_cv_rescue_candidates,
    build_polygon_candidate,
    build_probe_image,
    build_quality_metrics,
    clamp_center,
    clamp_idx,
    compute_view_rect,
    dedup_candidates,
    describe_rejects,
    display_to_source_view,
    draw_polygon_points,
    draw_prompt_points,
    evaluate_save_gate,
    extract_view_for_display,
    filter_candidates,
    format_prompt_points_for_meta,
    key_is,
    map_points_full_to_probe,
    map_probe_mask_to_full,
    map_probe_prob_to_full,
    mask_full_to_display_soft,
    overlay_mask_soft,
    refine_candidate_white_core,
    render_save_preview,
    run_sam_text_with_optional_boxes,
    save_sample,
    snap_positive_click_to_line,
)
from live_label_overlay import DEFAULT_CONFIG_PATH, ensure_dataset_layout, load_config, load_sam_model, str2bool, tiny_line_flag


WINDOW_NAME = "Guideline Repair Review"


def resolve_path(p: str) -> Path:
    raw = str(p).strip()
    if os.name == "nt" and raw.startswith("/mnt/") and len(raw) > 6 and raw[5].isalpha() and raw[6] == "/":
        drive = raw[5].upper()
        rest = raw[7:].replace("/", "\\")
        raw = f"{drive}:\\{rest}"
    elif os.name != "nt" and len(raw) >= 3 and raw[1:3] == ":\\" and raw[0].isalpha():
        drive = raw[0].lower()
        rest = raw[3:].replace("\\", "/")
        raw = f"/mnt/{drive}/{rest}"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def row_path_value(row: Dict, key: str, prefer_secondary: bool = False) -> str:
    if bool(prefer_secondary):
        secondary_key = f"secondary_{key}"
        secondary_value = str(row.get(secondary_key, "")).strip()
        if secondary_value:
            return secondary_value
    return str(row.get(key, "")).strip()


def load_summary(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise RuntimeError(f"Summary is not a JSON object: {path}")
    if "rows" in summary:
        return summary
    if "entries" in summary and "frames_dir" in summary and "output_root" in summary:
        return normalize_harvest_summary(summary, path)
    return summary


def normalize_harvest_summary(summary: Dict, summary_path: Path) -> Dict:
    output_root = resolve_path(str(summary.get("output_root", "")))
    rows: List[Dict] = []
    for entry in list(summary.get("entries", [])):
        if not isinstance(entry, dict):
            continue
        source_path = str(entry.get("source_path", "")).strip()
        if not source_path:
            continue
        decision = str(entry.get("decision", "")).strip().lower()
        output_path = str(entry.get("output_path", "")).strip()
        mask_output_path = str(entry.get("mask_output_path", "")).strip()
        source_index = int(entry.get("source_index", len(rows) + 1))
        pred_area = int(entry.get("mask_pixels", 0))
        overlay_path = ""
        if decision == "predicted" and output_path:
            overlay_path = output_path
        rows.append(
            {
                "index": int(source_index),
                "image_path": source_path,
                "original_path": source_path,
                "overlay_path": overlay_path,
                "mask_path": mask_output_path,
                "pred_area": int(pred_area),
                "score": entry.get("score"),
                "decision": decision,
                "output_path": output_path,
                "summary_path": str(summary_path.resolve()),
                "output_root": str(output_root),
            }
        )
    rows.sort(key=lambda row: int(row.get("index", 0)))
    normalized = dict(summary)
    normalized["rows"] = rows
    normalized["summary_type"] = "harvest"
    return normalized


def find_latest_summary(root: Path) -> Path:
    if root.is_file():
        return root
    harvest_summary = root / "harvest_summary.json"
    if harvest_summary.exists():
        return harvest_summary
    candidates = sorted(root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        summary = cand / "summary.json"
        if summary.exists():
            return summary
    raise FileNotFoundError(f"No summary.json found under {root}")


def load_resume_state(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_existing_source_image_paths(annotation_paths: List[Path]) -> set[str]:
    out: set[str] = set()
    for path in annotation_paths:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            src = str(row.get("source_image_path", "")).strip()
            if src:
                out.add(str(resolve_path(src)).lower())
    return out


def count_annotation_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    try:
        text = path.read_text(encoding="utf-8-sig")
    except Exception:
        return 0
    for line in text.splitlines():
        if line.strip():
            count += 1
    return count


def save_resume_state(
    path: Path,
    summary_path: Path,
    index: int,
    relabeled: List[int],
    interaction_mode: str = "prompt",
) -> None:
    payload = {
        "summary_path": str(summary_path.resolve()),
        "last_index": int(index),
        "relabeled_indices": [int(x) for x in relabeled],
        "interaction_mode": str(interaction_mode),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def source_to_display_view(
    src_xy: Tuple[int, int],
    display_wh: Tuple[int, int],
    view_rect: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int]]:
    sx, sy = int(src_xy[0]), int(src_xy[1])
    disp_w, disp_h = int(display_wh[0]), int(display_wh[1])
    x1, y1, x2, y2 = view_rect
    vw = max(1, x2 - x1 + 1)
    vh = max(1, y2 - y1 + 1)
    if not (x1 <= sx <= x2 and y1 <= sy <= y2):
        return None
    fx = float(sx - x1) / float(max(1, vw - 1))
    fy = float(sy - y1) / float(max(1, vh - 1))
    dx = int(round(fx * float(max(1, disp_w - 1))))
    dy = int(round(fy * float(max(1, disp_h - 1))))
    return dx, dy


def make_probe_like_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        fallback_polygon_min_points=int(args.fallback_polygon_min_points),
        fallback_polygon_enabled=True,
        fallback_refine_required=bool(args.fallback_refine_required),
        fallback_refine_min_area=int(args.fallback_refine_min_area),
        fallback_refine_min_elongation=float(args.fallback_refine_min_elongation),
        fallback_refine_dark_threshold=int(args.fallback_refine_dark_threshold),
        fallback_refine_ring_px=int(args.fallback_refine_ring_px),
        positive_hit_tol_px=int(args.positive_hit_tol_px),
        negative_hit_tol_px=int(args.negative_hit_tol_px),
        pos_soft_gate=bool(args.pos_soft_gate),
        pos_soft_radius_px=int(args.pos_soft_radius_px),
        pos_hard_max_dist_px=int(args.pos_hard_max_dist_px),
        pos_distance_penalty=float(args.pos_distance_penalty),
        min_white_ratio=float(args.min_white_ratio),
        save_gate_mode=str(args.save_gate_mode),
        save_min_white_ratio=float(args.save_min_white_ratio),
        save_min_elongation=float(args.save_min_elongation),
        save_max_fill_ratio=float(args.save_max_fill_ratio),
        save_max_pos_dist=float(args.save_max_pos_dist),
        dedup_iou=float(args.dedup_iou),
        cv_rescue_s_relax=int(args.cv_rescue_s_relax),
        cv_rescue_v_relax=int(args.cv_rescue_v_relax),
        cv_rescue_len_scale=float(args.cv_rescue_len_scale),
        cv_rescue_area_scale=float(args.cv_rescue_area_scale),
        cv_rescue_max_candidates=int(args.cv_rescue_max_candidates),
    )


def load_predicted_candidate(row: Dict, prefer_secondary_paths: bool = False) -> Optional[Dict]:
    mask_path = resolve_path(row_path_value(row, "mask_path", prefer_secondary=prefer_secondary_paths))
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask_u8 = (mask > 0).astype(np.uint8)
    area = int(mask_u8.sum())
    if area <= 0:
        return None
    return {
        "mask_full": mask_u8,
        "mask_prob_full": mask_u8.astype(np.float32),
        "source": "preloaded_infer_mask",
        "score": None,
        "rank_score": 0.0,
        "area": area,
        "white_ratio": 0.0,
        "pos_dist_min": 0.0,
        "pos_dist_max": 0.0,
        "neg_hits": 0,
    }


def resolve_checkpoint_path_or_pointer(raw_value: str) -> Path:
    path = resolve_path(raw_value)
    if path.suffix.lower() == ".txt":
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint pointer not found: {path}")
        pointer_value = path.read_text(encoding="utf-8").strip()
        if not pointer_value:
            raise RuntimeError(f"Checkpoint pointer is empty: {path}")
        path = resolve_path(pointer_value)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def ensure_specialist_runtime_loaded(runtime: Dict) -> None:
    if runtime.get("model") is not None:
        return
    ckpt_path = resolve_checkpoint_path_or_pointer(str(runtime["checkpoint_ref"]))
    model, ckpt = load_checkpoint_model(ckpt_path, device=runtime["device"])
    threshold = runtime.get("threshold_override")
    if threshold is None:
        threshold = ckpt.get("best_threshold", None)
    if threshold is None:
        threshold = 0.5
    runtime["checkpoint_path"] = ckpt_path
    runtime["model"] = model
    runtime["ckpt"] = ckpt
    runtime["threshold"] = float(threshold)


def run_specialist_preview_candidate(frame_bgr: np.ndarray, runtime: Dict) -> Tuple[Optional[Dict], str]:
    ensure_specialist_runtime_loaded(runtime)
    prob_np = predict_prob_multi_tta(
        model=runtime["model"],
        frame_bgr=frame_bgr,
        device=runtime["device"],
        tile_sizes=runtime["tile_sizes"],
        tile_overlaps=runtime["tile_overlaps"],
        tta_scales=runtime["tta_scales"],
        tta_hflip=bool(runtime["tta_hflip"]),
        use_amp=bool(runtime["use_amp"]),
    )
    pp_ctx = build_guideline_postprocess_context(frame_bgr)
    mask_bin = postprocess_guideline_mask(
        frame_bgr,
        prob_np,
        threshold=float(runtime["threshold"]),
        axis_complete=bool(runtime["axis_complete"]),
        axis_complete_max_gap=int(runtime["axis_complete_max_gap"]),
        axis_complete_max_extension=int(runtime["axis_complete_max_extension"]),
        axis_complete_min_white_ratio=float(runtime["axis_complete_min_white_ratio"]),
        axis_complete_min_dark_border_ratio=float(runtime["axis_complete_min_dark_border_ratio"]),
        context=pp_ctx,
    )
    mask_u8 = (mask_bin > 0).astype(np.uint8)
    area = int(mask_u8.sum())
    if area <= 0:
        return None, f"Tiny-line preview: no mask | thr={float(runtime['threshold']):.2f}"
    candidate = {
        "mask_full": mask_u8,
        "mask_prob_full": np.asarray(prob_np, dtype=np.float32),
        "source": "tinyline_specialist_cv",
        "score": None,
        "rank_score": 1.0,
        "area": area,
        "white_ratio": 0.0,
        "pos_dist_min": 0.0,
        "pos_dist_max": 0.0,
        "neg_hits": 0,
        "candidate_priority": 2,
        "probe_scale": 1.0,
        "prompt_count": 0,
    }
    return candidate, f"Tiny-line preview ready | area={area} | thr={float(runtime['threshold']):.2f}"


def build_display_image(
    row: Dict,
    frame_full_bgr: np.ndarray,
    show_overlay: bool,
    show_save_preview: bool,
    candidates: List[Dict],
    selected_idx: int,
    prompt_points: List[Dict],
    polygon_points: List[Tuple[int, int]],
    draw_candidates: bool,
    view_rect: Tuple[int, int, int, int],
    out_wh: Tuple[int, int],
    args,
) -> np.ndarray:
    if show_overlay and not show_save_preview:
        frame = cv2.imread(
            str(resolve_path(row_path_value(row, "overlay_path", prefer_secondary=bool(args.prefer_secondary_paths)))),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            frame = cv2.imread(
                str(resolve_path(row_path_value(row, "original_path", prefer_secondary=bool(args.prefer_secondary_paths)))),
                cv2.IMREAD_COLOR,
            )
    else:
        frame = cv2.imread(
            str(resolve_path(row_path_value(row, "original_path", prefer_secondary=bool(args.prefer_secondary_paths)))),
            cv2.IMREAD_COLOR,
        )
    if frame is None:
        raise FileNotFoundError(f"Failed to load display image for row: {row.get('image_path', '')}")

    selected: Optional[Dict] = None
    if candidates:
        selected = candidates[clamp_idx(int(selected_idx), len(candidates))]

    if show_save_preview and selected is not None:
        base = render_save_preview(frame_full_bgr, (selected["mask_full"] > 0).astype(np.uint8))
    else:
        base = frame

    viz = extract_view_for_display(base, view_rect, out_wh)
    if (not show_overlay) and bool(draw_candidates):
        draw_n = len(candidates) if int(args.draw_topk) <= 0 else max(1, min(int(args.draw_topk), len(candidates)))
        for i, cand in enumerate(candidates[:draw_n]):
            prob_src = cand["mask_prob_full"] if bool(args.display_prob_overlay) and ("mask_prob_full" in cand) else cand["mask_full"]
            mask_soft = mask_full_to_display_soft(
                prob_src,
                view_rect,
                out_wh,
                supersample=int(args.display_supersample),
            )
            if i == int(selected_idx):
                overlay_mask_soft(
                    viz,
                    mask_soft,
                    (0, 255, 255),
                    (0, 255, 255),
                    alpha=float(args.display_alpha_selected),
                    edge_thickness=int(args.display_edge_thickness),
                    gamma=float(args.display_prob_gamma),
                    edge_aa=bool(args.display_edge_aa),
                )
            else:
                overlay_mask_soft(
                    viz,
                    mask_soft,
                    (255, 255, 0),
                    (255, 180, 0),
                    alpha=float(args.display_alpha_other),
                    edge_thickness=int(args.display_edge_thickness),
                    gamma=float(args.display_prob_gamma),
                    edge_aa=bool(args.display_edge_aa),
                )
        if not bool(show_save_preview):
            draw_prompt_points(viz, prompt_points, view_rect, out_wh)
    if (not show_overlay) and (not bool(show_save_preview)):
        draw_polygon_points(viz, polygon_points, view_rect, out_wh)
    return viz


def build_candidates(
    frame_bgr: np.ndarray,
    model,
    processor,
    prompt: str,
    profile_cfg: Dict,
    white_cfg: Dict,
    device: str,
    prompt_points: List[Dict],
    polygon_points: List[Tuple[int, int]],
    view_rect: Tuple[int, int, int, int],
    probe_upscale: float,
    args,
) -> Tuple[List[Dict], str, Dict[str, int], float]:
    h, w = frame_bgr.shape[:2]
    probe_bgr, probe_scale = build_probe_image(
        frame_bgr,
        view_rect,
        float(probe_upscale),
        int(args.probe_max_side),
    )
    local_prompts = map_points_full_to_probe(prompt_points, view_rect, probe_scale, probe_bgr.shape[:2])

    candidates_probe = run_sam_text_with_optional_boxes(
        probe_bgr,
        model,
        processor,
        prompt,
        profile_cfg,
        device,
        local_prompts,
        int(args.prompt_box_pos_px),
        int(args.prompt_box_neg_px),
    )
    if local_prompts and not candidates_probe and bool(args.fallback_text_only):
        fb = run_sam_text_with_optional_boxes(
            probe_bgr,
            model,
            processor,
            prompt,
            profile_cfg,
            device,
            [],
            int(args.prompt_box_pos_px),
            int(args.prompt_box_neg_px),
        )
        for cand in fb:
            cand["source"] = "sam_text_fallback"
        candidates_probe = fb

    filtered_probe, reject_counts = filter_candidates(
        candidates_probe,
        probe_bgr,
        local_prompts,
        white_cfg,
        int(args.positive_hit_tol_px),
        int(args.negative_hit_tol_px),
        float(args.min_white_ratio),
        bool(args.pos_soft_gate),
        int(args.pos_soft_radius_px),
        int(args.pos_hard_max_dist_px),
        float(args.pos_distance_penalty),
    )
    if local_prompts and not filtered_probe and bool(args.pos_fallback_on_miss) and bool(args.fallback_text_only):
        miss_count = int(reject_counts.get("miss_pos", 0)) + int(reject_counts.get("miss_pos_hard", 0))
        total_rejects = int(sum(int(v) for v in reject_counts.values()))
        if total_rejects > 0 and miss_count >= max(1, total_rejects // 2):
            fb = run_sam_text_with_optional_boxes(
                probe_bgr,
                model,
                processor,
                prompt,
                profile_cfg,
                device,
                [],
                int(args.prompt_box_pos_px),
                int(args.prompt_box_neg_px),
            )
            for cand in fb:
                cand["source"] = "sam_text_pos_fallback"
            filtered_probe_fb, reject_counts_fb = filter_candidates(
                fb,
                probe_bgr,
                local_prompts,
                white_cfg,
                int(args.positive_hit_tol_px),
                int(args.negative_hit_tol_px),
                float(args.min_white_ratio),
                bool(args.pos_soft_gate),
                int(args.pos_soft_radius_px),
                int(round(float(args.pos_hard_max_dist_px) * float(args.pos_fallback_hard_scale))),
                float(args.pos_distance_penalty),
            )
            if filtered_probe_fb:
                filtered_probe = filtered_probe_fb
            for k, v in reject_counts_fb.items():
                reject_counts[f"fb_{k}"] = int(v)

    if not filtered_probe and bool(args.cv_rescue_on_empty):
        cv_candidates = build_cv_rescue_candidates(
            probe_bgr=probe_bgr,
            white_cfg=white_cfg,
            profile_cfg=profile_cfg,
            args=args,
        )
        if cv_candidates:
            cv_filtered, cv_rejects = filter_candidates(
                cv_candidates,
                probe_bgr,
                local_prompts,
                white_cfg,
                int(args.positive_hit_tol_px),
                int(args.negative_hit_tol_px),
                float(args.min_white_ratio),
                bool(args.pos_soft_gate),
                int(args.pos_soft_radius_px),
                int(args.pos_hard_max_dist_px),
                float(args.pos_distance_penalty),
            )
            if cv_filtered:
                filtered_probe = cv_filtered
            for k, v in cv_rejects.items():
                reject_counts[f"cv_{k}"] = int(v)

    if bool(args.refine_white_core):
        filtered_probe = [
            refine_candidate_white_core(
                probe_bgr,
                cand,
                white_cfg,
                threshold=float(args.refine_white_core_threshold),
                support_dilate_px=int(args.refine_support_dilate_px),
                min_area=int(args.refine_min_area),
            )
            for cand in filtered_probe
        ]
    filtered_probe = dedup_candidates(filtered_probe, float(args.dedup_iou))
    filtered_probe = filtered_probe[: max(1, int(args.candidate_max))]

    mapped: List[Dict] = []
    for cand in filtered_probe:
        full_mask = map_probe_mask_to_full(
            cand["mask"],
            view_rect,
            (h, w),
            probe_scale,
            map_threshold=float(args.map_threshold),
            smooth_kernel=int(args.map_smooth_kernel),
            area_preblur_sigma=float(args.map_area_preblur_sigma),
        )
        cc = dict(cand)
        cc["mask_full"] = full_mask
        if "mask_prob" in cand:
            cc["mask_prob_full"] = map_probe_prob_to_full(cand["mask_prob"], view_rect, (h, w), probe_scale)
        cc["view_rect"] = [int(v) for v in view_rect]
        cc["probe_scale"] = float(probe_scale)
        cc["prompt_count"] = int(len(local_prompts))
        mapped.append(cc)

    if bool(args.fallback_polygon_enabled) and len(polygon_points) >= int(args.fallback_polygon_min_points):
        poly_cand, poly_reason, poly_info = build_polygon_candidate(
            frame_full_bgr=frame_bgr,
            polygon_points=list(polygon_points),
            prompt_points=list(prompt_points),
            white_cfg=white_cfg,
            args=args,
        )
        if poly_cand is not None:
            poly_cand["candidate_priority"] = 1
            mapped.insert(0, poly_cand)
        else:
            reject_counts[f"poly_{poly_reason}"] = int(reject_counts.get(f"poly_{poly_reason}", 0)) + 1
            if bool(poly_info):
                pkey = f"poly_refine_{str(poly_info.get('reason', 'failed'))}"
                reject_counts[pkey] = int(reject_counts.get(pkey, 0)) + 1

    mapped.sort(
        key=lambda c: (
            -int(c.get("candidate_priority", 0)),
            -float(c.get("rank_score", -1e9)),
            -1.0 if c.get("score") is None else -float(c.get("score")),
            -float(c.get("white_ratio", 0.0)),
            int(c.get("area", 0)),
        )
    )
    mapped = mapped[: max(1, int(args.candidate_max))]
    return mapped, describe_rejects(reject_counts), reject_counts, float(probe_scale)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review inference overlays and repair misses with polygon labels.")
    parser.add_argument("--summary", type=str, default="guideline_line/CV_Test_overnight_winner")
    parser.add_argument("--decision-filter", type=str, choices=["all", "predicted", "no_prediction"], default="all")
    parser.add_argument("--dataset-root", type=str, default="guideline_line/data_zoomprobe")
    parser.add_argument("--quarantine-root", type=str, default="")
    parser.add_argument("--rejected-root", type=str, default="")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--profile", type=str, choices=["balanced", "precision"], default="precision")
    parser.add_argument("--resume-last", type=str2bool, default=True)
    parser.add_argument("--state-path", type=str, default="")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--display-scale", type=float, default=None)
    parser.add_argument("--zoom-step", type=float, default=1.18)
    parser.add_argument("--zoom-max", type=float, default=20.0)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--lora-path", type=str, default="none")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--probe-upscale", type=float, default=3.0)
    parser.add_argument("--probe-upscale-step", type=float, default=1.2)
    parser.add_argument("--probe-upscale-max", type=float, default=8.0)
    parser.add_argument("--probe-max-side", type=int, default=1600)
    parser.add_argument("--draw-topk", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=64)
    parser.add_argument("--prompt-box-pos-px", type=int, default=6)
    parser.add_argument("--prompt-box-neg-px", type=int, default=8)
    parser.add_argument("--fallback-polygon-enabled", type=str2bool, default=True)
    parser.add_argument("--fallback-polygon-min-points", type=int, default=3)
    parser.add_argument("--fallback-refine-required", type=str2bool, default=True)
    parser.add_argument("--fallback-refine-min-area", type=int, default=4)
    parser.add_argument("--fallback-refine-min-elongation", type=float, default=1.25)
    parser.add_argument("--fallback-refine-dark-threshold", type=int, default=122)
    parser.add_argument("--fallback-refine-ring-px", type=int, default=1)
    parser.add_argument("--positive-hit-tol-px", type=int, default=3)
    parser.add_argument("--negative-hit-tol-px", type=int, default=2)
    parser.add_argument("--pos-soft-gate", type=str2bool, default=True)
    parser.add_argument("--pos-soft-radius-px", type=int, default=7)
    parser.add_argument("--pos-hard-max-dist-px", type=int, default=18)
    parser.add_argument("--pos-distance-penalty", type=float, default=0.12)
    parser.add_argument("--pos-fallback-on-miss", type=str2bool, default=True)
    parser.add_argument("--pos-fallback-hard-scale", type=float, default=1.4)
    parser.add_argument("--snap-positive-to-line", type=str2bool, default=True)
    parser.add_argument("--snap-radius-px", type=int, default=10)
    parser.add_argument("--snap-min-white-score", type=float, default=0.15)
    parser.add_argument("--fallback-text-only", type=str2bool, default=True)
    parser.add_argument("--refine-white-core", type=str2bool, default=True)
    parser.add_argument("--refine-white-core-threshold", type=float, default=0.50)
    parser.add_argument("--refine-support-dilate-px", type=int, default=1)
    parser.add_argument("--refine-min-area", type=int, default=6)
    parser.add_argument("--map-threshold", type=float, default=0.47)
    parser.add_argument("--map-smooth-kernel", type=int, default=1)
    parser.add_argument("--map-area-preblur-sigma", type=float, default=0.2)
    parser.add_argument("--dedup-iou", type=float, default=0.72)
    parser.add_argument("--cv-rescue-on-empty", type=str2bool, default=True)
    parser.add_argument("--cv-rescue-s-relax", type=int, default=30)
    parser.add_argument("--cv-rescue-v-relax", type=int, default=28)
    parser.add_argument("--cv-rescue-len-scale", type=float, default=0.7)
    parser.add_argument("--cv-rescue-area-scale", type=float, default=0.7)
    parser.add_argument("--cv-rescue-max-candidates", type=int, default=16)
    parser.add_argument("--min-white-ratio", type=float, default=0.36)
    parser.add_argument("--save-gate-mode", type=str, choices=["strict", "soft"], default="strict")
    parser.add_argument("--save-min-white-ratio", type=float, default=0.36)
    parser.add_argument("--save-min-elongation", type=float, default=1.20)
    parser.add_argument("--save-max-fill-ratio", type=float, default=0.80)
    parser.add_argument("--save-max-pos-dist", type=float, default=18.0)
    parser.add_argument("--display-alpha-selected", type=float, default=0.34)
    parser.add_argument("--display-alpha-other", type=float, default=0.18)
    parser.add_argument("--display-supersample", type=int, default=2)
    parser.add_argument("--display-prob-overlay", type=str2bool, default=False)
    parser.add_argument("--display-prob-gamma", type=float, default=1.0)
    parser.add_argument("--display-edge-aa", type=str2bool, default=True)
    parser.add_argument("--display-edge-thickness", type=int, default=0)
    parser.add_argument("--specialist-checkpoint", type=str, default="guideline_line/tinyline_checkpoint_path.txt")
    parser.add_argument("--specialist-tile-sizes", type=str, default="640,896")
    parser.add_argument("--specialist-tile-overlaps", type=str, default="160,224")
    parser.add_argument("--specialist-tta-scales", type=str, default="1.0,1.15")
    parser.add_argument("--specialist-tta-hflip", type=str2bool, default=True)
    parser.add_argument("--specialist-axis-complete", type=str2bool, default=True)
    parser.add_argument("--specialist-axis-complete-max-gap", type=int, default=24)
    parser.add_argument("--specialist-axis-complete-max-extension", type=int, default=160)
    parser.add_argument("--specialist-axis-complete-min-white-ratio", type=float, default=0.42)
    parser.add_argument("--specialist-axis-complete-min-dark-border-ratio", type=float, default=0.05)
    parser.add_argument("--specialist-threshold", type=float, default=None)
    parser.add_argument("--specialist-amp", type=str2bool, default=True)
    parser.add_argument("--prefer-secondary-paths", type=str2bool, default=False)
    args = parser.parse_args()

    summary_arg = resolve_path(args.summary)
    summary_path = find_latest_summary(summary_arg)
    summary = load_summary(summary_path)
    rows = list(summary.get("rows", []))
    decision_filter = str(args.decision_filter).strip().lower()
    if decision_filter != "all":
        rows = [row for row in rows if str(row.get("decision", "")).strip().lower() == decision_filter]
    if not rows:
        raise RuntimeError(f"No rows found in summary: {summary_path}")

    config = load_config(resolve_path(args.config))
    prompt = str(args.prompt or config["default_prompt"])
    model_id = str(args.model_id or config["model_id"])
    profile_cfg = dict(config["profiles"][args.profile])
    white_cfg = dict(config["auto"]["white_cv"])
    tiny_cfg = dict(config["tiny_line"])
    display_scale = float(args.display_scale) if args.display_scale is not None else float(config.get("display_scale", 1.0))
    dataset_root = resolve_path(args.dataset_root)
    quarantine_root = resolve_path(args.quarantine_root) if str(args.quarantine_root).strip() else (dataset_root / "quarantine")
    rejected_root = resolve_path(args.rejected_root) if str(args.rejected_root).strip() else (dataset_root / "Rejected")
    dataset_dirs_main = ensure_dataset_layout(dataset_root)
    dataset_dirs_quarantine = ensure_dataset_layout(quarantine_root)
    dataset_dirs_rejected = ensure_dataset_layout(rejected_root)
    existing_source_paths = load_existing_source_image_paths(
        [
            Path(dataset_dirs_main["annotations"]),
            Path(dataset_dirs_quarantine["annotations"]),
            Path(dataset_dirs_rejected["annotations"]),
        ]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_sam_model(model_id, args.lora_path, device)
    specialist_tile_sizes = parse_csv_ints(args.specialist_tile_sizes, [640, 896])
    specialist_tile_overlaps = parse_csv_ints(args.specialist_tile_overlaps, [160, 224])
    if len(specialist_tile_overlaps) < len(specialist_tile_sizes):
        specialist_tile_overlaps.extend(
            [specialist_tile_overlaps[-1] if specialist_tile_overlaps else 160]
            * (len(specialist_tile_sizes) - len(specialist_tile_overlaps))
        )
    elif len(specialist_tile_overlaps) > len(specialist_tile_sizes):
        specialist_tile_overlaps = specialist_tile_overlaps[: len(specialist_tile_sizes)]
    specialist_runtime = {
        "checkpoint_ref": str(args.specialist_checkpoint),
        "device": torch.device(device),
        "tile_sizes": specialist_tile_sizes,
        "tile_overlaps": specialist_tile_overlaps,
        "tta_scales": parse_csv_floats(args.specialist_tta_scales, [1.0, 1.15]),
        "tta_hflip": bool(args.specialist_tta_hflip),
        "axis_complete": bool(args.specialist_axis_complete),
        "axis_complete_max_gap": int(args.specialist_axis_complete_max_gap),
        "axis_complete_max_extension": int(args.specialist_axis_complete_max_extension),
        "axis_complete_min_white_ratio": float(args.specialist_axis_complete_min_white_ratio),
        "axis_complete_min_dark_border_ratio": float(args.specialist_axis_complete_min_dark_border_ratio),
        "threshold_override": args.specialist_threshold,
        "use_amp": bool(args.specialist_amp),
        "model": None,
        "ckpt": None,
        "threshold": None,
        "checkpoint_path": None,
        "last_image_key": "",
        "last_candidate": None,
        "last_status": "",
    }

    state_path = resolve_path(args.state_path) if str(args.state_path).strip() else summary_path.with_name("repair_review_resume.json")
    resume = load_resume_state(state_path) if bool(args.resume_last) else {}
    if args.start_index is not None:
        idx = clamp_idx(int(args.start_index), len(rows))
    elif str(resume.get("summary_path", "")).lower() == str(summary_path.resolve()).lower():
        idx = clamp_idx(int(resume.get("last_index", 0)), len(rows))
    else:
        idx = 0
    relabeled_indices = set(int(x) for x in resume.get("relabeled_indices", []) if isinstance(x, int))

    probe_args = make_probe_like_args(args)
    saved_counts = {
        "main": count_annotation_rows(Path(dataset_dirs_main["annotations"])),
        "quarantine": count_annotation_rows(Path(dataset_dirs_quarantine["annotations"])),
        "rejected": count_annotation_rows(Path(dataset_dirs_rejected["annotations"])),
    }

    state = {
        "zoom": 1.0,
        "center": None,
        "show_overlay": True,
        "show_save_preview": False,
        "interaction_mode": str(resume.get("interaction_mode", "prompt")).lower().strip() or "prompt",
        "prompt_points": [],
        "polygon_points": [],
        "label_mode": POS_LABEL,
        "auto_enabled": False,
        "needs_infer": False,
        "force_infer": False,
        "candidates": [],
        "selected_idx": 0,
        "draw_candidates": False,
        "probe_upscale": float(max(1.0, args.probe_upscale)),
        "probe_scale_used": 1.0,
        "last_reject": "",
        "last_reject_counts": {},
        "external_preview_source": "",
        "status": "",
    }
    drag = {"active": False, "last_xy": (0, 0)}
    if str(state["interaction_mode"]) not in {"prompt", "polygon"}:
        state["interaction_mode"] = "prompt"

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def current_frame_and_dims(row: Dict) -> Tuple[np.ndarray, int, int]:
        frame = cv2.imread(
            str(resolve_path(row_path_value(row, "original_path", prefer_secondary=bool(args.prefer_secondary_paths)))),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            frame = cv2.imread(str(resolve_path(row_path_value(row, "image_path", prefer_secondary=False))), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Failed to load source/original image: {row.get('image_path', '')}")
        h, w = frame.shape[:2]
        return frame, w, h

    def reset_edit_state() -> None:
        state["prompt_points"] = []
        state["polygon_points"] = []
        state["candidates"] = []
        state["selected_idx"] = 0
        state["draw_candidates"] = False
        state["needs_infer"] = False
        state["force_infer"] = False
        state["probe_scale_used"] = 1.0
        state["show_save_preview"] = False
        state["last_reject"] = ""
        state["last_reject_counts"] = {}
        state["external_preview_source"] = ""

    def ensure_center(w: int, h: int) -> None:
        if state["center"] is None:
            state["center"] = (float(w) * 0.5, float(h) * 0.5)

    def on_mouse(event, x, y, flags, param):
        nonlocal idx
        row = rows[idx]
        frame, w, h = current_frame_and_dims(row)
        ensure_center(w, h)
        view_rect = compute_view_rect(w, h, state["zoom"], state["center"])
        disp_w = max(1, int(round(w * float(display_scale))))
        disp_h = max(1, int(round(h * float(display_scale))))
        disp_w = min(disp_w, 1600)
        disp_h = min(disp_h, 1000)

        if event == cv2.EVENT_LBUTTONDOWN:
            if not bool(state["show_overlay"]):
                src = display_to_source_view((x, y), (disp_w, disp_h), view_rect)
                if src is not None:
                    raw_x, raw_y = int(src[0]), int(src[1])
                    if str(state.get("interaction_mode", "prompt")) == "polygon":
                        state["polygon_points"].append((raw_x, raw_y))
                        state["status"] = f"Added polygon point #{len(state['polygon_points'])}"
                    else:
                        px, py = raw_x, raw_y
                        snap_msg = ""
                        if int(state["label_mode"]) == POS_LABEL and bool(args.snap_positive_to_line):
                            (px, py), snap_info = snap_positive_click_to_line(
                                frame,
                                (raw_x, raw_y),
                                radius_px=int(args.snap_radius_px),
                                min_white_score=float(args.snap_min_white_score),
                                dark_threshold=122,
                            )
                            if bool(snap_info.get("did_snap", 0.0)) and (px != raw_x or py != raw_y):
                                snap_msg = f" | snapped {raw_x},{raw_y}->{px},{py}"
                        point_rec = {"x": int(px), "y": int(py), "label": int(state["label_mode"])}
                        if int(state["label_mode"]) == POS_LABEL:
                            point_rec["raw_x"] = int(raw_x)
                            point_rec["raw_y"] = int(raw_y)
                        state["prompt_points"].append(point_rec)
                        state["needs_infer"] = bool(state["auto_enabled"])
                        state["force_infer"] = bool(state["auto_enabled"])
                        state["status"] = f"Added {'POS' if int(state['label_mode']) == POS_LABEL else 'NEG'} prompt{snap_msg}"
        elif event == cv2.EVENT_MBUTTONDOWN:
            drag["active"] = True
            drag["last_xy"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag["active"]:
            dx = x - drag["last_xy"][0]
            dy = y - drag["last_xy"][1]
            x1, y1, x2, y2 = view_rect
            vw = max(1, x2 - x1 + 1)
            vh = max(1, y2 - y1 + 1)
            fx = float(dx) / float(max(1, disp_w))
            fy = float(dy) / float(max(1, disp_h))
            state["center"] = clamp_center(w, h, state["zoom"], (state["center"][0] - fx * vw, state["center"][1] - fy * vh))
            drag["last_xy"] = (x, y)
            if bool(state["auto_enabled"]) and not bool(state["show_overlay"]):
                state["needs_infer"] = True
        elif event == cv2.EVENT_MBUTTONUP:
            drag["active"] = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not bool(state["show_overlay"]):
                reset_edit_state()
                state["status"] = "Cleared prompts/polygon, specialist preview, and pending detections"
        elif event == cv2.EVENT_MOUSEWHEEL:
            src_anchor = display_to_source_view((x, y), (disp_w, disp_h), view_rect)
            wheel = 1 if flags > 0 else -1
            old_zoom = float(state["zoom"])
            if wheel > 0:
                state["zoom"] = min(float(args.zoom_max), old_zoom * float(args.zoom_step))
            else:
                state["zoom"] = max(1.0, old_zoom / float(args.zoom_step))
            if src_anchor is not None:
                state["center"] = (float(src_anchor[0]), float(src_anchor[1]))
            state["center"] = clamp_center(w, h, state["zoom"], state["center"])
            if bool(state["auto_enabled"]) and not bool(state["show_overlay"]):
                state["needs_infer"] = True
            state["status"] = f"Zoom: x{float(state['zoom']):.2f}"

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        row = rows[idx]
        frame, w, h = current_frame_and_dims(row)
        ensure_center(w, h)
        state["center"] = clamp_center(w, h, state["zoom"], state["center"])
        disp_w = max(1, int(round(w * float(display_scale))))
        disp_h = max(1, int(round(h * float(display_scale))))
        disp_w = min(disp_w, 1600)
        disp_h = min(disp_h, 1000)
        view_rect = compute_view_rect(w, h, state["zoom"], state["center"])

        should_infer = (not bool(state["show_overlay"])) and (
            bool(state["force_infer"])
            or (
                (not str(state.get("external_preview_source", "")).strip())
                and bool(state["auto_enabled"])
                and bool(state["needs_infer"])
            )
        )
        if should_infer:
            mapped, reject_desc, reject_counts, probe_scale = build_candidates(
                frame_bgr=frame,
                model=model,
                processor=processor,
                prompt=prompt,
                profile_cfg=profile_cfg,
                white_cfg=white_cfg,
                device=device,
                prompt_points=list(state["prompt_points"]),
                polygon_points=list(state["polygon_points"]),
                view_rect=view_rect,
                probe_upscale=float(state["probe_upscale"]),
                args=args,
            )
            state["candidates"] = mapped
            state["selected_idx"] = clamp_idx(int(state["selected_idx"]), len(mapped)) if mapped else 0
            if mapped and any(int(c.get("candidate_priority", 0)) > 0 for c in mapped):
                state["selected_idx"] = 0
            state["draw_candidates"] = bool(mapped)
            state["probe_scale_used"] = float(probe_scale)
            state["last_reject"] = reject_desc
            state["last_reject_counts"] = dict(reject_counts)
            state["needs_infer"] = False
            state["force_infer"] = False
            if mapped:
                chosen = mapped[int(state["selected_idx"])]
                score_txt = "-" if chosen.get("score") is None else f"{float(chosen['score']):.2f}"
                state["status"] = (
                    f"Detected {len(mapped)} candidate(s) | selected {int(state['selected_idx']) + 1} | "
                    f"score={score_txt} | source={chosen.get('source', 'sam')}"
                )
            else:
                state["status"] = f"No valid mask in current view ({reject_desc})" if reject_desc else "No valid mask in current view"

        viz = build_display_image(
            row=row,
            frame_full_bgr=frame,
            show_overlay=bool(state["show_overlay"]),
            show_save_preview=bool(state["show_save_preview"]),
            candidates=list(state["candidates"]),
            selected_idx=int(state["selected_idx"]),
            prompt_points=list(state["prompt_points"]),
            polygon_points=list(state["polygon_points"]),
            draw_candidates=bool(state["draw_candidates"]),
            view_rect=view_rect,
            out_wh=(disp_w, disp_h),
            args=args,
        )

        selected_quality: Dict = {}
        selected_gate = {"pass": False, "reason": "none"}
        selected: Optional[Dict] = None
        if state["candidates"]:
            selected = state["candidates"][clamp_idx(int(state["selected_idx"]), len(state["candidates"]))]
        if selected is not None:
            selected_quality = build_quality_metrics(
                frame_bgr=frame,
                mask_u8=(selected["mask_full"] > 0).astype(np.uint8),
                prompt_points=list(state["prompt_points"]),
                white_cfg=white_cfg,
                pos_hit_tol_px=int(args.positive_hit_tol_px),
                neg_hit_tol_px=int(args.negative_hit_tol_px),
                pos_soft_gate=bool(args.pos_soft_gate),
                pos_soft_radius_px=int(args.pos_soft_radius_px),
                pos_hard_max_dist_px=int(args.pos_hard_max_dist_px),
                pos_distance_penalty=float(args.pos_distance_penalty),
            )
            selected_gate = evaluate_save_gate(selected_quality, probe_args)

        selected_diag = "Sel: -"
        if selected is not None:
            s_score = "-" if selected.get("score") is None else f"{float(selected.get('score')):.2f}"
            selected_diag = (
                f"Sel score={s_score} rank={float(selected.get('rank_score', 0.0)):.3f} "
                f"white={float(selected.get('white_ratio', 0.0)):.2f} "
                f"posDist=[{float(selected.get('pos_dist_min', 0.0)):.1f},{float(selected.get('pos_dist_max', 0.0)):.1f}] "
                f"negHits={int(selected.get('neg_hits', 0))}"
            )
        gate_diag = "SaveGate: -"
        if selected is not None:
            eligibility = "main-eligible" if bool(selected_gate.get("pass", False)) else "force-only"
            gate_diag = (
                f"SaveGate={'PASS' if bool(selected_gate.get('pass', False)) else 'BLOCK'} ({eligibility}) "
                f"reason={str(selected_gate.get('reason', 'none'))} "
                f"fill={float(selected_quality.get('fill_ratio', 1.0)):.2f} "
                f"elong={float(selected_quality.get('elongation', 1.0)):.2f}"
            )
        reject_diag = state["last_reject"] if state["last_reject"] else "top reject=-"
        pos_count = sum(1 for p in state["prompt_points"] if int(p["label"]) == POS_LABEL)
        neg_count = sum(1 for p in state["prompt_points"] if int(p["label"]) == NEG_LABEL)
        info_lines = [
            f"Image {idx + 1}/{len(rows)} | View={'OVERLAY' if bool(state['show_overlay']) else 'LABEL'} | Active={'POLYGON' if str(state['interaction_mode']) == 'polygon' else 'PROMPT'} | Auto={'ON' if bool(state['auto_enabled']) else 'OFF'}",
            f"Relabeled={'YES' if idx in relabeled_indices else 'NO'} | PredArea={int(row.get('pred_area', 0))} | Preview={'SAVE' if bool(state['show_save_preview']) else 'NORMAL'}",
            f"Handled in this summary: {len(relabeled_indices)} | Saved main: {int(saved_counts['main'])} | Quarantine: {int(saved_counts['quarantine'])} | Rejected: {int(saved_counts['rejected'])}",
            f"Prompt mode={'POS' if int(state['label_mode']) == POS_LABEL else 'NEG'} | Prompts: +{pos_count} / -{neg_count} | PolyPts={len(state['polygon_points'])} | Candidates={len(state['candidates'])} | Selected={int(state['selected_idx']) + 1 if state['candidates'] else 0}",
            f"File: {Path(row_path_value(row, 'image_path', prefer_secondary=False)).name}",
            f"OverlaySrc={'SECONDARY' if bool(args.prefer_secondary_paths) and str(row.get('secondary_overlay_path', '')).strip() else 'PRIMARY'}",
            selected_diag,
            gate_diag,
            reject_diag,
            str(state["status"]),
            "D/A next/prev | Shift+D/A jump 25 | L overlay/label | 1 tiny-line preview | P prompt/polygon | N POS/NEG | Wheel zoom | M-drag pan | Space detect | G auto | K cycle | +/- probe | Backspace undo | C clear polygon | R reject | R-click clear all | F/Enter finalize polygon | V preview | S save | Shift+S force | Q quit",
        ]
        y = 24
        for line in info_lines:
            cv2.putText(viz, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            y += 24

        cv2.imshow(WINDOW_NAME, viz)
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue
        if key in (ord("q"), ord("Q"), 27):
            break

        if key in (ord("d"), ord("D")):
            step = 25 if key == ord("D") else 1
            idx = clamp_idx(idx + step, len(rows))
            reset_edit_state()
            state["show_overlay"] = True
            state["status"] = ""
            save_resume_state(state_path, summary_path, idx, sorted(relabeled_indices), str(state["interaction_mode"]))
            continue
        if key in (ord("a"), ord("A")):
            step = 25 if key == ord("A") else 1
            idx = clamp_idx(idx - step, len(rows))
            reset_edit_state()
            state["show_overlay"] = True
            state["status"] = ""
            save_resume_state(state_path, summary_path, idx, sorted(relabeled_indices), str(state["interaction_mode"]))
            continue
        if key in (ord("l"), ord("L")):
            was_overlay = bool(state["show_overlay"])
            state["show_overlay"] = not bool(state["show_overlay"])
            state["show_save_preview"] = False
            if was_overlay and not bool(state["show_overlay"]):
                has_manual_state = bool(state["prompt_points"]) or bool(state["polygon_points"]) or bool(state["candidates"])
                if not has_manual_state:
                    preloaded = load_predicted_candidate(row, prefer_secondary_paths=bool(args.prefer_secondary_paths))
                    if preloaded is None and bool(args.prefer_secondary_paths):
                        preloaded = load_predicted_candidate(row, prefer_secondary_paths=False)
                    if preloaded is not None:
                        state["candidates"] = [preloaded]
                        state["selected_idx"] = 0
                        state["draw_candidates"] = False
                        state["probe_scale_used"] = 1.0
                        state["status"] = "View -> LABEL | preloaded predicted mask as candidate #1 | press K/Space to show or refresh"
                        continue
            state["status"] = f"View -> {'OVERLAY' if bool(state['show_overlay']) else 'LABEL'}"
            continue
        if key == ord("1"):
            try:
                image_key = str(
                    resolve_path(
                        row_path_value(row, "image_path", prefer_secondary=False)
                        or row_path_value(row, "original_path", prefer_secondary=False)
                    )
                )
                reset_edit_state()
                state["show_overlay"] = False
                if specialist_runtime["last_image_key"] == image_key:
                    candidate = specialist_runtime["last_candidate"]
                    status_msg = str(specialist_runtime["last_status"])
                else:
                    candidate, status_msg = run_specialist_preview_candidate(frame, specialist_runtime)
                    specialist_runtime["last_image_key"] = image_key
                    specialist_runtime["last_candidate"] = candidate
                    specialist_runtime["last_status"] = str(status_msg)
                state["external_preview_source"] = "tinyline_specialist"
                if candidate is not None:
                    state["candidates"] = [candidate]
                    state["selected_idx"] = 0
                    state["draw_candidates"] = True
                state["status"] = str(status_msg)
            except Exception as exc:
                state["status"] = f"Tiny-line preview failed: {exc}"
            continue
        if key in (ord("+"), ord("=")):
            state["probe_upscale"] = min(float(args.probe_upscale_max), float(state["probe_upscale"]) * float(args.probe_upscale_step))
            state["needs_infer"] = not bool(state["show_overlay"])
            state["status"] = f"Probe upscale x{float(state['probe_upscale']):.2f}"
            continue
        if key in (ord("-"), ord("_")):
            state["probe_upscale"] = max(1.0, float(state["probe_upscale"]) / float(args.probe_upscale_step))
            state["needs_infer"] = not bool(state["show_overlay"])
            state["status"] = f"Probe upscale x{float(state['probe_upscale']):.2f}"
            continue
        if key_is(key, "p", "P"):
            state["interaction_mode"] = "polygon" if str(state.get("interaction_mode", "prompt")) == "prompt" else "prompt"
            state["status"] = f"Mode -> {'POLYGON' if str(state['interaction_mode']) == 'polygon' else 'PROMPT'}"
            save_resume_state(state_path, summary_path, idx, sorted(relabeled_indices), str(state["interaction_mode"]))
            continue
        if key_is(key, "n", "N"):
            state["label_mode"] = NEG_LABEL if int(state["label_mode"]) == POS_LABEL else POS_LABEL
            state["status"] = f"Prompt mode -> {'POS' if int(state['label_mode']) == POS_LABEL else 'NEG'}"
            continue
        if key_is(key, "g", "G"):
            state["auto_enabled"] = not bool(state["auto_enabled"])
            if bool(state["auto_enabled"]) and not bool(state["show_overlay"]):
                state["needs_infer"] = True
            state["status"] = f"Auto detection {'enabled' if bool(state['auto_enabled']) else 'disabled'}"
            continue
        if key == 32:
            if bool(state["show_overlay"]):
                state["status"] = "Switch to LABEL view first"
            else:
                state["external_preview_source"] = ""
                state["force_infer"] = True
                state["status"] = "Running one-shot detection"
            continue
        if key_is(key, "k", "K"):
            if state["candidates"]:
                state["draw_candidates"] = True
                state["selected_idx"] = (int(state["selected_idx"]) + 1) % len(state["candidates"])
                chosen = state["candidates"][int(state["selected_idx"])]
                score_txt = "-" if chosen.get("score") is None else f"{float(chosen['score']):.2f}"
                state["status"] = f"Selected candidate {int(state['selected_idx']) + 1}/{len(state['candidates'])} | score={score_txt}"
            else:
                state["status"] = "No candidates to cycle"
            continue
        if key in (8, 127):
            if str(state.get("interaction_mode", "prompt")) == "polygon":
                if state["polygon_points"]:
                    state["polygon_points"].pop()
                    state["status"] = f"Removed polygon point, remaining {len(state['polygon_points'])}"
                else:
                    state["status"] = "No polygon point to undo"
            else:
                if state["prompt_points"]:
                    removed = state["prompt_points"].pop()
                    state["needs_infer"] = bool(state["auto_enabled"]) and not bool(state["show_overlay"])
                    state["status"] = f"Removed {'POS' if int(removed['label']) == POS_LABEL else 'NEG'} prompt"
                else:
                    state["status"] = "No prompt to undo"
            continue
        if key in (ord("c"), ord("C")):
            state["polygon_points"] = []
            state["status"] = "Cleared polygon points"
            continue
        if key in (ord("v"), ord("V")):
            state["show_save_preview"] = not bool(state["show_save_preview"])
            state["status"] = f"Save preview {'ON' if bool(state['show_save_preview']) else 'OFF'}"
            continue
        if key in (ord("f"), ord("F"), 13):
            if bool(state["show_overlay"]):
                state["status"] = "Switch to LABEL mode first"
                continue
            if str(state.get("interaction_mode", "prompt")) != "polygon":
                state["status"] = "Polygon finalize is only available in POLYGON mode"
                continue
            poly_cand, poly_reason, _ = build_polygon_candidate(
                frame_full_bgr=frame,
                polygon_points=list(state["polygon_points"]),
                prompt_points=list(state["prompt_points"]),
                white_cfg=white_cfg,
                args=probe_args,
            )
            if poly_cand is None:
                state["status"] = f"Polygon finalize failed ({poly_reason})"
                continue
            poly_cand["candidate_priority"] = 1
            state["candidates"] = [poly_cand]
            state["selected_idx"] = 0
            state["draw_candidates"] = True
            state["needs_infer"] = False
            state["force_infer"] = False
            state["last_reject"] = ""
            state["status"] = "Polygon candidate ready (source=polygon_refine)"
            continue

        normal_save_pressed = key == ord("s")
        force_save_pressed = key == ord("S")
        reject_pressed = key in (ord("r"), ord("R"))
        if normal_save_pressed or force_save_pressed or reject_pressed:
            candidate_to_save = selected
            quality_to_save = dict(selected_quality)
            gate_to_save = dict(selected_gate)
            source_image_key = str(resolve_path(str(row["image_path"]))).lower()
            if bool(state["show_overlay"]) and bool(args.prefer_secondary_paths or reject_pressed):
                preloaded = load_predicted_candidate(row, prefer_secondary_paths=True)
                if preloaded is None:
                    preloaded = load_predicted_candidate(row, prefer_secondary_paths=False)
                if preloaded is not None:
                    candidate_to_save = preloaded
                    quality_to_save = build_quality_metrics(
                        frame_bgr=frame,
                        mask_u8=(candidate_to_save["mask_full"] > 0).astype(np.uint8),
                        prompt_points=[],
                        white_cfg=white_cfg,
                        pos_hit_tol_px=int(args.positive_hit_tol_px),
                        neg_hit_tol_px=int(args.negative_hit_tol_px),
                        pos_soft_gate=bool(args.pos_soft_gate),
                        pos_soft_radius_px=int(args.pos_soft_radius_px),
                        pos_hard_max_dist_px=int(args.pos_hard_max_dist_px),
                        pos_distance_penalty=float(args.pos_distance_penalty),
                    )
                    gate_to_save = evaluate_save_gate(quality_to_save, probe_args)
            if bool(state["show_overlay"]):
                if candidate_to_save is None:
                    state["status"] = "Switch to LABEL view before saving"
                    continue
            if candidate_to_save is None:
                state["status"] = "No candidate to save/reject"
                continue
            if source_image_key in existing_source_paths:
                state["status"] = "Source image already labeled; duplicate save/reject blocked"
                continue

            gate_pass = bool(gate_to_save.get("pass", False))
            gate_reason = str(gate_to_save.get("reason", "none"))
            is_force = bool(force_save_pressed) and (not gate_pass) and (not reject_pressed)
            if (not reject_pressed) and str(args.save_gate_mode).lower().strip() == "strict" and (not gate_pass) and (not is_force):
                state["status"] = f"Blocked save ({gate_reason}) | use Shift+S to force"
                continue

            dataset_split = "main"
            dataset_dirs = dataset_dirs_main
            save_mode = "normal"
            if reject_pressed:
                dataset_split = "rejected"
                dataset_dirs = dataset_dirs_rejected
                save_mode = "reject"
            elif is_force:
                dataset_split = "quarantine"
                dataset_dirs = dataset_dirs_quarantine
                save_mode = "force"

            rec = save_sample(
                frame_full_bgr=frame,
                mask_full_u8=(candidate_to_save["mask_full"] > 0).astype(np.uint8),
                dataset_dirs=dataset_dirs,
                prompt=str(summary.get("checkpoint", prompt)),
                image_path=resolve_path(str(row["image_path"])),
                image_index=int(row["index"]) - 1,
                probe_scale=float(candidate_to_save.get("probe_scale", state["probe_scale_used"])),
                view_rect=view_rect,
                prompt_points_snapshot=format_prompt_points_for_meta(state["prompt_points"]),
                candidate_meta=candidate_to_save,
                tiny_cfg=tiny_cfg,
                pos_snap_enabled=bool(args.snap_positive_to_line),
                reject_summary=str(state["last_reject"]),
                save_gate_pass=bool(gate_pass),
                save_gate_reason=str(gate_reason),
                save_mode=str(save_mode),
                dataset_split=str(dataset_split),
                quality_metrics=quality_to_save,
                polygon_points=[tuple(xy) for xy in candidate_to_save.get("polygon_points", [])] if candidate_to_save.get("polygon_points") else None,
            )
            relabeled_indices.add(int(idx))
            existing_source_paths.add(source_image_key)
            saved_counts[str(dataset_split)] = int(saved_counts.get(str(dataset_split), 0)) + 1
            save_source = "overlay prediction" if bool(state["show_overlay"]) else "candidate"
            state["status"] = f"Saved {rec['id']} -> {dataset_split} ({save_source})"
            save_resume_state(state_path, summary_path, idx, sorted(relabeled_indices), str(state["interaction_mode"]))
            idx = clamp_idx(idx + 1, len(rows))
            reset_edit_state()
            state["show_overlay"] = True
            continue

    save_resume_state(state_path, summary_path, idx, sorted(relabeled_indices), str(state["interaction_mode"]))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
