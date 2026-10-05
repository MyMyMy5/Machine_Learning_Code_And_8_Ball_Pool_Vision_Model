from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.supervised.infer import run_supervised_inference


def _normalize_path(path_str: str) -> Path:
    if os.name != "nt" and path_str.startswith("C:\\"):
        return Path("/mnt/c/" + path_str[3:].replace("\\", "/"))
    if os.name == "nt" and path_str.startswith("/mnt/c/"):
        return Path("C:/" + path_str[len("/mnt/c/"):])
    return Path(path_str)


def _load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def _collect_split_items(index_path: Path, split_name: str) -> dict[str, dict[str, Path]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    items: dict[str, dict[str, Path]] = {}
    for item in index[split_name]:
        frame_id = str(item["id"])
        items.setdefault(
            frame_id,
            {
                "image_path": _normalize_path(str(item["image_path"])),
                "mask_path": _normalize_path(str(item["mask_path"])),
            },
        )
    return items


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized: dict[str, object] = {}
    for key, value in manifest.items():
        if key in {"metrics", "documentation", "review_bundle", "benchmark"}:
            normalized[key] = value
        elif key in {"micro_line_rescue", "colored_blob_rescue"} and isinstance(value, dict):
            normalized_value = dict(value)
            for nested_key in ("checkpoint", "checkpoint_path", "reranker_checkpoint", "reranker_checkpoint_path"):
                if nested_key in normalized_value and normalized_value[nested_key]:
                    normalized_value[nested_key] = str(_normalize_path(str(normalized_value[nested_key])))
            normalized[key] = normalized_value
        elif key.endswith("_checkpoint") or key.endswith("_reranker"):
            normalized[key] = _normalize_path(str(value))
        else:
            normalized[key] = value
    return normalized


def _compute_iou_and_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    iou = 0.0 if union == 0.0 else inter / union
    dice = (2.0 * inter + 1.0) / (float(pred.sum()) + float(gt.sum()) + 1.0)
    return iou, dice


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index-json", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "hard_val"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(_normalize_path(args.manifest))
    items = _collect_split_items(_normalize_path(args.index_json), args.split)
    output_root = _normalize_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    frame_ids = sorted(items)
    if args.limit is not None:
        frame_ids = frame_ids[: args.limit]

    for index, frame_id in enumerate(frame_ids, start=1):
        frame_output = output_root / frame_id
        if not (args.skip_existing and (frame_output / "report.json").exists()):
            run_supervised_inference(
                checkpoint_path=manifest["primary_checkpoint"],
                input_path=items[frame_id]["image_path"],
                output_dir=frame_output,
                image_size=384,
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
                post_final_candidate_selector_rescue_rules=manifest.get(
                    "post_final_candidate_selector_rescue_rules"
                ),
            )

        gt = _load_mask(items[frame_id]["mask_path"])
        pred = _load_mask(frame_output / "mask_final.png")
        iou, dice = _compute_iou_and_dice(pred, gt)
        winner = json.loads((frame_output / "report.json").read_text(encoding="utf-8"))["winner"]
        rows.append(
            {
                "id": frame_id,
                "iou": iou,
                "dice": dice,
                "winner": winner["candidate_id"],
                "score": float(winner["score"]),
                "source": winner["candidate_source"],
                "fallback": bool(winner.get("selected_via_fallback", False)),
                "rescue": bool(winner.get("selected_via_rescue", False)),
                "final_fallback": bool(winner.get("selected_via_final_fallback", False)),
                "secondary_rescue": bool(winner.get("selected_via_secondary_rescue", False)),
                "primary_recovery": bool(winner.get("selected_via_primary_recovery", False)),
                "reticle_recovery": bool(winner.get("selected_via_reticle_recovery", False)),
                "fallback_rescue": bool(winner.get("selected_via_fallback_rescue", False)),
                "reticle_fill_rescue": bool(winner.get("selected_via_reticle_fill_rescue", False)),
                "external_rescue": bool(winner.get("selected_via_external_rescue", False)),
                "external_mid_rescue": bool(winner.get("selected_via_external_mid_rescue", False)),
                "external_lowmid_rescue": bool(winner.get("selected_via_external_lowmid_rescue", False)),
                "external_reticle_rescue": bool(winner.get("selected_via_external_reticle_rescue", False)),
                "external_tiny_reticle_rescue": bool(winner.get("selected_via_external_tiny_reticle_rescue", False)),
                "external_reticle_large_rescue": bool(winner.get("selected_via_external_reticle_large_rescue", False)),
                "external_table_broad_rescue": bool(winner.get("selected_via_external_table_broad_rescue", False)),
                "external_blob_low_rescue": bool(winner.get("selected_via_external_blob_low_rescue", False)),
                "external_table_rescue": bool(winner.get("selected_via_external_table_rescue", False)),
                "external_blob_rescue": bool(winner.get("selected_via_external_blob_rescue", False)),
                "external_tiny_blob_rescue": bool(winner.get("selected_via_external_tiny_blob_rescue", False)),
            }
        )
        if index % 25 == 0:
            print(f"processed {index}/{len(frame_ids)}")

    rows.sort(key=lambda row: float(row["iou"]))
    summary = {
        "split": args.split,
        "count": len(rows),
        "mean_iou": float(np.mean([float(row["iou"]) for row in rows])) if rows else 0.0,
        "mean_dice": float(np.mean([float(row["dice"]) for row in rows])) if rows else 0.0,
        "zero_iou": int(sum(1 for row in rows if float(row["iou"]) == 0.0)),
        "fallback_count": int(sum(1 for row in rows if bool(row["fallback"]))),
        "rescue_count": int(sum(1 for row in rows if bool(row["rescue"]))),
        "final_fallback_count": int(sum(1 for row in rows if bool(row["final_fallback"]))),
        "secondary_rescue_count": int(sum(1 for row in rows if bool(row["secondary_rescue"]))),
        "primary_recovery_count": int(sum(1 for row in rows if bool(row["primary_recovery"]))),
        "reticle_recovery_count": int(sum(1 for row in rows if bool(row["reticle_recovery"]))),
        "fallback_rescue_count": int(sum(1 for row in rows if bool(row["fallback_rescue"]))),
        "reticle_fill_rescue_count": int(sum(1 for row in rows if bool(row["reticle_fill_rescue"]))),
        "external_rescue_count": int(sum(1 for row in rows if bool(row["external_rescue"]))),
        "external_mid_rescue_count": int(sum(1 for row in rows if bool(row["external_mid_rescue"]))),
        "external_lowmid_rescue_count": int(sum(1 for row in rows if bool(row["external_lowmid_rescue"]))),
        "external_reticle_rescue_count": int(sum(1 for row in rows if bool(row["external_reticle_rescue"]))),
        "external_tiny_reticle_rescue_count": int(sum(1 for row in rows if bool(row["external_tiny_reticle_rescue"]))),
        "external_reticle_large_rescue_count": int(sum(1 for row in rows if bool(row["external_reticle_large_rescue"]))),
        "external_table_broad_rescue_count": int(sum(1 for row in rows if bool(row["external_table_broad_rescue"]))),
        "external_blob_low_rescue_count": int(sum(1 for row in rows if bool(row["external_blob_low_rescue"]))),
        "external_table_rescue_count": int(sum(1 for row in rows if bool(row["external_table_rescue"]))),
        "external_blob_rescue_count": int(sum(1 for row in rows if bool(row["external_blob_rescue"]))),
        "external_tiny_blob_rescue_count": int(sum(1 for row in rows if bool(row["external_tiny_blob_rescue"]))),
        "worst25": rows[:25],
        "rows": rows,
    }
    summary_path = _normalize_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
