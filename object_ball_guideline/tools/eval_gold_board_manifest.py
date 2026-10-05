from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.supervised.manifest_inference import load_manifest, normalize_path, run_manifest_inference


def _load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.uint8)
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def _load_pred_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def _write_mask_only_output(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(output_path)


def _compute_iou_and_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_bool = pred > 0
    gt_bool = gt > 0
    inter = float(np.logical_and(pred_bool, gt_bool).sum())
    union = float(np.logical_or(pred_bool, gt_bool).sum())
    pred_pixels = float(pred_bool.sum())
    gt_pixels = float(gt_bool.sum())
    iou = 1.0 if union == 0.0 else inter / union
    dice = (2.0 * inter + 1.0) / (pred_pixels + gt_pixels + 1.0)
    return iou, dice


def _failure_type(*, gt_pixels: int, pred_pixels: int, iou: float, success_iou: float) -> str:
    if gt_pixels <= 0:
        return "true_negative" if pred_pixels <= 0 else "negative_false_positive"
    if iou >= success_iou:
        return "success"
    if pred_pixels <= 0:
        return "positive_no_prediction"
    if iou <= 0.0:
        return "positive_zero_iou_wrong_prediction"
    return "positive_partial_miss"


def _winner_flags(winner: dict[str, Any]) -> dict[str, bool]:
    return {
        key.removeprefix("selected_via_"): bool(value)
        for key, value in winner.items()
        if key.startswith("selected_via_")
    }


def _row_from_result(
    *,
    item: dict[str, Any],
    split_name: str,
    result: dict[str, object],
    gt: np.ndarray,
    final_mask: np.ndarray,
    output_dir: Path,
    success_iou: float,
) -> dict[str, Any]:
    winner = result["winner"]
    assert isinstance(winner, dict)
    iou, dice = _compute_iou_and_dice(final_mask, gt)
    gt_pixels = int(np.count_nonzero(gt))
    pred_pixels = int(np.count_nonzero(final_mask))
    features = winner.get("features", {})
    if not isinstance(features, dict):
        features = {}
    row: dict[str, Any] = {
        "id": str(item["id"]),
        "split": split_name,
        "kind": item.get("kind"),
        "source": item.get("source"),
        "image_path": str(item.get("image_path")),
        "mask_path": str(item.get("mask_path")) if item.get("mask_path") else None,
        "output_dir": str(output_dir),
        "iou": float(iou),
        "dice": float(dice),
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "failure_type": _failure_type(
            gt_pixels=gt_pixels,
            pred_pixels=pred_pixels,
            iou=float(iou),
            success_iou=success_iou,
        ),
        "winner_id": winner.get("candidate_id"),
        "winner_source": winner.get("candidate_source"),
        "winner_score": float(winner["score"]) if isinstance(winner.get("score"), (int, float)) else None,
        "winner_validity_score": (
            float(winner["validity_score"])
            if isinstance(winner.get("validity_score"), (int, float))
            else None
        ),
        "winner_ball_fill_fraction": (
            float(features["ball_fill_fraction"])
            if isinstance(features.get("ball_fill_fraction"), (int, float))
            else None
        ),
        "winner_connected_to_ball": (
            float(features["connected_to_ball"])
            if isinstance(features.get("connected_to_ball"), (int, float))
            else None
        ),
        "winner_flags": _winner_flags(winner),
    }
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if int(row["gt_pixels"]) > 0]
    negatives = [row for row in rows if int(row["gt_pixels"]) <= 0]
    negative_fps = [row for row in negatives if row["failure_type"] == "negative_false_positive"]
    positive_zero_iou = [row for row in positives if float(row["iou"]) <= 0.0]
    failure_counts = Counter(str(row["failure_type"]) for row in rows)
    source_counts = Counter(str(row.get("winner_source")) for row in rows)
    fp_source_counts = Counter(str(row.get("winner_source")) for row in negative_fps)
    flag_counts: Counter[str] = Counter()
    for row in rows:
        flags = row.get("winner_flags", {})
        if isinstance(flags, dict):
            flag_counts.update(key for key, value in flags.items() if value)

    def mean(key: str, subset: list[dict[str, Any]]) -> float:
        return float(np.mean([float(row[key]) for row in subset])) if subset else 0.0

    return {
        "count": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "mean_iou": mean("iou", rows),
        "mean_dice": mean("dice", rows),
        "positive_mean_iou": mean("iou", positives),
        "positive_mean_dice": mean("dice", positives),
        "positive_zero_iou": len(positive_zero_iou),
        "true_negative": failure_counts.get("true_negative", 0),
        "negative_false_positive": len(negative_fps),
        "negative_false_positive_rate": (len(negative_fps) / len(negatives)) if negatives else 0.0,
        "failure_counts": dict(sorted(failure_counts.items())),
        "winner_source_counts": dict(sorted(source_counts.items())),
        "negative_false_positive_source_counts": dict(sorted(fp_source_counts.items())),
        "winner_flag_counts": dict(sorted(flag_counts.items())),
    }


def _selected_items(gold_board: dict[str, Any], split_names: list[str]) -> list[tuple[str, dict[str, Any]]]:
    splits = gold_board.get("splits", {})
    items: list[tuple[str, dict[str, Any]]] = []
    for split_name in split_names:
        if split_name not in splits:
            available = ", ".join(sorted(splits))
            raise KeyError(f"Unknown gold-board split '{split_name}'. Available: {available}")
        items.extend((split_name, item) for item in splits[split_name])
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gold-board", default="runs/gold_board_v1/gold_board.json")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["harvest_holdout", "rejected_negatives", "flat_negatives"],
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-batch-size", type=int, default=1)
    parser.add_argument("--success-iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-output-images", action="store_true")
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write per-item report.json files; use for summary-only benchmark runs.",
    )
    parser.add_argument(
        "--mask-only-output",
        action="store_true",
        help="Write mask_final.png but skip overlay/recolor image generation.",
    )
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--save-recolor-preview", action="store_true")
    args = parser.parse_args()
    if args.no_report and args.skip_existing:
        parser.error("--no-report cannot be combined with --skip-existing because resume needs report.json")

    manifest = load_manifest(normalize_path(args.manifest))
    gold_board = json.loads(normalize_path(args.gold_board).read_text(encoding="utf-8"))
    output_root = normalize_path(args.output_root)
    summary_json = normalize_path(args.summary_json)
    assert output_root is not None
    assert summary_json is not None
    output_root.mkdir(parents=True, exist_ok=True)
    items = _selected_items(gold_board, args.splits)
    if args.limit is not None:
        items = items[: args.limit]

    rows: list[dict[str, Any]] = []
    for index, (split_name, item) in enumerate(items, start=1):
        image_path = normalize_path(item["image_path"])
        mask_path = normalize_path(item.get("mask_path"))
        if image_path is None:
            raise ValueError(f"gold-board item has no image_path: {item}")
        item_output_dir = output_root / split_name / str(item["id"])
        report_path = item_output_dir / "report.json"
        mask_output_path = item_output_dir / "mask_final.png"
        if args.skip_existing and report_path.exists() and mask_output_path.exists():
            winner_result = {
                "winner": json.loads(report_path.read_text(encoding="utf-8"))["winner"],
                "final_mask": _load_pred_mask(mask_output_path),
            }
        else:
            winner_result = run_manifest_inference(
                manifest=manifest,
                input_path=image_path,
                output_dir=item_output_dir,
                image_size=args.image_size,
                crop_batch_size=args.crop_batch_size,
                save_outputs=not args.no_output_images and not args.mask_only_output,
                save_intermediates=args.save_intermediates,
                save_report=not args.no_report,
                save_recolor_preview=args.save_recolor_preview,
                return_overlay=(not args.no_output_images and not args.mask_only_output)
                or args.save_recolor_preview,
            )
            if args.mask_only_output and not args.no_output_images:
                final_mask_for_write = np.asarray(winner_result["final_mask"], dtype=np.uint8)
                _write_mask_only_output(final_mask_for_write, mask_output_path)
        final_mask = np.asarray(winner_result["final_mask"], dtype=np.uint8)
        gt = _load_mask(mask_path, final_mask.shape)
        rows.append(
            _row_from_result(
                item=item,
                split_name=split_name,
                result=winner_result,
                gt=gt,
                final_mask=final_mask,
                output_dir=item_output_dir,
                success_iou=args.success_iou,
            )
        )
        if index % 25 == 0 or index == len(items):
            print(f"processed {index}/{len(items)}", flush=True)

    rows.sort(key=lambda row: (str(row["split"]), float(row["iou"]), -int(row["pred_pixels"])))
    by_split = {
        split_name: summarize_rows([row for row in rows if row["split"] == split_name])
        for split_name in args.splits
    }
    summary = {
        "manifest": str(args.manifest),
        "gold_board": str(args.gold_board),
        "splits": args.splits,
        "overall": summarize_rows(rows),
        "by_split": by_split,
        "worst25": sorted(rows, key=lambda row: (float(row["iou"]), -int(row["pred_pixels"])))[:25],
        "rows": rows,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(summary_json), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
