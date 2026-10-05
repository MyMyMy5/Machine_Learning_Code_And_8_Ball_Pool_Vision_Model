from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.supervised.final_veto import load_final_veto_checkpoint  # noqa: E402
from tools.eval_gold_board_manifest import summarize_rows  # noqa: E402


def _load_winner(row: dict[str, Any]) -> dict[str, object]:
    report_path = Path(str(row["output_dir"])) / "report.json"
    winner = json.loads(report_path.read_text(encoding="utf-8"))["winner"]
    if not isinstance(winner, dict):
        raise ValueError(f"Invalid winner payload in {report_path}")
    return _slim_winner(winner)


def _slim_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): slimmed
            for key, child in value.items()
            if (slimmed := _slim_value(child)) is not None
        }
    if isinstance(value, list):
        if len(value) <= 512 and all(isinstance(item, (int, float, bool, str)) or item is None for item in value):
            return value
        return None
    return None


def _slim_winner(winner: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): slimmed
        for key, value in winner.items()
        if (slimmed := _slim_value(value)) is not None
    }


def _row_with_veto_probability(row: dict[str, Any], *, veto_model) -> dict[str, Any]:
    pred_pixels = int(row.get("pred_pixels", 0))
    probability = 0.0
    if pred_pixels > 0:
        winner = _load_winner(row)
        probability = veto_model.veto_probability(winner, pred_pixels)
    enriched = dict(row)
    enriched["final_veto_probability"] = float(probability)
    return enriched


def _empty_mask_metrics(gt_pixels: int) -> tuple[float, float]:
    if gt_pixels <= 0:
        return 1.0, 1.0
    return 0.0, 1.0 / (float(gt_pixels) + 1.0)


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


def _simulate_row(
    row: dict[str, Any],
    *,
    veto_model,
    threshold: float,
    success_iou: float,
) -> dict[str, Any]:
    pred_pixels = int(row.get("pred_pixels", 0))
    probability = float(row.get("final_veto_probability", 0.0))
    vetoed = pred_pixels > 0 and probability >= threshold
    new_row = dict(row)
    new_row["final_veto_probability"] = float(probability)
    new_row["final_veto_threshold"] = float(threshold)
    new_row["selected_via_final_veto"] = bool(vetoed)
    if vetoed:
        gt_pixels = int(row.get("gt_pixels", 0))
        iou, dice = _empty_mask_metrics(gt_pixels)
        new_row["pre_final_veto_iou"] = float(row.get("iou", 0.0))
        new_row["pre_final_veto_dice"] = float(row.get("dice", 0.0))
        new_row["pre_final_veto_pred_pixels"] = pred_pixels
        new_row["iou"] = float(iou)
        new_row["dice"] = float(dice)
        new_row["pred_pixels"] = 0
        new_row["failure_type"] = _failure_type(
            gt_pixels=gt_pixels,
            pred_pixels=0,
            iou=float(iou),
            success_iou=success_iou,
        )
    return new_row


def _evaluate_threshold(
    rows: list[dict[str, Any]],
    *,
    veto_model,
    threshold: float,
    splits: list[str],
    success_iou: float,
) -> dict[str, Any]:
    simulated = [
        _simulate_row(row, veto_model=veto_model, threshold=threshold, success_iou=success_iou)
        for row in rows
    ]
    by_split = {
        split_name: summarize_rows([row for row in simulated if row["split"] == split_name])
        for split_name in splits
    }
    vetoed = [row for row in simulated if row.get("selected_via_final_veto")]
    vetoed_positive = [row for row in vetoed if int(row.get("gt_pixels", 0)) > 0]
    vetoed_negative = [row for row in vetoed if int(row.get("gt_pixels", 0)) <= 0]
    return {
        "threshold": float(threshold),
        "overall": summarize_rows(simulated),
        "by_split": by_split,
        "veto_count": len(vetoed),
        "vetoed_positive_count": len(vetoed_positive),
        "vetoed_negative_count": len(vetoed_negative),
        "vetoed_source_counts": dict(Counter(str(row.get("winner_source")) for row in vetoed)),
        "rows": simulated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--success-iou", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    veto_model = load_final_veto_checkpoint(Path(args.checkpoint), device=device)
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    rows = [
        _row_with_veto_probability(row, veto_model=veto_model)
        for row in summary["rows"]
    ]
    splits = list(summary.get("splits", sorted({str(row["split"]) for row in rows})))
    threshold = float(args.threshold) if args.threshold is not None else float(veto_model.threshold)
    result = _evaluate_threshold(
        rows,
        veto_model=veto_model,
        threshold=threshold,
        splits=splits,
        success_iou=args.success_iou,
    )
    result["source_summary"] = str(args.summary_json)
    result["checkpoint"] = str(args.checkpoint)

    if args.sweep:
        thresholds = np.linspace(0.05, 0.95, 19)
        result["threshold_sweep"] = [
            {
                "threshold": float(candidate),
                "overall": candidate_result["overall"],
                "by_split": candidate_result["by_split"],
                "veto_count": candidate_result["veto_count"],
                "vetoed_positive_count": candidate_result["vetoed_positive_count"],
                "vetoed_negative_count": candidate_result["vetoed_negative_count"],
            }
            for candidate in thresholds
            for candidate_result in [
                _evaluate_threshold(
                    rows,
                    veto_model=veto_model,
                    threshold=float(candidate),
                    splits=splits,
                    success_iou=args.success_iou,
                )
            ]
        ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(str(output))
    print(json.dumps({key: result[key] for key in ["threshold", "veto_count", "vetoed_positive_count", "vetoed_negative_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
