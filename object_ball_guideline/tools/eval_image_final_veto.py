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

from src.supervised.image_final_veto import (  # noqa: E402
    choose_veto_threshold,
    load_image_final_veto_checkpoint,
    load_veto_items,
    predict_probabilities,
)


def _summarize_at_threshold(items: list[dict[str, Any]], probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = probabilities >= float(threshold)
    targets = np.asarray([int(item["target"]) for item in items], dtype=np.int32)
    keep = targets == 0
    veto = targets == 1
    false_positive_veto = int(np.logical_and(preds, keep).sum())
    true_negative_veto = int(np.logical_and(preds, veto).sum())
    source_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    vetoed_rows: list[dict[str, Any]] = []
    for item, prob, pred in zip(items, probabilities, preds, strict=True):
        if not bool(pred):
            continue
        source_counter[str(item.get("winner_source"))] += 1
        split_counter[str(item.get("split"))] += 1
        vetoed_rows.append(
            {
                "id": item.get("id"),
                "split": item.get("split"),
                "winner_source": item.get("winner_source"),
                "target": int(item.get("target", 0)),
                "probability": float(prob),
                "pred_pixels": int(item.get("pred_pixels", 0)),
                "failure_type": item.get("failure_type"),
                "image_path": item.get("image_path"),
                "mask_path": item.get("mask_path"),
            }
        )
    return {
        "threshold": float(threshold),
        "count": len(items),
        "target_veto_total": int(veto.sum()),
        "keep_total": int(keep.sum()),
        "true_negative_veto": true_negative_veto,
        "false_positive_veto": false_positive_veto,
        "neg_veto_rate": true_negative_veto / max(1, int(veto.sum())),
        "pos_veto_rate": false_positive_veto / max(1, int(keep.sum())),
        "vetoed_source_counts": dict(sorted(source_counter.items())),
        "vetoed_split_counts": dict(sorted(split_counter.items())),
        "vetoed_rows": sorted(vetoed_rows, key=lambda row: (-float(row["probability"]), str(row["split"]), str(row["id"]))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, checkpoint = load_image_final_veto_checkpoint(args.checkpoint, device=device)
    config = dict(checkpoint.get("config", {}))
    image_size = int(args.image_size or config.get("image_size", 128))
    items = load_veto_items(args.dataset_json)
    probabilities, targets = predict_probabilities(
        model,
        items,
        image_size=image_size,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        include_global=bool(config.get("include_global", False)),
    )
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5))
    selected = _summarize_at_threshold(items, probabilities, threshold)
    best = choose_veto_threshold(probabilities, targets).to_dict()
    zero_positive_candidates: list[dict[str, Any]] = []
    for candidate in sorted(set(float(x) for x in np.concatenate([probabilities, [threshold, best["threshold"]]]))):
        summary = _summarize_at_threshold(items, probabilities, candidate)
        if int(summary["false_positive_veto"]) == 0:
            zero_positive_candidates.append(
                {
                    "threshold": float(candidate),
                    "true_negative_veto": int(summary["true_negative_veto"]),
                    "target_veto_total": int(summary["target_veto_total"]),
                    "neg_veto_rate": float(summary["neg_veto_rate"]),
                }
            )
    payload = {
        "dataset_json": str(args.dataset_json),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "image_size": image_size,
        "checkpoint_threshold": float(checkpoint.get("threshold", 0.5)),
        "selected_threshold_summary": selected,
        "best_threshold_on_dataset": best,
        "best_zero_positive_thresholds": sorted(
            zero_positive_candidates,
            key=lambda row: (-int(row["true_negative_veto"]), float(row["threshold"])),
        )[:20],
        "probability_rows": [
            {
                "id": item.get("id"),
                "split": item.get("split"),
                "winner_source": item.get("winner_source"),
                "target": int(item.get("target", 0)),
                "probability": float(prob),
                "pred_pixels": int(item.get("pred_pixels", 0)),
                "failure_type": item.get("failure_type"),
            }
            for item, prob in zip(items, probabilities, strict=True)
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["selected_threshold_summary", "best_threshold_on_dataset", "best_zero_positive_thresholds"]}, indent=2))
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
