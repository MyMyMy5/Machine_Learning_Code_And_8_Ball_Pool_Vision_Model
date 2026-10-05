from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _mask_path_for_row(row: dict[str, Any]) -> Path:
    return Path(str(row["output_dir"])) / "mask_final.png"


def _is_veto_target(row: dict[str, Any]) -> bool:
    return int(row.get("gt_pixels", 0)) <= 0 and int(row.get("pred_pixels", 0)) > 0


def _is_keep_target(row: dict[str, Any], *, min_keep_iou: float) -> bool:
    return (
        int(row.get("gt_pixels", 0)) > 0
        and int(row.get("pred_pixels", 0)) > 0
        and float(row.get("iou", 0.0)) >= float(min_keep_iou)
    )


def build_dataset(*, summary: dict[str, Any], sources: set[str] | None, min_keep_iou: float = 0.0) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    missing_masks: list[dict[str, str]] = []
    skipped_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    for row in summary.get("rows", []):
        if not isinstance(row, dict):
            continue
        winner_source = str(row.get("winner_source", ""))
        if sources is not None and winner_source not in sources:
            skipped_counts["source_filtered"] += 1
            continue
        if _is_veto_target(row):
            target = 1
        elif _is_keep_target(row, min_keep_iou=min_keep_iou):
            target = 0
        else:
            skipped_counts["empty_or_unusable"] += 1
            continue
        mask_path = _mask_path_for_row(row)
        if not mask_path.exists():
            missing_masks.append({"split": str(row.get("split")), "id": str(row.get("id")), "mask_path": str(mask_path)})
            continue
        item = {
            "id": str(row["id"]),
            "split": str(row["split"]),
            "kind": row.get("kind"),
            "source": row.get("source"),
            "image_path": str(row["image_path"]),
            "mask_path": str(mask_path),
            "winner_source": winner_source,
            "winner_id": row.get("winner_id"),
            "winner_score": row.get("winner_score"),
            "gt_pixels": int(row.get("gt_pixels", 0)),
            "pred_pixels": int(row.get("pred_pixels", 0)),
            "iou": row.get("iou"),
            "dice": row.get("dice"),
            "failure_type": row.get("failure_type"),
            "target": target,
        }
        items.append(item)
        source_counts[winner_source] += 1
        target_counts[str(target)] += 1

    return {
        "summary_json": summary.get("summary_json"),
        "sources": sorted(sources) if sources is not None else None,
        "min_keep_iou": float(min_keep_iou),
        "count": len(items),
        "target_veto_count": int(target_counts.get("1", 0)),
        "target_keep_count": int(target_counts.get("0", 0)),
        "source_counts": dict(sorted(source_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "missing_mask_count": len(missing_masks),
        "missing_masks": missing_masks[:100],
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument(
        "--min-keep-iou",
        type=float,
        default=0.0,
        help="Only positive rows with IoU at least this value are treated as keep examples.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["summary_json"] = str(summary_path)
    dataset = build_dataset(
        summary=summary,
        sources={str(source) for source in args.sources} if args.sources else None,
        min_keep_iou=float(args.min_keep_iou),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: dataset[key]
                for key in [
                    "count",
                    "target_veto_count",
                    "target_keep_count",
                    "source_counts",
                    "missing_mask_count",
                    "skipped_counts",
                ]
            },
            indent=2,
        )
    )
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
