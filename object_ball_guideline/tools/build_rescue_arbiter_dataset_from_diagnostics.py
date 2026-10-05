from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[object, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = (
            candidate.get("candidate_id"),
            candidate.get("candidate_source"),
            candidate.get("selector_pool"),
            candidate.get("pred_pixels"),
            round(float(candidate.get("score", 0.0)), 6)
            if isinstance(candidate.get("score"), (int, float))
            else None,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _candidate_key(candidate: dict[str, Any]) -> tuple[object, object]:
    return candidate.get("candidate_id"), candidate.get("selector_pool")


def _candidate_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    best = row.get("best_candidate")
    if isinstance(best, dict):
        candidates.append(best)
    for key in ("top_candidates_by_iou", "top_candidates_by_score"):
        values = row.get(key)
        if isinstance(values, list):
            candidates.extend(candidate for candidate in values if isinstance(candidate, dict))
    return _dedupe_candidates(candidates)


def build_dataset(
    diagnostics_paths: list[Path],
    *,
    positive_iou: float,
    negative_iou: float,
    min_improvement: float,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for diagnostics_path in diagnostics_paths:
        payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            raise ValueError(f"{diagnostics_path} does not contain a rows list")
        for row in rows:
            if not isinstance(row, dict):
                continue
            current = row.get("winner")
            if not isinstance(current, dict):
                continue
            current_key = _candidate_key(current)
            final_iou = float(row.get("final_iou", 0.0) or 0.0)
            for candidate in _candidate_items(row):
                if _candidate_key(candidate) == current_key:
                    continue
                candidate_iou = float(candidate.get("candidate_iou", 0.0) or 0.0)
                if candidate_iou >= positive_iou and candidate_iou >= final_iou + min_improvement:
                    target = 1
                elif candidate_iou <= negative_iou:
                    target = 0
                else:
                    continue
                items.append(
                    {
                        "id": str(row.get("id", "")),
                        "diagnostics_path": str(diagnostics_path),
                        "failure_type": row.get("failure_type"),
                        "split_source": row.get("split_source"),
                        "kind": row.get("kind"),
                        "final_iou": final_iou,
                        "candidate_iou": candidate_iou,
                        "target": target,
                        "current": current,
                        "candidate": candidate,
                    }
                )
    target_count = sum(int(item["target"]) for item in items)
    return {
        "diagnostics_paths": [str(path) for path in diagnostics_paths],
        "count": len(items),
        "target_positive_count": target_count,
        "target_negative_count": len(items) - target_count,
        "positive_iou": positive_iou,
        "negative_iou": negative_iou,
        "min_improvement": min_improvement,
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-json", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive-iou", type=float, default=0.5)
    parser.add_argument("--negative-iou", type=float, default=0.05)
    parser.add_argument("--min-improvement", type=float, default=0.25)
    args = parser.parse_args()

    dataset = build_dataset(
        [Path(path) for path in args.diagnostics_json],
        positive_iou=args.positive_iou,
        negative_iou=args.negative_iou,
        min_improvement=args.min_improvement,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: dataset[key]
                for key in ["count", "target_positive_count", "target_negative_count"]
            },
            indent=2,
        )
    )
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
