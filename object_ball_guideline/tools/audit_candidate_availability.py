from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _candidate_key(candidate: dict[str, Any] | None) -> tuple[str | None, str | None, str | None]:
    if not candidate:
        return (None, None, None)
    return (
        candidate.get("candidate_source"),
        candidate.get("selector_pool") or "primary",
        candidate.get("candidate_id"),
    )


def _point_distance(a: Any, b: Any) -> float | None:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 2 or len(b) < 2:
        return None
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _box_iou(a: Any, b: Any) -> float | None:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != 4 or len(b) != 4:
        return None
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return None if union <= 0.0 else inter / union


def _pred_pixel_delta(a: Any, b: Any) -> int | None:
    if not isinstance(a, int) or not isinstance(b, int):
        return None
    return abs(a - b)


def _matching_candidate(
    target: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    if not target:
        return None, "no_wide_best_candidate"

    target_key = _candidate_key(target)
    same_key: list[dict[str, Any]] = [
        candidate for candidate in candidates if _candidate_key(candidate) == target_key
    ]
    exact_key_geometry = [
        candidate
        for candidate in same_key
        if candidate.get("crop_box") == target.get("crop_box")
        and candidate.get("candidate_center") == target.get("candidate_center")
    ]
    if exact_key_geometry:
        return exact_key_geometry[0], "exact_key_and_geometry"

    same_source_geometry = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_source") == target.get("candidate_source")
        and candidate.get("crop_box") == target.get("crop_box")
        and candidate.get("candidate_center") == target.get("candidate_center")
    ]
    if same_source_geometry:
        same_pool = [
            candidate
            for candidate in same_source_geometry
            if (candidate.get("selector_pool") or "primary")
            == (target.get("selector_pool") or "primary")
        ]
        if same_pool:
            return same_pool[0], "same_source_pool_geometry"
        return same_source_geometry[0], "same_source_geometry_different_pool"

    same_source_pool = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_source") == target.get("candidate_source")
        and (candidate.get("selector_pool") or "primary") == (target.get("selector_pool") or "primary")
    ]
    if same_source_pool:
        scored: list[tuple[float, dict[str, Any]]] = []
        for candidate in same_source_pool:
            distance = _point_distance(candidate.get("candidate_center"), target.get("candidate_center"))
            box_iou = _box_iou(candidate.get("crop_box"), target.get("crop_box"))
            pixel_delta = _pred_pixel_delta(candidate.get("pred_pixels"), target.get("pred_pixels"))
            score = 0.0
            if distance is not None:
                score += max(0.0, 32.0 - distance) / 32.0
            if box_iou is not None:
                score += box_iou
            if pixel_delta is not None:
                score += max(0.0, 64.0 - float(pixel_delta)) / 64.0
            scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate = scored[0]
        if best_score >= 1.25:
            return best_candidate, "similar_source_pool_geometry"

    if same_key:
        return same_key[0], "same_key_geometry_mismatch"

    return None, "not_found"


def _top_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("best_candidate", "top_candidates_by_iou", "top_candidates_by_score"):
        value = row.get(key)
        values = value if isinstance(value, list) else [value]
        for candidate in values:
            if not isinstance(candidate, dict):
                continue
            marker = json.dumps(
                {
                    "key": _candidate_key(candidate),
                    "center": candidate.get("candidate_center"),
                    "box": candidate.get("crop_box"),
                    "pixels": candidate.get("pred_pixels"),
                },
                sort_keys=True,
            )
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append(candidate)
    return candidates


def _availability_label(
    *,
    wide_row: dict[str, Any],
    deployed_row: dict[str, Any] | None,
    matched_candidate: dict[str, Any] | None,
    match_type: str,
    success_iou: float,
) -> str:
    if deployed_row is None:
        return "missing_deployed_diagnostic"
    deployed_best_iou = float(deployed_row.get("best_candidate_iou") or 0.0)
    deployed_failure_type = str(deployed_row.get("failure_type"))
    if matched_candidate is not None:
        matched_iou = float(matched_candidate.get("candidate_iou") or 0.0)
        if match_type == "exact_key_and_geometry" and matched_iou >= success_iou:
            return "available_exact_high_iou"
        if match_type == "same_source_geometry_different_pool" and matched_iou >= success_iou:
            return "available_equivalent_different_pool_high_iou"
        if matched_iou >= success_iou:
            return "available_similar_high_iou"
        return "candidate_present_but_low_iou_or_zero"
    if deployed_best_iou >= success_iou:
        if deployed_failure_type == "selector_or_reranker_miss":
            return "alternate_high_iou_available"
        return "alternate_high_iou_not_selector"
    if float(wide_row.get("best_candidate_iou") or 0.0) >= success_iou:
        return "diagnostic_primary32_only_or_unavailable"
    return "not_relevant_low_iou"


def audit(
    *,
    wide_diagnostics: dict[str, Any],
    deployed_diagnostics: dict[str, Any],
    success_iou: float,
    top_n: int | None,
) -> dict[str, Any]:
    deployed_by_id = {str(row["id"]): row for row in deployed_diagnostics.get("rows", [])}
    wide_rows = [
        row
        for row in wide_diagnostics.get("rows", [])
        if row.get("failure_type") == "selector_or_reranker_miss"
    ]
    wide_rows.sort(key=lambda row: float(row.get("best_candidate_iou") or 0.0), reverse=True)
    if top_n is not None:
        wide_rows = wide_rows[:top_n]

    rows: list[dict[str, Any]] = []
    for wide_row in wide_rows:
        row_id = str(wide_row["id"])
        deployed_row = deployed_by_id.get(row_id)
        deployed_candidates = _top_candidates(deployed_row or {})
        matched_candidate, match_type = _matching_candidate(
            wide_row.get("best_candidate"),
            deployed_candidates,
        )
        availability = _availability_label(
            wide_row=wide_row,
            deployed_row=deployed_row,
            matched_candidate=matched_candidate,
            match_type=match_type,
            success_iou=success_iou,
        )
        rows.append(
            {
                "id": row_id,
                "availability": availability,
                "match_type": match_type,
                "wide_failure_type": wide_row.get("failure_type"),
                "deployed_failure_type": deployed_row.get("failure_type") if deployed_row else None,
                "wide_best_candidate_iou": wide_row.get("best_candidate_iou"),
                "deployed_best_candidate_iou": (
                    deployed_row.get("best_candidate_iou") if deployed_row else None
                ),
                "wide_final_pred_pixels": wide_row.get("final_pred_pixels"),
                "deployed_final_pred_pixels": (
                    deployed_row.get("final_pred_pixels") if deployed_row else None
                ),
                "wide_best_candidate": wide_row.get("best_candidate"),
                "deployed_best_candidate": (
                    deployed_row.get("best_candidate") if deployed_row else None
                ),
                "matched_deployed_candidate": matched_candidate,
                "wide_winner": wide_row.get("winner"),
                "deployed_winner": deployed_row.get("winner") if deployed_row else None,
            }
        )

    counts = Counter(row["availability"] for row in rows)
    deployed_failure_counts = Counter(
        str(row["deployed_failure_type"]) for row in rows if row.get("deployed_failure_type")
    )
    return {
        "wide_diagnostics": wide_diagnostics.get("count"),
        "deployed_diagnostics": deployed_diagnostics.get("count"),
        "audited_rows": len(rows),
        "success_iou": success_iou,
        "availability_counts": dict(sorted(counts.items())),
        "deployed_failure_counts_for_audited_rows": dict(sorted(deployed_failure_counts.items())),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-diagnostics", required=True)
    parser.add_argument("--deployed-diagnostics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--success-iou", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=None)
    args = parser.parse_args()

    wide = json.loads(Path(args.wide_diagnostics).read_text(encoding="utf-8"))
    deployed = json.loads(Path(args.deployed_diagnostics).read_text(encoding="utf-8"))
    result = audit(
        wide_diagnostics=wide,
        deployed_diagnostics=deployed,
        success_iou=args.success_iou,
        top_n=args.top_n,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
