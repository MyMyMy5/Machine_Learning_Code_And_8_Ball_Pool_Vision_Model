from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stages.propose_ball_crops import _detect_table_roi


def _normalize_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    if os.name != "nt" and path_str.startswith("C:\\"):
        return Path("/mnt/c/" + path_str[3:].replace("\\", "/"))
    if os.name == "nt" and path_str.startswith("/mnt/c/"):
        return Path("C:/" + path_str[len("/mnt/c/") :])
    return Path(path_str)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def _candidate_circles(report: dict[str, Any]) -> list[tuple[float, float, float, str]]:
    circles: list[tuple[float, float, float, str]] = []
    for candidate in report.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        center = candidate.get("candidate_center")
        radius = candidate.get("candidate_radius")
        if (
            isinstance(center, list)
            and len(center) == 2
            and isinstance(center[0], (int, float))
            and isinstance(center[1], (int, float))
            and isinstance(radius, (int, float))
        ):
            circles.append(
                (
                    float(center[0]),
                    float(center[1]),
                    max(1.0, float(radius)),
                    str(candidate.get("candidate_source")),
                )
            )
    winner = report.get("winner", {})
    if isinstance(winner, dict):
        center = winner.get("candidate_center")
        radius = winner.get("candidate_radius")
        if (
            isinstance(center, list)
            and len(center) == 2
            and isinstance(center[0], (int, float))
            and isinstance(center[1], (int, float))
            and isinstance(radius, (int, float))
        ):
            circles.append(
                (
                    float(center[0]),
                    float(center[1]),
                    max(1.0, float(radius)),
                    str(winner.get("candidate_source")),
                )
            )
    return circles


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _winner_feature(report: dict[str, Any], key: str) -> float | None:
    winner = report.get("winner", {})
    if not isinstance(winner, dict):
        return None
    features = winner.get("features", {})
    if not isinstance(features, dict):
        return None
    return _safe_float(features.get(key))


def _winner_flags(report: dict[str, Any]) -> list[str]:
    winner = report.get("winner", {})
    if not isinstance(winner, dict):
        return []
    return sorted(
        key.removeprefix("selected_via_")
        for key, value in winner.items()
        if key.startswith("selected_via_") and bool(value)
    )


def _mask_geometry(mask: np.ndarray) -> dict[str, float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return {
            "mask_bbox_w": 0.0,
            "mask_bbox_h": 0.0,
            "mask_bbox_area": 0.0,
            "mask_bbox_aspect": 0.0,
            "mask_centroid_x": -1.0,
            "mask_centroid_y": -1.0,
            "mask_component_count": 0.0,
            "mask_largest_component_fraction": 0.0,
        }
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox_w = float(x1 - x0 + 1)
    bbox_h = float(y1 - y0 + 1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)
    largest = float(component_areas.max(initial=0))
    pixels = float(xs.size)
    return {
        "mask_bbox_w": bbox_w,
        "mask_bbox_h": bbox_h,
        "mask_bbox_area": float(bbox_w * bbox_h),
        "mask_bbox_aspect": float(max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1.0)),
        "mask_centroid_x": float(xs.mean()),
        "mask_centroid_y": float(ys.mean()),
        "mask_component_count": float(max(0, num_labels - 1)),
        "mask_largest_component_fraction": float(largest / max(pixels, 1.0)),
    }


def _context_features(image: np.ndarray, mask: np.ndarray, report: dict[str, Any]) -> dict[str, float | int | str | None]:
    height, width = mask.shape[:2]
    pred_pixels = int(np.count_nonzero(mask))
    features: dict[str, float | int | str | None] = {
        "image_w": int(width),
        "image_h": int(height),
        "pred_pixels": pred_pixels,
    }
    features.update(_mask_geometry(mask))
    if pred_pixels <= 0:
        return features

    mask_bool = mask > 0
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rgb_on_mask = image[mask_bool].astype(np.float32)
    hsv_on_mask = hsv[mask_bool].astype(np.float32)
    white = ((hsv[..., 1] < 80) & (hsv[..., 2] > 150))
    features.update(
        {
            "mask_mean_r": float(rgb_on_mask[:, 0].mean()),
            "mask_mean_g": float(rgb_on_mask[:, 1].mean()),
            "mask_mean_b": float(rgb_on_mask[:, 2].mean()),
            "mask_mean_saturation": float(hsv_on_mask[:, 1].mean()),
            "mask_mean_value": float(hsv_on_mask[:, 2].mean()),
            "mask_white_fraction": float(np.mean(white[mask_bool])),
        }
    )

    try:
        x0, y0, x1, y1 = _detect_table_roi(image)
    except Exception:
        x0, y0, x1, y1 = (0, 0, width, height)
    in_table = np.zeros(mask.shape[:2], dtype=bool)
    in_table[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
    features.update(
        {
            "table_roi_x0": float(x0),
            "table_roi_y0": float(y0),
            "table_roi_x1": float(x1),
            "table_roi_y1": float(y1),
            "table_roi_area_fraction": float(((x1 - x0) * (y1 - y0)) / max(width * height, 1)),
            "mask_fraction_in_table_roi": float(np.count_nonzero(mask_bool & in_table) / max(pred_pixels, 1)),
        }
    )

    circles = _candidate_circles(report)
    features["candidate_count"] = len(circles)
    if not circles:
        features.update(
            {
                "min_candidate_center_distance": None,
                "min_candidate_boundary_distance": None,
                "min_abs_candidate_boundary_distance": None,
                "min_candidate_boundary_distance_norm": None,
                "mask_fraction_inside_any_candidate": 0.0,
                "mask_fraction_near_any_candidate": 0.0,
                "mask_fraction_on_candidate_boundary_band": 0.0,
                "nearest_candidate_source": None,
            }
        )
        return features

    ys, xs = np.where(mask_bool)
    point_x = xs.astype(np.float32)
    point_y = ys.astype(np.float32)
    any_inside = np.zeros(mask.shape[:2], dtype=bool)
    any_near = np.zeros(mask.shape[:2], dtype=bool)
    any_boundary = np.zeros(mask.shape[:2], dtype=bool)
    yy, xx = np.ogrid[:height, :width]
    best_boundary: tuple[float, float, float, str] | None = None
    best_center_distance: float | None = None
    for cx, cy, radius, source in circles:
        center_distance_pixels = np.sqrt((point_x - cx) ** 2 + (point_y - cy) ** 2)
        min_center = float(center_distance_pixels.min(initial=np.inf))
        min_boundary = float(min_center - radius)
        abs_boundary = abs(min_boundary)
        if best_boundary is None or abs_boundary < best_boundary[0]:
            best_boundary = (abs_boundary, min_boundary, min_boundary / max(radius, 1.0), source)
        if best_center_distance is None or min_center < best_center_distance:
            best_center_distance = min_center
        dist_grid_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        radius_sq = radius**2
        near_radius = radius + max(6.0, radius * 0.45)
        boundary_pad = max(4.0, radius * 0.25)
        any_inside |= dist_grid_sq <= radius_sq
        any_near |= dist_grid_sq <= near_radius**2
        any_boundary |= (dist_grid_sq >= max(1.0, radius - boundary_pad) ** 2) & (
            dist_grid_sq <= (radius + boundary_pad) ** 2
        )

    assert best_boundary is not None
    features.update(
        {
            "min_candidate_center_distance": float(best_center_distance if best_center_distance is not None else 0.0),
            "min_candidate_boundary_distance": float(best_boundary[1]),
            "min_abs_candidate_boundary_distance": float(best_boundary[0]),
            "min_candidate_boundary_distance_norm": float(best_boundary[2]),
            "mask_fraction_inside_any_candidate": float(np.count_nonzero(mask_bool & any_inside) / max(pred_pixels, 1)),
            "mask_fraction_near_any_candidate": float(np.count_nonzero(mask_bool & any_near) / max(pred_pixels, 1)),
            "mask_fraction_on_candidate_boundary_band": float(np.count_nonzero(mask_bool & any_boundary) / max(pred_pixels, 1)),
            "nearest_candidate_source": best_boundary[3],
        }
    )
    return features


def _row_output_paths(row: dict[str, Any]) -> tuple[Path | None, Path | None]:
    output_dir = _normalize_path(row.get("output_dir"))
    if output_dir is None:
        return None, None
    return output_dir / "report.json", output_dir / "mask_final.png"


def audit_summary(summary_json: Path, output_json: Path) -> dict[str, Any]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    audited_rows: list[dict[str, Any]] = []
    missing = Counter()
    for row in payload.get("rows", []):
        report_path, mask_path = _row_output_paths(row)
        image_path = _normalize_path(row.get("image_path"))
        if image_path is None or not image_path.exists():
            missing["image"] += 1
            continue
        if report_path is None or not report_path.exists():
            missing["report"] += 1
            continue
        if mask_path is None or not mask_path.exists():
            missing["mask"] += 1
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        image = _load_rgb(image_path)
        mask = _load_mask(mask_path)
        context = _context_features(image, mask, report)
        winner = report.get("winner", {})
        if not isinstance(winner, dict):
            winner = {}
        audited = dict(row)
        audited.update(
            {
                "report_path": str(report_path),
                "mask_final_path": str(mask_path),
                "winner_flags_full": _winner_flags(report),
                "winner_feature_area": _winner_feature(report, "area"),
                "winner_feature_skeleton_length": _winner_feature(report, "skeleton_length"),
                "winner_feature_average_width": _winner_feature(report, "average_width"),
                "winner_feature_ball_fill_fraction": _winner_feature(report, "ball_fill_fraction"),
                "winner_feature_connected_to_ball": _winner_feature(report, "connected_to_ball"),
                "winner_feature_outward_extension": _winner_feature(report, "outward_extension"),
                "pre_final_fallback_score": _safe_float(winner.get("pre_final_fallback_score")),
            }
        )
        audited.update(context)
        audited_rows.append(audited)
    result = {
        "summary_json": str(summary_json),
        "row_count": len(payload.get("rows", [])),
        "audited_count": len(audited_rows),
        "missing": dict(missing),
        "rows": audited_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    result = audit_summary(Path(args.summary_json), Path(args.output_json))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
