from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.pipeline.crop_utils import CropCandidate, build_crop_candidate
from src.pipeline.io_utils import ensure_dir, save_rgb_image


def _odd_kernel(size: int) -> int:
    return size if size % 2 == 1 else size + 1


def _roi_from_mask(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    min_area_fraction: float = 0.15,
    prefer_center: bool = False,
) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image_shape[:2]
    if not contours:
        return None
    center = (width // 2, height // 2)
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_fraction * width * height:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        contains_center = cv2.pointPolygonTest(contour, center, False) >= 0
        score = area + (width * height if prefer_center and contains_center else 0.0)
        if best is None or score > best[0]:
            best = (score, (x, y, w, h))
    if best is None:
        return None
    x, y, w, h = best[1]
    inset_x = max(6, int(round(w * 0.02)))
    inset_y = max(6, int(round(h * 0.02)))
    return (
        max(0, x + inset_x),
        max(0, y + inset_y),
        min(width, x + w - inset_x),
        min(height, y + h - inset_y),
    )


def _detect_table_roi_hsv(image: np.ndarray) -> tuple[int, int, int, int] | None:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = (
        (hsv[..., 0] >= 75)
        & (hsv[..., 0] <= 115)
        & (hsv[..., 1] >= 40)
        & (hsv[..., 2] >= 70)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    return _roi_from_mask(mask, image.shape[:2], min_area_fraction=0.15)


def _detect_table_roi_center_color(image: np.ndarray) -> tuple[int, int, int, int] | None:
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    sample_x0 = int(round(width * 0.28))
    sample_y0 = int(round(height * 0.28))
    sample_x1 = int(round(width * 0.72))
    sample_y1 = int(round(height * 0.72))
    sample = lab[sample_y0:sample_y1, sample_x0:sample_x1].reshape(-1, 3)
    if sample.size == 0:
        return None
    felt = np.median(sample, axis=0)
    distance = np.linalg.norm(lab - felt, axis=2)
    sample_distance = distance[sample_y0:sample_y1, sample_x0:sample_x1]
    threshold = max(14.0, float(np.percentile(sample_distance, 80)) * 1.35 + 6.0)

    mask = (distance <= threshold).astype(np.uint8) * 255
    side = min(height, width)
    open_k = _odd_kernel(max(3, int(round(side * 0.004))))
    close_k = _odd_kernel(max(21, int(round(side * 0.02))))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

    roi = _roi_from_mask(mask, image.shape[:2], min_area_fraction=0.2, prefer_center=True)
    if roi is None:
        return None
    x0, y0, x1, y1 = roi
    area_fraction = ((x1 - x0) * (y1 - y0)) / max(width * height, 1)
    if area_fraction > 0.92:
        return None
    return roi


def _detect_table_roi(image: np.ndarray) -> tuple[int, int, int, int]:
    # The table felt color is not stable across captures, so prefer a central-color ROI
    # and fall back to the older cyan felt heuristic when needed.
    roi = _detect_table_roi_center_color(image)
    if roi is not None:
        return roi
    roi = _detect_table_roi_hsv(image)
    if roi is not None:
        return roi
    height, width = image.shape[:2]
    return (0, 0, width, height)


def _inside_roi(center_x: int, center_y: int, roi: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = roi
    return x0 <= center_x <= x1 and y0 <= center_y <= y1


def _expand_roi(
    roi: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    margin_px: int,
) -> tuple[int, int, int, int]:
    if margin_px <= 0:
        return roi
    x0, y0, x1, y1 = roi
    height, width = image_shape[:2]
    return (
        max(0, x0 - margin_px),
        max(0, y0 - margin_px),
        min(width, x1 + margin_px),
        min(height, y1 + margin_px),
    )


def _compute_non_felt_mask(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    inner = lab[y0:y1, x0:x1].reshape(-1, 3)
    felt = np.median(inner, axis=0)
    distance = np.linalg.norm(lab - felt, axis=2)
    return (distance > 18.0).astype(np.uint8)


def _attached_white_score(image: np.ndarray, center_x: int, center_y: int, radius: int) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    white = ((hsv[..., 1] < 80) & (hsv[..., 2] > 150)).astype(np.uint8)
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    ball = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= (radius * 0.85) ** 2
    near = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= (radius * 2.2) ** 2
    attached = np.count_nonzero(white & near & ~ball)
    denom = max(1, np.count_nonzero(near & ~ball))
    return float(attached / denom)


def _detect_global_reticle(
    image: np.ndarray, roi: tuple[int, int, int, int]
) -> tuple[int, int, int] | None:
    x, y, x1, y1 = roi
    table = image[y:y1, x:x1]
    gray = cv2.cvtColor(table, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    hsv = cv2.cvtColor(table, cv2.COLOR_RGB2HSV)
    white = ((hsv[..., 1] < 80) & (hsv[..., 2] > 150)).astype(np.uint8)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=12,
        param1=80,
        param2=12,
        minRadius=6,
        maxRadius=22,
    )
    if circles is None:
        return None
    yy, xx = np.ogrid[: table.shape[0], : table.shape[1]]
    best: tuple[float, tuple[int, int, int]] | None = None
    for cx, cy, radius in np.round(circles[0]).astype(int):
        annulus = (
            ((xx - cx) ** 2 + (yy - cy) ** 2 <= (radius + 2) ** 2)
            & ((xx - cx) ** 2 + (yy - cy) ** 2 >= max(radius - 2, 1) ** 2)
        )
        inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= max(radius - 4, 2) ** 2
        ring_white = float(np.mean(white[annulus])) if np.any(annulus) else 0.0
        center_value = float(np.mean(gray[inner])) / 255.0 if np.any(inner) else 1.0
        score = ring_white + 0.3 * (1.0 - center_value)
        if best is None or score > best[0]:
            best = (score, (int(cx + x), int(cy + y), int(radius)))
    return best[1] if best is not None else None


def _reticle_axis_support_score(
    image: np.ndarray,
    ring_center: tuple[int, int],
    ball_center: tuple[int, int],
    ball_radius: int,
) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    white = ((hsv[..., 1] < 90) & (hsv[..., 2] > 150)).astype(np.uint8)
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    rcx, rcy = ring_center
    bcx, bcy = ball_center
    vector = np.array([bcx - rcx, bcy - rcy], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0:
        return 0.0
    unit = vector / norm
    rel_x = xx - rcx
    rel_y = yy - rcy
    t = rel_x * unit[0] + rel_y * unit[1]
    d = np.abs(rel_x * unit[1] - rel_y * unit[0])
    band = (
        (d <= max(2.5, ball_radius * 0.28))
        & (t >= 0)
        & (t <= norm + ball_radius * 1.2)
    )
    core = (xx - bcx) ** 2 + (yy - bcy) ** 2 <= max(2, int(round(ball_radius * 0.7))) ** 2
    return float(np.mean(white[band])) + 0.7 * float(np.mean((white & core)[core]))


def _circle_ball_score(
    image: np.ndarray,
    non_felt_mask: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
) -> float:
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    circle_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    if not np.any(circle_mask):
        return 0.0
    inner_radius = max(3, int(round(radius * 0.45)))
    inner_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= inner_radius**2
    non_felt_fraction = float(non_felt_mask[circle_mask].mean())
    inner_non_felt_fraction = float(non_felt_mask[inner_mask].mean())
    if inner_non_felt_fraction < 0.25:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    patch = gray[circle_mask]
    if patch.size == 0:
        return inner_non_felt_fraction
    contrast = float(np.clip(np.std(patch) / 64.0, 0.0, 1.0))
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    inner_sat = hsv[..., 1][inner_mask]
    inner_val = hsv[..., 2][inner_mask]
    sat_score = float(np.clip(np.median(inner_sat) / 255.0, 0.0, 1.0))
    if np.median(inner_sat) < 80 and np.mean(inner_val) < 145:
        return 0.0
    cue_like_penalty = 0.18 if np.median(inner_sat) < 120 and np.mean(inner_val) > 150 else 0.0
    score = (
        0.55 * inner_non_felt_fraction
        + 0.2 * non_felt_fraction
        + 0.15 * contrast
        + 0.10 * sat_score
        - cue_like_penalty
    )
    return float(np.clip(score, 0.0, 1.0))


def _dedupe_circles(circles: list[tuple[int, int, int, float]]) -> list[tuple[int, int, int, float]]:
    deduped: list[tuple[int, int, int, float]] = []
    for cx, cy, radius, score in sorted(circles, key=lambda item: item[3], reverse=True):
        keep = True
        for ecx, ecy, eradius, _ in deduped:
            if (cx - ecx) ** 2 + (cy - ecy) ** 2 < max(radius, eradius, 1) ** 2:
                keep = False
                break
        if keep:
            deduped.append((cx, cy, radius, score))
    return deduped


def _detect_hough_circles(
    image: np.ndarray,
    config: dict,
    roi: tuple[int, int, int, int],
    non_felt_mask: np.ndarray | None = None,
) -> list[tuple[int, int, int, float]]:
    x0, y0, x1, y1 = roi
    table = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(table, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    hough_cfg = config["hough"]
    if non_felt_mask is None:
        non_felt_mask = _compute_non_felt_mask(image, roi)
    radius_est = max(
        float(hough_cfg["min_radius"]),
        min(table.shape[:2]) / 36.0,
    )
    min_radius = max(int(round(radius_est * 0.65)), int(hough_cfg["min_radius"]))
    max_radius = min(int(round(radius_est * 1.5)), int(hough_cfg["max_radius"]))
    min_dist = max(int(round(radius_est * 1.8)), int(hough_cfg["min_radius"] * 2))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=float(hough_cfg["dp"]),
        minDist=min_dist,
        param1=float(hough_cfg["param1"]),
        param2=float(hough_cfg["param2"]),
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    raw_results: list[tuple[int, int, int, float]] = []
    if circles is not None:
        for cx, cy, radius in np.round(circles[0]).astype(int):
            global_x = int(cx + x0)
            global_y = int(cy + y0)
            if not _inside_roi(global_x, global_y, roi):
                continue
            score = _circle_ball_score(
                image,
                non_felt_mask,
                global_x,
                global_y,
                int(radius),
            )
            if score < 0.55:
                continue
            raw_results.append((global_x, global_y, int(radius), score))
    results = _dedupe_circles(raw_results)
    supported: list[tuple[int, int, int, float]] = []
    for cx, cy, radius, score in results:
        attached_white = _attached_white_score(image, cx, cy, radius)
        neighbor_count = sum(
            1
            for ox, oy, oradius, _ in results
            if (ox, oy, oradius) != (cx, cy, radius)
            and 0.5 * radius < np.hypot(ox - cx, oy - cy) < 2.5 * radius
            and 0.4 * radius <= oradius <= 1.6 * radius
        )
        boosted = (
            0.6 * score
            + 0.2 * min(1.0, attached_white * 8.0)
            + 0.2 * min(1.0, neighbor_count / 4.0)
        )
        supported.append((cx, cy, radius, float(np.clip(boosted, 0.0, 1.0))))
    return supported


def _detect_white_blob_circles(
    image: np.ndarray, roi: tuple[int, int, int, int]
) -> list[tuple[int, int, int, float]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] < 80) & (hsv[..., 2] > 170)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results: list[tuple[int, int, int, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if not _inside_roi(int(cx), int(cy), roi):
            continue
        white_score = min(1.0, area / max(np.pi * radius * radius, 1.0))
        results.append((int(cx), int(cy), int(radius), float(white_score)))
    return _dedupe_circles(results)


def _detect_colored_blob_circles(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    non_felt_mask: np.ndarray,
    config: dict | None = None,
) -> list[tuple[int, int, int, float]]:
    cfg = config or {}
    min_saturation = int(cfg.get("min_saturation", 120))
    min_value = int(cfg.get("min_value", 120))
    min_area = float(cfg.get("min_area", 30.0))
    max_area = float(cfg.get("max_area", 2500.0))
    min_radius = int(cfg.get("min_radius", 5))
    max_radius = int(cfg.get("max_radius", 35))
    min_score = float(cfg.get("min_score", 0.55))
    min_output_radius = int(cfg.get("min_output_radius", 8))
    score_cap = cfg.get("score_cap")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] > min_saturation) & (hsv[..., 2] > min_value)).astype(np.uint8) * 255
    roi_mask = np.zeros_like(mask)
    x0, y0, x1, y1 = roi
    roi_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results: list[tuple[int, int, int, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        (cx_float, cy_float), radius_float = cv2.minEnclosingCircle(contour)
        center_x = int(round(cx_float))
        center_y = int(round(cy_float))
        radius = int(round(radius_float))
        if radius < min_radius or radius > max_radius:
            continue
        if not _inside_roi(center_x, center_y, roi):
            continue
        score = _circle_ball_score(image, non_felt_mask, center_x, center_y, max(radius, 8))
        if score < min_score:
            continue
        output_score = min(score, float(score_cap)) if score_cap is not None else score
        results.append((center_x, center_y, max(radius, min_output_radius, 8), output_score))
    return _dedupe_circles(results)


def _detect_line_endpoint_circles(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    config: dict | None = None,
) -> list[tuple[int, int, int, float]]:
    cfg = config or {}
    min_gray = int(cfg.get("min_gray", 170))
    max_saturation = int(cfg.get("max_saturation", 70))
    min_area = int(cfg.get("min_area", 6))
    max_area = int(cfg.get("max_area", 1400))
    min_length = float(cfg.get("min_length", 8.0))
    max_width = float(cfg.get("max_width", 18.0))
    min_aspect = float(cfg.get("min_aspect", 2.0))
    radius = int(cfg.get("radius", 20))
    include_midpoint = bool(cfg.get("include_midpoint", False))

    x0, y0, x1, y1 = roi
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    roi_mask = np.zeros(gray.shape, dtype=np.uint8)
    roi_mask[y0:y1, x0:x1] = 1
    white = ((gray >= min_gray) & (hsv[..., 1] <= max_saturation) & (roi_mask > 0)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    results: list[tuple[int, int, int, float]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        box_long = float(max(width, height))
        box_short = float(max(min(width, height), 1))
        if box_long < min_length or box_short > max_width or (box_long / box_short) < min_aspect:
            continue

        ys, xs = np.nonzero(labels == label)
        if len(xs) < 2:
            continue
        coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        centered = coords - coords.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        projections = centered @ axis
        line_length = float(projections.max() - projections.min())
        if line_length < min_length:
            continue
        estimated_width = float(area / max(line_length, 1.0))
        if estimated_width > max_width:
            continue

        endpoint_indices = [int(np.argmin(projections)), int(np.argmax(projections))]
        centers = [(int(round(coords[index, 0])), int(round(coords[index, 1]))) for index in endpoint_indices]
        if include_midpoint:
            midpoint = coords.mean(axis=0)
            centers.append((int(round(midpoint[0])), int(round(midpoint[1]))))

        aspect_score = min(box_long / max(box_short, 1.0), 12.0) / 12.0
        length_score = min(line_length / 120.0, 1.0)
        score = float(np.clip(0.25 + 0.35 * aspect_score + 0.35 * length_score, 0.05, 0.95))
        for cx, cy in centers:
            if _inside_roi(cx, cy, roi):
                results.append((cx, cy, max(radius, 8), score))
    return _dedupe_circles(results)


def _fallback_grid_candidates(
    image: np.ndarray, config: dict, roi: tuple[int, int, int, int]
) -> list[tuple[int, int, int, float]]:
    grid_count = int(config["fallback"].get("grid_candidates", 6))
    height, width = image.shape[:2]
    radius = max(12, min(height, width) // 12)
    x0, y0, x1, y1 = roi
    xs = np.linspace(x0 + (x1 - x0) * 0.15, x0 + (x1 - x0) * 0.85, num=max(2, grid_count // 2))
    ys = np.linspace(y0 + (y1 - y0) * 0.2, y0 + (y1 - y0) * 0.8, num=2)
    candidates: list[tuple[int, int, int, float]] = []
    for idx, (cx, cy) in enumerate([(x, y) for y in ys for x in xs]):
        if idx >= grid_count:
            break
        candidates.append((int(cx), int(cy), radius, 0.05))
    return candidates


def generate_ball_candidates(
    image: np.ndarray, config: dict, output_dir: Path | None = None
) -> list[CropCandidate]:
    roi = _detect_table_roi(image)
    candidates: list[tuple[int, int, int, float, str]] = []
    non_felt_mask = _compute_non_felt_mask(image, roi)
    hough_balls = _detect_hough_circles(image, config, roi, non_felt_mask=non_felt_mask)
    for cx, cy, radius, score in hough_balls:
        candidates.append((cx, cy, radius, score, "table_hough"))

    reticle = _detect_global_reticle(image, roi)
    if reticle is not None:
        rcx, rcy, rr = reticle
        nearby = [
            (cx, cy, radius, score)
            for cx, cy, radius, score in hough_balls
            if 0.7 * rr < np.hypot(cx - rcx, cy - rcy) < 2.8 * rr
        ]
        nearby = sorted(
            nearby,
            key=lambda item: _reticle_axis_support_score(
                image, (rcx, rcy), (item[0], item[1]), item[2]
            ),
            reverse=True,
        )
        for rank, (cx, cy, radius, score) in enumerate(nearby[:3]):
            axis_score = _reticle_axis_support_score(image, (rcx, rcy), (cx, cy), radius)
            candidates.append(
                (
                    cx,
                    cy,
                    radius,
                    float(np.clip(0.7 * score + 0.6 * axis_score, 0.0, 1.0)),
                    f"reticle_global_{rank}",
                )
            )
    if len(candidates) < max(4, int(config.get("max_ball_candidates", 10)) // 2):
        for cx, cy, radius, score in _detect_white_blob_circles(image, roi):
            candidates.append((cx, cy, radius, score, "white_blob"))
    if bool(config.get("colored_blob_candidates", False)):
        for cx, cy, radius, score in _detect_colored_blob_circles(
            image, roi, non_felt_mask, config.get("colored_blob")
        ):
            candidates.append((cx, cy, radius, score, "colored_blob"))
    if bool(config.get("line_endpoint_candidates", False)):
        line_endpoint_cfg = config.get("line_endpoint") or {}
        line_endpoint_roi = _expand_roi(
            roi,
            image.shape[:2],
            int(line_endpoint_cfg.get("roi_expand_px", 0)) if isinstance(line_endpoint_cfg, dict) else 0,
        )
        for cx, cy, radius, score in _detect_line_endpoint_circles(
            image, line_endpoint_roi, line_endpoint_cfg
        ):
            candidates.append((cx, cy, radius, score, "line_endpoint"))
    if not candidates:
        for cx, cy, radius, score in _fallback_grid_candidates(image, config, roi):
            candidates.append((cx, cy, radius, score, "grid"))

    deduped = _dedupe_circles(
        [(cx, cy, radius, score) for cx, cy, radius, score, _ in candidates]
    )
    source_map = {(cx, cy, radius): source for cx, cy, radius, _, source in candidates}
    scored = sorted(deduped, key=lambda item: item[3], reverse=True)
    max_count = int(config.get("max_ball_candidates", 10))
    crop_scale = float(config.get("crop_scale", 4.25))
    padding_px = int(config.get("crop_padding_px", 24))
    results: list[CropCandidate] = []
    for index, (cx, cy, radius, score) in enumerate(scored[:max_count]):
        results.append(
            build_crop_candidate(
                candidate_id=f"crop_{index:02d}",
                center_x=cx,
                center_y=cy,
                radius=max(radius, 8),
                image_shape=image.shape[:2],
                crop_scale=crop_scale,
                padding_px=padding_px,
                source=source_map.get((cx, cy, radius), "unknown"),
                score=score,
                metadata={
                    "white_score": score,
                    "table_roi": list(roi),
                    "reticle_center": [reticle[0], reticle[1]] if reticle is not None else None,
                    "reticle_radius": reticle[2] if reticle is not None else None,
                },
            )
        )

    if not results:
        height, width = image.shape[:2]
        results.append(
            build_crop_candidate(
                candidate_id="crop_00",
                center_x=width // 2,
                center_y=height // 2,
                radius=max(12, min(height, width) // 10),
                image_shape=image.shape[:2],
                crop_scale=1.8,
                padding_px=0,
                source="scene_fallback",
                score=0.0,
                metadata={"table_roi": list(roi)},
            )
        )

    if output_dir is not None:
        ensure_dir(output_dir)
        for candidate in results:
            x0, y0, x1, y1 = candidate.crop_box
            save_rgb_image(
                output_dir / f"{candidate.candidate_id}_preview.png",
                image[y0:y1, x0:x1],
            )
    return results
