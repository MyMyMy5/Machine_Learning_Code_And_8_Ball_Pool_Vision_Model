from __future__ import annotations

import math

import cv2
import numpy as np

from src.pipeline.geometry import (
    branch_count,
    connected_components,
    mask_principal_axis,
    skeletonize_mask,
)


def ball_boundary_mask(ball_mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (thickness * 2 + 1, thickness * 2 + 1)
    )
    dilated = cv2.dilate((ball_mask > 0).astype(np.uint8), kernel)
    eroded = cv2.erode((ball_mask > 0).astype(np.uint8), kernel)
    return ((dilated > 0) & (eroded == 0)).astype(np.uint8)


def compute_mask_features(
    mask: np.ndarray,
    ball_mask: np.ndarray,
    ball_center: tuple[int, int],
    ball_radius: int,
    min_length_px: int = 18,
) -> dict[str, float]:
    binary = (mask > 0).astype(np.uint8)
    area = float(np.count_nonzero(binary))
    if area == 0:
        return {
            "area": 0.0,
            "skeleton_length": 0.0,
            "average_width": 0.0,
            "length_to_width_ratio": 0.0,
            "thinness": 0.0,
            "line_likeness": 0.0,
            "branch_count": 0.0,
            "component_count": 0.0,
            "continuity": 0.0,
            "ball_boundary_touch": 0.0,
            "ball_fill_fraction": 0.0,
            "connected_to_ball": 0.0,
            "outward_extension": 0.0,
            "shortness_penalty": 1.0,
            "blob_penalty": 1.0,
        }

    skeleton = skeletonize_mask(binary)
    skeleton_len = float(np.count_nonzero(skeleton))
    avg_width = float(area / max(skeleton_len, 1.0))
    length_to_width = float(skeleton_len / max(avg_width, 1e-6))
    thinness = float(np.clip((6.0 - avg_width) / 6.0, 0.0, 1.0))
    major, minor = mask_principal_axis(binary)
    axis_ratio = major / max(minor, 1e-6)
    line_likeness = float(
        np.clip(max(length_to_width / 25.0, axis_ratio / 20.0), 0.0, 1.0)
    )
    branches = float(branch_count(skeleton))
    components = float(connected_components(binary))
    continuity = float(
        np.clip(
            skeleton_len
            / max(skeleton_len + branches * 6.0 + max(components - 1.0, 0.0) * 10.0, 1.0),
            0.0,
            1.0,
        )
    )

    boundary = ball_boundary_mask(ball_mask, thickness=2)
    boundary_touch = float(
        np.count_nonzero((binary > 0) & (boundary > 0))
        / max(1, np.count_nonzero(boundary))
    )
    yy, xx = np.ogrid[: binary.shape[0], : binary.shape[1]]
    core_radius = max(2, int(round(ball_radius * 0.35)))
    core_mask = ((xx - ball_center[0]) ** 2 + (yy - ball_center[1]) ** 2 <= core_radius**2)
    ball_core_overlap = float(
        np.count_nonzero((binary > 0) & core_mask) / max(1, np.count_nonzero(core_mask))
    )
    ball_fill_fraction = float(
        np.count_nonzero((binary > 0) & (ball_mask > 0)) / max(area, 1.0)
    )
    connected_to_ball = (
        1.0
        if np.count_nonzero(
            (binary > 0)
            & (cv2.dilate(ball_mask, np.ones((5, 5), np.uint8)) > 0)
        )
        else 0.0
    )

    ys, xs = np.where(binary > 0)
    distances = np.sqrt((xs - ball_center[0]) ** 2 + (ys - ball_center[1]) ** 2)
    outward_extension = float(
        np.clip(
            (distances.max(initial=0.0) - ball_radius)
            / max(binary.shape[0], binary.shape[1], 1),
            0.0,
            1.0,
        )
    )
    shortness_penalty = float(
        np.clip(1.0 - (skeleton_len / max(float(min_length_px), 1.0)), 0.0, 1.0)
    )
    blob_penalty = float(np.clip(avg_width / 10.0, 0.0, 1.0))

    return {
        "area": area,
        "skeleton_length": skeleton_len,
        "average_width": avg_width,
        "length_to_width_ratio": length_to_width,
        "thinness": thinness,
        "line_likeness": line_likeness,
        "branch_count": branches,
        "component_count": components,
        "continuity": continuity,
        "ball_boundary_touch": boundary_touch,
        "ball_core_overlap": ball_core_overlap,
        "ball_fill_fraction": ball_fill_fraction,
        "connected_to_ball": connected_to_ball,
        "outward_extension": outward_extension,
        "shortness_penalty": shortness_penalty,
        "blob_penalty": blob_penalty,
    }


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return float(intersection / union) if union else 0.0


def line_distance_to_ball_center(
    line: list[list[float]], center: tuple[int, int]
) -> float:
    (x0, y0), (x1, y1) = line
    px, py = center
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0 and dy == 0:
        return float(math.hypot(px - x0, py - y0))
    t = ((px - x0) * dx + (py - y0) * dy) / max(dx * dx + dy * dy, 1e-6)
    t = min(1.0, max(0.0, t))
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    return float(math.hypot(px - proj_x, py - proj_y))
