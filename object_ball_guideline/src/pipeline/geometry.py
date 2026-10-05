from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
except ImportError:  # pragma: no cover - exercised only when scikit-image is absent
    _sk_skeletonize = None


def _cv_skeletonize(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
        residue = cv2.subtract(binary, opened)
        eroded = cv2.erode(binary, element)
        skeleton = cv2.bitwise_or(skeleton, residue)
        binary = eroded
        if cv2.countNonZero(binary) == 0:
            break
    return (skeleton > 0).astype(np.uint8)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    if _sk_skeletonize is not None:
        return _sk_skeletonize(mask > 0).astype(np.uint8)
    return _cv_skeletonize(mask)


def skeleton_length(mask: np.ndarray) -> float:
    return float(np.count_nonzero(skeletonize_mask(mask)))


def branch_count(skeleton: np.ndarray) -> int:
    if skeleton.dtype != np.uint8:
        skeleton = skeleton.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton, -1, kernel)
    return int(np.count_nonzero((skeleton > 0) & (neighbors >= 13)))


def connected_components(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return max(0, count - 1)


def line_segments_to_mask(
    shape: tuple[int, int],
    lines: Iterable[Iterable[Iterable[float]]],
    thickness: int = 1,
) -> np.ndarray:
    canvas = np.zeros(shape[:2], dtype=np.uint8)
    for line in lines:
        ((x0, y0), (x1, y1)) = line
        cv2.line(
            canvas,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            255,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    return (canvas > 0).astype(np.uint8)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    return cv2.dilate((mask > 0).astype(np.uint8), kernel)


def line_overlap_score(
    mask: np.ndarray, lines: list[list[list[float]]], dilation: int = 3
) -> float:
    if not lines:
        return 0.0
    skel = skeletonize_mask(mask)
    if not np.any(skel):
        return 0.0
    line_mask = line_segments_to_mask(skel.shape, lines, thickness=1)
    line_mask = dilate_mask(line_mask, dilation)
    overlap = np.count_nonzero((skel > 0) & (line_mask > 0))
    denom = max(1, np.count_nonzero(skel))
    return float(overlap / denom)


def dominant_line_length(lines: list[list[list[float]]]) -> float:
    best = 0.0
    for line in lines:
        ((x0, y0), (x1, y1)) = line
        best = max(best, math.hypot(x1 - x0, y1 - y0))
    return float(best)


def mask_principal_axis(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return 0.0, 0.0
    coords = np.column_stack([xs, ys]).astype(np.float32)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords.T)
    eigenvalues, _ = np.linalg.eigh(cov)
    eigenvalues = np.sort(np.clip(eigenvalues, a_min=1e-6, a_max=None))
    return float(eigenvalues[-1]), float(eigenvalues[0])
