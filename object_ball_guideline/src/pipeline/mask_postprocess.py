from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from src.pipeline.geometry import skeletonize_mask


def _kernel(radius: int) -> np.ndarray:
    radius = max(int(radius), 0)
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )


def keep_best_component(mask: np.ndarray, ball_mask: np.ndarray | None = None) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    best_index = 1
    best_score = -1.0
    dilated_ball = None
    if ball_mask is not None:
        dilated_ball = cv2.dilate(
            (ball_mask > 0).astype(np.uint8), np.ones((5, 5), dtype=np.uint8)
        )
    for label_index in range(1, count):
        component = (labels == label_index).astype(np.uint8)
        area = float(stats[label_index, cv2.CC_STAT_AREA])
        score = area
        if dilated_ball is not None:
            score += 10_000.0 * float(
                np.count_nonzero((component > 0) & (dilated_ball > 0))
            )
        if score > best_score:
            best_score = score
            best_index = label_index
    return (labels == best_index).astype(np.uint8)


def cleanup_mask(mask: np.ndarray, cleanup_cfg: dict, ball_mask: np.ndarray | None = None) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    open_radius = int(cleanup_cfg.get("morphology_open_radius", 0))
    close_radius = int(cleanup_cfg.get("morphology_close_radius", 0))
    if open_radius > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _kernel(open_radius))
    if close_radius > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _kernel(close_radius))
    min_area = int(cleanup_cfg.get("min_component_area", 0))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for label_index in range(1, count):
        if stats[label_index, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label_index] = 1
    if cleanup_cfg.get("keep_only_best_component", True):
        filtered = keep_best_component(filtered, ball_mask=ball_mask)
    return filtered.astype(np.uint8)


def prune_to_outgoing_branch(
    mask: np.ndarray, ball_center: tuple[int, int], ball_radius: int
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize_mask(binary)
    points = np.argwhere(skeleton > 0)
    if len(points) < 3:
        return binary

    point_set = {tuple(point) for point in points}

    def neighbors(point: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = point
        result = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                candidate = (y + dy, x + dx)
                if candidate in point_set:
                    result.append(candidate)
        return result

    degrees = {point: len(neighbors(point)) for point in point_set}
    branchpoints = [point for point, degree in degrees.items() if degree >= 3]
    endpoints = [point for point, degree in degrees.items() if degree == 1]
    if len(endpoints) < 1:
        return binary

    ball_yx = (int(round(ball_center[1])), int(round(ball_center[0])))
    if branchpoints:
        branch = min(
            branchpoints,
            key=lambda point: (point[0] - ball_yx[0]) ** 2 + (point[1] - ball_yx[1]) ** 2,
        )
    else:
        branch = min(
            point_set,
            key=lambda point: (point[0] - ball_yx[0]) ** 2 + (point[1] - ball_yx[1]) ** 2,
        )

    previous: dict[tuple[int, int], tuple[int, int] | None] = {branch: None}
    queue: deque[tuple[int, int]] = deque([branch])
    while queue:
        point = queue.popleft()
        for neighbor in neighbors(point):
            if neighbor not in previous:
                previous[neighbor] = point
                queue.append(neighbor)

    def path_to(endpoint: tuple[int, int]) -> list[tuple[int, int]]:
        if endpoint not in previous:
            return []
        path = []
        current: tuple[int, int] | None = endpoint
        while current is not None:
            path.append(current)
            current = previous[current]
        return path

    endpoint_paths = {endpoint: path_to(endpoint) for endpoint in endpoints}
    endpoint_paths = {endpoint: path for endpoint, path in endpoint_paths.items() if path}
    if len(endpoint_paths) < 1:
        return binary

    ball_radius = max(int(ball_radius), 4)

    def endpoint_score(endpoint: tuple[int, int]) -> tuple[int, float, int]:
        path = endpoint_paths[endpoint]
        dists = [
            float(np.hypot(point[1] - ball_center[0], point[0] - ball_center[1]))
            for point in path
        ]
        overlap = sum(distance <= ball_radius * 0.9 for distance in dists)
        min_dist = min(dists) if dists else float("inf")
        return (overlap, -min_dist, len(path))

    forward_endpoint = max(endpoint_paths, key=endpoint_score)
    best_overlap = endpoint_score(forward_endpoint)[0]
    if best_overlap == 0:
        return binary

    kept = np.zeros_like(binary)
    for point in endpoint_paths[forward_endpoint]:
        kept[point[0], point[1]] = 1
    kept = cv2.line(
        kept,
        (int(round(ball_center[0])), int(round(ball_center[1]))),
        (int(branch[1]), int(branch[0])),
        1,
        thickness=1,
        lineType=cv2.LINE_8,
    )
    avg_width = max(float(np.count_nonzero(binary)) / max(np.count_nonzero(skeleton), 1), 1.0)
    radius = max(1, int(round(avg_width / 2.0)))
    pruned = cv2.dilate(kept, _kernel(radius))
    return ((pruned > 0) & ((binary > 0) | (kept > 0))).astype(np.uint8)


def soft_alpha_mask(mask: np.ndarray, softness_px: int = 3) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if softness_px <= 0:
        return binary.astype(np.float32)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if float(distance.max()) == 0.0:
        return binary.astype(np.float32)
    alpha = np.clip(distance / float(max(softness_px, 1)), 0.0, 1.0)
    return alpha.astype(np.float32)
