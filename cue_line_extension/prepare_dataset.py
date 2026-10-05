"""Convert raw pool videos into training data with pseudo-label masks.

Usage example:
    python prepare_dataset.py data/raw_videos data/images data/masks \
        --frame-stride 5 --min-line-count 1

Put downloaded videos (mp4/avi/mkv) in the raw_videos directory before running.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
from typing import Iterable

import cv2 as cv
import numpy as np


LOW_WHITE = np.array([0, 0, 245], dtype=np.uint8)
HIGH_WHITE = np.array([180, 30, 255], dtype=np.uint8)
BRIGHT_MIN, BRIGHT_MAX = 200, 255
STRUCTURING_KERNEL = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
HOUGH_PARAMS = dict(rho=1, theta=np.pi / 180, threshold=40, minLineLength=15, maxLineGap=15)
MIN_LINE_LENGTH = 15
MAX_LINE_LENGTH = 100
MIN_LINE_ANGLE = 15
MAX_LINE_ANGLE = 85
LINE_EXTENSION_THICKNESS = 6




def find_cue_ball(frame_bgr: np.ndarray) -> tuple[int, int] | None:
    """Approximate the cue ball centre by isolating bright circular blobs."""
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    mask_white = cv.inRange(hsv, LOW_WHITE, HIGH_WHITE)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    opened = cv.morphologyEx(mask_white, cv.MORPH_OPEN, kernel, iterations=1)
    opened = cv.GaussianBlur(opened, (9, 9), 0)

    contours, _ = cv.findContours(opened, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best_center: tuple[int, int] | None = None
    best_score = 0.0

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 80 or area > 2000:
            continue
        (x, y), radius = cv.minEnclosingCircle(cnt)
        if radius <= 0:
            continue
        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity > best_score:
            best_score = circularity
            best_center = (int(round(x)), int(round(y)))

    return best_center


def _point_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    seg_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if seg_len_sq == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return math.hypot(px - proj_x, py - proj_y)



def point_saturation(frame_bgr: np.ndarray, x: int, y: int, radius: int = 6) -> float:
    h, w = frame_bgr.shape[:2]
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) / 255.0

def extend_segment_along_mask(x1: int, y1: int, x2: int, y2: int, mask: np.ndarray, thickness: int = LINE_EXTENSION_THICKNESS, margin_factor: float = 1.0, min_points: int = 20) -> tuple[int, int, int, int]:
    height, width = mask.shape[:2]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return x1, y1, x2, y2

    dir_x = dx / length
    dir_y = dy / length
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    margin = int(max(10, length * margin_factor))
    x_min = max(0, min(x1, x2) - margin)
    x_max = min(width, max(x1, x2) + margin)
    y_min = max(0, min(y1, y2) - margin)
    y_max = min(height, max(y1, y2) + margin)
    if x_min >= x_max or y_min >= y_max:
        return x1, y1, x2, y2

    region = mask[y_min:y_max, x_min:x_max]
    ys, xs = np.where(region > 0)
    if ys.size < min_points:
        return x1, y1, x2, y2

    xs = xs.astype(np.float32) + x_min
    ys = ys.astype(np.float32) + y_min
    px = xs - cx
    py = ys - cy

    perp = np.abs(px * (-dir_y) + py * dir_x)
    line_points_mask = perp <= thickness
    if not np.any(line_points_mask):
        return x1, y1, x2, y2

    proj = px[line_points_mask] * dir_x + py[line_points_mask] * dir_y
    if proj.size == 0:
        return x1, y1, x2, y2

    start_proj = proj.min()
    end_proj = proj.max()
    if end_proj - start_proj < length * 0.25:
        return x1, y1, x2, y2

    start_point = (cx + start_proj * dir_x, cy + start_proj * dir_y)
    end_point = (cx + end_proj * dir_x, cy + end_proj * dir_y)

    def clamp_point(px_val: float, py_val: float) -> tuple[int, int]:
        xi = int(round(px_val))
        yi = int(round(py_val))
        xi = max(0, min(width - 1, xi))
        yi = max(0, min(height - 1, yi))
        return xi, yi

    def project_to_border(x0: float, y0: float, vx: float, vy: float) -> tuple[int, int]:
        eps = 1e-6
        candidates: list[tuple[float, float, float]] = []
        if abs(vx) > eps:
            for xb in (0, width - 1):
                t = (xb - x0) / vx
                y = y0 + t * vy
                if t >= 0 and 0 <= y <= height - 1:
                    candidates.append((t, xb, y))
        if abs(vy) > eps:
            for yb in (0, height - 1):
                t = (yb - y0) / vy
                x = x0 + t * vx
                if t >= 0 and 0 <= x <= width - 1:
                    candidates.append((t, x, yb))
        if not candidates:
            return clamp_point(x0, y0)
        t, x, y = min(candidates, key=lambda c: c[0])
        return clamp_point(x, y)

    start_x, start_y = project_to_border(start_point[0], start_point[1], -dir_x, -dir_y)
    end_x, end_y = project_to_border(end_point[0], end_point[1], dir_x, dir_y)

    return start_x, start_y, end_x, end_y


def build_white_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    mask_white = cv.inRange(hsv, LOW_WHITE, HIGH_WHITE)

    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    mask_bright = cv.inRange(gray, BRIGHT_MIN, BRIGHT_MAX)
    mask_white = cv.bitwise_and(mask_white, mask_bright)

    mask_closed = cv.morphologyEx(mask_white, cv.MORPH_CLOSE, STRUCTURING_KERNEL, iterations=1)
    return mask_closed


def detect_line_mask(frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    mask = build_white_mask(frame_bgr)
    edges = cv.Canny(mask, 50, 150)
    line_segments = cv.HoughLinesP(edges, **HOUGH_PARAMS)

    line_mask = np.zeros(mask.shape, dtype=np.uint8)
    best_score = float('-inf')
    best_line = None

    cue_center = find_cue_ball(frame_bgr)

    if line_segments is not None:
        for (x1, y1, x2, y2) in line_segments[:, 0]:
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            angle = abs(math.degrees(math.atan2(dy, dx)))
            angle = angle if angle <= 90 else 180 - angle

            if not (MIN_LINE_LENGTH <= length <= MAX_LINE_LENGTH):
                continue
            if not (MIN_LINE_ANGLE <= angle <= MAX_LINE_ANGLE):
                continue

            ex1, ey1, ex2, ey2 = extend_segment_along_mask(x1, y1, x2, y2, mask)
            extended_length = math.hypot(ex2 - ex1, ey2 - ey1)

            dist_penalty = 0.0
            if cue_center is not None:
                dist_penalty = _point_segment_distance(cue_center[0], cue_center[1], ex1, ey1, ex2, ey2)

            sat1 = point_saturation(frame_bgr, ex1, ey1, radius=4)
            sat2 = point_saturation(frame_bgr, ex2, ey2, radius=4)

            if cue_center is not None:
                dist_penalty = _point_segment_distance(cue_center[0], cue_center[1], ex1, ey1, ex2, ey2)
                # reject lines that stay very close to the cue ball in a straight stub
                if dist_penalty < 4.0 and extended_length < 60:
                    continue
            else:
                dist_penalty = 0.0

            color_bonus = max(sat1, sat2)
            if color_bonus < 0.25:
                continue

            score = 0.2 * extended_length + 0.2 * length + 50.0 * color_bonus - 0.3 * dist_penalty
            if score > best_score:
                best_score = score
                best_line = (int(ex1), int(ey1), int(ex2), int(ey2))

    if best_line is not None:
        cv.line(line_mask, (best_line[0], best_line[1]), (best_line[2], best_line[3]), 255, 6, cv.LINE_AA)
        return line_mask, 1

    return line_mask, 0


def sanitize_stem(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_-]+", "_", name)
    sanitized = sanitized.strip("_")
    return sanitized or "clip"


def iter_videos(raw_dir: Path) -> Iterable[Path]:
    supported = (".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv")
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() in supported:
            yield path


def process_video(path: Path, images_dir: Path, masks_dir: Path, stride: int, min_line_count: int, store_overlay: bool) -> None:
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[WARN] Failed to open {path}")
        return

    base_name = sanitize_stem(path.stem)
    frame_idx = 0
    saved = 0

    overlay_dir = images_dir.parent / "overlays" if store_overlay else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        mask, line_count = detect_line_mask(frame)
        if line_count < min_line_count:
            frame_idx += 1
            continue

        frame_name = f"{base_name}_{frame_idx:06d}.png"
        image_path = images_dir / frame_name
        mask_path = masks_dir / frame_name

        cv.imwrite(str(image_path), frame)
        cv.imwrite(str(mask_path), mask)

        if overlay_dir is not None:
            overlay = frame.copy()
            overlay_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            overlay = cv.addWeighted(overlay, 0.7, overlay_mask, 0.6, 0)
            cv.imwrite(str(overlay_dir / frame_name), overlay)

        saved += 1
        frame_idx += 1

    cap.release()
    safe_name = sanitize_stem(path.stem)
    print(f"Processed {safe_name}: kept {saved} frames")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames and pseudo-label masks from pool videos")
    parser.add_argument("raw_dir", type=Path, help="Directory containing downloaded videos")
    parser.add_argument("images_dir", type=Path, help="Directory to save extracted frames")
    parser.add_argument("masks_dir", type=Path, help="Directory to save generated masks")
    parser.add_argument("--frame-stride", type=int, default=5, help="Keep every Nth frame (default: 5)")
    parser.add_argument("--min-line-count", type=int, default=1, help="Skip frames with fewer detected lines")
    parser.add_argument("--store-overlay", action="store_true", help="Also save colored overlays for inspection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.images_dir.mkdir(parents=True, exist_ok=True)
    args.masks_dir.mkdir(parents=True, exist_ok=True)

    videos = list(iter_videos(args.raw_dir))
    if not videos:
        print(f"No supported video files found in {args.raw_dir}")
        return

    for video_path in videos:
        process_video(video_path, args.images_dir, args.masks_dir, args.frame_stride, args.min_line_count, args.store_overlay)


if __name__ == "__main__":
    main()
