from __future__ import annotations

import cv2
import numpy as np


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    output = image.copy().astype(np.float32)
    color_arr = np.asarray(color, dtype=np.float32)
    binary = mask > 0
    output[binary] = output[binary] * (1.0 - alpha) + color_arr * alpha
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def draw_ball_candidates(image: np.ndarray, candidates: list) -> np.ndarray:
    output = image.copy()
    for candidate in candidates:
        center = (int(candidate.center_x), int(candidate.center_y))
        cv2.circle(output, center, int(candidate.radius), (0, 255, 255), 2)
        x0, y0, x1, y1 = candidate.crop_box
        cv2.rectangle(output, (x0, y0), (x1, y1), (255, 255, 0), 1)
        cv2.putText(
            output,
            candidate.candidate_id,
            (x0, max(16, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def draw_prompt_overlay(
    image: np.ndarray,
    box: list[int] | None = None,
    positive_points: list[list[float]] | None = None,
    negative_points: list[list[float]] | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    output = image.copy()
    if mask is not None:
        output = overlay_mask(output, mask, color=(0, 200, 0), alpha=0.35)
    if box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        cv2.rectangle(output, (x0, y0), (x1, y1), (255, 255, 0), 2)
    for point in positive_points or []:
        cv2.circle(
            output,
            (int(round(point[0])), int(round(point[1]))),
            4,
            (0, 255, 0),
            -1,
        )
    for point in negative_points or []:
        cv2.circle(
            output,
            (int(round(point[0])), int(round(point[1]))),
            4,
            (255, 0, 0),
            -1,
        )
    return output


def draw_lines_overlay(image: np.ndarray, lines: list[list[list[float]]]) -> np.ndarray:
    output = image.copy()
    for line in lines:
        (x0, y0), (x1, y1) = line
        cv2.line(
            output,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            (255, 128, 0),
            2,
            lineType=cv2.LINE_AA,
        )
    return output
