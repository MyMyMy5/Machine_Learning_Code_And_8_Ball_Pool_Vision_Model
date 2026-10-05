from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class CropCandidate:
    candidate_id: str
    center_x: int
    center_y: int
    radius: int
    crop_box: tuple[int, int, int, int]
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def local_center(self) -> tuple[int, int]:
        x0, y0, _, _ = self.crop_box
        return self.center_x - x0, self.center_y - y0


def clamp_box(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    return (
        max(0, x0),
        max(0, y0),
        min(width, x1),
        min(height, y1),
    )


def square_crop_box(
    center_x: int,
    center_y: int,
    radius: int,
    crop_scale: float,
    padding_px: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    half = int(round(radius * crop_scale + padding_px))
    return clamp_box(
        center_x - half,
        center_y - half,
        center_x + half,
        center_y + half,
        image_shape,
    )


def build_crop_candidate(
    candidate_id: str,
    center_x: int,
    center_y: int,
    radius: int,
    image_shape: tuple[int, int],
    crop_scale: float,
    padding_px: int,
    source: str,
    score: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> CropCandidate:
    crop_box = square_crop_box(
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        crop_scale=crop_scale,
        padding_px=padding_px,
        image_shape=image_shape,
    )
    return CropCandidate(
        candidate_id=candidate_id,
        center_x=int(center_x),
        center_y=int(center_y),
        radius=int(radius),
        crop_box=crop_box,
        source=source,
        score=float(score),
        metadata=metadata or {},
    )


def extract_crop(
    image: np.ndarray,
    candidate: CropCandidate,
    upscale_factor: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    x0, y0, x1, y1 = candidate.crop_box
    crop = image[y0:y1, x0:x1].copy()
    crop_h, crop_w = crop.shape[:2]
    if upscale_factor and abs(upscale_factor - 1.0) > 1e-6:
        resized = cv2.resize(
            crop,
            (
                max(1, int(round(crop_w * upscale_factor))),
                max(1, int(round(crop_h * upscale_factor))),
            ),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        resized = crop
    local_center = candidate.local_center()
    meta = {
        "candidate_id": candidate.candidate_id,
        "crop_box": [x0, y0, x1, y1],
        "crop_size": [crop_h, crop_w],
        "resize_size": [int(resized.shape[0]), int(resized.shape[1])],
        "upscale_factor": float(upscale_factor or 1.0),
        "local_center": [
            int(round(local_center[0] * (upscale_factor or 1.0))),
            int(round(local_center[1] * (upscale_factor or 1.0))),
        ],
        "local_radius": int(round(candidate.radius * (upscale_factor or 1.0))),
        "source": candidate.source,
        "candidate_score": candidate.score,
        "candidate_metadata": candidate.metadata,
    }
    return resized, meta


def local_ball_mask(crop_meta: dict[str, Any], padding: int = 0) -> np.ndarray:
    height, width = crop_meta["resize_size"]
    center_x, center_y = crop_meta["local_center"]
    radius = crop_meta["local_radius"] + padding
    yy, xx = np.ogrid[:height, :width]
    return (
        (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    ).astype(np.uint8)


def remap_mask_to_image(
    mask: np.ndarray, crop_meta: dict[str, Any], image_shape: tuple[int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = crop_meta["crop_box"]
    crop_h, crop_w = crop_meta["crop_size"]
    resize_h, resize_w = crop_meta["resize_size"]
    binary = (mask > 0).astype(np.uint8)
    if (resize_h, resize_w) != (crop_h, crop_w):
        binary = cv2.resize(
            binary, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
        )
        binary = (binary > 0).astype(np.uint8)
    full = np.zeros(image_shape[:2], dtype=np.uint8)
    full[y0:y1, x0:x1] = np.maximum(
        full[y0:y1, x0:x1], binary[: y1 - y0, : x1 - x0]
    )
    return full


def remap_points_to_image(
    points: list[list[float]] | np.ndarray, crop_meta: dict[str, Any]
) -> list[list[int]]:
    if len(points) == 0:
        return []
    x0, y0, _, _ = crop_meta["crop_box"]
    crop_h, crop_w = crop_meta["crop_size"]
    resize_h, resize_w = crop_meta["resize_size"]
    scale_x = crop_w / max(resize_w, 1)
    scale_y = crop_h / max(resize_h, 1)
    remapped: list[list[int]] = []
    for point_x, point_y in np.asarray(points):
        remapped.append(
            [
                int(round(x0 + point_x * scale_x)),
                int(round(y0 + point_y * scale_y)),
            ]
        )
    return remapped
