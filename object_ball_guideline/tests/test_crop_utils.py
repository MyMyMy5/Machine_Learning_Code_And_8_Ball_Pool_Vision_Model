from __future__ import annotations

import numpy as np

from src.pipeline.crop_utils import build_crop_candidate, extract_crop, local_ball_mask


def test_extract_crop_preserves_candidate_center() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    candidate = build_crop_candidate(
        candidate_id="crop_00",
        center_x=60,
        center_y=40,
        radius=10,
        image_shape=image.shape[:2],
        crop_scale=2.0,
        padding_px=4,
        source="test",
    )
    crop, meta = extract_crop(image, candidate, upscale_factor=2.0)
    assert crop.shape[:2] == tuple(meta["resize_size"])
    assert meta["local_center"] == [candidate.local_center()[0] * 2, candidate.local_center()[1] * 2]


def test_local_ball_mask_contains_center() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    candidate = build_crop_candidate(
        candidate_id="crop_00",
        center_x=40,
        center_y=40,
        radius=8,
        image_shape=image.shape[:2],
        crop_scale=2.0,
        padding_px=0,
        source="test",
    )
    _, meta = extract_crop(image, candidate)
    ball_mask = local_ball_mask(meta)
    cx, cy = meta["local_center"]
    assert ball_mask[cy, cx] == 1
