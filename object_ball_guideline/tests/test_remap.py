from __future__ import annotations

import numpy as np

from src.pipeline.crop_utils import remap_mask_to_image


def test_remap_mask_to_image_handles_upscaled_crop() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 8:12] = 1
    crop_meta = {
        "crop_box": [10, 12, 20, 22],
        "crop_size": [10, 10],
        "resize_size": [20, 20],
    }
    remapped = remap_mask_to_image(mask, crop_meta, (40, 40, 3))
    assert remapped.shape == (40, 40)
    assert remapped[15:20, 14:16].sum() > 0
