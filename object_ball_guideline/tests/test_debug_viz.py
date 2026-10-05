from __future__ import annotations

import numpy as np

from src.pipeline.debug_viz import overlay_mask


def test_overlay_mask_only_changes_masked_pixels() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:] = [10, 10, 10]
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    overlay = overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5)
    assert np.array_equal(overlay[0, 0], image[0, 0])
    assert not np.array_equal(overlay[6, 6], image[6, 6])
