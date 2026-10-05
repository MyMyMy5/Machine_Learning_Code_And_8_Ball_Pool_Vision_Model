from __future__ import annotations

import numpy as np

from src.pipeline.mask_postprocess import cleanup_mask


def test_cleanup_keeps_component_connected_to_ball() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:15, 10:25] = 1
    mask[30:33, 30:33] = 1
    ball_mask = np.zeros((40, 40), dtype=np.uint8)
    ball_mask[10:15, 10:15] = 1
    cleaned = cleanup_mask(
        mask,
        {
            "morphology_open_radius": 0,
            "morphology_close_radius": 0,
            "min_component_area": 4,
            "keep_only_best_component": True,
        },
        ball_mask=ball_mask,
    )
    assert cleaned[12, 12] == 1
    assert cleaned[31, 31] == 0
