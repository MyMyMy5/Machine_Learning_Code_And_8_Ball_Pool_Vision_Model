from __future__ import annotations

import numpy as np

from src.stages.propose_ball_crops import _detect_table_roi, generate_ball_candidates


def test_detect_table_roi_excludes_right_ui_on_dark_table() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:] = (18, 28, 48)

    # Table felt
    image[120:680, 120:1160] = (52, 52, 52)

    # Decorative rails around the felt
    image[90:120, 90:1190] = (180, 60, 40)
    image[680:710, 90:1190] = (180, 60, 40)
    image[120:680, 90:120] = (180, 60, 40)
    image[120:680, 1160:1190] = (180, 60, 40)

    # Right-side UI with bright circular elements that should not enter the ROI.
    image[110:660, 1195:1275] = (38, 96, 160)
    yy, xx = np.ogrid[:720, :1280]
    for cx, cy, radius in [(1235, 160, 26), (1235, 300, 24), (1235, 520, 22)]:
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        image[circle] = (240, 240, 240)

    x0, y0, x1, y1 = _detect_table_roi(image)
    assert x0 <= 145
    assert y0 <= 150
    assert x1 > 1120
    assert y1 > 650
    assert x1 < 1195


def test_colored_blob_candidates_can_recover_saturated_object_ball() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (42, 42, 42)
    yy, xx = np.ogrid[:240, :320]
    ball = (xx - 170) ** 2 + (yy - 120) ** 2 <= 12**2
    image[ball] = (235, 210, 28)

    candidates = generate_ball_candidates(
        image,
        {
            "max_ball_candidates": 10,
            "crop_scale": 4.25,
            "crop_padding_px": 24,
            "colored_blob_candidates": True,
            "hough": {
                "dp": 1.2,
                "param1": 110,
                "param2": 80,
                "min_radius": 30,
                "max_radius": 80,
            },
            "fallback": {"grid_candidates": 0},
        },
    )

    assert any(
        candidate.source == "colored_blob"
        and abs(candidate.center_x - 170) <= 2
        and abs(candidate.center_y - 120) <= 2
        for candidate in candidates
    )


def test_line_endpoint_candidates_add_thin_white_line_endpoints() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (42, 42, 42)
    image[120:123, 130:201] = (242, 242, 242)

    candidates = generate_ball_candidates(
        image,
        {
            "max_ball_candidates": 10,
            "crop_scale": 4.25,
            "crop_padding_px": 24,
            "line_endpoint_candidates": True,
            "line_endpoint": {
                "min_gray": 180,
                "max_saturation": 80,
                "min_length": 20,
                "radius": 18,
            },
            "hough": {
                "dp": 1.2,
                "param1": 110,
                "param2": 80,
                "min_radius": 30,
                "max_radius": 80,
            },
            "fallback": {"grid_candidates": 0},
        },
    )

    endpoint_candidates = [candidate for candidate in candidates if candidate.source == "line_endpoint"]
    assert len(endpoint_candidates) >= 2
    assert any(abs(candidate.center_x - 130) <= 3 and abs(candidate.center_y - 121) <= 3 for candidate in endpoint_candidates)
    assert any(abs(candidate.center_x - 200) <= 3 and abs(candidate.center_y - 121) <= 3 for candidate in endpoint_candidates)


def test_line_endpoint_candidates_can_expand_roi_for_near_rail_targets() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (18, 28, 48)
    image[60:180, 80:240] = (30, 95, 150)
    image[55:60, 70:250] = (170, 80, 30)
    image[180:185, 70:250] = (170, 80, 30)
    image[60:180, 70:80] = (170, 80, 30)
    image[60:180, 240:250] = (170, 80, 30)

    # A short outgoing target fragment just beyond the detected felt ROI edge.
    image[116:119, 245:259] = (242, 242, 242)

    base_candidates = generate_ball_candidates(
        image,
        {
            "max_ball_candidates": 20,
            "crop_scale": 4.25,
            "crop_padding_px": 24,
            "line_endpoint_candidates": True,
            "line_endpoint": {
                "min_gray": 180,
                "max_saturation": 80,
                "min_length": 8,
                "radius": 10,
            },
            "hough": {
                "dp": 1.2,
                "param1": 110,
                "param2": 80,
                "min_radius": 30,
                "max_radius": 80,
            },
            "fallback": {"grid_candidates": 0},
        },
    )
    expanded_candidates = generate_ball_candidates(
        image,
        {
            "max_ball_candidates": 20,
            "crop_scale": 4.25,
            "crop_padding_px": 24,
            "line_endpoint_candidates": True,
            "line_endpoint": {
                "min_gray": 180,
                "max_saturation": 80,
                "min_length": 8,
                "radius": 10,
                "roi_expand_px": 24,
            },
            "hough": {
                "dp": 1.2,
                "param1": 110,
                "param2": 80,
                "min_radius": 30,
                "max_radius": 80,
            },
            "fallback": {"grid_candidates": 0},
        },
    )

    assert not any(
        candidate.source == "line_endpoint" and candidate.center_x >= 244
        for candidate in base_candidates
    )
    assert any(
        candidate.source == "line_endpoint"
        and abs(candidate.center_x - 245) <= 3
        and abs(candidate.center_y - 117) <= 3
        for candidate in expanded_candidates
    )
