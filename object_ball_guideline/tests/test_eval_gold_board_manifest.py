from __future__ import annotations

import numpy as np

from tools.eval_gold_board_manifest import (
    _compute_iou_and_dice,
    _failure_type,
    _write_mask_only_output,
    summarize_rows,
)


def test_compute_iou_and_dice_treats_empty_prediction_and_empty_gt_as_perfect() -> None:
    pred = np.zeros((3, 3), dtype=np.uint8)
    gt = np.zeros((3, 3), dtype=np.uint8)

    iou, dice = _compute_iou_and_dice(pred, gt)

    assert iou == 1.0
    assert dice == 1.0


def test_failure_type_separates_negative_false_positive_from_true_negative() -> None:
    assert _failure_type(gt_pixels=0, pred_pixels=0, iou=1.0, success_iou=0.5) == "true_negative"
    assert (
        _failure_type(gt_pixels=0, pred_pixels=12, iou=0.0, success_iou=0.5)
        == "negative_false_positive"
    )


def test_summarize_rows_reports_positive_and_negative_gates() -> None:
    rows = [
        {
            "gt_pixels": 20,
            "pred_pixels": 20,
            "iou": 0.8,
            "dice": 0.9,
            "failure_type": "success",
            "winner_source": "table_hough",
            "winner_flags": {"fallback": True},
        },
        {
            "gt_pixels": 15,
            "pred_pixels": 0,
            "iou": 0.0,
            "dice": 0.1,
            "failure_type": "positive_no_prediction",
            "winner_source": "reticle_global_0",
            "winner_flags": {},
        },
        {
            "gt_pixels": 0,
            "pred_pixels": 0,
            "iou": 1.0,
            "dice": 1.0,
            "failure_type": "true_negative",
            "winner_source": "table_hough",
            "winner_flags": {"final_fallback": True},
        },
        {
            "gt_pixels": 0,
            "pred_pixels": 10,
            "iou": 0.0,
            "dice": 0.0,
            "failure_type": "negative_false_positive",
            "winner_source": "white_blob",
            "winner_flags": {},
        },
    ]

    summary = summarize_rows(rows)

    assert summary["count"] == 4
    assert summary["positive_count"] == 2
    assert summary["negative_count"] == 2
    assert summary["positive_zero_iou"] == 1
    assert summary["true_negative"] == 1
    assert summary["negative_false_positive"] == 1
    assert summary["negative_false_positive_rate"] == 0.5
    assert summary["negative_false_positive_source_counts"] == {"white_blob": 1}
    assert summary["winner_flag_counts"] == {"fallback": 1, "final_fallback": 1}


def test_write_mask_only_output_creates_parent_directory(tmp_path) -> None:
    output_path = tmp_path / "split" / "item" / "mask_final.png"
    mask = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)

    _write_mask_only_output(mask, output_path)

    assert output_path.exists()
