from __future__ import annotations

from tools.diagnose_guideline_failures import classify_failure


def test_classify_candidate_generation_miss_when_no_crop_contains_target() -> None:
    assert (
        classify_failure(
            gt_pixels=100,
            final_iou=0.0,
            final_pred_pixels=0,
            best_candidate_gt_pixels=3,
            best_candidate_iou=0.0,
            candidate_recall_pixels=12,
            success_iou=0.5,
        )
        == "candidate_generation_miss"
    )


def test_classify_segmentation_miss_when_crop_contains_target_but_no_mask_is_good() -> None:
    assert (
        classify_failure(
            gt_pixels=100,
            final_iou=0.1,
            final_pred_pixels=50,
            best_candidate_gt_pixels=90,
            best_candidate_iou=0.2,
            candidate_recall_pixels=12,
            success_iou=0.5,
        )
        == "segmentation_miss"
    )


def test_classify_selector_miss_when_good_candidate_loses() -> None:
    assert (
        classify_failure(
            gt_pixels=100,
            final_iou=0.1,
            final_pred_pixels=50,
            best_candidate_gt_pixels=90,
            best_candidate_iou=0.8,
            candidate_recall_pixels=12,
            success_iou=0.5,
        )
        == "selector_or_reranker_miss"
    )


def test_classify_negative_false_positive_for_zero_mask_gt() -> None:
    assert (
        classify_failure(
            gt_pixels=0,
            final_iou=0.0,
            final_pred_pixels=40,
            best_candidate_gt_pixels=0,
            best_candidate_iou=0.0,
            candidate_recall_pixels=12,
            success_iou=0.5,
        )
        == "negative_false_positive"
    )
