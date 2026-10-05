from __future__ import annotations

import numpy as np

from src.supervised.final_veto import (
    build_feature_spec,
    build_final_veto_features,
    choose_threshold,
)
from tools.eval_final_veto_summary import _empty_mask_metrics, _failure_type


def test_final_veto_features_include_source_group_and_flags() -> None:
    item = {
        "pred_pixels": 42,
        "target": 1,
        "winner": {
            "candidate_source": "reticle_global_0",
            "score": 0.25,
            "selected_via_reticle_recovery": True,
            "features": {"ball_fill_fraction": 0.0},
        },
    }
    spec = build_feature_spec([item])

    features = build_final_veto_features(item["winner"], 42, spec)

    assert features.shape == (spec.input_dim,)
    assert "score" in spec.numeric_names
    assert "features.ball_fill_fraction" in spec.numeric_names
    assert "reticle_global_0" in spec.source_names
    assert "reticle_global" in spec.source_group_names
    assert "reticle_recovery" in spec.flag_names


def test_choose_threshold_penalizes_keep_row_vetoes() -> None:
    probabilities = np.asarray([0.05, 0.10, 0.80, 0.90], dtype=np.float32)
    targets = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    threshold, metrics = choose_threshold(probabilities, targets)

    assert 0.10 < threshold <= 0.80
    assert metrics["neg_veto_rate"] == 1.0
    assert metrics["pos_veto_rate"] == 0.0


def test_empty_mask_metrics_match_positive_and_negative_semantics() -> None:
    assert _empty_mask_metrics(0) == (1.0, 1.0)
    assert _empty_mask_metrics(9) == (0.0, 0.1)
    assert _failure_type(gt_pixels=0, pred_pixels=0, iou=1.0, success_iou=0.5) == "true_negative"
    assert _failure_type(gt_pixels=12, pred_pixels=0, iou=0.0, success_iou=0.5) == "positive_no_prediction"
