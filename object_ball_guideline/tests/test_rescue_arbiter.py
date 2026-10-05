from __future__ import annotations

import numpy as np
import torch

from src.supervised.rescue_arbiter import (
    GuidelineRescueArbiter,
    build_feature_spec,
    build_rescue_arbiter_features,
    choose_threshold,
)


def test_rescue_arbiter_features_include_current_candidate_and_delta() -> None:
    current = {
        "candidate_source": "table_hough",
        "selector_pool": "primary",
        "score": 1.0,
        "pred_pixels": 100,
        "selected_via_rescue": True,
        "features": {"line_likeness": 0.5},
    }
    candidate = {
        "candidate_source": "white_blob",
        "selector_pool": "wide_rescue32",
        "score": 2.0,
        "pred_pixels": 80,
        "features": {"line_likeness": 1.0},
    }
    spec = build_feature_spec([{"current": current, "candidate": candidate, "target": 1}])

    features = build_rescue_arbiter_features(current, candidate, spec)

    assert features.shape == (spec.input_dim,)
    score_offset = spec.numeric_names.index("score") * 5
    assert features[score_offset] == 1.0
    assert features[score_offset + 2] == 2.0
    assert features[score_offset + 4] == 1.0


def test_choose_threshold_penalizes_false_positives() -> None:
    probabilities = np.asarray([0.95, 0.90, 0.70, 0.10], dtype=np.float32)
    targets = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    threshold, metrics = choose_threshold(probabilities, targets)

    assert threshold > 0.90
    assert metrics["fp"] == 0.0


def test_guideline_rescue_arbiter_forward_shape() -> None:
    model = GuidelineRescueArbiter(input_dim=5, hidden_dim=8)

    output = model(torch.zeros((3, 5), dtype=torch.float32))

    assert output.shape == (3,)
