from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.supervised.infer import (
    _apply_candidate_selector_rescue,
    _apply_post_final_candidate_selector_rescue,
    _apply_rescue_arbiter,
    _candidate_selector_rescue_is_eligible,
    _current_item_exemption_matches,
    _external_rescue_is_eligible,
    _final_context_reject_is_eligible,
    _image_final_veto_configs,
    _image_final_veto_min_pred_pixels_for_source,
    _image_final_veto_should_reject,
    _pre_final_fallback_score_reject_is_eligible,
    _primary_recovery_is_eligible,
    _rescue_config_should_run,
    _rescue_arbiter_candidate_is_eligible,
    _rescue_config_list,
    _micro_line_rescue_should_run,
    _run_selector_candidate_pool_rescue_configs,
    _secondary_rescue_is_eligible,
    _select_candidate_selector_rescue,
    _score_prediction,
    _split_candidate_selector_rules,
    _table_hough_final_fallback_reject_is_eligible,
    _winner_source_area_reject_is_eligible,
    _winner_source_area_score_reject_is_eligible,
)
from src.supervised.infer import _load_guideline_model
from src.supervised.conditioned import ConditionedGuidelineUNet
from src.supervised.model import GuidelineUNet


def test_secondary_rescue_accepts_high_ball_fill_fraction() -> None:
    item = {"features": {"ball_fill_fraction": 0.97}}
    assert _secondary_rescue_is_eligible(item, 0.9)


def test_load_guideline_model_supports_resnet18_unet_checkpoint(tmp_path: Path) -> None:
    from src.supervised.model import build_guideline_model

    checkpoint_path = tmp_path / "resnet18_unet.pt"
    model = build_guideline_model(model_type="guideline_resnet18_unet", base_channels=8, pretrained_encoder=False)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": {
                "model_type": "guideline_resnet18_unet",
                "base_channels": 8,
                "in_channels": 3,
            },
        },
        checkpoint_path,
    )
    loaded = _load_guideline_model(checkpoint_path, torch.device("cpu"))
    assert loaded.__class__.__name__ == "GuidelineResNet18UNet"


def test_secondary_rescue_rejects_low_ball_fill_fraction() -> None:
    item = {"features": {"ball_fill_fraction": 0.42}}
    assert not _secondary_rescue_is_eligible(item, 0.9)


def test_secondary_rescue_without_threshold_is_always_eligible() -> None:
    item = {"features": {"ball_fill_fraction": 0.0}}
    assert _secondary_rescue_is_eligible(item, None)


def test_current_item_exemption_matches_source_and_thresholds() -> None:
    item = {
        "candidate_source": "table_hough",
        "score": 3.87,
        "pred_pixels": 380,
        "confidence": 0.89,
        "features": {
            "ball_fill_fraction": 0.30,
            "connected_to_ball": 1.0,
            "line_likeness": 1.0,
            "outward_extension": 0.26,
        },
    }
    exemptions = [
        {
            "name": "tight_primary_table_object_line",
            "current_source_group": "table_hough",
            "min_score": 3.8,
            "max_score": 3.95,
            "min_pred_pixels": 360,
            "max_pred_pixels": 400,
            "min_confidence": 0.88,
            "min_ball_fill_fraction": 0.29,
            "max_ball_fill_fraction": 0.32,
            "min_connected_to_ball": 1.0,
            "min_line_likeness": 0.95,
            "min_outward_extension": 0.25,
            "max_outward_extension": 0.28,
        }
    ]
    matched, rule = _current_item_exemption_matches(item, exemptions)

    assert matched
    assert rule == exemptions[0]


def test_current_item_exemption_rejects_wrong_source_or_threshold() -> None:
    item = {
        "candidate_source": "white_blob",
        "score": 3.87,
        "pred_pixels": 380,
        "features": {"ball_fill_fraction": 0.30},
    }
    exemptions = [{"current_source_group": "table_hough", "min_score": 3.8, "max_score": 3.95}]

    matched, rule = _current_item_exemption_matches(item, exemptions)

    assert not matched
    assert rule is None


def test_score_prediction_accepts_custom_threshold() -> None:
    prob = np.array([[0.1, 0.2, 0.4]], dtype=np.float32)
    ball_mask = np.array([[0, 1, 0]], dtype=np.uint8)
    _, default_binary = _score_prediction(prob, ball_mask)
    _, lower_binary = _score_prediction(prob, ball_mask, threshold=0.2)
    assert int(default_binary.sum()) == 1
    assert int(lower_binary.sum()) == 2


def test_micro_line_rescue_requires_current_gate() -> None:
    current = {
        "candidate_source": "table_hough",
        "score": -3.9,
        "selected_via_final_fallback": True,
        "pred_pixels": 56,
        "features": {"ball_fill_fraction": 0.0},
    }
    config = {
        "current_source_group": "table_hough",
        "current_required_flag": "final_fallback",
        "min_current_score": -5.0,
        "max_current_score": 0.0,
        "min_current_pred_pixels": 40,
        "max_current_ball_fill_fraction": 0.1,
    }
    assert _micro_line_rescue_should_run(current, config)
    assert not _micro_line_rescue_should_run({**current, "selected_via_final_fallback": False}, config)
    assert not _micro_line_rescue_should_run({**current, "score": 0.2}, config)


def test_rescue_config_list_accepts_single_or_multiple_configs() -> None:
    single = {"checkpoint": "a.pt"}
    multiple = [{"checkpoint": "a.pt"}, {"checkpoint": "b.pt"}, "bad"]

    assert _rescue_config_list(single) == [single]
    assert _rescue_config_list(multiple) == [{"checkpoint": "a.pt"}, {"checkpoint": "b.pt"}]
    assert _rescue_config_list(None) == []


def test_primary_recovery_requires_matching_band_and_primary_fill() -> None:
    current_item = {"score": 2.1, "candidate_source": "table_hough", "features": {"ball_fill_fraction": 0.0}}
    primary_item = {"score": 1.8, "candidate_source": "table_hough", "features": {"ball_fill_fraction": 0.8}}
    assert _primary_recovery_is_eligible(
        current_item,
        primary_item,
        "table_hough",
        None,
        1.5,
        3.0,
        "table_hough",
        None,
        None,
        0.5,
    )


def test_primary_recovery_rejects_wrong_primary_source_or_fill() -> None:
    current_item = {"score": 2.1, "candidate_source": "table_hough", "features": {"ball_fill_fraction": 0.0}}
    wrong_source = {"score": 1.8, "candidate_source": "reticle_global_0", "features": {"ball_fill_fraction": 0.9}}
    low_fill = {"score": 1.8, "candidate_source": "table_hough", "features": {"ball_fill_fraction": 0.2}}
    assert not _primary_recovery_is_eligible(
        current_item,
        wrong_source,
        "table_hough",
        None,
        1.5,
        3.0,
        "table_hough",
        None,
        None,
        0.5,
    )
    assert not _primary_recovery_is_eligible(
        current_item,
        low_fill,
        "table_hough",
        None,
        1.5,
        3.0,
        "table_hough",
        None,
        None,
        0.5,
    )


def test_primary_recovery_accepts_group_and_primary_score_rule() -> None:
    current_item = {"score": 0.9, "candidate_source": "reticle_global_0", "features": {"ball_fill_fraction": 0.0}}
    primary_item = {"score": 1.7, "candidate_source": "reticle_global_1", "features": {"ball_fill_fraction": 0.0}}
    assert _primary_recovery_is_eligible(
        current_item,
        primary_item,
        None,
        "reticle_global",
        0.5,
        2.0,
        None,
        "reticle_global",
        1.5,
        None,
    )


def test_external_rescue_accepts_matching_band_and_fill() -> None:
    current_item = {"score": 4.1, "candidate_source": "table_hough", "features": {"ball_fill_fraction": 0.3}}
    assert _external_rescue_is_eligible(
        current_item,
        source_group="table_hough",
        min_score=1.0,
        max_score=6.0,
        current_max_ball_fill_fraction=0.65,
    )


def test_external_rescue_rejects_wrong_group_or_fill() -> None:
    wrong_group = {"score": 4.1, "candidate_source": "reticle_global_0", "features": {"ball_fill_fraction": 0.3}}
    high_fill = {"score": 4.1, "candidate_source": "table_hough", "features": {"ball_fill_fraction": 0.9}}
    assert not _external_rescue_is_eligible(
        wrong_group,
        source_group="table_hough",
        min_score=1.0,
        max_score=6.0,
        current_max_ball_fill_fraction=0.65,
    )
    assert not _external_rescue_is_eligible(
        high_fill,
        source_group="table_hough",
        min_score=1.0,
        max_score=6.0,
        current_max_ball_fill_fraction=0.65,
    )


def test_table_hough_final_fallback_reject_requires_source_flag_and_area() -> None:
    item = {"candidate_source": "table_hough", "selected_via_final_fallback": True}
    assert _table_hough_final_fallback_reject_is_eligible(
        item,
        pred_pixels=120,
        min_pred_pixels=80,
    )
    assert not _table_hough_final_fallback_reject_is_eligible(
        item,
        pred_pixels=79,
        min_pred_pixels=80,
    )
    assert not _table_hough_final_fallback_reject_is_eligible(
        {"candidate_source": "white_blob", "selected_via_final_fallback": True},
        pred_pixels=120,
        min_pred_pixels=80,
    )
    assert not _table_hough_final_fallback_reject_is_eligible(
        {"candidate_source": "table_hough"},
        pred_pixels=120,
        min_pred_pixels=80,
    )


def test_table_hough_final_fallback_reject_accepts_low_score() -> None:
    item = {
        "candidate_source": "table_hough",
        "selected_via_final_fallback": True,
        "score": -0.2,
    }
    assert _table_hough_final_fallback_reject_is_eligible(
        item,
        pred_pixels=14,
        min_pred_pixels=80,
        max_score=0.0,
    )
    assert not _table_hough_final_fallback_reject_is_eligible(
        {**item, "score": 0.1},
        pred_pixels=14,
        min_pred_pixels=80,
        max_score=0.0,
    )


def test_pre_final_fallback_score_reject_uses_pre_fallback_score() -> None:
    item = {"pre_final_fallback_score": -1.1, "score": 0.7}
    assert _pre_final_fallback_score_reject_is_eligible(
        item,
        pred_pixels=40,
        max_score=-0.923089,
    )
    assert not _pre_final_fallback_score_reject_is_eligible(
        {**item, "pre_final_fallback_score": -0.2},
        pred_pixels=40,
        max_score=-0.923089,
    )
    assert not _pre_final_fallback_score_reject_is_eligible(
        item,
        pred_pixels=0,
        max_score=-0.923089,
    )
    assert not _pre_final_fallback_score_reject_is_eligible(
        item,
        pred_pixels=40,
        max_score=None,
    )


def test_final_context_reject_matches_exact_source_and_thresholds() -> None:
    rules = [
        {
            "name": "pure_white_table_hough",
            "source": "table_hough",
            "min_mask_white_fraction": 0.965,
            "min_min_candidate_center_distance": 14.8,
        }
    ]
    assert _final_context_reject_is_eligible(
        {"candidate_source": "table_hough"},
        context_features={
            "mask_white_fraction": 0.99,
            "min_candidate_center_distance": 15.0,
        },
        source_rules=rules,
    )[0]
    assert not _final_context_reject_is_eligible(
        {"candidate_source": "table_hough"},
        context_features={
            "mask_white_fraction": 0.99,
            "min_candidate_center_distance": 12.0,
        },
        source_rules=rules,
    )[0]
    assert not _final_context_reject_is_eligible(
        {"candidate_source": "white_blob"},
        context_features={
            "mask_white_fraction": 0.99,
            "min_candidate_center_distance": 15.0,
        },
        source_rules=rules,
    )[0]


def test_candidate_selector_rescue_matches_current_and_candidate_thresholds() -> None:
    current = {
        "candidate_id": "crop_00",
        "candidate_source": "table_hough",
        "score": 0.72,
        "selected_via_rescue": True,
        "features": {"ball_fill_fraction": 0.0},
    }
    candidate = {
        "candidate_id": "crop_01",
        "candidate_source": "reticle_global_0",
        "score": 0.49,
        "heuristic_score": 2.9,
        "pred_pixels": 80,
        "features": {
            "ball_fill_fraction": 0.91,
            "connected_to_ball": 1.0,
        },
    }
    rule = {
        "current_source_group": "table_hough",
        "current_required_flag": "rescue",
        "min_current_score": 0.5,
        "max_current_ball_fill_fraction": 0.05,
        "candidate_source_group": "reticle_global",
        "min_candidate_score": 0.4,
        "min_candidate_heuristic_score": 2.5,
        "min_candidate_pred_pixels": 60,
        "min_candidate_ball_fill_fraction": 0.8,
        "min_candidate_connected_to_ball": 1.0,
    }
    assert _candidate_selector_rescue_is_eligible(current, candidate, rule)
    assert not _candidate_selector_rescue_is_eligible(
        {**current, "selected_via_rescue": False},
        candidate,
        rule,
    )


def test_select_candidate_selector_rescue_uses_selection_key() -> None:
    current = {
        "candidate_id": "crop_04",
        "candidate_source": "table_hough",
        "score": 0.82,
        "selected_via_rescue": True,
        "features": {"ball_fill_fraction": 0.6},
    }
    short_candidate = {
        "candidate_id": "crop_05",
        "candidate_source": "table_hough",
        "score": -0.1,
        "heuristic_score": 3.1,
        "pred_pixels": 50,
        "features": {
            "ball_fill_fraction": 0.32,
            "connected_to_ball": 1.0,
            "outward_extension": 0.2,
        },
    }
    longer_candidate = {
        **short_candidate,
        "candidate_id": "crop_03",
        "pred_pixels": 83,
    }
    rule = {
        "current_source_group": "table_hough",
        "current_required_flag": "selected_via_rescue",
        "min_current_score": 0.5,
        "max_current_score": 1.0,
        "candidate_source_group": "table_hough",
        "min_candidate_score": -0.2,
        "min_candidate_heuristic_score": 2.5,
        "min_candidate_pred_pixels": 70,
        "min_candidate_ball_fill_fraction": 0.25,
        "max_candidate_ball_fill_fraction": 0.45,
        "min_candidate_connected_to_ball": 1.0,
        "min_candidate_outward_extension": 0.15,
        "selection_key": "pred_pixels",
    }
    selected, matched_rule = _select_candidate_selector_rescue(
        current,
        [current, short_candidate, longer_candidate],
        [rule],
    )
    assert selected is longer_candidate
    assert matched_rule is rule


def test_candidate_selector_rescue_can_require_candidate_pool() -> None:
    current = {
        "candidate_id": "crop_03",
        "candidate_source": "table_hough",
        "score": 1.78,
        "selected_via_fallback": True,
        "features": {"ball_fill_fraction": 0.35},
    }
    active_candidate = {
        "candidate_id": "crop_19",
        "candidate_source": "table_hough",
        "score": 0.1,
        "heuristic_score": 3.2,
        "pred_pixels": 99,
        "selector_pool": "fallback",
        "features": {"ball_fill_fraction": 0.75, "connected_to_ball": 1.0},
    }
    pooled_candidate = {**active_candidate, "selector_pool": "secondary_rescue"}
    rule = {
        "current_source_group": "table_hough",
        "current_required_flag": "fallback",
        "candidate_pool": "secondary_rescue",
        "candidate_source_group": "table_hough",
        "min_candidate_heuristic_score": 3.0,
        "min_candidate_ball_fill_fraction": 0.7,
    }
    assert not _candidate_selector_rescue_is_eligible(current, active_candidate, rule)
    assert _candidate_selector_rescue_is_eligible(current, pooled_candidate, rule)


def test_candidate_selector_allows_same_crop_id_from_different_pool() -> None:
    current = {
        "candidate_id": "crop_13",
        "candidate_source": "white_blob",
        "score": 1.0,
    }
    pooled_candidate = {
        "candidate_id": "crop_13",
        "candidate_source": "white_blob",
        "selector_pool": "wide_rescue32",
        "score": 2.1,
        "features": {"connected_to_ball": 1.0},
    }
    rule = {
        "candidate_pool": "wide_rescue32",
        "candidate_source": "white_blob",
        "min_candidate_score": 2.0,
        "min_candidate_connected_to_ball": 1.0,
    }
    assert _candidate_selector_rescue_is_eligible(current, pooled_candidate, rule)


def test_rescue_config_without_current_source_runs_for_any_source() -> None:
    assert _rescue_config_should_run(
        {"candidate_source": "reticle_global_0", "score": 2.0},
        {"name": "pool_only_rescue"},
    )


def test_split_candidate_selector_rules_keeps_pool_specific_rules_separate() -> None:
    active_rule = {"name": "active"}
    pooled_rule = {"name": "pooled", "candidate_pool": "secondary_rescue"}
    active_rules, pooled_rules = _split_candidate_selector_rules([active_rule, pooled_rule])
    assert active_rules == [active_rule]
    assert pooled_rules == [pooled_rule]


def test_post_external_selector_candidate_pool_rescue_runs_after_external_flags(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_generate_ball_candidates(_image, config, output_dir=None):
        calls.append({"config": config, "output_dir": output_dir})
        return ["candidate"]

    def fake_score_candidates(**_kwargs):
        return (
            {"candidate_id": "crop_00"},
            [],
            [{"candidate_id": "crop_00", "candidate_source": "colored_blob", "score": 1.25}],
        )

    monkeypatch.setattr("src.supervised.infer.generate_ball_candidates", fake_generate_ball_candidates)
    monkeypatch.setattr("src.supervised.infer._load_guideline_model", lambda *_args: object())
    monkeypatch.setattr("src.supervised.infer._score_candidates", fake_score_candidates)

    current = {
        "candidate_id": "external_main_reticle_rescue",
        "candidate_source": "external_main_reticle_rescue",
        "score": 0.91,
        "selected_via_external_reticle_rescue": True,
    }
    pool_items = _run_selector_candidate_pool_rescue_configs(
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        output_dir=tmp_path,
        current_item=current,
        configs=[
            {
                "name": "late_colored_blob",
                "pool": "late_colored_blob",
                "checkpoint": str(tmp_path / "model.pt"),
                "use_reranker": False,
                "current_source": "external_main_reticle_rescue",
                "current_required_flags": ["external_reticle_rescue"],
                "colored_blob_candidates": True,
                "colored_blob": {"min_saturation": 120},
            }
        ],
        image_size=384,
        device=torch.device("cpu"),
        save_intermediates=False,
        crop_batch_size=8,
        amp_enabled=False,
        stage_name="post_external",
    )

    assert calls[0]["config"]["colored_blob_candidates"] is True
    assert calls[0]["config"]["colored_blob"] == {"min_saturation": 120}
    assert pool_items[0]["selector_pool"] == "late_colored_blob"
    assert pool_items[0]["selector_candidate_pool_rescue"] is True
    assert pool_items[0]["post_external_selector_candidate_pool_rescue"] is True
    assert pool_items[0]["pre_post_external_selector_candidate_pool_rescue_source"] == "external_main_reticle_rescue"
    assert pool_items[0]["pre_post_external_selector_candidate_pool_rescue_score"] == 0.91


def test_post_external_selector_candidate_pool_rescue_respects_current_flags(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.supervised.infer.generate_ball_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pool should not run")),
    )

    pool_items = _run_selector_candidate_pool_rescue_configs(
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        output_dir=tmp_path,
        current_item={
            "candidate_id": "external_main_reticle_rescue",
            "candidate_source": "external_main_reticle_rescue",
            "score": 0.91,
        },
        configs=[
            {
                "name": "late_colored_blob",
                "checkpoint": str(tmp_path / "model.pt"),
                "current_source": "external_main_reticle_rescue",
                "current_required_flags": ["external_reticle_rescue"],
            }
        ],
        image_size=384,
        device=torch.device("cpu"),
        save_intermediates=False,
        crop_batch_size=8,
        amp_enabled=False,
        stage_name="post_external",
    )

    assert pool_items == []


def test_apply_candidate_selector_rescue_tags_post_external_stage() -> None:
    current = {
        "candidate_id": "external_main_reticle_rescue",
        "candidate_source": "external_main_reticle_rescue",
        "score": 0.91,
    }
    candidate = {
        "candidate_id": "crop_08",
        "candidate_source": "table_hough",
        "score": 1.1,
        "heuristic_score": 3.4,
        "pred_pixels": 90,
        "features": {"connected_to_ball": 1.0, "ball_fill_fraction": 1.0},
    }
    selected = _apply_candidate_selector_rescue(
        current,
        [candidate],
        [],
        [
            {
                "current_source": "external_main_reticle_rescue",
                "candidate_source_group": "table_hough",
                "min_candidate_connected_to_ball": 1.0,
            }
        ],
        stage_name="post_external",
    )
    assert selected["candidate_id"] == "crop_08"
    assert selected["selected_via_post_external_candidate_selector_rescue"] is True
    assert selected["pre_post_external_candidate_selector_rescue_source"] == "external_main_reticle_rescue"


def test_post_final_candidate_selector_rescue_matches_zeroed_final_mask() -> None:
    current = {
        "candidate_id": "crop_00",
        "candidate_source": "table_hough",
        "score": -0.12,
        "selected_via_final_fallback": True,
        "selected_via_table_hough_final_fallback_reject": True,
        "features": {"ball_fill_fraction": 0.0},
    }
    candidate = {
        "candidate_id": "crop_01",
        "candidate_source": "reticle_global_0",
        "selector_pool": "primary",
        "score": 1.1,
        "heuristic_score": 2.65,
        "pred_pixels": 28,
        "features": {
            "ball_fill_fraction": 1.0,
            "connected_to_ball": 1.0,
            "outward_extension": 0.0,
        },
    }
    selected = _apply_post_final_candidate_selector_rescue(
        current,
        [candidate],
        [candidate],
        [
            {
                "name": "zeroed_table_to_filled_reticle",
                "current_source_group": "table_hough",
                "current_required_flags": ["final_fallback", "table_hough_final_fallback_reject"],
                "max_current_final_pred_pixels": 0,
                "candidate_pool": "primary",
                "candidate_source": "reticle_global_0",
                "min_candidate_score": 1.0,
                "min_candidate_heuristic_score": 2.5,
                "max_candidate_pred_pixels": 35,
                "min_candidate_ball_fill_fraction": 0.9,
                "min_candidate_connected_to_ball": 1.0,
            }
        ],
        final_pred_pixels=0,
    )

    assert selected["candidate_id"] == "crop_01"
    assert selected["selected_via_post_final_candidate_selector_rescue"] is True
    assert selected["pre_post_final_candidate_selector_rescue_final_pred_pixels"] == 0


def test_post_final_candidate_selector_rescue_blocks_nonzero_final_mask_by_default() -> None:
    current = {
        "candidate_id": "crop_00",
        "candidate_source": "table_hough",
        "score": 0.5,
    }
    candidate = {
        "candidate_id": "crop_01",
        "candidate_source": "table_hough",
        "score": 0.6,
        "features": {"line_likeness": 1.0},
    }
    rule = {
        "current_source_group": "table_hough",
        "candidate_source_group": "table_hough",
        "min_candidate_line_likeness": 1.0,
    }

    blocked = _apply_post_final_candidate_selector_rescue(
        current,
        [candidate],
        [],
        [rule],
        final_pred_pixels=12,
    )
    allowed = _apply_post_final_candidate_selector_rescue(
        current,
        [candidate],
        [],
        [{**rule, "allow_nonzero_final_mask": True}],
        final_pred_pixels=12,
    )

    assert blocked["candidate_id"] == "crop_00"
    assert allowed["candidate_id"] == "crop_01"
    assert allowed["selected_via_post_final_candidate_selector_rescue"] is True


def test_apply_rescue_arbiter_selects_high_probability_candidate(monkeypatch) -> None:
    current = {
        "candidate_id": "crop_00",
        "candidate_source": "table_hough",
        "score": 0.5,
        "features": {"ball_fill_fraction": 0.0},
    }
    low_candidate = {
        "candidate_id": "crop_01",
        "candidate_source": "table_hough",
        "score": 0.8,
        "pred_pixels": 40,
        "features": {"line_likeness": 1.0},
    }
    high_candidate = {
        "candidate_id": "crop_02",
        "candidate_source": "table_hough",
        "score": 1.1,
        "pred_pixels": 90,
        "features": {"line_likeness": 1.0},
    }

    class FakeArbiter:
        threshold = 0.75

        def candidate_probability(self, _current, candidate) -> float:
            return 0.9 if candidate["candidate_id"] == "crop_02" else 0.1

    monkeypatch.setattr("src.supervised.infer._load_rescue_arbiter_cached", lambda *_args: FakeArbiter())

    selected = _apply_rescue_arbiter(
        current,
        [current, low_candidate, high_candidate],
        {
            "checkpoint": "fake.pt",
            "candidate_source_group": "table_hough",
            "min_candidate_pred_pixels": 30,
        },
        device=torch.device("cpu"),
    )

    assert selected["candidate_id"] == "crop_02"
    assert selected["selected_via_rescue_arbiter"] is True
    assert selected["rescue_arbiter_probability"] == 0.9


def test_rescue_arbiter_candidate_filter_allows_omitted_source() -> None:
    current = {"candidate_id": "crop_00", "candidate_source": "table_hough", "selector_pool": "primary"}
    candidate = {
        "candidate_id": "crop_01",
        "candidate_source": "white_blob",
        "selector_pool": "wide_rescue32",
        "pred_pixels": 80,
    }

    assert _rescue_arbiter_candidate_is_eligible(
        current,
        candidate,
        {"candidate_pool": "wide_rescue32", "min_candidate_pred_pixels": 10},
    )


def test_winner_source_area_reject_matches_exact_or_group_source() -> None:
    assert _winner_source_area_reject_is_eligible(
        {"candidate_source": "external_main_rescue"},
        pred_pixels=450,
        source_min_pred_pixels={"external_main_rescue": 400},
    )
    assert _winner_source_area_reject_is_eligible(
        {"candidate_source": "reticle_global_0"},
        pred_pixels=120,
        source_min_pred_pixels={"reticle_global": 100},
    )
    assert not _winner_source_area_reject_is_eligible(
        {"candidate_source": "external_main_rescue"},
        pred_pixels=399,
        source_min_pred_pixels={"external_main_rescue": 400},
    )


def test_winner_source_area_reject_supports_tight_exemptions() -> None:
    item = {
        "candidate_source": "table_hough",
        "score": 6.5,
        "confidence": 0.91,
        "features": {
            "ball_fill_fraction": 0.2,
            "connected_to_ball": 1.0,
            "line_likeness": 1.0,
        },
    }
    exemption = {
        "name": "large_attached_table_target",
        "min_score": 6.0,
        "max_score": 7.0,
        "min_confidence": 0.9,
        "min_ball_fill_fraction": 0.1,
        "max_ball_fill_fraction": 0.3,
        "min_connected_to_ball": 1.0,
        "min_line_likeness": 0.95,
        "min_pred_pixels": 600,
        "max_pred_pixels": 700,
    }
    assert not _winner_source_area_reject_is_eligible(
        item,
        pred_pixels=650,
        source_min_pred_pixels={"table_hough": 600},
        source_exemptions={"table_hough": [exemption]},
    )
    assert _winner_source_area_reject_is_eligible(
        item,
        pred_pixels=800,
        source_min_pred_pixels={"table_hough": 600},
        source_exemptions={"table_hough": [exemption]},
    )


def test_image_final_veto_supports_tight_source_exemptions() -> None:
    item = {
        "candidate_source": "table_hough",
        "score": 6.1,
        "confidence": 0.91,
        "candidate_radius": 24,
        "reticle_distance": 24.9,
        "features": {
            "ball_fill_fraction": 0.23,
            "connected_to_ball": 1.0,
            "line_likeness": 1.0,
            "outward_extension": 0.37,
        },
    }
    should_reject, probability = _image_final_veto_should_reject(
        image=np.zeros((10, 60, 3), dtype=np.uint8),
        final_mask=np.ones((10, 60), dtype=np.uint8),
        current_item=item,
        config={
            "checkpoint": "does_not_need_to_exist_when_exempt.pt",
            "sources": ["table_hough"],
            "source_exemptions": {
                "table_hough": [
                    {
                        "name": "large_attached_table_target",
                        "min_score": 6.0,
                        "max_score": 6.2,
                        "min_pred_pixels": 600,
                        "max_pred_pixels": 600,
                        "min_confidence": 0.9,
                        "min_ball_fill_fraction": 0.2,
                        "max_ball_fill_fraction": 0.25,
                        "min_connected_to_ball": 1.0,
                        "min_line_likeness": 0.95,
                        "min_outward_extension": 0.36,
                        "max_outward_extension": 0.38,
                    }
                ]
            },
        },
        device=torch.device("cpu"),
    )
    assert should_reject is False
    assert probability is None


def test_winner_source_area_score_reject_matches_exact_or_group_source() -> None:
    rules = {
        "table_hough": {"min_pred_pixels": 54, "max_score": 0.054},
        "reticle_global": {"min_pred_pixels": 320, "max_score": 1.3},
    }
    assert _winner_source_area_score_reject_is_eligible(
        {"candidate_source": "table_hough", "score": 0.053},
        pred_pixels=54,
        source_rules=rules,
    )
    assert _winner_source_area_score_reject_is_eligible(
        {"candidate_source": "reticle_global_0", "score": 1.2},
        pred_pixels=400,
        source_rules=rules,
    )
    assert not _winner_source_area_score_reject_is_eligible(
        {"candidate_source": "table_hough", "score": 0.2},
        pred_pixels=90,
        source_rules=rules,
    )
    assert not _winner_source_area_score_reject_is_eligible(
        {"candidate_source": "table_hough", "score": 0.053},
        pred_pixels=53,
        source_rules=rules,
    )


def test_winner_source_area_score_reject_supports_feature_exemptions() -> None:
    rules = {
        "table_hough": {
            "min_pred_pixels": 54,
            "max_score": 0.054,
            "exemptions": [
                {
                    "name": "confident_thin_line",
                    "min_confidence": 0.84,
                    "min_line_likeness": 0.8,
                }
            ],
        }
    }
    assert not _winner_source_area_score_reject_is_eligible(
        {
            "candidate_source": "table_hough",
            "score": 0.01,
            "confidence": 0.88,
            "features": {"line_likeness": 1.0},
        },
        pred_pixels=80,
        source_rules=rules,
    )
    assert _winner_source_area_score_reject_is_eligible(
        {
            "candidate_source": "table_hough",
            "score": 0.01,
            "confidence": 0.88,
            "features": {"line_likeness": 0.4},
        },
        pred_pixels=80,
        source_rules=rules,
    )


def test_image_final_veto_min_pred_pixels_can_be_source_specific() -> None:
    config = {
        "min_pred_pixels": 80,
        "source_min_pred_pixels": {
            "external_main_rescue": 400,
            "reticle_global": 120,
        },
    }
    assert _image_final_veto_min_pred_pixels_for_source(config, "external_main_rescue") == 400
    assert _image_final_veto_min_pred_pixels_for_source(config, "reticle_global_0") == 120
    assert _image_final_veto_min_pred_pixels_for_source(config, "white_blob") == 80


def test_image_final_veto_configs_accepts_single_or_multiple_configs() -> None:
    single = {"checkpoint": "a.pt"}
    multiple = [{"checkpoint": "a.pt"}, {"checkpoint": "b.pt"}, "bad"]

    assert _image_final_veto_configs(single) == [single]
    assert _image_final_veto_configs(multiple) == [{"checkpoint": "a.pt"}, {"checkpoint": "b.pt"}]
    assert _image_final_veto_configs(None) == []


def test_load_guideline_model_infers_non_default_base_channels(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "wide_guideline.pt"
    model = GuidelineUNet(base_channels=48)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": {"base_channels": 48},
        },
        checkpoint_path,
    )
    loaded = _load_guideline_model(checkpoint_path, torch.device("cpu"))
    stem_weight = loaded.stem.block[0].weight
    assert stem_weight.shape[0] == 48


def test_load_guideline_model_loads_conditioned_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "conditioned_guideline.pt"
    model = ConditionedGuidelineUNet(base_channels=8)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": {
                "model_type": "conditioned_guideline_unet",
                "base_channels": 8,
                "in_channels": 4,
            },
        },
        checkpoint_path,
    )
    loaded = _load_guideline_model(checkpoint_path, torch.device("cpu"))
    outputs = loaded(torch.zeros((1, 4, 32, 32), dtype=torch.float32))
    assert outputs["mask_logits"].shape == (1, 1, 32, 32)
    assert outputs["validity_logits"].shape == (1, 1)
