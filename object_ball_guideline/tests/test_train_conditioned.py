from __future__ import annotations

import torch

from src.supervised.conditioned import ConditionedGuidelineUNet
from src.supervised.model import GuidelineUNet
from src.supervised.train_conditioned import _candidate_faithful_entries, inflate_rgb_state_for_conditioned_model


def test_inflate_rgb_checkpoint_copies_rgb_stem_and_zero_initializes_heatmap_channel() -> None:
    rgb_model = GuidelineUNet(base_channels=8)
    conditioned = ConditionedGuidelineUNet(base_channels=8)
    inflated = inflate_rgb_state_for_conditioned_model(rgb_model.state_dict(), conditioned.state_dict())

    assert inflated["stem.block.0.weight"].shape[1] == 4
    assert torch.equal(inflated["stem.block.0.weight"][:, :3], rgb_model.state_dict()["stem.block.0.weight"])
    assert torch.count_nonzero(inflated["stem.block.0.weight"][:, 3]).item() == 0
    assert torch.equal(inflated["head.weight"], rgb_model.state_dict()["head.weight"])


def test_candidate_faithful_entries_remove_center_fallback_positive_kinds() -> None:
    entries = [
        {"kind": "view_positive", "id": "view"},
        {"kind": "bbox_positive", "id": "bbox"},
        {"kind": "candidate_positive", "id": "candidate"},
        {"kind": "candidate_negative", "id": "candidate_neg"},
        {"kind": "rejected_negative", "id": "reject"},
        {"kind": "external_negative", "id": "external"},
    ]

    filtered = _candidate_faithful_entries(entries)

    assert [entry["id"] for entry in filtered] == ["candidate", "candidate_neg", "reject", "external"]
