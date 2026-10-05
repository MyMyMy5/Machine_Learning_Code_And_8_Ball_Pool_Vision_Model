from __future__ import annotations

import json
from pathlib import Path

import torch

from src.supervised.train import DiceBCE, _load_exclude_ids


def test_load_exclude_ids_reads_json_list(tmp_path: Path) -> None:
    path = tmp_path / "exclude.json"
    path.write_text(json.dumps(["a", 2]), encoding="utf-8")

    assert _load_exclude_ids(path) == ["a", "2"]


def test_thin_line_loss_is_finite_and_penalizes_line_pixels() -> None:
    logits = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    targets = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    targets[:, :, 3, 2:6] = 1.0

    plain = DiceBCE(pos_weight=2.0)(logits, targets)
    weighted = DiceBCE(
        pos_weight=2.0,
        line_weight=3.0,
        neighborhood_weight=0.5,
        line_abs_weight=0.2,
    )(logits, targets)

    assert torch.isfinite(weighted)
    assert weighted > plain
