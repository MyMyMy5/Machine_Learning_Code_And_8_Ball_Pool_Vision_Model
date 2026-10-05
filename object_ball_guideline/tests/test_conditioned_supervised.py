from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.supervised.conditioned import ConditionedGuidelineUNet
from src.supervised.conditioned_data import ConditionedZoomProbeCropDataset, build_candidate_heatmap


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = [20, 40, 60]
    Image.fromarray(image, mode="RGB").save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30:34, 30:48] = 255
    Image.fromarray(mask, mode="L").save(path)


def test_candidate_heatmap_places_peak_at_local_candidate_center() -> None:
    heatmap = build_candidate_heatmap(
        crop_box=[16, 16, 48, 48],
        candidate_center=[32, 24],
        candidate_radius=4,
        output_size=32,
    )

    assert heatmap.shape == (32, 32)
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    assert abs(x - 16) <= 1
    assert abs(y - 8) <= 1
    assert float(heatmap.max()) == 1.0


def test_conditioned_dataset_returns_four_channels_and_validity_label(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    _write_rgb(image_path)
    _write_mask(mask_path)

    entries = [
        {
            "kind": "candidate_positive",
            "id": "positive",
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "crop_box": [16, 16, 48, 48],
            "candidate_center": [32, 32],
            "candidate_radius": 4,
            "zero_mask": False,
        },
        {
            "kind": "external_negative",
            "id": "negative",
            "image_path": str(image_path),
            "mask_path": None,
            "crop_box": [0, 0, 32, 32],
            "candidate_center": [16, 16],
            "candidate_radius": 4,
            "zero_mask": True,
        },
    ]

    dataset = ConditionedZoomProbeCropDataset(entries, image_size=32, augment=False, seed=1)
    positive = dataset[0]
    negative = dataset[1]

    assert positive["image"].shape == (4, 32, 32)
    assert positive["mask"].shape == (1, 32, 32)
    assert positive["validity"].shape == (1,)
    assert float(positive["validity"].item()) == 1.0
    assert float(negative["validity"].item()) == 0.0
    assert float(negative["mask"].sum().item()) == 0.0


def test_conditioned_dataset_can_reject_positive_without_candidate_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    _write_rgb(image_path)
    _write_mask(mask_path)

    dataset = ConditionedZoomProbeCropDataset(
        [
            {
                "kind": "view_positive",
                "id": "positive_without_candidate",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "crop_box": [16, 16, 48, 48],
                "zero_mask": False,
            }
        ],
        image_size=32,
        augment=False,
        seed=1,
        require_candidate_metadata=True,
    )

    try:
        dataset[0]
    except ValueError as exc:
        assert "lacks candidate metadata" in str(exc)
    else:
        raise AssertionError("Expected candidate-metadata validation to fail")


def test_conditioned_model_outputs_mask_and_validity_logits() -> None:
    model = ConditionedGuidelineUNet(base_channels=8)
    outputs = model(torch.zeros((2, 4, 32, 32), dtype=torch.float32))

    assert outputs["mask_logits"].shape == (2, 1, 32, 32)
    assert outputs["validity_logits"].shape == (2, 1)
