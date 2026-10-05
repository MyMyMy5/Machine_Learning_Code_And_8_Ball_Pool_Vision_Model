from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.crop_utils import CropCandidate
from src.supervised.config import DatasetConfig
from src.supervised.data import ZoomProbeCropDataset, build_zoomprobe_index


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[20:44, 20:44] = [210, 210, 210]
    Image.fromarray(image, mode="RGB").save(path)


def _write_mask(path: Path, box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((64, 64), dtype=np.uint8)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 255
    Image.fromarray(mask, mode="L").save(path)


def _append_annotation(root: Path, sample_id: str, image_path: Path, mask_path: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "annotations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": sample_id,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "view_xyxy": [16, 16, 48, 48],
                }
            )
            + "\n"
        )


def test_build_zoomprobe_index_combines_rejected_and_flat_negatives(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_zoomprobe"
    quarantine_root = data_root / "quarantine"
    rejected_root = data_root / "Rejected"
    flat_negative_root = tmp_path / "negative_selected"
    output_root = tmp_path / "run"

    positive_image = data_root / "images" / "positive.png"
    positive_mask = data_root / "masks" / "positive.png"
    _write_rgb(positive_image)
    _write_mask(positive_mask, (24, 30, 42, 34))
    _append_annotation(data_root, "positive", positive_image, positive_mask)
    (quarantine_root / "annotations.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (quarantine_root / "annotations.jsonl").write_text("", encoding="utf-8")

    rejected_image = rejected_root / "images" / "bad_prediction.png"
    rejected_mask = rejected_root / "masks" / "bad_prediction.png"
    _write_rgb(rejected_image)
    _write_mask(rejected_mask, (8, 8, 16, 20))
    _append_annotation(rejected_root, "bad_prediction", rejected_image, rejected_mask)

    flat_negative = flat_negative_root / "negative.png"
    _write_rgb(flat_negative)

    def _fake_candidates(image, config):
        return [
            CropCandidate(
                candidate_id="candidate_0",
                center_x=32,
                center_y=32,
                radius=8,
                crop_box=(8, 8, 56, 56),
                source="test",
            )
        ]

    monkeypatch.setattr("src.supervised.data.generate_ball_candidates", _fake_candidates)

    index = build_zoomprobe_index(
        DatasetConfig(
            main_root=data_root,
            quarantine_root=quarantine_root,
            output_root=output_root,
            negative_roots=[rejected_root, flat_negative_root],
            negative_crops_per_image=2,
        )
    )

    train_kinds = [entry["kind"] for entry in index["train"]]
    assert "rejected_negative" in train_kinds
    assert "external_negative" in train_kinds

    negative_entries = [entry for entry in index["train"] if entry["kind"] in {"rejected_negative", "external_negative"}]
    assert negative_entries
    assert all(entry["zero_mask"] for entry in negative_entries)
    rejected_entries = [entry for entry in negative_entries if entry["kind"] == "rejected_negative"]
    assert [entry["id"] for entry in rejected_entries] == ["bad_prediction_0"]
    assert rejected_entries[0]["crop_box"] == [8, 8, 56, 56]

    dataset = ZoomProbeCropDataset(negative_entries, image_size=32, augment=False, seed=1)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        assert float(sample["mask"].sum()) == 0.0


def test_build_zoomprobe_index_rebuilds_when_annotations_change(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_zoomprobe"
    quarantine_root = data_root / "quarantine"
    output_root = tmp_path / "run"
    (quarantine_root / "annotations.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (quarantine_root / "annotations.jsonl").write_text("", encoding="utf-8")

    def _fake_candidates(image, config):
        return []

    monkeypatch.setattr("src.supervised.data.generate_ball_candidates", _fake_candidates)

    first_image = data_root / "images" / "first.png"
    first_mask = data_root / "masks" / "first.png"
    _write_rgb(first_image)
    _write_mask(first_mask, (20, 30, 40, 34))
    _append_annotation(data_root, "first", first_image, first_mask)

    config = DatasetConfig(main_root=data_root, quarantine_root=quarantine_root, output_root=output_root)
    first_index = build_zoomprobe_index(config)
    first_count = len(first_index["train"]) + len(first_index["val"])

    second_image = data_root / "images" / "second.png"
    second_mask = data_root / "masks" / "second.png"
    _write_rgb(second_image)
    _write_mask(second_mask, (18, 28, 38, 32))
    _append_annotation(data_root, "second", second_image, second_mask)

    second_index = build_zoomprobe_index(config)
    second_count = len(second_index["train"]) + len(second_index["val"])
    assert second_count > first_count


def test_build_zoomprobe_index_can_add_mask_centered_context_positives(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_zoomprobe"
    quarantine_root = data_root / "quarantine"
    output_root = tmp_path / "run"
    (quarantine_root / "annotations.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (quarantine_root / "annotations.jsonl").write_text("", encoding="utf-8")

    def _fake_candidates(image, config):
        return []

    monkeypatch.setattr("src.supervised.data.generate_ball_candidates", _fake_candidates)

    positive_image = data_root / "images" / "positive.png"
    positive_mask = data_root / "masks" / "positive.png"
    _write_rgb(positive_image)
    _write_mask(positive_mask, (24, 30, 42, 34))
    _append_annotation(data_root, "positive", positive_image, positive_mask)

    index = build_zoomprobe_index(
        DatasetConfig(
            main_root=data_root,
            quarantine_root=quarantine_root,
            output_root=output_root,
            positive_context_sides=[32],
        )
    )

    context_entries = [entry for entry in index["train"] if entry["kind"] == "context_positive"]
    assert len(context_entries) == 1
    assert context_entries[0]["crop_box"] == [17, 16, 49, 48]
    assert context_entries[0]["context_side"] == 32


def test_build_zoomprobe_index_excludes_negative_root_ids(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_zoomprobe"
    quarantine_root = data_root / "quarantine"
    rejected_root = data_root / "Rejected"
    flat_negative_root = tmp_path / "negative_selected"
    output_root = tmp_path / "run"
    (data_root / "annotations.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (data_root / "annotations.jsonl").write_text("", encoding="utf-8")
    (quarantine_root / "annotations.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (quarantine_root / "annotations.jsonl").write_text("", encoding="utf-8")

    rejected_image = rejected_root / "images" / "holdout_reject.png"
    rejected_mask = rejected_root / "masks" / "holdout_reject.png"
    _write_rgb(rejected_image)
    _write_mask(rejected_mask, (8, 8, 16, 20))
    _append_annotation(rejected_root, "holdout_reject", rejected_image, rejected_mask)

    flat_negative_root.mkdir(parents=True)
    _write_rgb(flat_negative_root / "holdout_flat.png")
    _write_rgb(flat_negative_root / "train_flat.png")

    def _fake_candidates(image, config):
        return [
            CropCandidate(
                candidate_id="candidate_0",
                center_x=32,
                center_y=32,
                radius=8,
                crop_box=(8, 8, 56, 56),
                source="test",
            )
        ]

    monkeypatch.setattr("src.supervised.data.generate_ball_candidates", _fake_candidates)

    index = build_zoomprobe_index(
        DatasetConfig(
            main_root=data_root,
            quarantine_root=quarantine_root,
            output_root=output_root,
            negative_roots=[rejected_root, flat_negative_root],
            exclude_ids=["holdout_reject", "holdout_flat"],
        )
    )

    ids = {entry["id"] for entry in index["train"]}
    assert "holdout_reject" not in ids
    assert "holdout_flat_0" not in ids
    assert "train_flat_0" in ids
