from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.supervised.infer import _predict_crops
from tools import filter_guideline_harvest
from tools import run_supervised_best_harvest


class _ZeroModel(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (batch.shape[0], 1, batch.shape[2], batch.shape[3]),
            dtype=batch.dtype,
            device=batch.device,
        )


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:] = color
    Image.fromarray(image, mode="RGB").save(path)


def _write_mask(path: Path, active: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((8, 8), dtype=np.uint8)
    if active:
        mask[2:6, 2:6] = 255
    Image.fromarray(mask, mode="L").save(path)


def test_predict_crops_batches_on_cpu() -> None:
    crops = [
        np.zeros((12, 14, 3), dtype=np.uint8),
        np.full((10, 16, 3), 255, dtype=np.uint8),
    ]
    outputs = _predict_crops(
        _ZeroModel(),
        crops,
        image_size=8,
        device=torch.device("cpu"),
        crop_batch_size=2,
        amp_enabled=False,
    )
    assert len(outputs) == 2
    assert outputs[0].shape == (12, 14)
    assert outputs[1].shape == (10, 16)
    assert np.isclose(float(outputs[0].max()), 127.0 / 255.0)


def test_filter_guideline_harvest_backfills_flat_layout(tmp_path: Path, monkeypatch) -> None:
    frames_dir = tmp_path / "frames"
    inference_dir = tmp_path / "legacy_infer"
    output_root = tmp_path / "harvest"

    detected_frame = frames_dir / "video_a" / "frame001.jpg"
    missed_frame = frames_dir / "video_b" / "frame001.jpg"
    _write_rgb(detected_frame, (10, 20, 30))
    _write_rgb(missed_frame, (40, 50, 60))

    detected_stem = "video_a__frame001"
    missed_stem = "video_b__frame001"
    detected_dir = inference_dir / detected_stem
    missed_dir = inference_dir / missed_stem
    _write_rgb(detected_dir / "overlay_final.png", (1, 2, 3))
    _write_rgb(missed_dir / "overlay_final.png", (4, 5, 6))
    _write_mask(detected_dir / "mask_final.png", active=True)
    _write_mask(missed_dir / "mask_final.png", active=False)
    (detected_dir / "report.json").write_text(json.dumps({"winner": {"score": 1.25}}), encoding="utf-8")
    (missed_dir / "report.json").write_text(json.dumps({"winner": {"score": -0.5}}), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "filter_guideline_harvest.py",
            "--frames-dir",
            str(frames_dir),
            "--inference-dir",
            str(inference_dir),
            "--output-root",
            str(output_root),
            "--recurse-frames",
        ],
    )
    assert filter_guideline_harvest.main() == 0

    assert (output_root / "Predicted" / "video_a__frame001.png").exists()
    assert (output_root / "Masks_Predicted" / "video_a__frame001.png").exists()
    assert (output_root / "No_Prediction" / "video_b__frame001.jpg").exists()
    assert not (output_root / "Masks_Not_Detected").exists()


def test_run_harvest_routes_detected_and_not_detected(tmp_path: Path, monkeypatch) -> None:
    frames_dir = tmp_path / "frames"
    output_root = tmp_path / "harvest"
    manifest_path = tmp_path / "manifest.json"

    positive = frames_dir / "match_a" / "positive.jpg"
    negative = frames_dir / "match_b" / "negative.jpg"
    _write_rgb(positive, (100, 100, 100))
    _write_rgb(negative, (120, 120, 120))

    manifest_path.write_text(
        json.dumps(
            {
                "primary_checkpoint": str(tmp_path / "guideline_unet_best.pt"),
                "primary_reranker": str(tmp_path / "guideline_reranker_best.pt"),
            }
        ),
        encoding="utf-8",
    )

    call_count = {"value": 0}

    def _fake_infer(**kwargs):
        call_count["value"] += 1
        input_path = kwargs["input_path"]
        if input_path.name == "positive.jpg":
            return {
                "winner": {"score": 2.0},
                "candidates": [],
                "final_mask": np.ones((8, 8), dtype=np.uint8),
                "final_overlay": np.full((8, 8, 3), 200, dtype=np.uint8),
            }
        return {
            "winner": {"score": -1.0},
            "candidates": [],
            "final_mask": np.zeros((8, 8), dtype=np.uint8),
            "final_overlay": np.full((8, 8, 3), 50, dtype=np.uint8),
        }

    monkeypatch.setattr(run_supervised_best_harvest, "run_supervised_inference", _fake_infer)

    summary = run_supervised_best_harvest.run_harvest(
        frames_dir=frames_dir,
        output_root=output_root,
        manifest_path=manifest_path,
        patterns=["*.jpg"],
        recurse=True,
        image_size=384,
        skip_existing=False,
        detection_min_score=0.0,
        detection_min_mask_pixels=1,
        crop_batch_size=32,
        amp_enabled=True,
    )

    assert summary["predicted_count"] == 1
    assert summary["no_prediction_count"] == 1
    assert (output_root / "Predicted" / "match_a__positive.png").exists()
    assert (output_root / "Masks_Predicted" / "match_a__positive.png").exists()
    assert (output_root / "No_Prediction" / "match_b__negative.jpg").exists()

    summary_skip = run_supervised_best_harvest.run_harvest(
        frames_dir=frames_dir,
        output_root=output_root,
        manifest_path=manifest_path,
        patterns=["*.jpg"],
        recurse=True,
        image_size=384,
        skip_existing=True,
        detection_min_score=0.0,
        detection_min_mask_pixels=1,
        crop_batch_size=32,
        amp_enabled=True,
    )

    assert summary_skip["processed_count"] == 0
    assert call_count["value"] == 2


def test_run_harvest_full_manifest_uses_manifest_stack(tmp_path: Path, monkeypatch) -> None:
    frames_dir = tmp_path / "frames"
    output_root = tmp_path / "harvest"
    manifest_path = tmp_path / "manifest.json"
    frame = frames_dir / "match_a" / "frame.jpg"
    _write_rgb(frame, (100, 100, 100))

    manifest_path.write_text(
        json.dumps(
            {
                "primary_checkpoint": str(tmp_path / "guideline_unet_best.pt"),
                "primary_reranker": str(tmp_path / "guideline_reranker_best.pt"),
                "post_external_candidate_selector_rescue_rules": [{"name": "post_rule"}],
            }
        ),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    def _fake_full_manifest(**kwargs):
        calls.update(kwargs)
        return {
            "winner": {"score": 2.0},
            "candidates": [],
            "final_mask": np.ones((8, 8), dtype=np.uint8),
            "final_overlay": np.full((8, 8, 3), 200, dtype=np.uint8),
        }

    def _unexpected_fast_path(**_kwargs):
        raise AssertionError("full-manifest harvest should not use the fast primary-only path")

    monkeypatch.setattr(run_supervised_best_harvest, "run_manifest_inference", _fake_full_manifest)
    monkeypatch.setattr(run_supervised_best_harvest, "run_supervised_inference", _unexpected_fast_path)

    summary = run_supervised_best_harvest.run_harvest(
        frames_dir=frames_dir,
        output_root=output_root,
        manifest_path=manifest_path,
        patterns=["*.jpg"],
        recurse=True,
        image_size=384,
        skip_existing=False,
        detection_min_score=0.0,
        detection_min_mask_pixels=1,
        crop_batch_size=32,
        amp_enabled=True,
        full_manifest=True,
    )

    assert summary["full_manifest"] is True
    assert summary["predicted_count"] == 1
    assert calls["manifest"]["post_external_candidate_selector_rescue_rules"] == [{"name": "post_rule"}]
    assert calls["save_report"] is False
