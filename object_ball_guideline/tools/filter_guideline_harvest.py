from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from tools.harvest_common import collect_image_files, normalize_path, output_stem


def _mask_pixels(mask_path: Path) -> int:
    return int((np.asarray(Image.open(mask_path).convert("L")) > 0).sum())


def _ensure_output_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "predicted": root / "Predicted",
        "masks_predicted": root / "Masks_Predicted",
        "no_prediction": root / "No_Prediction",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _clear_previous_outputs(paths: dict[str, Path]) -> None:
    _remove_if_exists(paths["predicted_overlay"])
    _remove_if_exists(paths["predicted_mask"])
    _remove_if_exists(paths["no_prediction_image"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--detection-min-score", type=float, default=-999.0)
    parser.add_argument("--detection-min-mask-pixels", type=int, default=1)
    parser.add_argument("--recurse-frames", action="store_true")
    args = parser.parse_args()

    frames_dir = normalize_path(args.frames_dir)
    inference_dir = normalize_path(args.inference_dir)
    output_root = normalize_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frame_files = collect_image_files(frames_dir, ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"], bool(args.recurse_frames))
    if not frame_files:
        raise SystemExit(f"No frame images found under: {frames_dir}")

    split_dirs = _ensure_output_dirs(output_root)
    kept: list[dict[str, object]] = []
    missed: list[dict[str, object]] = []

    for frame_path in frame_files:
        stem = output_stem(frame_path, frames_dir)
        item_dir = inference_dir / stem
        report_path = item_dir / "report.json"
        mask_path = item_dir / "mask_final.png"
        overlay_path = item_dir / "overlay_final.png"
        if not report_path.exists() or not mask_path.exists() or not overlay_path.exists():
            continue

        report = json.loads(report_path.read_text(encoding="utf-8"))
        winner = report["winner"]
        score = float(winner["score"])
        mask_pixels = _mask_pixels(mask_path)
        is_detected = score >= float(args.detection_min_score) and mask_pixels >= int(args.detection_min_mask_pixels)
        output_paths = {
            "predicted_overlay": split_dirs["predicted"] / f"{stem}.png",
            "predicted_mask": split_dirs["masks_predicted"] / f"{stem}.png",
            "no_prediction_image": split_dirs["no_prediction"] / f"{stem}{frame_path.suffix.lower()}",
        }
        _clear_previous_outputs(output_paths)

        row = {
            "stem": stem,
            "source_path": str(frame_path),
            "report_path": str(report_path),
            "score": score,
            "mask_pixels": mask_pixels,
        }
        if is_detected:
            shutil.copy2(overlay_path, output_paths["predicted_overlay"])
            shutil.copy2(mask_path, output_paths["predicted_mask"])
            row["decision"] = "predicted"
            row["output_path"] = str(output_paths["predicted_overlay"])
            row["mask_output_path"] = str(output_paths["predicted_mask"])
            kept.append(row)
        else:
            shutil.copy2(frame_path, output_paths["no_prediction_image"])
            row["decision"] = "no_prediction"
            row["output_path"] = str(output_paths["no_prediction_image"])
            row["mask_output_path"] = None
            missed.append(row)

    summary = {
        "frames_dir": str(frames_dir),
        "inference_dir": str(inference_dir),
        "output_root": str(output_root),
        "detection_min_score": float(args.detection_min_score),
        "detection_min_mask_pixels": int(args.detection_min_mask_pixels),
        "predicted_count": len(kept),
        "no_prediction_count": len(missed),
        "predicted": kept,
        "no_prediction": missed,
    }
    summary_path = output_root / "harvest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
