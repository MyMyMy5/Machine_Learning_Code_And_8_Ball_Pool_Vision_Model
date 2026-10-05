from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.crop_utils import extract_crop, local_ball_mask, remap_mask_to_image  # noqa: E402
from src.pipeline.mask_postprocess import cleanup_mask  # noqa: E402
from src.pipeline.metrics import compute_mask_features  # noqa: E402
from src.stages.propose_ball_crops import generate_ball_candidates  # noqa: E402
from src.supervised.infer import (  # noqa: E402
    _candidate_reticle_distance,
    _load_guideline_model,
    _predict_crop,
    _rank_prediction,
    _score_prediction,
)


def _normalize_path(path_str: str) -> Path:
    if os.name != "nt" and path_str.startswith("C:\\"):
        return Path("/mnt/c/" + path_str[3:].replace("\\", "/"))
    if os.name == "nt" and path_str.startswith("/mnt/c/"):
        return Path("C:/" + path_str[len("/mnt/c/"):])
    return Path(path_str)


def _load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def _compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return 0.0 if union == 0.0 else inter / union


def _unique_positive_frames(index: dict[str, list[dict[str, object]]], split_name: str) -> list[dict[str, str]]:
    frames: dict[str, dict[str, str]] = {}
    for item in index[split_name]:
        if item.get("zero_mask") or not item.get("mask_path"):
            continue
        if str(item.get("kind", "")) not in {"view_positive", "bbox_positive", "candidate_positive"}:
            continue
        frame_id = str(item["id"])
        if frame_id.startswith("neg::"):
            continue
        frames.setdefault(
            frame_id,
            {
                "frame_id": frame_id,
                "image_path": str(item["image_path"]),
                "mask_path": str(item["mask_path"]),
            },
        )
    return [frames[key] for key in sorted(frames)]


def _negative_frames_from_gold_board(
    gold_board: dict[str, object],
    split_names: list[str],
) -> list[dict[str, str]]:
    splits = gold_board.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError("gold board does not contain a splits object")
    frames: dict[str, dict[str, str]] = {}
    for split_name in split_names:
        rows = splits.get(split_name, [])
        if not isinstance(rows, list):
            raise ValueError(f"gold board split {split_name!r} is not a list")
        for row in rows:
            if not isinstance(row, dict):
                continue
            kind = str(row.get("kind", ""))
            if kind not in {"image_negative", "flat_negative"}:
                continue
            frame_id = str(row["id"])
            frames.setdefault(
                frame_id,
                {
                    "frame_id": frame_id,
                    "image_path": str(row["image_path"]),
                    "mask_path": "",
                },
            )
    return [frames[key] for key in sorted(frames)]


def _stable_fraction(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _build_frame_candidates(
    *,
    image: np.ndarray,
    gt_mask: np.ndarray,
    model: torch.nn.Module,
    image_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    candidates = generate_ball_candidates(
        image,
        {
            "max_ball_candidates": 10,
            "crop_scale": 4.25,
            "crop_padding_px": 24,
            "upscale_factor": 1.0,
            "hough": {
                "dp": 1.2,
                "min_dist_factor": 1.5,
                "param1": 110,
                "param2": 18,
                "min_radius": 7,
                "max_radius": 80,
            },
            "fallback": {"grid_candidates": 6, "white_ball_bias": True},
        },
    )
    frame_candidates: list[dict[str, object]] = []
    for candidate in candidates:
        crop_image, crop_meta = extract_crop(image, candidate)
        ball_mask = local_ball_mask(crop_meta)
        prob = _predict_crop(model, crop_image, image_size, device)
        raw_score, binary = _score_prediction(prob, ball_mask)
        cleaned = cleanup_mask(
            binary,
            {
                "morphology_open_radius": 0,
                "morphology_close_radius": 1,
                "min_component_area": 4,
                "keep_only_best_component": True,
            },
            ball_mask=ball_mask,
        )
        features = compute_mask_features(
            cleaned,
            ball_mask,
            tuple(crop_meta["local_center"]),
            int(crop_meta["local_radius"]),
        )
        score = _rank_prediction(raw_score, candidate.score, features)
        full_mask = remap_mask_to_image(cleaned, crop_meta, image.shape)
        frame_candidates.append(
            {
                "candidate_id": candidate.candidate_id,
                "source": candidate.source,
                "candidate_score": float(candidate.score),
                "raw_score": float(raw_score),
                "confidence": float(prob[cleaned > 0].mean()) if np.count_nonzero(cleaned) else 0.0,
                "pred_pixels": int(np.count_nonzero(cleaned)),
                "reticle_distance": _candidate_reticle_distance(candidate) or -1.0,
                "iou": _compute_iou(full_mask, gt_mask),
                "score": float(score),
                "features": {key: float(value) for key, value in features.items()},
            }
        )
    return frame_candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--split-limit", type=int, default=None)
    parser.add_argument("--negative-gold-board", default=None)
    parser.add_argument("--negative-splits", nargs="*", default=[])
    parser.add_argument("--negative-val-fraction", type=float, default=0.2)
    parser.add_argument("--negative-train-limit", type=int, default=None)
    parser.add_argument("--negative-val-limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _normalize_path(args.checkpoint)
    model = _load_guideline_model(checkpoint_path, device)
    index = json.loads(_normalize_path(args.index_json).read_text(encoding="utf-8"))

    output: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "hard": []}
    split_map = {"train": "train", "val": "val", "hard": "hard_val"}

    for out_name, src_name in split_map.items():
        frames = _unique_positive_frames(index, src_name)
        if args.split_limit is not None:
            frames = frames[: args.split_limit]
        for frame_index, frame in enumerate(frames, start=1):
            image = np.asarray(Image.open(_normalize_path(frame["image_path"])).convert("RGB"))
            gt_mask = _load_mask(_normalize_path(frame["mask_path"]))
            frame_candidates = _build_frame_candidates(
                image=image,
                gt_mask=gt_mask,
                model=model,
                image_size=args.image_size,
                device=device,
            )
            output[out_name].append({"frame_id": frame["frame_id"], "candidates": frame_candidates})
            if frame_index % 25 == 0:
                print(f"{out_name}: processed {frame_index}/{len(frames)}")

    if args.negative_gold_board:
        negative_splits = args.negative_splits or ["rejected_negatives", "flat_negatives"]
        gold_board = json.loads(_normalize_path(args.negative_gold_board).read_text(encoding="utf-8"))
        negative_frames = _negative_frames_from_gold_board(gold_board, negative_splits)
        train_negatives = []
        val_negatives = []
        for frame in negative_frames:
            if _stable_fraction(frame["frame_id"]) < args.negative_val_fraction:
                val_negatives.append(frame)
            else:
                train_negatives.append(frame)
        if args.negative_train_limit is not None:
            train_negatives = train_negatives[: args.negative_train_limit]
        if args.negative_val_limit is not None:
            val_negatives = val_negatives[: args.negative_val_limit]

        output["negative_train"] = []
        output["negative_val"] = []
        for out_name, frames in (("negative_train", train_negatives), ("negative_val", val_negatives)):
            for frame_index, frame in enumerate(frames, start=1):
                image = np.asarray(Image.open(_normalize_path(frame["image_path"])).convert("RGB"))
                gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                frame_candidates = _build_frame_candidates(
                    image=image,
                    gt_mask=gt_mask,
                    model=model,
                    image_size=args.image_size,
                    device=device,
                )
                output[out_name].append({"frame_id": frame["frame_id"], "candidates": frame_candidates})
                if frame_index % 25 == 0:
                    print(f"{out_name}: processed {frame_index}/{len(frames)}")

    output_path = _normalize_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output), encoding="utf-8")
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
