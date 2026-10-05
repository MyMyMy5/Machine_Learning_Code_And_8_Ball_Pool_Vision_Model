from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.pipeline.crop_utils import extract_crop, remap_mask_to_image
from src.stages.propose_ball_crops import generate_ball_candidates
from src.supervised.config import DatasetConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize_path(path: str | Path) -> Path:
    path = str(path)
    if len(path) >= 3 and path[1:3] == ":\\":
        native_path = Path(path)
        if native_path.exists():
            return native_path
        return Path(f"/mnt/{path[0].lower()}/" + path[3:].replace("\\", "/"))
    return Path(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _file_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _dataset_signature(config: DatasetConfig) -> dict[str, Any]:
    negative_roots = []
    for root in config.negative_roots:
        root = _normalize_path(root)
        annotation_path = root / "annotations.jsonl"
        if annotation_path.exists():
            negative_roots.append(
                {
                    "root": str(root),
                    "kind": "annotations",
                    "annotations": _file_signature(annotation_path),
                }
            )
        else:
            files = _image_files(root)
            latest_mtime_ns = max((path.stat().st_mtime_ns for path in files), default=0)
            total_size = sum(path.stat().st_size for path in files)
            negative_roots.append(
                {
                    "root": str(root),
                    "kind": "flat_images",
                    "count": len(files),
                    "total_size": total_size,
                    "latest_mtime_ns": latest_mtime_ns,
                }
            )
    return {
        "main_annotations": _file_signature(config.main_root / "annotations.jsonl"),
        "quarantine_annotations": _file_signature(config.quarantine_root / "annotations.jsonl"),
        "negative_roots": negative_roots,
        "exclude_ids": sorted(str(item) for item in config.exclude_ids),
        "positive_context_sides": [int(side) for side in config.positive_context_sides],
        "image_size": config.image_size,
        "crop_padding": config.crop_padding,
        "negatives_per_image": config.negatives_per_image,
        "negative_crops_per_image": config.negative_crops_per_image,
        "min_positive_pixels": config.min_positive_pixels,
        "train_fraction": config.train_fraction,
        "val_fraction": config.val_fraction,
        "seed": config.seed,
        "negative_root_semantics": "image_level_zero_mask_v2",
    }


def _square_box_from_xyxy(box: list[int], image_shape: tuple[int, int], padding_frac: float) -> list[int]:
    x0, y0, x1, y1 = box
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    side = int(round(max(width, height) * (1.0 + padding_frac)))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    half = side / 2.0
    ix0 = max(0, int(round(cx - half)))
    iy0 = max(0, int(round(cy - half)))
    ix1 = min(image_shape[1], int(round(cx + half)))
    iy1 = min(image_shape[0], int(round(cy + half)))
    return [ix0, iy0, ix1, iy1]


def _square_box_from_center_side(
    center_x: float,
    center_y: float,
    side: int,
    image_shape: tuple[int, int],
) -> list[int]:
    side = max(1, int(side))
    half = side / 2.0
    x0 = int(round(center_x - half))
    y0 = int(round(center_y - half))
    x1 = x0 + side
    y1 = y0 + side
    height, width = image_shape[:2]
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return [max(0, x0), max(0, y0), min(width, x1), min(height, y1)]


def _mask_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _crop_mask(mask: np.ndarray, crop_box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    return (mask[y0:y1, x0:x1] > 0).astype(np.uint8)


def _candidate_config() -> dict[str, Any]:
    return {
        "max_ball_candidates": 12,
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
    }


def _full_image_crop(image_shape: tuple[int, int]) -> list[int]:
    height, width = image_shape[:2]
    side = min(height, width)
    x0 = max(0, (width - side) // 2)
    y0 = max(0, (height - side) // 2)
    return [x0, y0, x0 + side, y0 + side]


def _resize_pair(image: np.ndarray, mask: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    pil_mask = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    pil_image = TF.resize(pil_image, [image_size, image_size], antialias=True)
    pil_mask = TF.resize(pil_mask, [image_size, image_size], interpolation=TF.InterpolationMode.NEAREST)
    return np.asarray(pil_image), (np.asarray(pil_mask) > 127).astype(np.uint8)


def _resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    pil_image = TF.resize(pil_image, [image_size, image_size], antialias=True)
    return np.asarray(pil_image)


def _augment_pair(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    pil_mask = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")

    if rng.random() < 0.5:
        pil_image = TF.hflip(pil_image)
        pil_mask = TF.hflip(pil_mask)
    if rng.random() < 0.5:
        pil_image = TF.vflip(pil_image)
        pil_mask = TF.vflip(pil_mask)

    angle = rng.uniform(-10.0, 10.0)
    pil_image = TF.rotate(pil_image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=[0, 0, 0])
    pil_mask = TF.rotate(pil_mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)

    brightness = 0.75 + 0.5 * rng.random()
    contrast = 0.75 + 0.5 * rng.random()
    saturation = 0.70 + 0.7 * rng.random()
    hue = rng.uniform(-0.08, 0.08)
    pil_image = TF.adjust_brightness(pil_image, brightness)
    pil_image = TF.adjust_contrast(pil_image, contrast)
    pil_image = TF.adjust_saturation(pil_image, saturation)
    pil_image = TF.adjust_hue(pil_image, hue)

    image_np = np.asarray(pil_image)
    mask_np = (np.asarray(pil_mask) > 127).astype(np.uint8)
    return image_np, mask_np


def _augment_image_with_zero_mask(image: np.ndarray, rng: random.Random) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")

    if rng.random() < 0.5:
        pil_image = TF.hflip(pil_image)
    if rng.random() < 0.5:
        pil_image = TF.vflip(pil_image)

    angle = rng.uniform(-10.0, 10.0)
    pil_image = TF.rotate(pil_image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=[0, 0, 0])

    brightness = 0.75 + 0.5 * rng.random()
    contrast = 0.75 + 0.5 * rng.random()
    saturation = 0.70 + 0.7 * rng.random()
    hue = rng.uniform(-0.08, 0.08)
    pil_image = TF.adjust_brightness(pil_image, brightness)
    pil_image = TF.adjust_contrast(pil_image, contrast)
    pil_image = TF.adjust_saturation(pil_image, saturation)
    pil_image = TF.adjust_hue(pil_image, hue)
    return np.asarray(pil_image)


def _is_zero_mask_entry(entry: dict[str, Any]) -> bool:
    return bool(entry.get("zero_mask")) or entry.get("kind") == "candidate_negative"


def build_zoomprobe_index(config: DatasetConfig) -> dict[str, list[dict[str, Any]]]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    index_path = config.output_root / "index.json"
    meta_path = config.output_root / "index_meta.json"
    signature = _dataset_signature(config)
    if index_path.exists() and meta_path.exists():
        stored_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if stored_meta.get("signature") == signature:
            return json.loads(index_path.read_text(encoding="utf-8"))

    if index_path.exists():
        index_path.unlink()
    if meta_path.exists():
        meta_path.unlink()

    main_annotations = _load_jsonl(config.main_root / "annotations.jsonl")
    quarantine_annotations = _load_jsonl(config.quarantine_root / "annotations.jsonl")
    exclude_ids = set(str(item) for item in config.exclude_ids)
    if exclude_ids:
        main_annotations = [record for record in main_annotations if str(record["id"]) not in exclude_ids]
        quarantine_annotations = [record for record in quarantine_annotations if str(record["id"]) not in exclude_ids]
    rng = random.Random(config.seed)
    main_ids = [record["id"] for record in main_annotations]
    rng.shuffle(main_ids)
    split_point = int(round(len(main_ids) * config.train_fraction))
    train_ids = set(main_ids[:split_point])
    val_ids = set(main_ids[split_point:])

    index: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "hard_val": []}

    def append_entries(records: list[dict[str, Any]], split_name: str, negative_limit: int) -> None:
        for record in tqdm(records, desc=f"index {split_name} labels", leave=False):
            image_path = _normalize_path(record["image_path"])
            mask_path = _normalize_path(record["mask_path"])
            image = np.asarray(Image.open(image_path).convert("RGB"))
            mask = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)
            if mask.sum() == 0:
                continue
            mask_box = _mask_box(mask)
            view_box = record.get("view_xyxy") or mask_box
            view_crop = _square_box_from_xyxy(view_box, image.shape[:2], config.crop_padding)
            index[split_name].append(
                {
                    "kind": "view_positive",
                    "id": record["id"],
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "crop_box": view_crop,
                    "mask_pixels": int(mask.sum()),
                }
            )
            bbox_crop = _square_box_from_xyxy(mask_box, image.shape[:2], config.crop_padding)
            index[split_name].append(
                {
                    "kind": "bbox_positive",
                    "id": record["id"],
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "crop_box": bbox_crop,
                    "mask_pixels": int(mask.sum()),
                }
            )
            seen_positive_crops = {tuple(view_crop), tuple(bbox_crop)}
            mask_cx = (mask_box[0] + mask_box[2]) / 2.0
            mask_cy = (mask_box[1] + mask_box[3]) / 2.0
            mask_side = max(mask_box[2] - mask_box[0], mask_box[3] - mask_box[1])
            for side in sorted({int(item) for item in config.positive_context_sides if int(item) > mask_side}):
                context_crop = _square_box_from_center_side(mask_cx, mask_cy, side, image.shape[:2])
                crop_key = tuple(context_crop)
                if crop_key in seen_positive_crops:
                    continue
                seen_positive_crops.add(crop_key)
                context_mask = _crop_mask(mask, context_crop)
                if context_mask.sum() < config.min_positive_pixels:
                    continue
                index[split_name].append(
                    {
                        "kind": "context_positive",
                        "id": record["id"],
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "crop_box": context_crop,
                        "mask_pixels": int(context_mask.sum()),
                        "context_side": int(side),
                    }
                )
            candidates = generate_ball_candidates(image, _candidate_config())
            negatives = []
            for candidate in candidates:
                crop_image, crop_meta = extract_crop(image, candidate)
                crop_mask = _crop_mask(mask, list(crop_meta["crop_box"]))
                if crop_mask.sum() >= config.min_positive_pixels:
                    index[split_name].append(
                        {
                            "kind": "candidate_positive",
                            "id": record["id"],
                            "image_path": str(image_path),
                            "mask_path": str(mask_path),
                            "crop_box": list(crop_meta["crop_box"]),
                            "candidate_center": [int(candidate.center_x), int(candidate.center_y)],
                            "candidate_radius": int(candidate.radius),
                            "candidate_source": candidate.source,
                            "mask_pixels": int(crop_mask.sum()),
                        }
                    )
                elif crop_mask.sum() == 0:
                    negatives.append(
                        {
                            "kind": "candidate_negative",
                            "id": record["id"],
                            "image_path": str(image_path),
                            "mask_path": str(mask_path),
                            "crop_box": list(crop_meta["crop_box"]),
                            "candidate_center": [int(candidate.center_x), int(candidate.center_y)],
                            "candidate_radius": int(candidate.radius),
                            "candidate_source": candidate.source,
                            "mask_pixels": 0,
                            "zero_mask": True,
                        }
                    )
            rng.shuffle(negatives)
            index[split_name].extend(negatives[:negative_limit])

    def append_image_level_negative(image_path: Path, sample_id: str, kind: str) -> None:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        candidates = generate_ball_candidates(image, _candidate_config())
        entries = []
        for candidate in candidates[: max(1, config.negative_crops_per_image)]:
            _, crop_meta = extract_crop(image, candidate)
            entries.append((candidate, list(crop_meta["crop_box"])))
        if not entries:
            crop_box = _full_image_crop(image.shape[:2])
            entries.append((None, crop_box))
        for idx, (candidate, crop_box) in enumerate(entries[: max(1, config.negative_crops_per_image)]):
            center = (
                [int(candidate.center_x), int(candidate.center_y)]
                if candidate is not None
                else [int(round((crop_box[0] + crop_box[2]) / 2.0)), int(round((crop_box[1] + crop_box[3]) / 2.0))]
            )
            radius = int(candidate.radius) if candidate is not None else int(round(max(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]) * 0.08))
            index["train"].append(
                {
                    "kind": kind,
                    "id": f"{sample_id}_{idx}",
                    "image_path": str(image_path),
                    "mask_path": None,
                    "crop_box": crop_box,
                    "candidate_center": center,
                    "candidate_radius": radius,
                    "candidate_source": candidate.source if candidate is not None else "full_image",
                    "mask_pixels": 0,
                    "zero_mask": True,
                }
            )

    def append_rejected_records(records: list[dict[str, Any]], root: Path) -> None:
        for record in tqdm(records, desc=f"index rejected negatives {root.name}", leave=False):
            sample_id = str(record.get("id", ""))
            if sample_id in exclude_ids:
                continue
            image_path = _normalize_path(record.get("image_path") or root / "images" / f"{sample_id}.png")
            if not image_path.exists():
                continue
            append_image_level_negative(image_path, sample_id or image_path.stem, "rejected_negative")

    def append_flat_negative_images(image_paths: list[Path]) -> None:
        for image_path in tqdm(image_paths, desc="index flat negatives", leave=False):
            if image_path.stem in exclude_ids:
                continue
            append_image_level_negative(image_path, image_path.stem, "external_negative")

    def append_negative_roots() -> None:
        for root in config.negative_roots:
            root = _normalize_path(root)
            if not root.exists():
                continue
            annotations = _load_jsonl(root / "annotations.jsonl")
            if annotations:
                append_rejected_records(annotations, root)
            else:
                append_flat_negative_images(_image_files(root))

    append_entries([r for r in main_annotations if r["id"] in train_ids], "train", config.negatives_per_image)
    append_entries([r for r in main_annotations if r["id"] in val_ids], "val", config.negatives_per_image)
    append_entries(quarantine_annotations, "hard_val", 1)
    append_negative_roots()

    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps({"signature": signature}, indent=2), encoding="utf-8")
    return index


class ZoomProbeCropDataset(Dataset):
    def __init__(
        self,
        entries: list[dict[str, Any]],
        image_size: int,
        augment: bool,
        seed: int,
    ) -> None:
        self.entries = entries
        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        image = np.asarray(Image.open(entry["image_path"]).convert("RGB"))
        x0, y0, x1, y1 = entry["crop_box"]
        crop_image = image[y0:y1, x0:x1]

        if _is_zero_mask_entry(entry):
            crop_image = _resize_image(crop_image, self.image_size)
            if self.augment:
                crop_image = _augment_image_with_zero_mask(crop_image, self.rng)
            mask_tensor = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        else:
            mask = (np.asarray(Image.open(entry["mask_path"]).convert("L")) > 0).astype(np.uint8)
            crop_mask = mask[y0:y1, x0:x1]
            crop_image, crop_mask = _resize_pair(crop_image, crop_mask, self.image_size)
            if self.augment:
                crop_image, crop_mask = _augment_pair(crop_image, crop_mask, self.rng)
            mask_tensor = torch.from_numpy(crop_mask[None, ...].astype(np.float32))

        crop_image = np.array(crop_image, dtype=np.uint8, copy=True, order="C")
        image_tensor = torch.from_numpy(crop_image).permute(2, 0, 1).float().div_(255.0)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": entry["id"],
            "kind": entry["kind"],
        }
