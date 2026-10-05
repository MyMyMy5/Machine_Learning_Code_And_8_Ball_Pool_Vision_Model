from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src.supervised.data import _is_zero_mask_entry


def build_candidate_heatmap(
    *,
    crop_box: list[int],
    candidate_center: list[int] | None,
    candidate_radius: int | float | None,
    output_size: int,
) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in crop_box]
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)
    if candidate_center is None:
        local_x = crop_w / 2.0
        local_y = crop_h / 2.0
    else:
        local_x = float(candidate_center[0]) - x0
        local_y = float(candidate_center[1]) - y0
    scaled_x = local_x * float(output_size) / crop_w
    scaled_y = local_y * float(output_size) / crop_h
    radius = float(candidate_radius or max(crop_w, crop_h) * 0.08)
    sigma = max(1.5, radius * float(output_size) / max(crop_w, crop_h) * 0.55)
    yy, xx = np.mgrid[:output_size, :output_size]
    heatmap = np.exp(-(((xx - scaled_x) ** 2 + (yy - scaled_y) ** 2) / (2.0 * sigma**2)))
    heatmap /= max(float(heatmap.max()), 1e-6)
    return heatmap.astype(np.float32)


def _resize_image_mask_heatmap(
    image: np.ndarray,
    mask: np.ndarray,
    heatmap: np.ndarray,
    image_size: int,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    pil_mask = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    pil_heatmap = Image.fromarray((np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    pil_image = TF.resize(pil_image, [image_size, image_size], antialias=True)
    pil_mask = TF.resize(pil_mask, [image_size, image_size], interpolation=TF.InterpolationMode.NEAREST)
    pil_heatmap = TF.resize(pil_heatmap, [image_size, image_size], interpolation=TF.InterpolationMode.BILINEAR)
    return pil_image, pil_mask, pil_heatmap


def _augment_triplet(
    image: Image.Image,
    mask: Image.Image,
    heatmap: Image.Image,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    if rng.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
        heatmap = TF.hflip(heatmap)
    if rng.random() < 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
        heatmap = TF.vflip(heatmap)
    angle = rng.uniform(-10.0, 10.0)
    image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=[0, 0, 0])
    mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
    heatmap = TF.rotate(heatmap, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    image = TF.adjust_brightness(image, 0.75 + 0.5 * rng.random())
    image = TF.adjust_contrast(image, 0.75 + 0.5 * rng.random())
    image = TF.adjust_saturation(image, 0.70 + 0.7 * rng.random())
    image = TF.adjust_hue(image, rng.uniform(-0.08, 0.08))
    return image, mask, heatmap


class ConditionedZoomProbeCropDataset(Dataset):
    def __init__(
        self,
        entries: list[dict[str, Any]],
        image_size: int,
        augment: bool,
        seed: int,
        require_candidate_metadata: bool = False,
    ) -> None:
        self.entries = entries
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.rng = random.Random(seed)
        self.require_candidate_metadata = bool(require_candidate_metadata)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        image = np.asarray(Image.open(Path(entry["image_path"])).convert("RGB"))
        x0, y0, x1, y1 = [int(v) for v in entry["crop_box"]]
        crop_image = image[y0:y1, x0:x1]
        zero_mask = _is_zero_mask_entry(entry)
        if zero_mask:
            crop_mask = np.zeros(crop_image.shape[:2], dtype=np.uint8)
        else:
            if self.require_candidate_metadata and entry.get("candidate_center") is None:
                raise ValueError(f"Conditioned positive entry lacks candidate metadata: {entry['id']}")
            mask = (np.asarray(Image.open(Path(entry["mask_path"])).convert("L")) > 0).astype(np.uint8)
            crop_mask = mask[y0:y1, x0:x1]
        heatmap = build_candidate_heatmap(
            crop_box=[x0, y0, x1, y1],
            candidate_center=entry.get("candidate_center"),
            candidate_radius=entry.get("candidate_radius"),
            output_size=self.image_size,
        )
        pil_image, pil_mask, pil_heatmap = _resize_image_mask_heatmap(crop_image, crop_mask, heatmap, self.image_size)
        if self.augment:
            pil_image, pil_mask, pil_heatmap = _augment_triplet(pil_image, pil_mask, pil_heatmap, self.rng)

        image_np = np.array(pil_image, dtype=np.uint8, copy=True, order="C")
        mask_np = (np.asarray(pil_mask) > 127).astype(np.float32)
        heatmap_np = (np.asarray(pil_heatmap).astype(np.float32) / 255.0)[None, ...]
        rgb_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().div_(255.0)
        heatmap_tensor = torch.from_numpy(np.array(heatmap_np, dtype=np.float32, copy=True, order="C"))
        return {
            "image": torch.cat([rgb_tensor, heatmap_tensor], dim=0),
            "mask": torch.from_numpy(mask_np[None, ...]),
            "validity": torch.tensor([0.0 if zero_mask else 1.0], dtype=torch.float32),
            "id": entry["id"],
            "kind": entry["kind"],
        }
