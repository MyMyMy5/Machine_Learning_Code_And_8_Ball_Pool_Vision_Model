from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _normalize_path(path: str | Path) -> Path:
    path_str = str(path).replace("\\", "/")
    if os.name == "nt" and path_str.startswith("/mnt/c/"):
        return Path("C:/" + path_str[len("/mnt/c/") :])
    if os.name != "nt" and len(path_str) >= 3 and path_str[1:3] == ":/":
        return Path("/mnt/" + path_str[0].lower() + "/" + path_str[3:].replace("\\", "/"))
    return Path(path_str)


def _load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(_normalize_path(path)).convert("RGB"), dtype=np.uint8)


def _load_mask(path: str | Path) -> np.ndarray:
    return (np.asarray(Image.open(_normalize_path(path)).convert("L")) > 0).astype(np.uint8)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expanded_square_box(
    bbox: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    min_size: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    size = int(math.ceil(max(width, height, min_size) * float(padding)))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    left = int(round(cx - size / 2))
    top = int(round(cy - size / 2))
    right = left + size
    bottom = top + size
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height
    left = max(0, left)
    top = max(0, top)
    right = max(left + 1, min(image_width, right))
    bottom = max(top + 1, min(image_height, bottom))
    return left, top, right, bottom


def _resize_channel(array: np.ndarray, size: int, *, mode: str) -> np.ndarray:
    pil = Image.fromarray(array)
    resample = Image.Resampling.NEAREST if mode == "nearest" else Image.Resampling.BILINEAR
    return np.asarray(pil.resize((size, size), resample=resample))


def build_veto_tensor(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    image_size: int,
    min_crop_size: int = 96,
    crop_padding: float = 2.6,
    include_global: bool = False,
) -> torch.Tensor:
    bbox = _mask_bbox(mask)
    if bbox is None:
        raise ValueError("image veto tensor requires a non-empty final mask")
    height, width = image.shape[:2]
    left, top, right, bottom = _expanded_square_box(
        bbox,
        image_width=width,
        image_height=height,
        min_size=min_crop_size,
        padding=crop_padding,
    )
    image_crop = image[top:bottom, left:right]
    mask_crop = mask[top:bottom, left:right]
    rgb = _resize_channel(image_crop, image_size, mode="bilinear").astype(np.float32) / 255.0
    pred_mask = _resize_channel((mask_crop * 255).astype(np.uint8), image_size, mode="nearest").astype(np.float32) / 255.0
    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)
    white_prior = ((min_rgb > 0.62) & ((max_rgb - min_rgb) < 0.23)).astype(np.float32)
    channels = [
        rgb.transpose(2, 0, 1),
        pred_mask[None, ...],
        white_prior[None, ...],
    ]
    if include_global:
        global_rgb = _resize_channel(image, image_size, mode="bilinear").astype(np.float32) / 255.0
        global_mask = _resize_channel((mask * 255).astype(np.uint8), image_size, mode="nearest").astype(np.float32) / 255.0
        global_max_rgb = global_rgb.max(axis=2)
        global_min_rgb = global_rgb.min(axis=2)
        global_white_prior = ((global_min_rgb > 0.62) & ((global_max_rgb - global_min_rgb) < 0.23)).astype(np.float32)
        channels.extend(
            [
                global_rgb.transpose(2, 0, 1),
                global_mask[None, ...],
                global_white_prior[None, ...],
            ]
        )
    tensor = np.concatenate(channels, axis=0)
    return torch.from_numpy(tensor.astype(np.float32))


class ImageFinalVetoDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        image_size: int,
        min_crop_size: int = 96,
        crop_padding: float = 2.6,
        augment: bool = False,
        cache_tensors: bool = False,
        include_global: bool = False,
    ) -> None:
        self.items = items
        self.image_size = int(image_size)
        self.min_crop_size = int(min_crop_size)
        self.crop_padding = float(crop_padding)
        self.augment = bool(augment)
        self.include_global = bool(include_global)
        self.cached_tensors: list[torch.Tensor] | None = None
        if cache_tensors:
            self.cached_tensors = [self._load_tensor(item) for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def _load_tensor(self, item: dict[str, Any]) -> torch.Tensor:
        image = _load_rgb(str(item["image_path"]))
        mask = _load_mask(str(item["mask_path"]))
        return build_veto_tensor(
            image,
            mask,
            image_size=self.image_size,
            min_crop_size=self.min_crop_size,
            crop_padding=self.crop_padding,
            include_global=self.include_global,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        tensor = self.cached_tensors[index].clone() if self.cached_tensors is not None else self._load_tensor(item)
        if self.augment:
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])
            brightness = 1.0 + random.uniform(-0.08, 0.08)
            tensor[:3] = torch.clamp(tensor[:3] * brightness, 0.0, 1.0)
        target = torch.tensor(float(item["target"]), dtype=torch.float32)
        return tensor, target


class GuidelineImageFinalVeto(nn.Module):
    def __init__(self, in_channels: int = 5, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.15),
            nn.Linear(width * 4, width * 2),
            nn.SiLU(inplace=True),
            nn.Linear(width * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


@dataclass(frozen=True)
class VetoMetrics:
    threshold: float
    accuracy: float
    neg_veto_rate: float
    pos_veto_rate: float
    true_negative_veto: int
    false_positive_veto: int
    target_veto_total: int
    keep_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "neg_veto_rate": self.neg_veto_rate,
            "pos_veto_rate": self.pos_veto_rate,
            "true_negative_veto": self.true_negative_veto,
            "false_positive_veto": self.false_positive_veto,
            "target_veto_total": self.target_veto_total,
            "keep_total": self.keep_total,
        }


def choose_veto_threshold(probabilities: np.ndarray, targets: np.ndarray) -> VetoMetrics:
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99], dtype=np.float32),
                probabilities.astype(np.float32),
                np.asarray([float(probabilities.max() + 1e-6)] if probabilities.size else [1.0], dtype=np.float32),
            ]
        )
    )
    best: VetoMetrics | None = None
    best_key = -1e9
    veto_targets = targets >= 0.5
    keep_targets = ~veto_targets
    target_total = int(veto_targets.sum())
    keep_total = int(keep_targets.sum())
    for threshold in candidates:
        preds = probabilities >= float(threshold)
        true_negative_veto = int(np.logical_and(preds, veto_targets).sum())
        false_positive_veto = int(np.logical_and(preds, keep_targets).sum())
        neg_veto_rate = true_negative_veto / max(1, target_total)
        pos_veto_rate = false_positive_veto / max(1, keep_total)
        accuracy = float((preds == veto_targets).mean()) if len(targets) else 0.0
        metrics = VetoMetrics(
            threshold=float(threshold),
            accuracy=accuracy,
            neg_veto_rate=float(neg_veto_rate),
            pos_veto_rate=float(pos_veto_rate),
            true_negative_veto=true_negative_veto,
            false_positive_veto=false_positive_veto,
            target_veto_total=target_total,
            keep_total=keep_total,
        )
        zero_positive_bonus = 1.0 if false_positive_veto == 0 else 0.0
        key = neg_veto_rate + zero_positive_bonus - 4.0 * pos_veto_rate + 0.02 * accuracy
        if key > best_key:
            best = metrics
            best_key = key
    assert best is not None
    return best


def predict_probabilities(
    model: nn.Module,
    items: list[dict[str, Any]],
    *,
    image_size: int,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    cache_tensors: bool = True,
    include_global: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = ImageFinalVetoDataset(
        items,
        image_size=image_size,
        augment=False,
        cache_tensors=cache_tensors,
        include_global=include_global,
    )
    effective_workers = 0 if cache_tensors else num_workers
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=device.type == "cuda",
    )
    return predict_probabilities_from_loader(model, loader, device=device)


def predict_probabilities_from_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[float] = []
    targets: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device, non_blocking=True))
            probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
            targets.extend(batch_y.cpu().tolist())
    return np.asarray(probabilities, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def train_image_final_veto(
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    *,
    image_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    width: int,
    device: torch.device,
    num_workers: int = 0,
    cache_tensors: bool = True,
    include_global: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    in_channels = 10 if include_global else 5
    train_dataset = ImageFinalVetoDataset(
        train_items,
        image_size=image_size,
        augment=True,
        cache_tensors=cache_tensors,
        include_global=include_global,
    )
    effective_workers = 0 if cache_tensors else num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_dataset = ImageFinalVetoDataset(
        val_items,
        image_size=image_size,
        augment=False,
        cache_tensors=cache_tensors,
        include_global=include_global,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=device.type == "cuda",
    )
    model = GuidelineImageFinalVeto(in_channels=in_channels, width=width).to(device)
    positives = sum(int(item["target"]) for item in train_items)
    negatives = len(train_items) - positives
    pos_weight = torch.tensor([max(1.0, negatives / max(1, positives))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_key = -1e9
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * int(batch_y.numel())
            seen += int(batch_y.numel())

        val_probs, val_targets = predict_probabilities_from_loader(
            model,
            device=device,
            loader=val_loader,
        )
        threshold_metrics = choose_veto_threshold(val_probs, val_targets)
        key = threshold_metrics.neg_veto_rate - 3.0 * threshold_metrics.pos_veto_rate
        if threshold_metrics.false_positive_veto == 0:
            key += 0.75
        if key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_metrics = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, seen),
                "selection_key": float(key),
                **threshold_metrics.to_dict(),
            }

    assert best_state is not None
    assert best_metrics is not None
    checkpoint = {
        "model": best_state,
        "threshold": float(best_metrics["threshold"]),
        "config": {
            "image_size": int(image_size),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "width": int(width),
            "in_channels": int(in_channels),
            "include_global": bool(include_global),
            "train_count": len(train_items),
            "val_count": len(val_items),
        },
        "metrics": best_metrics,
    }
    return checkpoint, best_metrics


def load_image_final_veto_checkpoint(path: str | Path, *, device: torch.device) -> tuple[GuidelineImageFinalVeto, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    config = dict(checkpoint.get("config", {}))
    model = GuidelineImageFinalVeto(
        in_channels=int(config.get("in_channels", 10 if config.get("include_global", False) else 5)),
        width=int(config.get("width", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def load_veto_items(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["items"])
