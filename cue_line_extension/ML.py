"""Training scaffold for a cue-line segmentation model.

This module provides a PyTorch pipeline that can be used to train a
lightweight segmentation network which predicts the white aiming line in
8-ball pool screenshots. It covers dataset loading, data augmentation,
model definition, loss calculation, training/validation loops, and basic
checkpointing. Plug in your labelled dataset of frames + binary masks to
get started.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF, InterpolationMode
from PIL import Image

import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrainConfig:
    dataset_root: Path
    output_dir: Path
    image_size: Tuple[int, int] = (512, 512)
    batch_size: int = 4
    num_workers: int = 4
    num_epochs: int = 20
    learning_rate: float = 3e-4
    scheduler_t_max: int = 0
    scheduler_min_lr: float = 1e-5
    weight_decay: float = 1e-4
    pos_weight: float = 2.0
    mixed_precision: bool = True
    save_every: int = 5
    val_split: float = 0.2
    seed: int = 42
    resume_checkpoint: Optional[Path] = None
    device: str = "auto"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LineDataset(Dataset):
    """Dataset that expects `images/` and `masks/` folders under root."""

    def __init__(self, dataset_root: Path, items: Iterable[Path], image_size: Tuple[int, int], augment: bool) -> None:
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.augment = augment
        self.image_paths = list(items)
        self.image_paths.sort()
        self.mask_dir = dataset_root / "masks"
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
        self.noise_std = 0.02

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name} at {mask_path}")

        with Image.open(image_path).convert("RGB") as img:
            with Image.open(mask_path).convert("L") as mask:
                img = img.resize(self.image_size, Image.BILINEAR)
                mask = mask.resize(self.image_size, Image.NEAREST)

                if self.augment:
                    img, mask = self._random_augment(img, mask)

                img_tensor = TF.to_tensor(img)
                if self.augment:
                    img_tensor = torch.clamp(img_tensor + torch.randn_like(img_tensor) * self.noise_std, 0.0, 1.0)
                img_tensor = self.normalize(img_tensor)
                mask_tensor = TF.to_tensor(mask)
                mask_tensor = (mask_tensor > 0.5).float()
                return img_tensor, mask_tensor

    def _random_augment(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if torch.rand(1).item() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if torch.rand(1).item() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        angle, translations, scale, shear = transforms.RandomAffine.get_params(
            degrees=(-18, 18),
            translate=(0.08, 0.08),
            scale_ranges=(0.95, 1.05),
            shears=(-3, 3),
            img_size=img.size,
        )
        img = TF.affine(
            img,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )

        if torch.rand(1).item() > 0.5:
            width, height = img.size
            start_points = [(0, 0), (width, 0), (width, height), (0, height)]
            jitter = 0.02
            offsets = [
                (
                    x + torch.empty(1).uniform_(-jitter, jitter).item() * width,
                    y + torch.empty(1).uniform_(-jitter, jitter).item() * height,
                )
                for (x, y) in start_points
            ]
            img = TF.perspective(img, start_points, offsets, InterpolationMode.BILINEAR)
            mask = TF.perspective(mask, start_points, offsets, InterpolationMode.NEAREST)

        img = self.color_jitter(img)
        return img, mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Small UNet variant suited for real-time inference."""

    def __init__(self, base_channels: int = 38) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))

        bottleneck = self.bottleneck(self.pool(c4))

        d4 = self.up4(bottleneck)
        d4 = torch.cat([d4, c4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, c3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, c2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, c1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Loss / Metrics
# ---------------------------------------------------------------------------


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0, pos_weight: float = 2.0) -> None:
        super().__init__()
        self.smooth = smooth
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        targets_flat = targets.view(targets.size(0), -1)
        probs_flat = probs.view(probs.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = 1 - ((2 * intersection + self.smooth) / (probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth))
        return bce_loss + dice.mean()


@torch.no_grad()
def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-5) / (union + 1e-5)
    return dice.mean()


# ---------------------------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(root: Path, val_split: float, seed: int) -> Tuple[list[Path], list[Path]]:
    image_dir = root / "images"
    paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
    if not paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    for path_list in (list(image_dir.glob("*.PNG")), list(image_dir.glob("*.JPG")), list(image_dir.glob("*.JPEG"))):
        paths.extend(path_list)
    unique_paths = {p.resolve(): p for p in paths}
    paths = list(unique_paths.values())
    paths.sort(key=lambda p: p.name.lower())
    rng = random.Random(seed)
    rng.shuffle(paths)
    val_size = max(1, int(len(paths) * val_split)) if len(paths) > 1 else 0
    if val_size == 0:
        return paths, []
    val_paths = paths[:val_size]
    train_paths = paths[val_size:]
    return train_paths, val_paths


def create_loaders(config: TrainConfig, *, pin_memory: bool) -> Tuple[DataLoader, DataLoader | None]:
    train_paths, val_paths = split_dataset(config.dataset_root, config.val_split, config.seed)
    train_ds = LineDataset(config.dataset_root, train_paths, config.image_size, augment=True)
    val_ds = LineDataset(config.dataset_root, val_paths, config.image_size, augment=False) if val_paths else None

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=pin_memory and config.num_workers > 0,
    )

    val_loader = (
        DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=pin_memory and config.num_workers > 0,
        )
        if val_ds is not None
        else None
    )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    pos_weight: float,
) -> Tuple[float, float]:
    model.train()
    criterion = DiceBCELoss(pos_weight=pos_weight)
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_dice += dice_coefficient(logits.detach(), masks).item() * images.size(0)

    num_samples = len(loader.dataset)
    return total_loss / num_samples, total_dice / num_samples


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, pos_weight: float) -> Tuple[float, float]:
    model.eval()
    criterion = DiceBCELoss(pos_weight=pos_weight)
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item() * images.size(0)
            total_dice += dice_coefficient(logits, masks).item() * images.size(0)

    num_samples = len(loader.dataset)
    return total_loss / num_samples, total_dice / num_samples


# ---------------------------------------------------------------------------
# Checkpointing / Logging
# ---------------------------------------------------------------------------


def save_checkpoint(state: dict, output_dir: Path, epoch: int, tag: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    ckpt_path = output_dir / f"checkpoint_epoch_{epoch:03d}{suffix}.pth"
    torch.save(state, ckpt_path)


def log_metrics(output_dir: Path, epoch: int, train_loss: float, train_dice: float, val_loss: float, val_dice: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_dice": train_dice,
        "val_loss": val_loss,
        "val_dice": val_dice,
    }
    log_path = output_dir / "metrics.jsonl"
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    if config.device == "cpu":
        device = torch.device("cpu")
    elif config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)

    pin_memory = device.type == "cuda"
    train_loader, val_loader = create_loaders(config, pin_memory=pin_memory)
    train_count = len(train_loader.dataset)
    val_count = len(val_loader.dataset) if val_loader is not None else 0
    print(f"Loaded dataset from {config.dataset_root} -> train={train_count} samples, val={val_count} samples")

    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    t_max = config.scheduler_t_max if config.scheduler_t_max > 0 else config.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, t_max), eta_min=config.scheduler_min_lr)
    scaler = torch.amp.GradScaler("cuda") if (config.mixed_precision and device.type == "cuda") else None

    start_epoch = 1
    best_val_loss = float("inf")

    if config.resume_checkpoint is not None:
        checkpoint = torch.load(config.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resuming from {config.resume_checkpoint} at epoch {start_epoch}")

    for epoch in range(start_epoch, config.num_epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, scaler, device, config.pos_weight)

        if val_loader is not None:
            val_loss, val_dice = evaluate(model, val_loader, device, config.pos_weight)
        else:
            val_loss, val_dice = float('nan'), float('nan')

        log_metrics(config.output_dir, epoch, train_loss, train_dice, val_loss, val_dice)

        print(
            f"Epoch {epoch:02d}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

        scheduler.step()

        checkpoint_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
        }

        if val_loader is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_payload["best_val_loss"] = best_val_loss
            save_checkpoint(checkpoint_payload, config.output_dir, epoch, tag="best")

        if config.save_every and epoch % config.save_every == 0:
            save_checkpoint(checkpoint_payload, config.output_dir, epoch)

    if checkpoint_payload is not None:
        save_checkpoint(checkpoint_payload, config.output_dir, epoch, tag="final")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a cue-line segmentation model")
    parser.add_argument("dataset_root", type=Path, help="Directory with images/ and masks/ subfolders")
    parser.add_argument("output_dir", type=Path, help="Directory to store checkpoints and logs")
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=2.0, help="Positive class weight for BCE loss")
    parser.add_argument("--scheduler-t-max", type=int, default=0, help="Period for cosine annealing (0 defaults to num_epochs)")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5, help="Minimum learning rate for cosine annealing")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to run training on")
    args = parser.parse_args()

    return TrainConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler_t_max=args.scheduler_t_max,
        scheduler_min_lr=args.scheduler_min_lr,
        pos_weight=args.pos_weight,
        mixed_precision=not args.disable_amp,
        val_split=args.val_split,
        seed=args.seed,
        resume_checkpoint=args.resume,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
