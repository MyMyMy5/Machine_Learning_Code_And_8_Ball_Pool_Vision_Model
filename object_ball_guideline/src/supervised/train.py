from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.pipeline.io_utils import ensure_dir, save_json
from src.supervised.config import DatasetConfig, SupervisedConfig
from src.supervised.data import ZoomProbeCropDataset, build_zoomprobe_index
from src.supervised.model import build_guideline_model


def _configure_torch_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _amp_dtype(name: str) -> torch.dtype:
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {name}")


def _checkpoint_base_channels(checkpoint: dict[str, object]) -> int:
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, dict) and "base_channels" in model_config:
        return int(model_config["base_channels"])
    model_state = checkpoint.get("model")
    if isinstance(model_state, dict):
        stem_weight = model_state.get("stem.block.0.weight")
        if isinstance(stem_weight, torch.Tensor):
            return int(stem_weight.shape[0])
    return 32


def _checkpoint_model_type(checkpoint: dict[str, object]) -> str:
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, dict):
        return str(model_config.get("model_type", "guideline_unet"))
    return "guideline_unet"


def _load_exclude_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"exclude ids file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of ids in {path}")
    return [str(item) for item in payload]


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (2.0 * intersection + 1.0) / (union + 1.0)


class DiceBCE(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        line_weight: float = 0.0,
        neighborhood_weight: float = 0.0,
        line_abs_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.line_weight = float(line_weight)
        self.neighborhood_weight = float(neighborhood_weight)
        self.line_abs_weight = float(line_abs_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.tensor([self.pos_weight], dtype=torch.float32, device=targets.device)
        weights = None
        if self.line_weight > 0.0 or self.neighborhood_weight > 0.0:
            dilated = F.max_pool2d(targets, kernel_size=3, stride=1, padding=1)
            weights = 1.0 + self.line_weight * targets + self.neighborhood_weight * (dilated - targets).clamp_min(0.0)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, weight=weights)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()
        if weights is None or self.line_abs_weight <= 0.0:
            return bce + dice
        line_abs = ((probs - targets).abs() * weights).mean()
        return bce + dice + self.line_abs_weight * line_abs


def _segmentation_criterion(config: SupervisedConfig) -> DiceBCE:
    return DiceBCE(
        pos_weight=config.train.pos_weight,
        line_weight=config.train.line_weight,
        neighborhood_weight=config.train.neighborhood_weight,
        line_abs_weight=config.train.line_abs_weight,
    )


def build_dataloaders(config: SupervisedConfig) -> dict[str, DataLoader]:
    index = build_zoomprobe_index(config.dataset)
    loaders = {}
    for split_name, augment in [("train", True), ("val", False), ("hard_val", False)]:
        dataset = ZoomProbeCropDataset(
            entries=index[split_name],
            image_size=config.dataset.image_size,
            augment=augment,
            seed=config.dataset.seed,
        )
        is_train = split_name == "train"
        num_workers = config.train.num_workers if is_train else min(config.train.num_workers, 4)
        loader_kwargs = {
            "batch_size": config.train.batch_size if is_train else config.inference.batch_size,
            "shuffle": is_train,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "drop_last": False,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = config.train.persistent_workers and is_train
            loader_kwargs["prefetch_factor"] = config.train.prefetch_factor
        loaders[split_name] = DataLoader(dataset, **loader_kwargs)
    return loaders


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
    pos_weight: float,
) -> dict[str, float]:
    model.eval()
    losses = []
    dices = []
    criterion = DiceBCE(pos_weight=pos_weight).to(device)
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            logits = model(images)
            losses.append(float(criterion(logits, masks).item()))
            dices.extend(dice_score(logits, masks).cpu().tolist())
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "dice": float(sum(dices) / max(1, len(dices))),
    }


def train(config: SupervisedConfig) -> Path:
    output_root = ensure_dir(config.dataset.output_root)
    loaders = build_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_torch_runtime(device)
    model = build_guideline_model(
        model_type=config.train.model_type,
        base_channels=config.train.base_channels,
        pretrained_encoder=config.train.pretrained_encoder,
    ).to(device)
    if config.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    amp_dtype = _amp_dtype(config.train.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    criterion = _segmentation_criterion(config).to(device)
    best_val = -1.0
    best_path = output_root / config.train.checkpoint_name
    history: list[dict[str, float]] = []
    start_epoch = 1
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location="cpu")
        checkpoint_base_channels = _checkpoint_base_channels(checkpoint)
        checkpoint_model_type = _checkpoint_model_type(checkpoint)
        if checkpoint_model_type != config.train.model_type:
            raise ValueError(
                f"Checkpoint model_type={checkpoint_model_type} does not match "
                f"requested model_type={config.train.model_type}"
            )
        if checkpoint_base_channels != config.train.base_channels:
            raise ValueError(
                f"Checkpoint base_channels={checkpoint_base_channels} does not match "
                f"requested base_channels={config.train.base_channels}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        stored_history = checkpoint.get("history")
        if isinstance(stored_history, list):
            history = stored_history
            best_val = max((float(item.get("val_dice", -1.0)) for item in history), default=-1.0)
            start_epoch = len(history) + 1

    for epoch in range(start_epoch, config.train.epochs + 1):
        model.train()
        epoch_losses = []
        progress = tqdm(loaders["train"], desc=f"epoch {epoch}/{config.train.epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            if config.train.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{epoch_losses[-1]:.4f}")

        train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        val_metrics = evaluate(model, loaders["val"], device, config.train.channels_last, config.train.pos_weight)
        hard_metrics = evaluate(model, loaders["hard_val"], device, config.train.channels_last, config.train.pos_weight)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "hard_val_loss": hard_metrics["loss"],
                "hard_val_dice": hard_metrics["dice"],
            }
        )
        if val_metrics["dice"] > best_val:
            best_val = val_metrics["dice"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": {
                        "model_type": config.train.model_type,
                        "base_channels": config.train.base_channels,
                        "in_channels": 3,
                    },
                    "config": json.loads(json.dumps(config, default=lambda o: str(o))),
                    "history": history,
                },
                best_path,
            )
        save_json(output_root / "history.json", {"history": history})
    return best_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--model-type", default="guideline_unet")
    parser.add_argument("--pretrained-encoder", action="store_true")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--line-weight", type=float, default=None)
    parser.add_argument("--neighborhood-weight", type=float, default=None)
    parser.add_argument("--line-abs-weight", type=float, default=None)
    parser.add_argument("--negative-root", action="append", default=[])
    parser.add_argument("--negative-crops-per-image", type=int, default=4)
    parser.add_argument(
        "--positive-context-side",
        action="append",
        type=int,
        default=[],
        help="Add mask-centered positive training crops with this side length in source pixels. Repeatable.",
    )
    parser.add_argument("--exclude-ids-json", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    config = SupervisedConfig(
        dataset=DatasetConfig(
            main_root=data_root,
            quarantine_root=data_root / "quarantine",
            output_root=Path(args.output_root),
            negative_roots=[Path(item) for item in args.negative_root],
            exclude_ids=_load_exclude_ids(Path(args.exclude_ids_json)) if args.exclude_ids_json else [],
            positive_context_sides=[int(item) for item in args.positive_context_side],
            image_size=args.image_size,
            negative_crops_per_image=args.negative_crops_per_image,
        )
    )
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.num_workers = args.num_workers
    config.train.prefetch_factor = args.prefetch_factor
    config.train.model_type = args.model_type
    config.train.pretrained_encoder = bool(args.pretrained_encoder)
    config.train.base_channels = args.base_channels
    if args.learning_rate is not None:
        config.train.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.train.weight_decay = args.weight_decay
    if args.pos_weight is not None:
        config.train.pos_weight = args.pos_weight
    if args.line_weight is not None:
        config.train.line_weight = args.line_weight
    if args.neighborhood_weight is not None:
        config.train.neighborhood_weight = args.neighborhood_weight
    if args.line_abs_weight is not None:
        config.train.line_abs_weight = args.line_abs_weight

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        checkpoint_base_channels = _checkpoint_base_channels(ckpt)
        checkpoint_model_type = _checkpoint_model_type(ckpt)
        if checkpoint_model_type != config.train.model_type:
            raise ValueError(
                f"Resume checkpoint model_type={checkpoint_model_type} does not match "
                f"requested model_type={config.train.model_type}"
            )
        if checkpoint_base_channels != config.train.base_channels:
            raise ValueError(
                f"Resume checkpoint base_channels={checkpoint_base_channels} does not match "
                f"requested base_channels={config.train.base_channels}"
            )
        # Warm-start by copying the previous best weights into the fresh run.
        ensure_dir(config.dataset.output_root)
        warm_path = Path(args.resume)
        target_path = config.dataset.output_root / config.train.checkpoint_name
        if warm_path.resolve() != target_path.resolve() and not target_path.exists():
            ckpt = dict(ckpt)
            ckpt.pop("history", None)
            torch.save(ckpt, target_path)
    checkpoint = train(config)
    print(str(checkpoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
