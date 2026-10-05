from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.pipeline.io_utils import ensure_dir, save_json
from src.supervised.conditioned import (
    ConditionedGuidelineLoss,
    ConditionedGuidelineUNet,
    conditioned_dice_score,
)
from src.supervised.conditioned_data import ConditionedZoomProbeCropDataset
from src.supervised.config import DatasetConfig, SupervisedConfig
from src.supervised.data import build_zoomprobe_index
from src.supervised.train import _amp_dtype, _checkpoint_base_channels, _configure_torch_runtime

CONDITIONED_CANDIDATE_KINDS = {
    "candidate_positive",
    "candidate_negative",
    "rejected_negative",
    "external_negative",
}


def inflate_rgb_state_for_conditioned_model(
    rgb_state: dict[str, torch.Tensor],
    conditioned_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    inflated = dict(conditioned_state)
    for key, value in rgb_state.items():
        if key not in inflated:
            continue
        target = inflated[key]
        if key == "stem.block.0.weight" and value.ndim == 4 and target.ndim == 4 and value.shape[1] == 3 and target.shape[1] == 4:
            merged = target.clone()
            merged[:, :3] = value
            merged[:, 3:] = 0.0
            inflated[key] = merged
        elif tuple(value.shape) == tuple(target.shape):
            inflated[key] = value
    return inflated


def _load_exclude_ids(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    raise ValueError(f"Expected list of ids in {path}")


def _candidate_faithful_entries(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    return [entry for entry in entries if str(entry.get("kind")) in CONDITIONED_CANDIDATE_KINDS]


def build_conditioned_dataloaders(config: SupervisedConfig, *, candidate_only: bool = False) -> dict[str, DataLoader]:
    index = build_zoomprobe_index(config.dataset)
    loaders: dict[str, DataLoader] = {}
    for split_name, augment in [("train", True), ("val", False), ("hard_val", False)]:
        entries = index[split_name]
        if candidate_only:
            entries = _candidate_faithful_entries(entries)
        dataset = ConditionedZoomProbeCropDataset(
            entries=entries,
            image_size=config.dataset.image_size,
            augment=augment,
            seed=config.dataset.seed,
            require_candidate_metadata=candidate_only,
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


def evaluate_conditioned(
    model: ConditionedGuidelineUNet,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
    pos_weight: float,
) -> dict[str, float]:
    model.eval()
    criterion = ConditionedGuidelineLoss(pos_weight=pos_weight).to(device)
    losses = []
    dices = []
    validity_correct = 0
    validity_count = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            validity = batch["validity"].to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            outputs = model(images)
            loss_parts = criterion(outputs, masks, validity)
            losses.append(float(loss_parts["loss"].item()))
            dices.extend(conditioned_dice_score(outputs["mask_logits"], masks).cpu().tolist())
            validity_pred = (torch.sigmoid(outputs["validity_logits"]) >= 0.5).float()
            validity_correct += int((validity_pred == validity).sum().item())
            validity_count += int(validity.numel())
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "dice": float(sum(dices) / max(1, len(dices))),
        "validity_accuracy": float(validity_correct / max(1, validity_count)),
    }


def _warm_start_conditioned(model: ConditionedGuidelineUNet, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_base_channels = _checkpoint_base_channels(checkpoint)
    model_base_channels = int(model.stem.block[0].weight.shape[0])
    if checkpoint_base_channels != model_base_channels:
        raise ValueError(
            f"Resume checkpoint base_channels={checkpoint_base_channels} does not match "
            f"requested base_channels={model_base_channels}"
        )
    source_state = checkpoint["model"]
    if checkpoint.get("model_config", {}).get("model_type") == "conditioned_guideline_unet":
        model.load_state_dict(source_state, strict=True)
    else:
        model.load_state_dict(inflate_rgb_state_for_conditioned_model(source_state, model.state_dict()), strict=False)


def train_conditioned(config: SupervisedConfig, resume: Path | None = None, *, candidate_only: bool = False) -> Path:
    output_root = ensure_dir(config.dataset.output_root)
    loaders = build_conditioned_dataloaders(config, candidate_only=candidate_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_torch_runtime(device)
    model = ConditionedGuidelineUNet(base_channels=config.train.base_channels).to(device)
    if config.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if resume is not None:
        _warm_start_conditioned(model, resume)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    amp_dtype = _amp_dtype(config.train.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    criterion = ConditionedGuidelineLoss(pos_weight=config.train.pos_weight).to(device)
    best_val = -1.0
    best_path = output_root / "guideline_conditioned_unet_best.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        epoch_losses = []
        progress = tqdm(loaders["train"], desc=f"conditioned epoch {epoch}/{config.train.epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            validity = batch["validity"].to(device, non_blocking=True)
            if config.train.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                outputs = model(images)
                loss_parts = criterion(outputs, masks, validity)
                loss = loss_parts["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{epoch_losses[-1]:.4f}")

        train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        val_metrics = evaluate_conditioned(model, loaders["val"], device, config.train.channels_last, config.train.pos_weight)
        hard_metrics = evaluate_conditioned(model, loaders["hard_val"], device, config.train.channels_last, config.train.pos_weight)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_validity_accuracy": val_metrics["validity_accuracy"],
                "hard_val_loss": hard_metrics["loss"],
                "hard_val_dice": hard_metrics["dice"],
                "hard_val_validity_accuracy": hard_metrics["validity_accuracy"],
            }
        )
        if val_metrics["dice"] > best_val:
            best_val = val_metrics["dice"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": {
                        "model_type": "conditioned_guideline_unet",
                        "base_channels": config.train.base_channels,
                        "in_channels": 4,
                    },
                    "config": json.loads(json.dumps(config, default=lambda o: str(o))),
                    "candidate_only": candidate_only,
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
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--negative-root", action="append", default=[])
    parser.add_argument("--negative-crops-per-image", type=int, default=4)
    parser.add_argument("--exclude-ids-json", default=None)
    parser.add_argument("--candidate-only", action="store_true")
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
            image_size=args.image_size,
            negative_crops_per_image=args.negative_crops_per_image,
        )
    )
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.num_workers = args.num_workers
    config.train.prefetch_factor = args.prefetch_factor
    config.train.base_channels = args.base_channels
    if args.learning_rate is not None:
        config.train.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.train.weight_decay = args.weight_decay
    if args.pos_weight is not None:
        config.train.pos_weight = args.pos_weight
    checkpoint = train_conditioned(
        config,
        resume=Path(args.resume) if args.resume else None,
        candidate_only=args.candidate_only,
    )
    print(str(checkpoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
