from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
    main_root: Path
    quarantine_root: Path
    output_root: Path
    negative_roots: list[Path] = field(default_factory=list)
    exclude_ids: list[str] = field(default_factory=list)
    positive_context_sides: list[int] = field(default_factory=list)
    image_size: int = 384
    crop_padding: float = 0.35
    negatives_per_image: int = 3
    negative_crops_per_image: int = 4
    min_positive_pixels: int = 12
    train_fraction: float = 0.84
    val_fraction: float = 0.16
    seed: int = 1337


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 24
    num_workers: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    model_type: str = "guideline_unet"
    pretrained_encoder: bool = False
    base_channels: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    pos_weight: float = 18.0
    line_weight: float = 0.0
    neighborhood_weight: float = 0.0
    line_abs_weight: float = 0.0
    grad_clip_norm: float = 1.0
    amp_dtype: str = "bfloat16"
    channels_last: bool = True
    checkpoint_name: str = "guideline_unet_best.pt"
    train_preview_count: int = 8


@dataclass
class InferenceConfig:
    image_size: int = 384
    batch_size: int = 16
    prob_threshold: float = 0.35
    candidate_limit: int = 10
    prefer_connected: bool = True


@dataclass
class SupervisedConfig:
    dataset: DatasetConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
