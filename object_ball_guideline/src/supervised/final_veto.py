from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


def _source_group(source: object) -> str:
    source_str = str(source or "")
    if source_str.startswith("reticle_global"):
        return "reticle_global"
    if source_str.startswith("external_main"):
        return "external_main"
    if source_str.startswith("external_tiny"):
        return "external_tiny"
    return source_str


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _flatten_numeric(prefix: str, value: object, out: dict[str, float]) -> None:
    number = _safe_float(value)
    if number is not None:
        out[prefix] = number
        return
    if isinstance(value, dict):
        for key, child in value.items():
            key_str = str(key)
            if key_str in {"mask", "crop_meta", "full_frame_mask"}:
                continue
            _flatten_numeric(f"{prefix}.{key_str}" if prefix else key_str, child, out)
        return
    if isinstance(value, (list, tuple)) and len(value) <= 6:
        for index, child in enumerate(value):
            child_number = _safe_float(child)
            if child_number is not None:
                out[f"{prefix}.{index}"] = child_number


def _winner_numeric_values(winner: dict[str, object], pred_pixels: int) -> dict[str, float]:
    values: dict[str, float] = {
        "final_pred_pixels": float(pred_pixels),
        "log_final_pred_pixels": math.log1p(max(int(pred_pixels), 0)),
    }
    for key, value in winner.items():
        key_str = str(key)
        if key_str in {"candidate_id", "candidate_source", "mask", "crop_meta", "full_frame_mask"}:
            continue
        if key_str.startswith("selected_via_"):
            continue
        _flatten_numeric(key_str, value, values)
    return values


def _winner_flags(winner: dict[str, object]) -> set[str]:
    return {
        key.removeprefix("selected_via_")
        for key, value in winner.items()
        if key.startswith("selected_via_") and bool(value)
    }


@dataclass(frozen=True)
class FinalVetoFeatureSpec:
    numeric_names: tuple[str, ...]
    source_names: tuple[str, ...]
    source_group_names: tuple[str, ...]
    flag_names: tuple[str, ...]

    @property
    def input_dim(self) -> int:
        return (
            len(self.numeric_names) * 2
            + len(self.source_names)
            + len(self.source_group_names)
            + len(self.flag_names)
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "numeric_names": list(self.numeric_names),
            "source_names": list(self.source_names),
            "source_group_names": list(self.source_group_names),
            "flag_names": list(self.flag_names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FinalVetoFeatureSpec":
        return cls(
            numeric_names=tuple(str(item) for item in payload.get("numeric_names", [])),
            source_names=tuple(str(item) for item in payload.get("source_names", [])),
            source_group_names=tuple(str(item) for item in payload.get("source_group_names", [])),
            flag_names=tuple(str(item) for item in payload.get("flag_names", [])),
        )


def build_feature_spec(items: list[dict[str, object]]) -> FinalVetoFeatureSpec:
    numeric_names: set[str] = set()
    source_names: set[str] = set()
    source_group_names: set[str] = set()
    flag_names: set[str] = set()
    for item in items:
        winner = item["winner"]
        assert isinstance(winner, dict)
        pred_pixels = int(item.get("pred_pixels", 0))
        numeric_names.update(_winner_numeric_values(winner, pred_pixels))
        source = str(winner.get("candidate_source", ""))
        source_names.add(source)
        source_group_names.add(_source_group(source))
        flag_names.update(_winner_flags(winner))
    return FinalVetoFeatureSpec(
        numeric_names=tuple(sorted(numeric_names)),
        source_names=tuple(sorted(source_names)),
        source_group_names=tuple(sorted(source_group_names)),
        flag_names=tuple(sorted(flag_names)),
    )


def build_final_veto_features(
    winner: dict[str, object],
    pred_pixels: int,
    spec: FinalVetoFeatureSpec,
) -> np.ndarray:
    numeric_values = _winner_numeric_values(winner, pred_pixels)
    vector: list[float] = []
    for name in spec.numeric_names:
        if name in numeric_values:
            vector.append(float(numeric_values[name]))
            vector.append(1.0)
        else:
            vector.append(0.0)
            vector.append(0.0)

    source = str(winner.get("candidate_source", ""))
    source_group = _source_group(source)
    flags = _winner_flags(winner)
    vector.extend(1.0 if source == name else 0.0 for name in spec.source_names)
    vector.extend(1.0 if source_group == name else 0.0 for name in spec.source_group_names)
    vector.extend(1.0 if name in flags else 0.0 for name in spec.flag_names)
    return np.asarray(vector, dtype=np.float32)


class GuidelineFinalVeto(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class LoadedFinalVeto:
    model: GuidelineFinalVeto
    feature_spec: FinalVetoFeatureSpec
    feature_mean: np.ndarray
    feature_std: np.ndarray
    threshold: float
    device: torch.device

    def veto_probability(self, winner: dict[str, object], pred_pixels: int) -> float:
        features = build_final_veto_features(winner, pred_pixels, self.feature_spec)
        normalized = (features - self.feature_mean) / self.feature_std
        tensor = torch.from_numpy(normalized).to(self.device)[None, ...]
        with torch.no_grad():
            logit = float(self.model(tensor).item())
        return float(torch.sigmoid(torch.tensor(logit)).item())

    def should_veto(self, winner: dict[str, object], pred_pixels: int) -> tuple[bool, float]:
        probability = self.veto_probability(winner, pred_pixels)
        return probability >= self.threshold, probability


def load_final_veto_checkpoint(
    checkpoint_path,
    device: torch.device | None = None,
) -> LoadedFinalVeto:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device)
    feature_spec = FinalVetoFeatureSpec.from_dict(state["feature_spec"])
    feature_mean = np.asarray(state["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(state["feature_std"], dtype=np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    model = GuidelineFinalVeto(
        input_dim=int(feature_mean.shape[0]),
        hidden_dim=int(state.get("config", {}).get("hidden_dim", 64)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return LoadedFinalVeto(
        model=model,
        feature_spec=feature_spec,
        feature_mean=feature_mean,
        feature_std=feature_std,
        threshold=float(state.get("threshold", 0.5)),
        device=device,
    )


def train_final_veto(
    train_items: list[dict[str, object]],
    val_items: list[dict[str, object]],
    *,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    device: torch.device | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if not train_items:
        raise ValueError("final veto training requires at least one train item")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_spec = build_feature_spec(train_items)

    train_x = np.stack(
        [
            build_final_veto_features(dict(item["winner"]), int(item.get("pred_pixels", 0)), feature_spec)
            for item in train_items
        ]
    ).astype(np.float32)
    train_y = np.asarray([float(item["target"]) for item in train_items], dtype=np.float32)
    feature_mean = train_x.mean(axis=0).astype(np.float32)
    feature_std = train_x.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0

    train_tensor = torch.from_numpy((train_x - feature_mean) / feature_std).float().to(device)
    target_tensor = torch.from_numpy(train_y).float().to(device)
    positives = float(train_y.sum())
    negatives = float(len(train_y) - train_y.sum())
    pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))], dtype=torch.float32, device=device)

    model = GuidelineFinalVeto(input_dim=feature_spec.input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    val_x = None
    val_y = None
    if val_items:
        val_x = np.stack(
            [
                build_final_veto_features(dict(item["winner"]), int(item.get("pred_pixels", 0)), feature_spec)
                for item in val_items
            ]
        ).astype(np.float32)
        val_y = np.asarray([float(item["target"]) for item in val_items], dtype=np.float32)
        val_tensor = torch.from_numpy((val_x - feature_mean) / feature_std).float().to(device)
    else:
        val_tensor = train_tensor
        val_y = train_y

    best_key = -1e9
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, object] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_tensor)
        loss = criterion(logits, target_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(val_tensor)).detach().cpu().numpy()
        threshold, metrics = choose_threshold(val_probs, val_y)
        selection_key = float(metrics["neg_veto_rate"]) - 2.0 * float(metrics["pos_veto_rate"])
        if selection_key > best_key:
            best_key = selection_key
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "threshold": float(threshold),
                "selection_key": selection_key,
                **metrics,
            }

    assert best_state is not None
    assert best_metrics is not None
    checkpoint = {
        "model": best_state,
        "feature_spec": feature_spec.to_dict(),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "threshold": float(best_metrics["threshold"]),
        "metrics": best_metrics,
        "config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "hidden_dim": hidden_dim,
            "train_count": len(train_items),
            "val_count": len(val_items),
        },
    }
    return checkpoint, best_metrics


def choose_threshold(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, float]]:
    if probabilities.size == 0:
        return 0.5, {
            "accuracy": 0.0,
            "pos_veto_rate": 0.0,
            "neg_veto_rate": 0.0,
            "true_positive_veto": 0.0,
            "false_positive_veto": 0.0,
        }
    candidates = np.unique(np.concatenate([[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9], probabilities]))
    best_threshold = 0.5
    best_key = -1e9
    best_metrics: dict[str, float] = {}
    for threshold in candidates:
        preds = probabilities >= float(threshold)
        veto_targets = targets >= 0.5
        non_veto_targets = ~veto_targets
        true_positive_veto = float(np.logical_and(preds, veto_targets).sum())
        false_positive_veto = float(np.logical_and(preds, non_veto_targets).sum())
        pos_total = float(non_veto_targets.sum())
        neg_total = float(veto_targets.sum())
        pos_veto_rate = false_positive_veto / max(1.0, pos_total)
        neg_veto_rate = true_positive_veto / max(1.0, neg_total)
        accuracy = float((preds == veto_targets).mean())
        # Prefer suppressing negatives but strongly penalize vetoes on keep-labeled rows.
        key = neg_veto_rate - 2.0 * pos_veto_rate + 0.05 * accuracy
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = {
                "accuracy": accuracy,
                "pos_veto_rate": pos_veto_rate,
                "neg_veto_rate": neg_veto_rate,
                "true_positive_veto": true_positive_veto,
                "false_positive_veto": false_positive_veto,
            }
    return best_threshold, best_metrics
