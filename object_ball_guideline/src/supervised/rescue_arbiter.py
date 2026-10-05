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


def _numeric_values(item: dict[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in (
        "score",
        "heuristic_score",
        "reranker_score",
        "raw_score",
        "candidate_score",
        "confidence",
        "validity_score",
        "pred_pixels",
        "reticle_distance",
        "candidate_radius",
    ):
        value = _safe_float(item.get(key))
        if value is not None:
            values[key] = value
    pred_pixels = _safe_float(item.get("pred_pixels"))
    if pred_pixels is not None:
        values["log_pred_pixels"] = math.log1p(max(pred_pixels, 0.0))
    features = item.get("features")
    if isinstance(features, dict):
        for key, value in features.items():
            number = _safe_float(value)
            if number is not None:
                values[str(key)] = number
    return values


def _flags(item: dict[str, object]) -> set[str]:
    return {
        key.removeprefix("selected_via_")
        for key, value in item.items()
        if key.startswith("selected_via_") and bool(value)
    }


@dataclass(frozen=True)
class RescueArbiterFeatureSpec:
    numeric_names: tuple[str, ...]
    source_names: tuple[str, ...]
    source_group_names: tuple[str, ...]
    pool_names: tuple[str, ...]
    current_flag_names: tuple[str, ...]
    candidate_flag_names: tuple[str, ...]

    @property
    def input_dim(self) -> int:
        return (
            len(self.numeric_names) * 5
            + len(self.source_names) * 2
            + len(self.source_group_names) * 2
            + len(self.pool_names) * 2
            + len(self.current_flag_names)
            + len(self.candidate_flag_names)
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "numeric_names": list(self.numeric_names),
            "source_names": list(self.source_names),
            "source_group_names": list(self.source_group_names),
            "pool_names": list(self.pool_names),
            "current_flag_names": list(self.current_flag_names),
            "candidate_flag_names": list(self.candidate_flag_names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RescueArbiterFeatureSpec":
        return cls(
            numeric_names=tuple(str(item) for item in payload.get("numeric_names", [])),
            source_names=tuple(str(item) for item in payload.get("source_names", [])),
            source_group_names=tuple(str(item) for item in payload.get("source_group_names", [])),
            pool_names=tuple(str(item) for item in payload.get("pool_names", [])),
            current_flag_names=tuple(str(item) for item in payload.get("current_flag_names", [])),
            candidate_flag_names=tuple(str(item) for item in payload.get("candidate_flag_names", [])),
        )


def build_feature_spec(items: list[dict[str, object]]) -> RescueArbiterFeatureSpec:
    numeric_names: set[str] = set()
    source_names: set[str] = set()
    source_group_names: set[str] = set()
    pool_names: set[str] = set()
    current_flag_names: set[str] = set()
    candidate_flag_names: set[str] = set()
    for item in items:
        current = item["current"]
        candidate = item["candidate"]
        assert isinstance(current, dict)
        assert isinstance(candidate, dict)
        numeric_names.update(_numeric_values(current))
        numeric_names.update(_numeric_values(candidate))
        for endpoint in (current, candidate):
            source = str(endpoint.get("candidate_source", ""))
            source_names.add(source)
            source_group_names.add(_source_group(source))
            pool = str(endpoint.get("selector_pool") or "none")
            pool_names.add(pool)
        current_flag_names.update(_flags(current))
        candidate_flag_names.update(_flags(candidate))
    return RescueArbiterFeatureSpec(
        numeric_names=tuple(sorted(numeric_names)),
        source_names=tuple(sorted(source_names)),
        source_group_names=tuple(sorted(source_group_names)),
        pool_names=tuple(sorted(pool_names)),
        current_flag_names=tuple(sorted(current_flag_names)),
        candidate_flag_names=tuple(sorted(candidate_flag_names)),
    )


def build_rescue_arbiter_features(
    current: dict[str, object],
    candidate: dict[str, object],
    spec: RescueArbiterFeatureSpec,
) -> np.ndarray:
    current_values = _numeric_values(current)
    candidate_values = _numeric_values(candidate)
    vector: list[float] = []
    for name in spec.numeric_names:
        current_value = current_values.get(name)
        candidate_value = candidate_values.get(name)
        has_current = current_value is not None
        has_candidate = candidate_value is not None
        current_float = float(current_value) if has_current else 0.0
        candidate_float = float(candidate_value) if has_candidate else 0.0
        vector.extend(
            [
                current_float,
                1.0 if has_current else 0.0,
                candidate_float,
                1.0 if has_candidate else 0.0,
                candidate_float - current_float if has_current and has_candidate else 0.0,
            ]
        )

    current_source = str(current.get("candidate_source", ""))
    candidate_source = str(candidate.get("candidate_source", ""))
    current_group = _source_group(current_source)
    candidate_group = _source_group(candidate_source)
    current_pool = str(current.get("selector_pool") or "none")
    candidate_pool = str(candidate.get("selector_pool") or "none")
    current_flags = _flags(current)
    candidate_flags = _flags(candidate)
    vector.extend(1.0 if current_source == name else 0.0 for name in spec.source_names)
    vector.extend(1.0 if candidate_source == name else 0.0 for name in spec.source_names)
    vector.extend(1.0 if current_group == name else 0.0 for name in spec.source_group_names)
    vector.extend(1.0 if candidate_group == name else 0.0 for name in spec.source_group_names)
    vector.extend(1.0 if current_pool == name else 0.0 for name in spec.pool_names)
    vector.extend(1.0 if candidate_pool == name else 0.0 for name in spec.pool_names)
    vector.extend(1.0 if name in current_flags else 0.0 for name in spec.current_flag_names)
    vector.extend(1.0 if name in candidate_flags else 0.0 for name in spec.candidate_flag_names)
    return np.asarray(vector, dtype=np.float32)


class GuidelineRescueArbiter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(hidden_dim, max(24, hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(24, hidden_dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class LoadedRescueArbiter:
    model: GuidelineRescueArbiter
    feature_spec: RescueArbiterFeatureSpec
    feature_mean: np.ndarray
    feature_std: np.ndarray
    threshold: float
    device: torch.device

    def candidate_probability(self, current: dict[str, object], candidate: dict[str, object]) -> float:
        features = build_rescue_arbiter_features(current, candidate, self.feature_spec)
        normalized = (features - self.feature_mean) / self.feature_std
        tensor = torch.from_numpy(normalized).to(self.device)[None, ...]
        with torch.no_grad():
            logit = float(self.model(tensor).item())
        return float(torch.sigmoid(torch.tensor(logit)).item())


def load_rescue_arbiter_checkpoint(
    checkpoint_path,
    device: torch.device | None = None,
) -> LoadedRescueArbiter:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device)
    feature_spec = RescueArbiterFeatureSpec.from_dict(state["feature_spec"])
    feature_mean = np.asarray(state["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(state["feature_std"], dtype=np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    model = GuidelineRescueArbiter(
        input_dim=int(feature_mean.shape[0]),
        hidden_dim=int(state.get("config", {}).get("hidden_dim", 96)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return LoadedRescueArbiter(
        model=model,
        feature_spec=feature_spec,
        feature_mean=feature_mean,
        feature_std=feature_std,
        threshold=float(state.get("threshold", 0.5)),
        device=device,
    )


def choose_threshold(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, float]]:
    if probabilities.size == 0:
        return 0.5, {
            "true_positive_rate": 0.0,
            "false_positive_rate": 0.0,
            "precision": 0.0,
            "f1": 0.0,
        }
    thresholds = sorted(set(float(value) for value in probabilities), reverse=True)
    best_threshold = thresholds[0]
    best_key = -1e9
    best_metrics: dict[str, float] = {}
    positive_count = float(np.sum(targets > 0.5))
    negative_count = float(np.sum(targets <= 0.5))
    for threshold in thresholds:
        pred = probabilities >= threshold
        tp = float(np.sum(pred & (targets > 0.5)))
        fp = float(np.sum(pred & (targets <= 0.5)))
        fn = float(np.sum((~pred) & (targets > 0.5)))
        true_positive_rate = tp / max(positive_count, 1.0)
        false_positive_rate = fp / max(negative_count, 1.0)
        precision = tp / max(tp + fp, 1.0)
        f1 = (2.0 * precision * true_positive_rate) / max(precision + true_positive_rate, 1e-6)
        key = true_positive_rate - 4.0 * false_positive_rate + 0.25 * precision
        if key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = {
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
                "precision": precision,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "selection_key": key,
            }
    return float(best_threshold), best_metrics


def train_rescue_arbiter(
    train_items: list[dict[str, object]],
    val_items: list[dict[str, object]],
    *,
    epochs: int = 500,
    learning_rate: float = 8e-4,
    weight_decay: float = 1e-4,
    hidden_dim: int = 96,
    device: torch.device | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if not train_items:
        raise ValueError("rescue arbiter training requires at least one train item")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_spec = build_feature_spec(train_items)
    train_x = np.stack(
        [
            build_rescue_arbiter_features(dict(item["current"]), dict(item["candidate"]), feature_spec)
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

    model = GuidelineRescueArbiter(input_dim=feature_spec.input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if val_items:
        val_x = np.stack(
            [
                build_rescue_arbiter_features(dict(item["current"]), dict(item["candidate"]), feature_spec)
                for item in val_items
            ]
        ).astype(np.float32)
        val_y = np.asarray([float(item["target"]) for item in val_items], dtype=np.float32)
        val_tensor = torch.from_numpy((val_x - feature_mean) / feature_std).float().to(device)
    else:
        val_y = train_y
        val_tensor = train_tensor

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
        key = float(metrics["selection_key"])
        if key > best_key:
            best_key = key
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "threshold": float(threshold),
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
