from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


SOURCE_NAMES = (
    "table_hough",
    "reticle_global_0",
    "reticle_global_1",
    "reticle_global_2",
    "white_blob",
)

NUMERIC_FEATURE_NAMES = (
    "log_raw_score",
    "candidate_score",
    "confidence",
    "log_pred_pixels",
    "reticle_distance",
    "has_reticle_distance",
    "area",
    "skeleton_length",
    "average_width",
    "length_to_width_ratio",
    "thinness",
    "line_likeness",
    "branch_count",
    "component_count",
    "continuity",
    "ball_boundary_touch",
    "ball_core_overlap",
    "ball_fill_fraction",
    "connected_to_ball",
    "outward_extension",
    "shortness_penalty",
    "blob_penalty",
)


def build_reranker_features(
    *,
    candidate_source: str,
    candidate_score: float,
    raw_score: float,
    confidence: float,
    pred_pixels: int,
    reticle_distance: float | None,
    mask_features: dict[str, float],
) -> np.ndarray:
    if reticle_distance is None or reticle_distance < 0:
        reticle_value = 0.0
        has_reticle = 0.0
    else:
        reticle_value = min(float(reticle_distance), 12.0)
        has_reticle = 1.0

    numeric_values = {
        "log_raw_score": math.log1p(max(float(raw_score), 0.0)),
        "candidate_score": float(candidate_score),
        "confidence": float(confidence),
        "log_pred_pixels": math.log1p(max(int(pred_pixels), 0)),
        "reticle_distance": reticle_value,
        "has_reticle_distance": has_reticle,
        "area": float(mask_features.get("area", 0.0)),
        "skeleton_length": float(mask_features.get("skeleton_length", 0.0)),
        "average_width": float(mask_features.get("average_width", 0.0)),
        "length_to_width_ratio": float(mask_features.get("length_to_width_ratio", 0.0)),
        "thinness": float(mask_features.get("thinness", 0.0)),
        "line_likeness": float(mask_features.get("line_likeness", 0.0)),
        "branch_count": float(mask_features.get("branch_count", 0.0)),
        "component_count": float(mask_features.get("component_count", 0.0)),
        "continuity": float(mask_features.get("continuity", 0.0)),
        "ball_boundary_touch": float(mask_features.get("ball_boundary_touch", 0.0)),
        "ball_core_overlap": float(mask_features.get("ball_core_overlap", 0.0)),
        "ball_fill_fraction": float(mask_features.get("ball_fill_fraction", 0.0)),
        "connected_to_ball": float(mask_features.get("connected_to_ball", 0.0)),
        "outward_extension": float(mask_features.get("outward_extension", 0.0)),
        "shortness_penalty": float(mask_features.get("shortness_penalty", 0.0)),
        "blob_penalty": float(mask_features.get("blob_penalty", 0.0)),
    }
    vector = [numeric_values[name] for name in NUMERIC_FEATURE_NAMES]
    vector.extend(1.0 if candidate_source == source else 0.0 for source in SOURCE_NAMES)
    return np.asarray(vector, dtype=np.float32)


class GuidelineReranker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_1: int = 64, hidden_dim_2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class LoadedReranker:
    model: GuidelineReranker
    feature_mean: np.ndarray
    feature_std: np.ndarray
    device: torch.device

    def score(self, features: np.ndarray) -> np.ndarray:
        normalized = (features.astype(np.float32) - self.feature_mean) / self.feature_std
        tensor = torch.from_numpy(normalized).to(self.device)
        with torch.no_grad():
            scores = self.model(tensor).squeeze(-1).cpu().numpy()
        return scores.astype(np.float32)


def load_reranker_checkpoint(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> LoadedReranker:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device)
    feature_mean = np.asarray(state["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(state["feature_std"], dtype=np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    model = GuidelineReranker(input_dim=int(feature_mean.shape[0])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return LoadedReranker(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
    )


def _frame_to_features(frame: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    candidates = frame["candidates"]
    assert isinstance(candidates, list)
    xs = []
    ys = []
    for candidate in candidates:
        assert isinstance(candidate, dict)
        xs.append(
            build_reranker_features(
                candidate_source=str(candidate["source"]),
                candidate_score=float(candidate["candidate_score"]),
                raw_score=float(candidate["raw_score"]),
                confidence=float(candidate["confidence"]),
                pred_pixels=int(candidate["pred_pixels"]),
                reticle_distance=float(candidate["reticle_distance"]),
                mask_features=dict(candidate["features"]),
            )
        )
        ys.append(float(candidate["iou"]))
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def _prepare_split(frames: list[dict[str, object]], mean: np.ndarray, std: np.ndarray) -> list[tuple[torch.Tensor, torch.Tensor]]:
    prepared = []
    for frame in frames:
        xs, ys = _frame_to_features(frame)
        prepared.append(
            (
                torch.from_numpy((xs - mean) / std),
                torch.from_numpy(ys),
            )
        )
    return prepared


def _evaluate(model: GuidelineReranker, frames: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    model.eval()
    total = 0.0
    device = next(model.parameters()).device
    with torch.no_grad():
        for xs, ious in frames:
            scores = model(xs.to(device)).squeeze(-1).cpu()
            total += float(ious[int(torch.argmax(scores).item())].item())
    return total / max(len(frames), 1)


def _negative_reject_rate(
    model: GuidelineReranker,
    frames: list[tuple[torch.Tensor, torch.Tensor]],
    threshold: float = 0.0,
) -> float:
    if not frames:
        return 0.0
    model.eval()
    device = next(model.parameters()).device
    kept = 0
    with torch.no_grad():
        for xs, _ in frames:
            scores = model(xs.to(device)).squeeze(-1).cpu()
            kept += int(torch.max(scores).item() < threshold)
    return kept / len(frames)


def train_reranker(
    dataset: dict[str, list[dict[str, object]]],
    *,
    selection_weight_hard: float = 1.0,
    selection_weight_negative: float = 0.0,
    epochs: int = 180,
    learning_rate: float = 1.5e-3,
    weight_decay: float = 5e-5,
    alpha: float = 10.0,
    aux_mse_weight: float = 0.05,
    negative_frame_weight: float = 0.35,
    device: torch.device | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_frames = dataset["train"]
    train_x = np.concatenate([_frame_to_features(frame)[0] for frame in train_frames], axis=0)
    feature_mean = train_x.mean(axis=0).astype(np.float32)
    feature_std = train_x.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0

    prepared_train = _prepare_split(train_frames, feature_mean, feature_std)
    prepared_val = _prepare_split(dataset["val"], feature_mean, feature_std)
    prepared_hard = _prepare_split(dataset["hard"], feature_mean, feature_std)
    prepared_negative_train = _prepare_split(dataset.get("negative_train", []), feature_mean, feature_std)
    prepared_negative_val = _prepare_split(dataset.get("negative_val", []), feature_mean, feature_std)
    prepared_train.extend(prepared_negative_train)

    model = GuidelineReranker(input_dim=int(feature_mean.shape[0])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_key = -1e9
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, object] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(prepared_train)).tolist()
        running_loss = 0.0
        used_frames = 0
        for index in order:
            xs, ious = prepared_train[index]
            xs = xs.to(device)
            ious = ious.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(xs).squeeze(-1)
            if torch.max(ious).item() <= 0.0:
                loss = negative_frame_weight * torch.mean(torch.nn.functional.softplus(scores))
            else:
                target = torch.softmax(ious * alpha, dim=0)
                loss = torch.sum(-target * torch.log_softmax(scores, dim=0))
                if aux_mse_weight > 0.0:
                    mse = torch.mean((torch.sigmoid(scores) - ious) ** 2)
                    loss = loss + aux_mse_weight * mse
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            used_frames += 1

        val_iou = _evaluate(model, prepared_val)
        hard_iou = _evaluate(model, prepared_hard)
        negative_reject = _negative_reject_rate(model, prepared_negative_val, threshold=0.0)
        selection_key = (
            val_iou
            + selection_weight_hard * hard_iou
            + selection_weight_negative * negative_reject
        )
        if selection_key > best_key:
            best_key = selection_key
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {
                "epoch": epoch,
                "train_loss": running_loss / max(used_frames, 1),
                "val_iou": val_iou,
                "hard_iou": hard_iou,
                "negative_reject_rate_at_zero": negative_reject,
                "selection_key": selection_key,
                "selection_weight_hard": selection_weight_hard,
                "selection_weight_negative": selection_weight_negative,
            }

    assert best_state is not None
    assert best_metrics is not None
    checkpoint = {
        "model": best_state,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "metrics": best_metrics,
        "config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "alpha": alpha,
            "aux_mse_weight": aux_mse_weight,
            "negative_frame_weight": negative_frame_weight,
            "selection_weight_hard": selection_weight_hard,
            "selection_weight_negative": selection_weight_negative,
            "source_names": list(SOURCE_NAMES),
            "numeric_feature_names": list(NUMERIC_FEATURE_NAMES),
        },
    }
    return checkpoint, best_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--selection-weight-hard", type=float, default=1.0)
    parser.add_argument("--selection-weight-negative", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--aux-mse-weight", type=float, default=0.05)
    parser.add_argument("--negative-frame-weight", type=float, default=0.35)
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    checkpoint, metrics = train_reranker(
        dataset,
        selection_weight_hard=args.selection_weight_hard,
        selection_weight_negative=args.selection_weight_negative,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        aux_mse_weight=args.aux_mse_weight,
        negative_frame_weight=args.negative_frame_weight,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    metrics_path = Path(args.metrics_json) if args.metrics_json else output_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
