from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.supervised.reranker import SOURCE_NAMES, build_reranker_features, load_reranker_checkpoint


SELECTED_NUMERIC_FEATURES = (
    "score",
    "raw_score",
    "candidate_score",
    "confidence",
    "pred_pixels",
    "reticle_distance",
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


def _selected_candidate_vector(item: dict[str, object]) -> np.ndarray:
    features = dict(item["features"])
    vector = [
        float(item["score"]),
        float(item["raw_score"]),
        float(item["candidate_score"]),
        float(item["confidence"]),
        float(item["pred_pixels"]),
        float(item["reticle_distance"]),
        float(features.get("area", 0.0)),
        float(features.get("skeleton_length", 0.0)),
        float(features.get("average_width", 0.0)),
        float(features.get("length_to_width_ratio", 0.0)),
        float(features.get("thinness", 0.0)),
        float(features.get("line_likeness", 0.0)),
        float(features.get("branch_count", 0.0)),
        float(features.get("component_count", 0.0)),
        float(features.get("continuity", 0.0)),
        float(features.get("ball_boundary_touch", 0.0)),
        float(features.get("ball_core_overlap", 0.0)),
        float(features.get("ball_fill_fraction", 0.0)),
        float(features.get("connected_to_ball", 0.0)),
        float(features.get("outward_extension", 0.0)),
        float(features.get("shortness_penalty", 0.0)),
        float(features.get("blob_penalty", 0.0)),
    ]
    vector.extend(1.0 if item["candidate_source"] == source else 0.0 for source in SOURCE_NAMES)
    return np.asarray(vector, dtype=np.float32)


def build_meta_selector_features(
    primary_item: dict[str, object],
    fallback_item: dict[str, object],
) -> np.ndarray:
    primary = _selected_candidate_vector(primary_item)
    fallback = _selected_candidate_vector(fallback_item)
    delta = primary - fallback
    summary = np.asarray(
        [
            float(primary_item["score"]),
            float(fallback_item["score"]),
            float(primary_item["score"]) - float(fallback_item["score"]),
        ],
        dtype=np.float32,
    )
    return np.concatenate([primary, fallback, delta, summary]).astype(np.float32)


class GuidelineMetaSelector(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LoadedMetaSelector:
    def __init__(
        self,
        model: GuidelineMetaSelector,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        device: torch.device,
    ) -> None:
        self.model = model
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.feature_std[self.feature_std < 1e-6] = 1.0
        self.device = device

    def choose_fallback(
        self,
        primary_item: dict[str, object],
        fallback_item: dict[str, object],
    ) -> tuple[bool, float]:
        features = build_meta_selector_features(primary_item, fallback_item)
        normalized = (features - self.feature_mean) / self.feature_std
        tensor = torch.from_numpy(normalized).to(self.device)[None, ...]
        with torch.no_grad():
            logit = float(self.model(tensor).item())
        return logit >= 0.0, logit


def load_meta_selector_checkpoint(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> LoadedMetaSelector:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device)
    feature_mean = np.asarray(state["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(state["feature_std"], dtype=np.float32)
    model = GuidelineMetaSelector(input_dim=int(feature_mean.shape[0])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return LoadedMetaSelector(model, feature_mean, feature_std, device)


def _candidate_to_features(candidate: dict[str, object]) -> np.ndarray:
    return build_reranker_features(
        candidate_source=str(candidate["source"]),
        candidate_score=float(candidate["candidate_score"]),
        raw_score=float(candidate["raw_score"]),
        confidence=float(candidate["confidence"]),
        pred_pixels=int(candidate["pred_pixels"]),
        reticle_distance=float(candidate["reticle_distance"]),
        mask_features=dict(candidate["features"]),
    )


def _summarize_frame(frame: dict[str, object], reranker) -> dict[str, object]:
    candidates = frame["candidates"]
    assert isinstance(candidates, list)
    matrix = np.stack([_candidate_to_features(candidate) for candidate in candidates]).astype(np.float32)
    scores = reranker.score(matrix)
    index = int(np.argmax(scores))
    candidate = candidates[index]
    features = dict(candidate["features"])
    return {
        "score": float(scores[index]),
        "raw_score": float(candidate["raw_score"]),
        "candidate_score": float(candidate["candidate_score"]),
        "candidate_source": str(candidate["source"]),
        "confidence": float(candidate["confidence"]),
        "pred_pixels": int(candidate["pred_pixels"]),
        "reticle_distance": float(candidate["reticle_distance"]),
        "features": {key: float(value) for key, value in features.items()},
        "iou": float(candidate["iou"]),
    }


def _evaluate_meta_selector(
    model: GuidelineMetaSelector,
    mean: np.ndarray,
    std: np.ndarray,
    items: list[dict[str, object]],
) -> float:
    if not items:
        return 0.0
    model.eval()
    total = 0.0
    with torch.no_grad():
        for item in items:
            normalized = (item["features"] - mean) / std
            logit = float(model(torch.from_numpy(normalized)[None, ...].float()).item())
            total += item["fallback_iou"] if logit >= 0.0 else item["primary_iou"]
    return total / len(items)


def train_meta_selector(
    primary_dataset: dict[str, list[dict[str, object]]],
    fallback_dataset: dict[str, list[dict[str, object]]],
    primary_reranker_checkpoint: Path,
    fallback_reranker_checkpoint: Path,
    *,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    epochs: int = 300,
) -> tuple[dict[str, object], dict[str, object]]:
    primary_reranker = load_reranker_checkpoint(primary_reranker_checkpoint, device=torch.device("cpu"))
    fallback_reranker = load_reranker_checkpoint(fallback_reranker_checkpoint, device=torch.device("cpu"))

    def build_split(split_name: str) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for primary_frame, fallback_frame in zip(primary_dataset[split_name], fallback_dataset[split_name], strict=True):
            assert primary_frame["frame_id"] == fallback_frame["frame_id"]
            primary_item = _summarize_frame(primary_frame, primary_reranker)
            fallback_item = _summarize_frame(fallback_frame, fallback_reranker)
            items.append(
                {
                    "frame_id": primary_frame["frame_id"],
                    "features": build_meta_selector_features(primary_item, fallback_item),
                    "target": 1.0 if fallback_item["iou"] > primary_item["iou"] else 0.0,
                    "primary_iou": primary_item["iou"],
                    "fallback_iou": fallback_item["iou"],
                }
            )
        return items

    train_items = build_split("train")
    val_items = build_split("val")
    hard_items = build_split("hard")

    train_x = np.stack([item["features"] for item in train_items]).astype(np.float32)
    feature_mean = train_x.mean(axis=0).astype(np.float32)
    feature_std = train_x.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0

    train_tensor = torch.from_numpy((train_x - feature_mean) / feature_std).float()
    train_targets = torch.from_numpy(np.asarray([item["target"] for item in train_items], dtype=np.float32))

    model = GuidelineMetaSelector(input_dim=int(feature_mean.shape[0]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_key = -1e9
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, object] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_tensor)
        loss = criterion(logits, train_targets)
        loss.backward()
        optimizer.step()

        val_iou = _evaluate_meta_selector(model, feature_mean, feature_std, val_items)
        hard_iou = _evaluate_meta_selector(model, feature_mean, feature_std, hard_items)
        selection_key = val_iou + hard_iou
        if selection_key > best_key:
            best_key = selection_key
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_iou": val_iou,
                "hard_iou": hard_iou,
                "selection_key": selection_key,
            }

    assert best_state is not None
    assert best_metrics is not None
    checkpoint = {
        "model": best_state,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "metrics": best_metrics,
        "config": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "selected_numeric_features": list(SELECTED_NUMERIC_FEATURES),
            "source_names": list(SOURCE_NAMES),
        },
    }
    return checkpoint, best_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dataset-json", required=True)
    parser.add_argument("--fallback-dataset-json", required=True)
    parser.add_argument("--primary-reranker", required=True)
    parser.add_argument("--fallback-reranker", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    primary_dataset = json.loads(Path(args.primary_dataset_json).read_text(encoding="utf-8"))
    fallback_dataset = json.loads(Path(args.fallback_dataset_json).read_text(encoding="utf-8"))
    checkpoint, metrics = train_meta_selector(
        primary_dataset,
        fallback_dataset,
        primary_reranker_checkpoint=Path(args.primary_reranker),
        fallback_reranker_checkpoint=Path(args.fallback_reranker),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
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
