from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.supervised.rescue_arbiter import train_rescue_arbiter  # noqa: E402


def _split_items(
    items: list[dict[str, object]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    positives = [item for item in items if int(item.get("target", 0)) == 1]
    negatives = [item for item in items if int(item.get("target", 0)) == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_bucket(bucket: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        val_count = max(1, int(round(len(bucket) * val_fraction))) if len(bucket) > 1 else 0
        return bucket[val_count:], bucket[:val_count]

    pos_train, pos_val = split_bucket(positives)
    neg_train, neg_val = split_bucket(negatives)
    train_items = pos_train + neg_train
    val_items = pos_val + neg_val
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    train_items, val_items = _split_items(
        list(dataset["items"]),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint, _metrics = train_rescue_arbiter(
        train_items,
        val_items,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        device=device,
    )
    checkpoint["train_dataset"] = str(args.dataset_json)
    checkpoint["metrics"] = {
        **checkpoint["metrics"],
        "device": str(device),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "train_positive_count": sum(int(item["target"]) for item in train_items),
        "val_positive_count": sum(int(item["target"]) for item in val_items),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    metrics_path = Path(args.metrics_json) if args.metrics_json else output.with_suffix(".json")
    metrics_path.write_text(json.dumps(checkpoint["metrics"], indent=2), encoding="utf-8")
    print(json.dumps(checkpoint["metrics"], indent=2))
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
