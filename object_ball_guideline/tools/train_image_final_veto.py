from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.supervised.image_final_veto import train_image_final_veto  # noqa: E402


def _split_items(
    items: list[dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    keep_items = [item for item in items if int(item.get("target", 0)) == 0]
    veto_items = [item for item in items if int(item.get("target", 0)) == 1]
    rng.shuffle(keep_items)
    rng.shuffle(veto_items)

    def split_bucket(bucket: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if len(bucket) <= 1:
            return bucket, []
        val_count = max(1, int(round(len(bucket) * val_fraction)))
        return bucket[val_count:], bucket[:val_count]

    keep_train, keep_val = split_bucket(keep_items)
    veto_train, veto_val = split_bucket(veto_items)
    train_items = keep_train + veto_train
    val_items = keep_val + veto_val
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--include-global", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    items = list(dataset["items"])
    if not any(int(item.get("target", 0)) == 1 for item in items):
        raise ValueError("image final-veto training needs at least one veto-target item")
    if not any(int(item.get("target", 0)) == 0 for item in items):
        raise ValueError("image final-veto training needs at least one keep-target item")

    train_items, val_items = _split_items(items, val_fraction=args.val_fraction, seed=args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint, metrics = train_image_final_veto(
        train_items,
        val_items,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        width=args.width,
        device=device,
        num_workers=args.num_workers,
        include_global=args.include_global,
    )
    checkpoint["train_dataset"] = str(args.dataset_json)
    checkpoint["metrics"] = {
        **checkpoint["metrics"],
        "device": str(device),
        "seed": int(args.seed),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "train_veto_count": sum(int(item["target"]) for item in train_items),
        "val_veto_count": sum(int(item["target"]) for item in val_items),
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
