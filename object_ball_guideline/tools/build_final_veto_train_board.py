from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _dedupe_by_image(entries: list[dict[str, Any]], *, kind: str, source: str, mask_required: bool) -> list[dict[str, Any]]:
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for entry in entries:
        image_path = str(entry.get("image_path", ""))
        if not image_path or image_path in seen:
            continue
        if mask_required and not entry.get("mask_path"):
            continue
        seen.add(image_path)
        items.append(
            {
                "id": str(entry["id"]).split("_crop", 1)[0],
                "kind": kind,
                "source": source,
                "image_path": image_path,
                "mask_path": str(entry["mask_path"]) if entry.get("mask_path") else None,
            }
        )
    return items


def build_board(index: dict[str, Any], *, split: str, limit_per_split: int | None = None) -> dict[str, Any]:
    rows = list(index[split])
    positive_entries = [
        entry
        for entry in rows
        if str(entry.get("kind", "")).endswith("_positive") and entry.get("mask_path")
    ]
    rejected_entries = [entry for entry in rows if entry.get("kind") == "rejected_negative"]
    flat_entries = [entry for entry in rows if entry.get("kind") == "external_negative"]

    positives = _dedupe_by_image(
        positive_entries,
        kind="positive",
        source=f"{split}_positive",
        mask_required=True,
    )
    rejected = _dedupe_by_image(
        rejected_entries,
        kind="rejected_negative",
        source=f"{split}_rejected_negative",
        mask_required=False,
    )
    flat = _dedupe_by_image(
        flat_entries,
        kind="flat_negative",
        source=f"{split}_flat_negative",
        mask_required=False,
    )
    if limit_per_split is not None:
        positives = positives[:limit_per_split]
        rejected = rejected[:limit_per_split]
        flat = flat[:limit_per_split]
    return {
        "source_index_split": split,
        "splits": {
            f"{split}_positives": positives,
            f"{split}_rejected_negatives": rejected,
            f"{split}_flat_negatives": flat,
        },
        "summary": {
            f"{split}_positives": len(positives),
            f"{split}_rejected_negatives": len(rejected),
            f"{split}_flat_negatives": len(flat),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-json", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "hard_val"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-split", type=int, default=None)
    args = parser.parse_args()

    index = json.loads(Path(args.index_json).read_text(encoding="utf-8"))
    board = build_board(index, split=args.split, limit_per_split=args.limit_per_split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(board, indent=2), encoding="utf-8")
    print(json.dumps(board["summary"], indent=2))
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
