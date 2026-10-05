from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _base_item_index(gold_board: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    splits = gold_board.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError("gold board has no splits object")
    for split_name, items in splits.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and "id" in item:
                index[(str(split_name), str(item["id"]))] = item
    return index


def _source_from_selector_item(item: dict[str, Any]) -> str:
    winner = item.get("winner")
    if isinstance(winner, dict):
        return str(winner.get("candidate_source", ""))
    return str(item.get("winner_source", ""))


def _source_from_summary_row(row: dict[str, Any]) -> str:
    return str(row.get("winner_source", ""))


def _should_keep_summary_row(
    row: dict[str, Any],
    *,
    sources: set[str],
    include_true_negatives: bool,
) -> bool:
    if _source_from_summary_row(row) not in sources:
        return False
    if int(row.get("pred_pixels", 0)) > 0:
        return True
    if include_true_negatives and int(row.get("gt_pixels", 0)) <= 0:
        return True
    return False


def _item_from_base(base: dict[str, Any], *, source_note: str) -> dict[str, Any]:
    return {
        "id": str(base["id"]),
        "kind": base.get("kind"),
        "source": source_note,
        "image_path": str(base["image_path"]),
        "mask_path": str(base["mask_path"]) if base.get("mask_path") else None,
    }


def build_from_selector_dataset(
    *,
    gold_board: dict[str, Any],
    selector_dataset: dict[str, Any],
    sources: set[str],
    max_items_per_split: int | None,
) -> dict[str, Any]:
    base_index = _base_item_index(gold_board)
    selected: dict[str, list[dict[str, Any]]] = {}
    missing: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    for selector_item in selector_dataset.get("items", []):
        if not isinstance(selector_item, dict):
            continue
        source = _source_from_selector_item(selector_item)
        if source not in sources:
            continue
        split_name = str(selector_item["split"])
        item_id = str(selector_item["id"])
        base = base_index.get((split_name, item_id))
        if base is None:
            missing.append({"split": split_name, "id": item_id})
            continue
        split_items = selected.setdefault(split_name, [])
        if max_items_per_split is not None and len(split_items) >= max_items_per_split:
            continue
        split_items.append(_item_from_base(base, source_note=f"selector_source:{source}"))
        source_counts[source] += 1
        target_counts[str(selector_item.get("target", 0))] += 1

    return {
        "source": "selector_dataset",
        "sources": sorted(sources),
        "splits": selected,
        "summary": {split: len(items) for split, items in selected.items()},
        "source_counts": dict(sorted(source_counts.items())),
        "selector_target_counts": dict(sorted(target_counts.items())),
        "missing_count": len(missing),
        "missing": missing[:100],
    }


def build_from_summary(
    *,
    gold_board: dict[str, Any],
    summary: dict[str, Any],
    sources: set[str],
    include_true_negatives: bool,
    max_items_per_split: int | None,
) -> dict[str, Any]:
    base_index = _base_item_index(gold_board)
    selected: dict[str, list[dict[str, Any]]] = {}
    missing: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()

    for row in summary.get("rows", []):
        if not isinstance(row, dict):
            continue
        if not _should_keep_summary_row(row, sources=sources, include_true_negatives=include_true_negatives):
            continue
        split_name = str(row["split"])
        item_id = str(row["id"])
        base = base_index.get((split_name, item_id))
        if base is None:
            missing.append({"split": split_name, "id": item_id})
            continue
        split_items = selected.setdefault(split_name, [])
        if max_items_per_split is not None and len(split_items) >= max_items_per_split:
            continue
        source = _source_from_summary_row(row)
        split_items.append(_item_from_base(base, source_note=f"summary_source:{source}"))
        source_counts[source] += 1
        failure_counts[str(row.get("failure_type", ""))] += 1

    return {
        "source": "summary",
        "sources": sorted(sources),
        "splits": selected,
        "summary": {split: len(items) for split, items in selected.items()},
        "source_counts": dict(sorted(source_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "missing_count": len(missing),
        "missing": missing[:100],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-board", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--selector-dataset", default=None)
    selector.add_argument("--summary-json", default=None)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-true-negatives", action="store_true")
    parser.add_argument("--max-items-per-split", type=int, default=None)
    args = parser.parse_args()

    gold_board = _load_json(args.gold_board)
    sources = {str(source) for source in args.sources}
    if args.selector_dataset:
        payload = build_from_selector_dataset(
            gold_board=gold_board,
            selector_dataset=_load_json(args.selector_dataset),
            sources=sources,
            max_items_per_split=args.max_items_per_split,
        )
    else:
        payload = build_from_summary(
            gold_board=gold_board,
            summary=_load_json(args.summary_json),
            sources=sources,
            include_true_negatives=bool(args.include_true_negatives),
            max_items_per_split=args.max_items_per_split,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["summary", "source_counts", "missing_count"]}, indent=2))
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
