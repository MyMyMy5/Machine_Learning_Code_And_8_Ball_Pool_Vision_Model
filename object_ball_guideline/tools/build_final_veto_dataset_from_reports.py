from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    winner = payload.get("winner")
    if not isinstance(winner, dict):
        raise ValueError(f"Invalid winner payload in {path}")
    return _slim_winner(winner)


def _slim_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): slimmed
            for key, child in value.items()
            if (slimmed := _slim_value(child)) is not None
        }
    if isinstance(value, list):
        if len(value) <= 512 and all(isinstance(item, (int, float, bool, str)) or item is None for item in value):
            return value
        return None
    return None


def _slim_winner(winner: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): slimmed
        for key, value in winner.items()
        if (slimmed := _slim_value(value)) is not None
    }


def _winner_pred_pixels(winner: dict[str, Any]) -> int:
    pred_pixels = winner.get("pred_pixels")
    if isinstance(pred_pixels, (int, float)):
        return int(pred_pixels)
    pred_area_values = [
        int(value)
        for key, value in winner.items()
        if (
            isinstance(value, (int, float))
            and not str(key).startswith("pre_")
            and (str(key).endswith("_pred_area") or str(key).endswith("_pred_pixels"))
        )
    ]
    return max(pred_area_values, default=0)


def _is_positive_item(split_name: str, item: dict[str, Any]) -> bool:
    if item.get("mask_path"):
        return True
    if str(item.get("kind", "")).lower().startswith("positive"):
        return True
    return "positive" in split_name


def build_dataset(*, gold_board: dict[str, Any], eval_roots: list[Path]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    splits = gold_board.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError("gold board has no splits object")

    for split_name, split_items in splits.items():
        if not isinstance(split_items, list):
            continue
        for item in split_items:
            if not isinstance(item, dict):
                continue
            item_id = str(item["id"])
            report_path = next(
                (
                    eval_root / split_name / item_id / "report.json"
                    for eval_root in eval_roots
                    if (eval_root / split_name / item_id / "report.json").exists()
                ),
                None,
            )
            if report_path is None:
                missing.append({"split": split_name, "id": item_id})
                continue
            winner = _load_report(report_path)
            pred_pixels = _winner_pred_pixels(winner)
            is_positive = _is_positive_item(split_name, item)
            items.append(
                {
                    "id": item_id,
                    "split": split_name,
                    "kind": item.get("kind"),
                    "source": item.get("source"),
                    "summary_path": None,
                    "output_dir": str(report_path.parent),
                    "gt_pixels": 1 if is_positive else 0,
                    "pred_pixels": pred_pixels,
                    "iou": None,
                    "dice": None,
                    "failure_type": (
                        "positive_unscored"
                        if is_positive
                        else ("negative_false_positive" if pred_pixels > 0 else "true_negative")
                    ),
                    "target": int((not is_positive) and pred_pixels > 0),
                    "winner": winner,
                }
            )

    target_count = sum(int(item["target"]) for item in items)
    return {
        "eval_roots": [str(path) for path in eval_roots],
        "count": len(items),
        "target_veto_count": target_count,
        "target_keep_count": len(items) - target_count,
        "missing_count": len(missing),
        "missing": missing,
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-board", required=True)
    parser.add_argument("--eval-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gold_board = json.loads(Path(args.gold_board).read_text(encoding="utf-8"))
    dataset = build_dataset(
        gold_board=gold_board,
        eval_roots=[Path(path) for path in args.eval_root],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: dataset[key]
                for key in ["count", "target_veto_count", "target_keep_count", "missing_count"]
            },
            indent=2,
        )
    )
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
