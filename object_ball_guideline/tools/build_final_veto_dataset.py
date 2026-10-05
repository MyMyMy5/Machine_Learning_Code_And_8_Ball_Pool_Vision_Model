from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_winner(row: dict[str, Any]) -> dict[str, object]:
    report_path = Path(str(row["output_dir"])) / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    winner = payload["winner"]
    if not isinstance(winner, dict):
        raise ValueError(f"Invalid winner payload in {report_path}")
    return winner


def _target_for_row(row: dict[str, Any]) -> int:
    gt_pixels = int(row.get("gt_pixels", 0))
    pred_pixels = int(row.get("pred_pixels", 0))
    return int(gt_pixels <= 0 and pred_pixels > 0)


def build_dataset(summary_paths: list[Path]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in summary.get("rows", []):
            if not isinstance(row, dict):
                continue
            winner = _load_winner(row)
            items.append(
                {
                    "id": str(row["id"]),
                    "split": str(row["split"]),
                    "kind": row.get("kind"),
                    "source": row.get("source"),
                    "summary_path": str(summary_path),
                    "output_dir": str(row["output_dir"]),
                    "gt_pixels": int(row.get("gt_pixels", 0)),
                    "pred_pixels": int(row.get("pred_pixels", 0)),
                    "iou": float(row.get("iou", 0.0)),
                    "dice": float(row.get("dice", 0.0)),
                    "failure_type": str(row.get("failure_type", "")),
                    "target": _target_for_row(row),
                    "winner": winner,
                }
            )
    target_count = sum(int(item["target"]) for item in items)
    return {
        "summary_paths": [str(path) for path in summary_paths],
        "count": len(items),
        "target_veto_count": target_count,
        "target_keep_count": len(items) - target_count,
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = build_dataset([Path(item) for item in args.summary_json])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(json.dumps({key: dataset[key] for key in ["count", "target_veto_count", "target_keep_count"]}, indent=2))
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
