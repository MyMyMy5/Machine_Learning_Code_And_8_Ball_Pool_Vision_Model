from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.crop_utils import build_crop_candidate  # noqa: E402


def _normalize_path(path_str: str) -> Path:
    if os.name != "nt" and path_str.startswith("C:\\"):
        return Path("/mnt/c/" + path_str[3:].replace("\\", "/"))
    if os.name == "nt" and path_str.startswith("/mnt/c/"):
        return Path("C:/" + path_str[len("/mnt/c/"):])
    return Path(path_str)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-index", required=True)
    parser.add_argument("--eval-summary", required=True)
    parser.add_argument("--eval-output-root", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "hard"])
    parser.add_argument("--max-iou", type=float, default=0.25)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--crop-scale", type=float, default=4.25)
    parser.add_argument("--padding-px", type=int, default=24)
    args = parser.parse_args()

    base_index_path = _normalize_path(args.base_index)
    eval_summary_path = _normalize_path(args.eval_summary)
    eval_output_root = _normalize_path(args.eval_output_root)
    output_index_path = _normalize_path(args.output_index)
    output_summary_path = _normalize_path(args.output_summary)

    base_index = json.loads(base_index_path.read_text(encoding="utf-8"))
    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))

    train_lookup: dict[str, dict[str, str]] = {}
    for item in base_index["train"]:
        train_lookup.setdefault(
            str(item["id"]),
            {
                "image_path": str(item["image_path"]),
                "mask_path": str(item["mask_path"]),
            },
        )

    rows = eval_summary["rows"]
    filtered = [
        row
        for row in rows
        if not str(row["id"]).startswith("neg::") and float(row["iou"]) <= float(args.max_iou)
    ]
    if args.max_entries is not None:
        filtered = filtered[: args.max_entries]

    residual_entries: list[dict[str, object]] = []
    for row in filtered:
        frame_id = str(row["id"])
        report_path = eval_output_root / frame_id / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))["winner"]
        image_path = _normalize_path(train_lookup[frame_id]["image_path"])
        image = np.asarray(Image.open(image_path).convert("RGB"))
        center_x, center_y = [int(value) for value in report["candidate_center"]]
        radius = int(report["candidate_radius"])
        candidate = build_crop_candidate(
            candidate_id=str(report["winner"]) if "winner" in report else str(report["candidate_id"]),
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            image_shape=image.shape[:2],
            crop_scale=float(args.crop_scale),
            padding_px=int(args.padding_px),
            source=str(report["candidate_source"]),
            score=float(report["score"]),
        )
        residual_entries.append(
            {
                "kind": "residual_policy_negative",
                "id": frame_id,
                "image_path": train_lookup[frame_id]["image_path"],
                "mask_path": train_lookup[frame_id]["mask_path"],
                "crop_box": list(candidate.crop_box),
                "mask_pixels": 0,
                "winner_score": float(report["score"]),
                "winner_source": str(report["candidate_source"]),
                "winner_iou": float(row["iou"]),
            }
        )

    merged_index = dict(base_index)
    merged_index["train"] = list(base_index["train"]) + residual_entries

    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_index_path.write_text(json.dumps(merged_index, indent=2), encoding="utf-8")
    output_summary_path.write_text(
        json.dumps(
            {
                "source_eval_summary": str(eval_summary_path),
                "source_eval_output_root": str(eval_output_root),
                "split": args.split,
                "max_iou": float(args.max_iou),
                "count": len(residual_entries),
                "entries": residual_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(output_index_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
