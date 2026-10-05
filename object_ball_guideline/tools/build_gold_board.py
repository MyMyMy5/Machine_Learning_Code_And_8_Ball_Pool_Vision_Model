from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_RE = re.compile(r"_f(\d+)(?:\.[^.]+)?$", re.IGNORECASE)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_root_path(path: str | Path) -> Path:
    path_text = str(path)
    if os.name != "nt" and len(path_text) >= 3 and path_text[1:3] in {":\\", ":/"}:
        native_path = Path(path_text)
        if native_path.exists():
            return native_path
        return Path(f"/mnt/{path_text[0].lower()}/" + path_text[3:].replace("\\", "/"))
    return Path(path_text)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_item_path(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def frame_group_key(path: str | Path, bucket_size: int = 25) -> str:
    path = Path(str(path))
    stem = path.stem
    match = FRAME_RE.search(path.name)
    if not match:
        return str(path.parent / stem)
    frame_index = int(match.group(1))
    bucket = frame_index // max(1, int(bucket_size))
    video_stem = FRAME_RE.sub("", path.name)
    return f"{path.parent}/{video_stem}:bucket{bucket:08d}"


def _stable_fraction(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _old_split_items(index: dict[str, Any], split: str) -> list[dict[str, object]]:
    items: dict[str, dict[str, object]] = {}
    for row in index.get(split, []):
        sample_id = str(row["id"])
        items.setdefault(
            sample_id,
            {
                "id": sample_id,
                "kind": "positive",
                "image_path": _normalize_item_path(row.get("image_path")),
                "mask_path": _normalize_item_path(row.get("mask_path")),
                "source": split,
            },
        )
    return sorted(items.values(), key=lambda item: str(item["id"]))


def _positive_item(record: dict[str, Any], source: str) -> dict[str, object]:
    source_image = str(record.get("source_image_path") or record.get("image_path") or record["id"])
    return {
        "id": str(record["id"]),
        "kind": "positive",
        "image_path": _normalize_item_path(record.get("image_path")),
        "mask_path": _normalize_item_path(record.get("mask_path")),
        "source_image_path": source_image,
        "group_key": frame_group_key(source_image),
        "source": source,
    }


def _negative_item(record: dict[str, Any], source: str) -> dict[str, object]:
    source_image = str(record.get("source_image_path") or record.get("image_path") or record["id"])
    return {
        "id": str(record["id"]),
        "kind": "image_negative",
        "image_path": _normalize_item_path(record.get("image_path")),
        "mask_path": None,
        "source_image_path": source_image,
        "group_key": frame_group_key(source_image),
        "source": source,
    }


def _flat_negative_items(root: Path) -> list[dict[str, object]]:
    if not root.exists():
        return []
    items = []
    for path in sorted(item for item in root.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES):
        items.append(
            {
                "id": path.stem,
                "kind": "flat_negative",
                "image_path": str(path),
                "mask_path": None,
                "source_image_path": str(path),
                "group_key": frame_group_key(path),
                "source": "flat_negative",
            }
        )
    return items


def _harvest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    harvested = []
    for record in records:
        source = str(record.get("source_image_path") or "")
        created_at = str(record.get("created_at") or "")
        if "video_frames_model_harvest" in source or "video_infer_harvest" in source or created_at.startswith("2026-04"):
            harvested.append(record)
    return harvested


def _select_group_holdout(items: list[dict[str, object]], holdout_fraction: float) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in items:
        grouped.setdefault(str(item["group_key"]), []).append(item)
    selected = []
    for key in sorted(grouped):
        if _stable_fraction(key) < holdout_fraction:
            selected.extend(grouped[key])
    if not selected and items:
        first_key = sorted(grouped)[0]
        selected.extend(grouped[first_key])
    return sorted(selected, key=lambda item: str(item["id"]))


def build_gold_board(
    *,
    index_json: Path,
    data_root: Path,
    rejected_root: Path,
    negative_root: Path,
    output_root: Path,
    holdout_fraction: float = 0.2,
    frame_bucket_size: int = 25,
) -> dict[str, object]:
    global frame_group_key
    original_frame_group_key = frame_group_key

    def _bucketed(path: str | Path, bucket_size: int = frame_bucket_size) -> str:
        return original_frame_group_key(path, bucket_size=bucket_size)

    frame_group_key = _bucketed
    try:
        index_json = _normalize_root_path(index_json)
        data_root = _normalize_root_path(data_root)
        rejected_root = _normalize_root_path(rejected_root)
        negative_root = _normalize_root_path(negative_root)
        output_root = _normalize_root_path(output_root)

        old_index = _read_json(index_json)
        old_val = _old_split_items(old_index, "val")
        old_hard = _old_split_items(old_index, "hard_val")
        old_ids = {str(item["id"]) for item in old_val + old_hard}

        records = _read_jsonl(data_root / "annotations.jsonl")
        harvest_items = [
            _positive_item(record, "harvest_holdout")
            for record in _harvest_records(records)
            if str(record.get("id")) not in old_ids
        ]
        harvest_holdout = _select_group_holdout(harvest_items, holdout_fraction)

        rejected_items = [
            _negative_item(record, "rejected_negative")
            for record in _read_jsonl(rejected_root / "annotations.jsonl")
        ]
        rejected_holdout = _select_group_holdout(rejected_items, holdout_fraction)
        flat_negatives = _flat_negative_items(negative_root)
        flat_negative_holdout = _select_group_holdout(flat_negatives, holdout_fraction)

        board = {
            "version": 3,
            "policy": {
                "positive_rule": "valid guideline is the object-ball outgoing guideline, starting from or belonging to the target ball after cue-ball contact",
                "negative_rule": "Rejected and selected negative folders are image-level zero-mask negatives: the model should detect nothing anywhere in those frames",
                "holdout_fraction": float(holdout_fraction),
                "frame_bucket_size": int(frame_bucket_size),
            },
            "splits": {
                "old_val": old_val,
                "old_hard_val": old_hard,
                "harvest_holdout": harvest_holdout,
                "rejected_negatives": rejected_holdout,
                "flat_negatives": flat_negative_holdout,
            },
        }
        excluded_ids = sorted(
            {
                str(item["id"])
                for split_items in board["splits"].values()
                for item in split_items
            }
        )
        _write_json(output_root / "gold_board.json", board)
        _write_json(output_root / "train_exclude_ids.json", excluded_ids)
        return board
    finally:
        frame_group_key = original_frame_group_key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-json", default="runs/supervised_train_v2/index.json")
    parser.add_argument("--data-root", default="C:/My_Project/SAM_3/guideline_line/data_zoomprobe")
    parser.add_argument("--rejected-root", default="C:/My_Project/SAM_3/guideline_line/data_zoomprobe/Rejected")
    parser.add_argument("--negative-root", default="C:/My_Project/8_BALL_POOL/data/images/negative_selected")
    parser.add_argument("--output-root", default="runs/gold_board_v1")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--frame-bucket-size", type=int, default=25)
    args = parser.parse_args()
    board = build_gold_board(
        index_json=Path(args.index_json),
        data_root=Path(args.data_root),
        rejected_root=Path(args.rejected_root),
        negative_root=Path(args.negative_root),
        output_root=Path(args.output_root),
        holdout_fraction=args.holdout_fraction,
        frame_bucket_size=args.frame_bucket_size,
    )
    summary = {name: len(items) for name, items in board["splits"].items()}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
