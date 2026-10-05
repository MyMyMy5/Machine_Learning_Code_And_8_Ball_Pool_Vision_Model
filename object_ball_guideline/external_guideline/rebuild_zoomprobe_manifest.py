from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def normalize_row(row: Dict, dataset_split: str, save_mode: str) -> Dict:
    out = dict(row)
    out["dataset_split"] = str(out.get("dataset_split", dataset_split) or dataset_split)
    out["save_mode"] = str(out.get("save_mode", save_mode) or save_mode)
    return out


def sort_key(row: Dict):
    return (
        str(row.get("created_at", "")),
        str(row.get("source_image_path", "")),
        str(row.get("id", "")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild merged zoomprobe manifest and deterministic train/val split.")
    parser.add_argument("--source-root", type=Path, default=Path("guideline_line/data_zoomprobe"))
    parser.add_argument("--output-root", type=Path, default=Path("guideline_line/data_zoomprobe_merged"))
    parser.add_argument("--include-quarantine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    main_rows = [
        normalize_row(row, dataset_split="main", save_mode="normal")
        for row in read_jsonl(source_root / "annotations.jsonl")
    ]
    quarantine_rows: List[Dict] = []
    if bool(args.include_quarantine):
        quarantine_rows = [
            normalize_row(row, dataset_split="quarantine", save_mode="force")
            for row in read_jsonl(source_root / "quarantine" / "annotations.jsonl")
        ]

    merged = main_rows + quarantine_rows
    merged.sort(key=sort_key)
    if not merged:
        raise RuntimeError("No annotation rows found to merge.")

    ids = [str(row.get("id", "")).strip() for row in merged if str(row.get("id", "")).strip()]
    if len(ids) != len(merged):
        raise RuntimeError("Some merged rows are missing 'id'.")

    split_index = max(1, min(len(ids) - 1, int(round(len(ids) * (1.0 - float(args.val_ratio))))))
    train_ids = ids[:split_index]
    val_ids = ids[split_index:]

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "annotations.jsonl", merged)
    (output_root / "splits.json").write_text(
        json.dumps(
            {
                "train": train_ids,
                "val": val_ids,
                "counts": {
                    "total": len(merged),
                    "train": len(train_ids),
                    "val": len(val_ids),
                    "main": len(main_rows),
                    "quarantine": len(quarantine_rows),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Merged annotations: {len(merged)}")
    print(f"Main: {len(main_rows)} | Quarantine: {len(quarantine_rows)}")
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")
    print(f"annotations.jsonl -> {output_root / 'annotations.jsonl'}")
    print(f"splits.json -> {output_root / 'splits.json'}")


if __name__ == "__main__":
    main()
