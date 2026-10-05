from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from live_label_overlay import ensure_dataset_layout


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def tiny_line_flag(mask_u8: np.ndarray, tiny_cfg: Dict) -> bool:
    mask = (mask_u8 > 0).astype(np.uint8)
    area = int(mask.sum())
    if area <= 0:
        return False
    h, w = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return False
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    area_ratio = float(area) / float(max(1, h * w))
    max_side = max(int(bw), int(bh))
    return (
        area_ratio <= float(tiny_cfg.get("max_area_ratio", 0.002))
        or max_side <= int(tiny_cfg.get("max_bbox_side", 90))
    )


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def append_jsonl(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def existing_ids(ann_path: Path) -> set:
    ids = set()
    for rec in read_jsonl(ann_path):
        rid = str(rec.get("id", "")).strip()
        if rid:
            ids.add(rid)
    return ids


def existing_source_sample_ids(ann_path: Path) -> set:
    out = set()
    for rec in read_jsonl(ann_path):
        sid = str(rec.get("source_sample_id", "")).strip()
        if sid:
            out.add(sid)
    return out


def make_record(
    source_rec: Dict,
    imported_id: str,
    prompt: str,
    tiny_cfg: Dict,
) -> Optional[Dict]:
    src_img = Path(str(source_rec.get("source_image_path", "")).strip())
    src_mask = Path(str(source_rec.get("mask_path", "")).strip())
    if not src_img.exists() or not src_mask.exists():
        return None

    img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    m = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
    if img is None or m is None or img.shape[:2] != m.shape[:2]:
        return None
    h, w = img.shape[:2]
    mask_u8 = (m > 127).astype(np.uint8)
    if int(mask_u8.sum()) <= 0:
        return None

    roi_xyxy = [0, 0, int(w - 1), int(h - 1)]
    return {
        "id": str(imported_id),
        "image_path": str(src_img.resolve()),
        "mask_path": str(src_mask.resolve()),
        "prompt": str(prompt),
        "monitor_index": -1,
        "roi_xyxy": roi_xyxy,
        "capture_upscale": 1.0,
        "tiny_line": bool(tiny_line_flag(mask_u8, tiny_cfg)),
        "refined": True,
        "created_at": str(source_rec.get("created_at") or utc_now_iso()),
        "source_mode": "color_analysis_import",
        "source_sample_id": str(source_rec.get("id", "")),
        "source_image_path": str(src_img.resolve()),
        "source_mask_path": str(src_mask.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import shape_color_analyzer labels into SAM3 training annotations.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--source-jsonl", type=str, default="guideline_line/data/color_analysis/samples.jsonl")
    parser.add_argument("--target-data-root", type=str, default="guideline_line/data_colortrain")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--id-prefix", type=str, default="CA_")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    tiny_cfg = cfg.get("tiny_line", {})
    prompt = str(args.prompt or cfg.get("default_prompt", "white aiming line"))

    source_path = Path(args.source_jsonl)
    if not source_path.exists():
        raise FileNotFoundError(f"Source JSONL not found: {source_path}")
    source_rows = read_jsonl(source_path)
    if not source_rows:
        raise RuntimeError("No rows found in source JSONL.")

    target_root = Path(args.target_data_root)
    target_dirs = ensure_dataset_layout(target_root)
    ann_path = target_dirs["annotations"]

    ids_taken = existing_ids(ann_path)
    source_taken = existing_source_sample_ids(ann_path)
    pending: List[Dict] = []
    skipped_exists = 0
    skipped_invalid = 0

    for src in source_rows:
        sid = str(src.get("id", "")).strip()
        if not sid:
            skipped_invalid += 1
            continue
        if sid in source_taken:
            skipped_exists += 1
            continue
        rid = f"{str(args.id_prefix)}{sid}"
        if rid in ids_taken:
            skipped_exists += 1
            continue
        rec = make_record(src, rid, prompt=prompt, tiny_cfg=tiny_cfg)
        if rec is None:
            skipped_invalid += 1
            continue
        pending.append(rec)
        ids_taken.add(rid)
        source_taken.add(sid)

    if not args.dry_run:
        append_jsonl(ann_path, pending)

    print(f"Source rows: {len(source_rows)}")
    print(f"Imported rows: {len(pending)}")
    print(f"Skipped existing: {skipped_exists}")
    print(f"Skipped invalid: {skipped_invalid}")
    print(f"Target annotations: {ann_path}")
    print(f"Target data root: {target_root.resolve()}")
    if args.dry_run:
        print("Dry run: no files were modified.")


if __name__ == "__main__":
    main()
