from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from cv_guideline_common import resolve_path, utc_now_iso, write_json


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a positive replay manifest from relabeled indices saved by review_infer_repairs.py."
    )
    parser.add_argument("--summary", type=str, required=True, help="Summary JSON used by the repair review.")
    parser.add_argument("--resume-state", type=str, required=True, help="repair_review_resume.json path.")
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--name", type=str, default="repaired_positive_manifest")
    parser.add_argument(
        "--annotation-jsonl",
        type=str,
        nargs="*",
        default=[
            "guideline_line/data_zoomprobe/annotations.jsonl",
            "guideline_line/data_zoomprobe/quarantine/annotations.jsonl",
        ],
        help="Annotation JSONL paths used to resolve saved dataset rows by source_image_path.",
    )
    args = parser.parse_args()

    summary_path = resolve_path(args.summary)
    state_path = resolve_path(args.resume_state)
    output_path = resolve_path(args.output_manifest)

    summary = load_json(summary_path)
    state = load_json(state_path)
    rows = list(summary.get("rows", []))
    relabeled_indices = {int(x) for x in state.get("relabeled_indices", []) if isinstance(x, int)}
    selected_sources: dict[str, int] = {}
    for idx, row in enumerate(rows):
        if idx not in relabeled_indices:
            continue
        source_image_path = str(row.get("image_path", "")).strip()
        if not source_image_path:
            continue
        selected_sources[str(resolve_path(source_image_path)).lower()] = int(idx)

    annotation_rows: List[Dict] = []
    for raw_path in list(args.annotation_jsonl or []):
        ann_path = resolve_path(raw_path)
        if not ann_path.exists():
            continue
        for line in ann_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            annotation_rows.append(obj)

    selected_rows: List[Dict] = []
    images: List[str] = []
    positive_sample_ids: List[str] = []
    resolved_by_source: Dict[str, Dict] = {}
    for row in annotation_rows:
        source_image_path = str(row.get("source_image_path", "")).strip()
        if not source_image_path:
            continue
        source_key = str(resolve_path(source_image_path)).lower()
        if source_key not in selected_sources:
            continue
        previous = resolved_by_source.get(source_key)
        if previous is None or str(row.get("created_at", "")) > str(previous.get("created_at", "")):
            resolved_by_source[source_key] = row

    missing_sources: List[str] = []
    for source_key, review_index in sorted(selected_sources.items(), key=lambda item: item[1]):
        row = resolved_by_source.get(source_key)
        if row is None:
            missing_sources.append(source_key)
            continue
        image_path = str(row.get("image_path", "")).strip()
        mask_path = str(row.get("mask_path", "")).strip()
        sample_id = str(row.get("id", "")).strip()
        if not image_path or not mask_path or not sample_id:
            missing_sources.append(source_key)
            continue
        images.append(image_path)
        positive_sample_ids.append(sample_id)
        selected_rows.append(
            {
                "index": int(review_index),
                "sample_id": sample_id,
                "source_image_path": str(row.get("source_image_path", "")).strip(),
                "image_path": image_path,
                "mask_path": mask_path,
                "created_at": str(row.get("created_at", "")).strip(),
                "dataset_split": str(row.get("dataset_split", "")).strip(),
                "save_mode": str(row.get("save_mode", "")).strip(),
            }
        )

    manifest = {
        "created_at": utc_now_iso(),
        "meta": {
            "name": str(args.name).strip() or "repaired_positive_manifest",
            "source_summary": str(summary_path),
            "resume_state": str(state_path),
            "positive_count": int(len(selected_rows)),
            "relabeled_indices": sorted(relabeled_indices),
            "positive_sample_ids": positive_sample_ids,
            "missing_source_image_paths": missing_sources,
        },
        "images": images,
        "rows": selected_rows,
    }
    write_json(output_path, manifest)
    print(str(output_path))
    print(json.dumps({"positive_count": len(selected_rows)}, indent=2))


if __name__ == "__main__":
    main()
