from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Set

from cv_guideline_common import list_images, read_jsonl, resolve_path, utc_now_iso, write_json


def canonical_path_str(path_value: str) -> str:
    return str(resolve_path(path_value)).lower()


def collect_labeled_source_paths(annotation_paths: Sequence[Path]) -> Set[str]:
    out: Set[str] = set()
    for ann_path in annotation_paths:
        if not ann_path.exists():
            continue
        for row in read_jsonl(ann_path):
            src = str(row.get("source_image_path", "")).strip()
            if not src:
                continue
            out.add(canonical_path_str(src))
    return out


def find_summary_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.name.lower() == "summary.json" else []
    return sorted(p for p in root.rglob("summary.json") if p.is_file())


def collect_summary_image_paths(summary_roots: Sequence[Path]) -> Set[str]:
    out: Set[str] = set()
    for root in summary_roots:
        if not root.exists():
            continue
        for summary_path in find_summary_files(root):
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            for row in payload.get("rows", []):
                img_path = str(row.get("image_path", "")).strip()
                if not img_path:
                    continue
                out.add(canonical_path_str(img_path))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean mining manifest that excludes already-labeled source images and previous mining samples.")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recursive", type=str, default="false")
    parser.add_argument("--annotations", type=str, nargs="*", default=[])
    parser.add_argument("--exclude-summary-roots", type=str, nargs="*", default=[])
    args = parser.parse_args()

    recursive = str(args.recursive).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    image_dir = resolve_path(args.image_dir)
    output_manifest = resolve_path(args.output_manifest)
    annotation_paths = [resolve_path(x) for x in args.annotations]
    summary_roots = [resolve_path(x) for x in args.exclude_summary_roots]

    all_images = list_images(image_dir, recursive=recursive)
    labeled_source_paths = collect_labeled_source_paths(annotation_paths)
    prior_summary_paths = collect_summary_image_paths(summary_roots)
    exclude_paths = set(labeled_source_paths)
    exclude_paths.update(prior_summary_paths)

    eligible = [p for p in all_images if str(p).lower() not in exclude_paths]
    n = max(1, min(int(args.num_images), len(eligible)))
    rng = random.Random(int(args.seed))
    selected = rng.sample(eligible, n) if eligible else []

    manifest: Dict[str, object] = {
        "created_at": utc_now_iso(),
        "image_dir": str(image_dir),
        "num_images_requested": int(args.num_images),
        "num_images_selected": int(len(selected)),
        "seed": int(args.seed),
        "recursive": bool(recursive),
        "annotations": [str(p) for p in annotation_paths],
        "exclude_summary_roots": [str(p) for p in summary_roots],
        "exclusion_stats": {
            "all_images": int(len(all_images)),
            "excluded_labeled_source_paths": int(len(labeled_source_paths)),
            "excluded_prior_summary_paths": int(len(prior_summary_paths)),
            "excluded_union": int(len(exclude_paths)),
            "eligible_images": int(len(eligible)),
        },
        "images": [str(p) for p in selected],
    }
    write_json(output_manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
