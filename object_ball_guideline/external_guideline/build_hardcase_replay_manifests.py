from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_subset_benchmarks import score_negative_distractor, write_manifest
from cv_guideline_common import (
    build_negative_records,
    load_annotation_records,
    load_split_ids,
    random_split_ids,
    resolve_path,
    split_by_ids,
    split_negatives,
    str2bool,
    utc_now_iso,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only hardcase replay manifests for guideline segmentation.")
    parser.add_argument("--annotations", type=str, default="guideline_line/data_zoomprobe_merged/annotations.jsonl")
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--negative-image-dir", type=str, required=True)
    parser.add_argument("--include-main", type=str2bool, default=True)
    parser.add_argument("--include-quarantine", type=str2bool, default=True)
    parser.add_argument("--include-legacy", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio-fallback", type=float, default=0.15)
    parser.add_argument("--ultra-tiny-thresh", type=float, default=12.0)
    parser.add_argument("--short-thresh", type=float, default=20.0)
    parser.add_argument("--distractor-top-k", type=int, default=48)
    parser.add_argument("--output-dir", type=str, default="guideline_line/hardcase_replay")
    args = parser.parse_args()

    annotations_path = resolve_path(args.annotations)
    splits_path = resolve_path(args.splits)
    negative_dir = resolve_path(args.negative_image_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, load_stats = load_annotation_records(
        annotations_path=annotations_path,
        include_main=bool(args.include_main),
        include_quarantine=bool(args.include_quarantine),
        include_legacy=bool(args.include_legacy),
    )
    train_ids, val_ids = load_split_ids(splits_path)
    split_mode_used = "existing"
    if not train_ids or not val_ids:
        train_ids, val_ids = random_split_ids(
            [r.sample_id for r in records],
            val_ratio=float(args.val_ratio_fallback),
            seed=int(args.seed),
        )
        split_mode_used = "fallback_random"

    train_pos, _val_pos = split_by_ids(records, train_ids, val_ids)
    neg_all = build_negative_records(negative_dir, prefix="NEG_EXT")
    train_neg, _val_neg = split_negatives(neg_all, val_ratio=float(args.val_ratio_fallback), seed=int(args.seed))

    ultra_tiny_pos = [r for r in train_pos if float(r.mask_major_len) < float(args.ultra_tiny_thresh)]
    short_pos = [r for r in train_pos if float(r.mask_major_len) < float(args.short_thresh)]

    distractor_rows = [score_negative_distractor(r.image_path) for r in train_neg]
    distractor_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    top_k = max(1, min(int(args.distractor_top_k), len(distractor_rows))) if distractor_rows else 0
    distractor_top = distractor_rows[:top_k]
    distractor_paths = [Path(str(row["image_path"])) for row in distractor_top]

    ultra_manifest = output_dir / "train_ultra_tiny_lt12.json"
    short_manifest = output_dir / "train_short_lt20.json"
    distractor_manifest = output_dir / f"train_distractor_negatives_top{top_k}.json"
    catalog_path = output_dir / "hardcase_replay_catalog.json"

    write_manifest(
        ultra_manifest,
        [r.image_path for r in ultra_tiny_pos],
        {
            "name": "train_ultra_tiny_lt12",
            "split_mode_used": split_mode_used,
            "splits": str(splits_path),
            "negative_image_dir": str(negative_dir),
            "positive_count": int(len(ultra_tiny_pos)),
            "negative_count": 0,
            "ultra_tiny_thresh": float(args.ultra_tiny_thresh),
            "positive_sample_ids": [r.sample_id for r in ultra_tiny_pos],
        },
    )
    write_manifest(
        short_manifest,
        [r.image_path for r in short_pos],
        {
            "name": "train_short_lt20",
            "split_mode_used": split_mode_used,
            "splits": str(splits_path),
            "negative_image_dir": str(negative_dir),
            "positive_count": int(len(short_pos)),
            "negative_count": 0,
            "short_thresh": float(args.short_thresh),
            "positive_sample_ids": [r.sample_id for r in short_pos],
        },
    )
    write_manifest(
        distractor_manifest,
        distractor_paths,
        {
            "name": f"train_distractor_negatives_top{top_k}",
            "split_mode_used": split_mode_used,
            "splits": str(splits_path),
            "negative_image_dir": str(negative_dir),
            "positive_count": 0,
            "negative_count": int(len(distractor_paths)),
            "selection": "Top-ranked training negatives by white-distractor heuristic.",
        },
    )

    catalog = {
        "created_at": utc_now_iso(),
        "annotations": str(annotations_path),
        "splits": str(splits_path),
        "negative_image_dir": str(negative_dir),
        "split_mode_used": split_mode_used,
        "load_stats": load_stats,
        "counts": {
            "train_positive_total": int(len(train_pos)),
            "train_negative_total": int(len(train_neg)),
            "train_ultra_tiny_lt12": int(len(ultra_tiny_pos)),
            "train_short_lt20": int(len(short_pos)),
            "train_distractor_top_k": int(len(distractor_top)),
        },
        "manifests": {
            "train_ultra_tiny_lt12": str(ultra_manifest),
            "train_short_lt20": str(short_manifest),
            f"train_distractor_negatives_top{top_k}": str(distractor_manifest),
        },
        "positive_subsets": {
            "train_ultra_tiny_lt12": [
                {
                    "sample_id": r.sample_id,
                    "image_path": str(r.image_path),
                    "mask_major_len": float(r.mask_major_len),
                    "source": r.source,
                }
                for r in ultra_tiny_pos
            ],
            "train_short_lt20": [
                {
                    "sample_id": r.sample_id,
                    "image_path": str(r.image_path),
                    "mask_major_len": float(r.mask_major_len),
                    "source": r.source,
                }
                for r in short_pos
            ],
        },
        "distractor_negatives": distractor_top,
    }
    write_json(catalog_path, catalog)

    print(f"Wrote: {ultra_manifest}")
    print(f"Wrote: {short_manifest}")
    print(f"Wrote: {distractor_manifest}")
    print(f"Catalog: {catalog_path}")
    print(
        json.dumps(
            {
                "train_positive_total": len(train_pos),
                "train_negative_total": len(train_neg),
                "train_ultra_tiny_lt12": len(ultra_tiny_pos),
                "train_short_lt20": len(short_pos),
                "train_distractor_top_k": len(distractor_top),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
