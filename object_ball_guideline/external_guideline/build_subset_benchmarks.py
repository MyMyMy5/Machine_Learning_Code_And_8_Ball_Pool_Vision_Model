from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from cv_guideline_common import (
    build_guideline_postprocess_context,
    build_negative_records,
    load_annotation_records,
    load_bgr,
    load_split_ids,
    random_split_ids,
    resolve_path,
    split_by_ids,
    split_negatives,
    str2bool,
    utc_now_iso,
    write_json,
)


def circularity_from_contour(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0.0 or perimeter <= 1e-6:
        return 0.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


def score_negative_distractor(image_path: Path) -> Dict[str, object]:
    image = load_bgr(image_path)
    ctx = build_guideline_postprocess_context(image)
    mask = (ctx.white_like > 0).astype(np.uint8)
    if int(mask.sum()) <= 0:
        return {
            "image_path": str(image_path),
            "score": 0.0,
            "white_area": 0,
            "component_count": 0,
            "top_components": [],
        }

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comp_rows: List[Dict[str, float]] = []
    total_score = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 12.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = float(max(1, w * h))
        fill_ratio = area / bbox_area
        circ = max(0.0, min(1.25, circularity_from_contour(contour)))
        mean_side = 0.5 * float(w + h)
        area_term = min(area / 900.0, 1.0)
        size_term = min(mean_side / 48.0, 1.0)
        compact_term = 1.0 - min(abs(fill_ratio - 0.62) / 0.62, 1.0)
        circular_term = max(0.0, min(circ, 1.0))
        score = (0.45 * area_term) + (0.20 * size_term) + (0.20 * compact_term) + (0.15 * circular_term)
        total_score += float(score)
        comp_rows.append(
            {
                "area": float(area),
                "bbox_w": float(w),
                "bbox_h": float(h),
                "fill_ratio": float(fill_ratio),
                "circularity": float(circ),
                "score": float(score),
            }
        )

    comp_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    top_components = comp_rows[:5]
    white_area = int(mask.sum())
    white_ratio = float(white_area) / float(mask.shape[0] * mask.shape[1])
    total_score += min(white_ratio / 0.02, 1.0) * 0.25

    return {
        "image_path": str(image_path),
        "score": float(total_score),
        "white_area": int(white_area),
        "white_ratio": float(white_ratio),
        "component_count": int(len(comp_rows)),
        "top_components": top_components,
    }


def write_manifest(
    output_path: Path,
    image_paths: Sequence[Path],
    meta: Dict[str, object],
) -> None:
    payload = {
        "created_at": utc_now_iso(),
        "num_images": int(len(image_paths)),
        "images": [str(p) for p in image_paths],
        "meta": meta,
    }
    write_json(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed subset benchmark manifests for guideline segmentation.")
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
    parser.add_argument("--output-dir", type=str, default="guideline_line/benchmarks")
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
    if not train_ids or not val_ids:
        train_ids, val_ids = random_split_ids(
            [r.sample_id for r in records],
            val_ratio=float(args.val_ratio_fallback),
            seed=int(args.seed),
        )
    _train_pos, val_pos = split_by_ids(records, train_ids, val_ids)
    neg_all = build_negative_records(negative_dir, prefix="NEG_EXT")
    _train_neg, val_neg = split_negatives(neg_all, val_ratio=float(args.val_ratio_fallback), seed=int(args.seed))

    ultra_tiny_pos = [r for r in val_pos if float(r.mask_major_len) < float(args.ultra_tiny_thresh)]
    short_pos = [r for r in val_pos if float(r.mask_major_len) < float(args.short_thresh)]

    distractor_rows = [score_negative_distractor(r.image_path) for r in val_neg]
    distractor_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    top_k = max(1, min(int(args.distractor_top_k), len(distractor_rows)))
    distractor_top = distractor_rows[:top_k]
    distractor_paths = [Path(str(row["image_path"])) for row in distractor_top]

    ultra_manifest = output_dir / "ultra_tiny_lt12_plus_valneg.json"
    short_manifest = output_dir / "short_lt20_plus_valneg.json"
    distractor_manifest = output_dir / f"distractor_negatives_top{top_k}.json"
    catalog_path = output_dir / "subset_benchmark_catalog.json"

    write_manifest(
        ultra_manifest,
        [r.image_path for r in ultra_tiny_pos] + [r.image_path for r in val_neg],
        {
            "name": "ultra_tiny_lt12_plus_valneg",
            "splits": str(splits_path),
            "negative_image_dir": str(negative_dir),
            "positive_count": int(len(ultra_tiny_pos)),
            "negative_count": int(len(val_neg)),
            "ultra_tiny_thresh": float(args.ultra_tiny_thresh),
            "positive_sample_ids": [r.sample_id for r in ultra_tiny_pos],
        },
    )
    write_manifest(
        short_manifest,
        [r.image_path for r in short_pos] + [r.image_path for r in val_neg],
        {
            "name": "short_lt20_plus_valneg",
            "splits": str(splits_path),
            "negative_image_dir": str(negative_dir),
            "positive_count": int(len(short_pos)),
            "negative_count": int(len(val_neg)),
            "short_thresh": float(args.short_thresh),
            "positive_sample_ids": [r.sample_id for r in short_pos],
        },
    )
    write_manifest(
        distractor_manifest,
        distractor_paths,
        {
            "name": f"distractor_negatives_top{top_k}",
            "splits": str(splits_path),
            "negative_image_dir": str(negative_dir),
            "positive_count": 0,
            "negative_count": int(len(distractor_paths)),
            "selection": "Top-ranked validation negatives by white-distractor heuristic.",
        },
    )

    catalog = {
        "created_at": utc_now_iso(),
        "annotations": str(annotations_path),
        "splits": str(splits_path),
        "negative_image_dir": str(negative_dir),
        "load_stats": load_stats,
        "counts": {
            "val_positive_total": int(len(val_pos)),
            "val_negative_total": int(len(val_neg)),
            "ultra_tiny_lt12": int(len(ultra_tiny_pos)),
            "short_lt20": int(len(short_pos)),
            "distractor_top_k": int(len(distractor_top)),
        },
        "manifests": {
            "ultra_tiny_lt12_plus_valneg": str(ultra_manifest),
            "short_lt20_plus_valneg": str(short_manifest),
            f"distractor_negatives_top{top_k}": str(distractor_manifest),
        },
        "positive_subsets": {
            "ultra_tiny_lt12": [
                {
                    "sample_id": r.sample_id,
                    "image_path": str(r.image_path),
                    "mask_major_len": float(r.mask_major_len),
                    "source": r.source,
                }
                for r in ultra_tiny_pos
            ],
            "short_lt20": [
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
                "val_positive_total": len(val_pos),
                "val_negative_total": len(val_neg),
                "ultra_tiny_lt12": len(ultra_tiny_pos),
                "short_lt20": len(short_pos),
                "distractor_top_k": len(distractor_top),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
