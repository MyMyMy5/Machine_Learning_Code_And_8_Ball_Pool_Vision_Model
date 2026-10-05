from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from inference_runtime import AsyncInferenceRuntime, InferenceJob, build_process_pool, run_async_pipeline
from shared_runtime_postprocess import eval_postprocess_shared, share_array

from cv_guideline_common import (
    AnnotationRecord,
    NegativeRecord,
    build_centerline_reference,
    build_guideline_postprocess_context,
    build_negative_records,
    load_benchmark_manifest,
    load_annotation_records,
    load_bgr,
    load_checkpoint_model,
    load_mask_u8,
    load_split_ids,
    parse_csv_floats,
    parse_csv_ints,
    precision_recall_f1,
    predict_prob_multi_tta,
    random_split_ids,
    resolve_path,
    safe_div,
    split_by_ids,
    split_negatives,
    str2bool,
    sweep_postprocessed_thresholds,
    utc_now_iso,
    write_json,
)


def render_overlay(frame_bgr: np.ndarray, mask_u8: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    out = frame_bgr.copy()
    m = (mask_u8 > 0)
    if np.any(m):
        c = np.array(color, dtype=np.float32)
        out[m] = np.clip(out[m].astype(np.float32) * (1.0 - alpha) + c * alpha, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1, cv2.LINE_AA)
    return out


def evaluate_thresholds(
    model: torch.nn.Module,
    items: Sequence[Tuple[str, Optional[AnnotationRecord], Optional[NegativeRecord]]],
    device: torch.device,
    tile_sizes: Sequence[int],
    tile_overlaps: Sequence[int],
    tta_scales: Sequence[float],
    tta_hflip: bool,
    thresholds: Sequence[float],
    use_amp: bool,
    axis_complete: bool,
    axis_complete_max_gap: int,
    axis_complete_max_extension: int,
    axis_complete_min_white_ratio: float,
    axis_complete_min_dark_border_ratio: float,
    save_preview_dir: Optional[Path] = None,
    preview_limit: int = 60,
    infer_runtime: str = "direct",
    loader_workers: int = 4,
    postprocess_workers: int = 4,
    prefetch_items: int = 16,
    max_batch_tiles: int = 32,
    max_wait_ms: int = 4,
    bucket_by_tile: bool = True,
    postprocess_backend: str = "thread",
) -> Dict:
    per_thr = {
        float(t): {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "cl_tp": 0,
            "cl_fp": 0,
            "cl_fn": 0,
            "neg_frames": 0,
            "neg_fp_frames": 0,
        }
        for t in thresholds
    }

    if save_preview_dir is not None:
        save_preview_dir.mkdir(parents=True, exist_ok=True)
    preview_saved = 0

    runtime_mode = str(infer_runtime).strip().lower()
    runtime_stats: Dict[str, float] = {}
    if runtime_mode == "async":
        runtime = AsyncInferenceRuntime(
            model=model,
            device=device,
            max_batch_tiles=int(max_batch_tiles),
            max_wait_ms=int(max_wait_ms),
            bucket_by_tile=bool(bucket_by_tile),
        )
        postprocess_executor = None
        try:
            pbar = tqdm(total=len(items), desc="eval", leave=False)

            def load_item(idx: int, item: Tuple[str, Optional[AnnotationRecord], Optional[NegativeRecord]]) -> Dict:
                kind, pos_rec, neg_rec = item
                if kind == "pos" and pos_rec is not None:
                    image = load_bgr(pos_rec.image_path)
                    gt = load_mask_u8(pos_rec.mask_path)
                    return {
                        "index": idx,
                        "kind": kind,
                        "image": image,
                        "gt": gt,
                        "gt_ref": build_centerline_reference(gt, tol_px=2),
                        "sample_id": pos_rec.sample_id,
                        "is_negative": False,
                    }
                if kind == "neg" and neg_rec is not None:
                    image = load_bgr(neg_rec.image_path)
                    return {
                        "index": idx,
                        "kind": kind,
                        "image": image,
                        "gt": np.zeros(image.shape[:2], dtype=np.uint8),
                        "gt_ref": None,
                        "sample_id": neg_rec.sample_id,
                        "is_negative": True,
                    }
                raise RuntimeError(f"Invalid eval item at index={idx}")

            def build_job(idx: int, loaded: Dict) -> InferenceJob:
                image = loaded["image"]
                return InferenceJob(
                    job_id=int(idx),
                    image_path=str(loaded["sample_id"]),
                    image_bgr=image,
                    original_hw=tuple(int(x) for x in image.shape[:2]),
                    tile_sizes=tile_sizes,
                    tile_overlaps=tile_overlaps,
                    tta_scales=tta_scales,
                    tta_hflip=bool(tta_hflip),
                    use_amp=bool(use_amp),
                )

            def postprocess_item(idx: int, loaded: Dict, prob_np: np.ndarray) -> Dict:
                image = loaded["image"]
                gt = loaded["gt"]
                gt_ref = loaded.get("gt_ref")
                is_negative = bool(loaded["is_negative"])
                pp_ctx = build_guideline_postprocess_context(image)
                preview_thr = float(thresholds[len(thresholds) // 2]) if save_preview_dir is not None and int(idx) < int(preview_limit) else None
                local_thr, preview_mask = sweep_postprocessed_thresholds(
                    frame_bgr=image,
                    prob_np=prob_np,
                    thresholds=thresholds,
                    gt_u8=gt,
                    gt_ref=gt_ref,
                    is_negative=is_negative,
                    axis_complete=bool(axis_complete),
                    axis_complete_max_gap=int(axis_complete_max_gap),
                    axis_complete_max_extension=int(axis_complete_max_extension),
                    axis_complete_min_white_ratio=float(axis_complete_min_white_ratio),
                    axis_complete_min_dark_border_ratio=float(axis_complete_min_dark_border_ratio),
                    context=pp_ctx,
                    preview_threshold=preview_thr,
                )

                preview_canvas = None
                if save_preview_dir is not None and int(idx) < int(preview_limit) and preview_mask is not None:
                    t_prev = preview_thr if preview_thr is not None else float(thresholds[len(thresholds) // 2])
                    ov_pred = render_overlay(image, preview_mask, (0, 255, 255), alpha=0.32)
                    ov_gt = render_overlay(image, gt, (0, 255, 0), alpha=0.22)
                    preview_canvas = np.hstack([image, ov_gt, ov_pred])
                    cv2.putText(
                        preview_canvas,
                        f"{loaded['sample_id']} | kind={loaded['kind']} | thr={t_prev:.2f}",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                return {
                    "index": int(idx),
                    "sample_id": str(loaded["sample_id"]),
                    "kind": str(loaded["kind"]),
                    "per_thr": local_thr,
                    "preview_canvas": preview_canvas,
                }

            def submit_process_shared(executor, idx: int, loaded: Dict, prob_np: np.ndarray):
                image_spec, image_shm = share_array(loaded["image"])
                gt_spec, gt_shm = share_array(loaded["gt"])
                prob_spec, prob_shm = share_array(np.asarray(prob_np, dtype=np.float32))
                preview_path = ""
                preview_thr = None
                if save_preview_dir is not None and int(idx) < int(preview_limit):
                    preview_thr = float(thresholds[len(thresholds) // 2])
                    preview_path = str(save_preview_dir / f"{int(idx) + 1:04d}_{loaded['sample_id']}.png")
                payload = {
                    "index": int(idx),
                    "sample_id": str(loaded["sample_id"]),
                    "kind": str(loaded["kind"]),
                    "image_spec": image_spec,
                    "gt_spec": gt_spec,
                    "prob_spec": prob_spec,
                    "thresholds": [float(t) for t in thresholds],
                    "is_negative": bool(loaded["is_negative"]),
                    "axis_complete": bool(axis_complete),
                    "axis_complete_max_gap": int(axis_complete_max_gap),
                    "axis_complete_max_extension": int(axis_complete_max_extension),
                    "axis_complete_min_white_ratio": float(axis_complete_min_white_ratio),
                    "axis_complete_min_dark_border_ratio": float(axis_complete_min_dark_border_ratio),
                    "preview_threshold": preview_thr,
                    "preview_path": preview_path,
                }
                fut = executor.submit(eval_postprocess_shared, payload)

                def _cleanup(_fut):
                    for shm in (image_shm, gt_shm, prob_shm):
                        try:
                            shm.close()
                        finally:
                            try:
                                shm.unlink()
                            except FileNotFoundError:
                                pass

                fut.add_done_callback(_cleanup)
                return fut

            backend = str(postprocess_backend).strip().lower()
            if backend == "process_shared":
                postprocess_executor = build_process_pool(max_workers=int(postprocess_workers))

            results, runtime_stats = run_async_pipeline(
                items=items,
                load_fn=load_item,
                build_job_fn=build_job,
                postprocess_fn=postprocess_item,
                runtime=runtime,
                loader_workers=int(loader_workers),
                postprocess_workers=int(postprocess_workers),
                prefetch_items=int(prefetch_items),
                postprocess_executor=postprocess_executor,
                postprocess_submit_fn=submit_process_shared if backend == "process_shared" else None,
                progress_callback=lambda done, total, stats: (
                    pbar.update(max(0, int(done) - int(pbar.n))),
                    pbar.set_postfix(
                        img=f"{done}/{total}",
                        ips=f"{stats.get('images_per_sec', 0.0):.2f}",
                        tps=f"{stats.get('tiles_per_sec', 0.0):.1f}",
                        b=f"{stats.get('avg_tile_batch', 0.0):.1f}",
                    ),
                ),
            )
            pbar.close()
        finally:
            runtime.shutdown()
            if postprocess_executor is not None:
                postprocess_executor.shutdown(wait=True)

        for row in results:
            for thr in thresholds:
                t = float(thr)
                src = row["per_thr"][t]
                st = per_thr[t]
                st["tp"] += int(src["tp"])
                st["fp"] += int(src["fp"])
                st["fn"] += int(src["fn"])
                st["cl_tp"] += int(src["cl_tp"])
                st["cl_fp"] += int(src["cl_fp"])
                st["cl_fn"] += int(src["cl_fn"])
                st["neg_frames"] += int(src["neg_frames"])
                st["neg_fp_frames"] += int(src["neg_fp_frames"])
            if save_preview_dir is not None and preview_saved < int(preview_limit) and row["preview_canvas"] is not None:
                cv2.imwrite(
                    str(save_preview_dir / f"{int(row['index']) + 1:04d}_{row['sample_id']}.png"),
                    row["preview_canvas"],
                )
                preview_saved += 1
    else:
        pbar = tqdm(items, desc="eval", leave=False)
        for idx, (kind, pos_rec, neg_rec) in enumerate(pbar, start=1):
            if kind == "pos" and pos_rec is not None:
                image = load_bgr(pos_rec.image_path)
                gt = load_mask_u8(pos_rec.mask_path)
                gt_ref = build_centerline_reference(gt, tol_px=2)
                sample_id = pos_rec.sample_id
                is_negative = False
            elif kind == "neg" and neg_rec is not None:
                image = load_bgr(neg_rec.image_path)
                gt = np.zeros(image.shape[:2], dtype=np.uint8)
                gt_ref = None
                sample_id = neg_rec.sample_id
                is_negative = True
            else:
                continue

            prob_np = predict_prob_multi_tta(
                model=model,
                frame_bgr=image,
                device=device,
                tile_sizes=tile_sizes,
                tile_overlaps=tile_overlaps,
                tta_scales=tta_scales,
                tta_hflip=tta_hflip,
                use_amp=use_amp,
            )
            pp_ctx = build_guideline_postprocess_context(image)
            preview_thr = float(thresholds[len(thresholds) // 2]) if save_preview_dir is not None and preview_saved < int(preview_limit) else None
            local_thr, preview_mask = sweep_postprocessed_thresholds(
                frame_bgr=image,
                prob_np=prob_np,
                thresholds=thresholds,
                gt_u8=gt,
                gt_ref=gt_ref,
                is_negative=is_negative,
                axis_complete=bool(axis_complete),
                axis_complete_max_gap=int(axis_complete_max_gap),
                axis_complete_max_extension=int(axis_complete_max_extension),
                axis_complete_min_white_ratio=float(axis_complete_min_white_ratio),
                axis_complete_min_dark_border_ratio=float(axis_complete_min_dark_border_ratio),
                context=pp_ctx,
                preview_threshold=preview_thr,
            )
            for thr in thresholds:
                t = float(thr)
                src = local_thr[t]
                st = per_thr[t]
                st["tp"] += int(src["tp"])
                st["fp"] += int(src["fp"])
                st["fn"] += int(src["fn"])
                st["cl_tp"] += int(src["cl_tp"])
                st["cl_fp"] += int(src["cl_fp"])
                st["cl_fn"] += int(src["cl_fn"])
                st["neg_frames"] += int(src["neg_frames"])
                st["neg_fp_frames"] += int(src["neg_fp_frames"])

            if save_preview_dir is not None and preview_saved < int(preview_limit) and preview_mask is not None:
                t_prev = preview_thr if preview_thr is not None else float(thresholds[len(thresholds) // 2])
                ov_pred = render_overlay(image, preview_mask, (0, 255, 255), alpha=0.32)
                ov_gt = render_overlay(image, gt, (0, 255, 0), alpha=0.22)
                canvas = np.hstack([image, ov_gt, ov_pred])
                cv2.putText(canvas, f"{sample_id} | kind={kind} | thr={t_prev:.2f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(save_preview_dir / f"{idx:04d}_{sample_id}.png"), canvas)
                preview_saved += 1

    by_thr: Dict[str, Dict] = {}
    best_thr = None
    best_score = -1e9
    best_metrics = None
    for thr in thresholds:
        t = float(thr)
        st = per_thr[t]
        p, r, f1 = precision_recall_f1(st["tp"], st["fp"], st["fn"])
        cp, cr, cf1 = precision_recall_f1(st["cl_tp"], st["cl_fp"], st["cl_fn"])
        fp_frame = safe_div(st["neg_fp_frames"], st["neg_frames"]) if st["neg_frames"] > 0 else 0.0
        score = (0.50 * p) + (0.30 * f1) + (0.20 * cf1) - (0.20 * fp_frame)
        row = {
            "pixel_precision": float(p),
            "pixel_recall": float(r),
            "pixel_f1": float(f1),
            "centerline_precision": float(cp),
            "centerline_recall": float(cr),
            "centerline_f1": float(cf1),
            "fp_per_frame": float(fp_frame),
            "tp": int(st["tp"]),
            "fp": int(st["fp"]),
            "fn": int(st["fn"]),
            "neg_frames": int(st["neg_frames"]),
            "neg_fp_frames": int(st["neg_fp_frames"]),
            "composite_score": float(score),
        }
        by_thr[str(round(t, 4))] = row
        if (score > best_score) or (
            abs(score - best_score) < 1e-9 and p > float((best_metrics or {}).get("pixel_precision", -1.0))
        ):
            best_score = float(score)
            best_thr = float(t)
            best_metrics = row

    return {
        "best_threshold": float(best_thr if best_thr is not None else thresholds[0]),
        "best": best_metrics if best_metrics is not None else {},
        "by_threshold": by_thr,
        "preview_saved": int(preview_saved),
        "runtime_mode": runtime_mode,
        "runtime_stats": runtime_stats,
        "runtime_postprocess_backend": str(postprocess_backend).strip().lower() if runtime_mode == "async" else "thread",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate non-LoRA cue-line segmentation checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--annotations", type=str, default="guideline_line/data_zoomprobe_merged/annotations.jsonl")
    parser.add_argument("--splits", type=str, default="guideline_line/data_zoomprobe_merged/splits.json")
    parser.add_argument("--negative-image-dir", type=str, default="")
    parser.add_argument("--report-out", type=str, default="")
    parser.add_argument("--save-previews", type=str2bool, default=False)
    parser.add_argument("--include-main", type=str2bool, default=True)
    parser.add_argument("--include-quarantine", type=str2bool, default=True)
    parser.add_argument("--include-legacy", type=str2bool, default=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--tile-sizes", type=str, default="640,896")
    parser.add_argument("--tile-overlaps", type=str, default="160,224")
    parser.add_argument("--tta-scales", type=str, default="1.0,1.15")
    parser.add_argument("--tta-hflip", type=str2bool, default=True)
    parser.add_argument("--axis-complete", type=str2bool, default=True)
    parser.add_argument("--axis-complete-max-gap", type=int, default=24)
    parser.add_argument("--axis-complete-max-extension", type=int, default=160)
    parser.add_argument("--axis-complete-min-white-ratio", type=float, default=0.42)
    parser.add_argument("--axis-complete-min-dark-border-ratio", type=float, default=0.05)
    parser.add_argument("--benchmark-manifest", type=str, default="")
    parser.add_argument("--infer-runtime", type=str, default="direct", choices=["direct", "async"])
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--postprocess-workers", type=int, default=4)
    parser.add_argument("--postprocess-backend", type=str, default="thread", choices=["thread", "process_shared"])
    parser.add_argument("--prefetch-items", type=int, default=16)
    parser.add_argument("--max-batch-tiles", type=int, default=32)
    parser.add_argument("--max-wait-ms", type=int, default=4)
    parser.add_argument("--bucket-by-tile", type=str2bool, default=True)
    parser.add_argument("--save-runtime-stats", type=str2bool, default=True)
    parser.add_argument("--runtime-fallback-direct", type=str2bool, default=True)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio-fallback", type=float, default=0.15)
    args = parser.parse_args()

    ckpt_path = resolve_path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    annotations_path = resolve_path(args.annotations)
    splits_path = resolve_path(args.splits)
    report_out = resolve_path(args.report_out) if str(args.report_out).strip() else (ckpt_path.parent / "eval_report.json")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    tile_sizes = parse_csv_ints(args.tile_sizes, [int(args.tile_size), max(int(args.tile_size), 896)])
    tile_overlaps = parse_csv_ints(args.tile_overlaps, [int(args.tile_overlap), max(int(args.tile_overlap), 224)])
    if len(tile_overlaps) < len(tile_sizes):
        tile_overlaps.extend([tile_overlaps[-1] if tile_overlaps else 160] * (len(tile_sizes) - len(tile_overlaps)))
    elif len(tile_overlaps) > len(tile_sizes):
        tile_overlaps = tile_overlaps[: len(tile_sizes)]
    tta_scales = parse_csv_floats(args.tta_scales, [1.0, 1.15])

    records, load_stats = load_annotation_records(
        annotations_path=annotations_path,
        include_main=bool(args.include_main),
        include_quarantine=bool(args.include_quarantine),
        include_legacy=bool(args.include_legacy),
    )
    train_ids, val_ids = load_split_ids(splits_path)
    if not train_ids or not val_ids:
        t_ids, v_ids = random_split_ids([r.sample_id for r in records], val_ratio=float(args.val_ratio_fallback), seed=int(args.seed))
        train_ids, val_ids = t_ids, v_ids
    _train_pos, val_pos = split_by_ids(records, train_ids, val_ids)
    if not val_pos:
        raise RuntimeError("No validation positives available after split filtering.")

    neg_dir = resolve_path(args.negative_image_dir) if str(args.negative_image_dir).strip() else None
    neg_all = build_negative_records(neg_dir, prefix="NEG_EXT") if neg_dir is not None else []
    _train_neg, val_neg = split_negatives(neg_all, val_ratio=float(args.val_ratio_fallback), seed=int(args.seed))

    benchmark_manifest = resolve_path(args.benchmark_manifest) if str(args.benchmark_manifest).strip() else None
    if benchmark_manifest is not None and benchmark_manifest.exists():
        manifest_images = set(str(p) for p in load_benchmark_manifest(benchmark_manifest))
        val_pos = [r for r in val_pos if str(r.image_path) in manifest_images]
        val_neg = [r for r in val_neg if str(r.image_path) in manifest_images]
        if not val_pos and not val_neg:
            raise RuntimeError(f"Benchmark manifest {benchmark_manifest} matched no validation items.")
    items: List[Tuple[str, Optional[AnnotationRecord], Optional[NegativeRecord]]] = []
    for r in val_pos:
        items.append(("pos", r, None))
    for r in val_neg:
        items.append(("neg", None, r))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint_model(ckpt_path, device=device)
    thresholds = [round(x, 2) for x in np.arange(0.30, 0.8001, 0.05)]
    preview_dir = (report_out.parent / "eval_previews") if bool(args.save_previews) else None
    try:
        out = evaluate_thresholds(
            model=model,
            items=items,
            device=device,
            tile_sizes=tile_sizes,
            tile_overlaps=tile_overlaps,
            tta_scales=tta_scales,
            tta_hflip=bool(args.tta_hflip),
            thresholds=thresholds,
            use_amp=bool(args.amp),
            axis_complete=bool(args.axis_complete),
            axis_complete_max_gap=int(args.axis_complete_max_gap),
            axis_complete_max_extension=int(args.axis_complete_max_extension),
            axis_complete_min_white_ratio=float(args.axis_complete_min_white_ratio),
            axis_complete_min_dark_border_ratio=float(args.axis_complete_min_dark_border_ratio),
            save_preview_dir=preview_dir,
            preview_limit=80,
            infer_runtime=str(args.infer_runtime),
            loader_workers=int(args.loader_workers),
            postprocess_workers=int(args.postprocess_workers),
            prefetch_items=int(args.prefetch_items),
            max_batch_tiles=int(args.max_batch_tiles),
            max_wait_ms=int(args.max_wait_ms),
            bucket_by_tile=bool(args.bucket_by_tile),
            postprocess_backend=str(args.postprocess_backend),
        )
    except Exception:
        if not bool(args.runtime_fallback_direct) or str(args.infer_runtime).strip().lower() != "async":
            raise
        print("Eval async runtime failed; retrying in direct mode.")
        out = evaluate_thresholds(
            model=model,
            items=items,
            device=device,
            tile_sizes=tile_sizes,
            tile_overlaps=tile_overlaps,
            tta_scales=tta_scales,
            tta_hflip=bool(args.tta_hflip),
            thresholds=thresholds,
            use_amp=bool(args.amp),
            axis_complete=bool(args.axis_complete),
            axis_complete_max_gap=int(args.axis_complete_max_gap),
            axis_complete_max_extension=int(args.axis_complete_max_extension),
            axis_complete_min_white_ratio=float(args.axis_complete_min_white_ratio),
            axis_complete_min_dark_border_ratio=float(args.axis_complete_min_dark_border_ratio),
            save_preview_dir=preview_dir,
            preview_limit=80,
            infer_runtime="direct",
            loader_workers=int(args.loader_workers),
            postprocess_workers=int(args.postprocess_workers),
            prefetch_items=int(args.prefetch_items),
            max_batch_tiles=int(args.max_batch_tiles),
            max_wait_ms=int(args.max_wait_ms),
            bucket_by_tile=bool(args.bucket_by_tile),
            postprocess_backend="thread",
        )

    report = {
        "timestamp": utc_now_iso(),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "model_family": str(ckpt.get("model_family", "custom")),
        "model_name": str(ckpt.get("model_name", "")),
        "backbone": str(ckpt.get("backbone", "resnet34")),
        "img_size": int(ckpt.get("img_size", -1)),
        "records_val_positive": len(val_pos),
        "records_val_negative": len(val_neg),
        "records_total": len(items),
        "best_threshold": float(out["best_threshold"]),
        "best_metrics": out["best"],
        "metrics_by_threshold": out["by_threshold"],
        "data_load_stats": load_stats,
        "paths": {
            "annotations": str(annotations_path),
            "splits": str(splits_path),
            "negative_image_dir": str(neg_dir) if neg_dir is not None else "",
            "preview_dir": str(preview_dir) if preview_dir is not None else "",
            "benchmark_manifest": str(benchmark_manifest) if benchmark_manifest is not None else "",
        },
        "tile_config": {"tile_sizes": tile_sizes, "tile_overlaps": tile_overlaps},
        "tta_config": {"tta_scales": tta_scales, "tta_hflip": bool(args.tta_hflip)},
        "runtime": {
            "infer_runtime": str(out.get("runtime_mode", str(args.infer_runtime))),
            "loader_workers": int(args.loader_workers),
            "postprocess_workers": int(args.postprocess_workers),
            "postprocess_backend": str(out.get("runtime_postprocess_backend", str(args.postprocess_backend))),
            "prefetch_items": int(args.prefetch_items),
            "max_batch_tiles": int(args.max_batch_tiles),
            "max_wait_ms": int(args.max_wait_ms),
            "bucket_by_tile": bool(args.bucket_by_tile),
            "runtime_stats": out.get("runtime_stats", {}),
        },
    }
    write_json(report_out, report)
    if bool(args.save_runtime_stats):
        write_json(
            report_out.parent / "runtime_stats.json",
            {
                "updated_at": utc_now_iso(),
                "runtime_mode": str(out.get("runtime_mode", str(args.infer_runtime))),
                "runtime_stats": out.get("runtime_stats", {}),
            },
        )
    print(f"Device: {device}")
    print(f"Val positives: {len(val_pos)} | Val negatives: {len(val_neg)}")
    print(f"Best threshold: {float(out['best_threshold']):.2f}")
    bm = out["best"]
    print(
        f"P={float(bm.get('pixel_precision', 0.0)):.4f} "
        f"R={float(bm.get('pixel_recall', 0.0)):.4f} "
        f"F1={float(bm.get('pixel_f1', 0.0)):.4f} "
        f"cF1={float(bm.get('centerline_f1', 0.0)):.4f} "
        f"FP/frame={float(bm.get('fp_per_frame', 0.0)):.4f}"
    )
    print(f"Report: {report_out}")
    if preview_dir is not None:
        print(f"Previews: {preview_dir}")


if __name__ == "__main__":
    main()
