from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from inference_runtime import AsyncInferenceRuntime, InferenceJob, build_process_pool, run_async_pipeline
from shared_runtime_postprocess import infer_postprocess_shared, share_array

from cv_guideline_common import (
    build_guideline_postprocess_context,
    build_benchmark_manifest,
    list_images,
    load_benchmark_manifest,
    load_bgr,
    load_checkpoint_model,
    parse_csv_floats,
    parse_csv_ints,
    postprocess_guideline_mask,
    predict_prob_multi_tta,
    resolve_path,
    str2bool,
    utc_now_iso,
    write_json,
)


def render_overlay(frame_bgr: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.32) -> np.ndarray:
    out = frame_bgr.copy()
    m = mask_u8 > 0
    if np.any(m):
        color = np.array([0, 255, 255], dtype=np.float32)
        out[m] = np.clip(out[m].astype(np.float32) * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiled inference with non-LoRA cue-line detector and export overlays.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="guideline_line/CV_Test")
    parser.add_argument("--num-images", type=int, default=70)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--save-prob-maps", type=str2bool, default=False)
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
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--recursive", type=str2bool, default=False)
    parser.add_argument("--timestamp-subdir", type=str2bool, default=True)
    args = parser.parse_args()

    ckpt_path = resolve_path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    image_dir = resolve_path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    out_root = resolve_path(args.output_root)
    if bool(args.timestamp_subdir):
        import datetime as dt

        out_root = out_root / f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tile_sizes = parse_csv_ints(args.tile_sizes, [int(args.tile_size), max(int(args.tile_size), 896)])
    tile_overlaps = parse_csv_ints(args.tile_overlaps, [int(args.tile_overlap), max(int(args.tile_overlap), 224)])
    if len(tile_overlaps) < len(tile_sizes):
        tile_overlaps.extend([tile_overlaps[-1] if tile_overlaps else 160] * (len(tile_sizes) - len(tile_overlaps)))
    elif len(tile_overlaps) > len(tile_sizes):
        tile_overlaps = tile_overlaps[: len(tile_sizes)]
    tta_scales = parse_csv_floats(args.tta_scales, [1.0, 1.15])

    benchmark_manifest = resolve_path(args.benchmark_manifest) if str(args.benchmark_manifest).strip() else None
    if benchmark_manifest is not None and benchmark_manifest.exists():
        selected = load_benchmark_manifest(benchmark_manifest)
    else:
        all_images = list_images(image_dir, recursive=bool(args.recursive))
        if not all_images:
            raise RuntimeError(f"No images found in: {image_dir}")
        n = max(1, min(int(args.num_images), len(all_images)))
        rng = random.Random(int(args.seed))
        selected = rng.sample(all_images, n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint_model(ckpt_path, device=device)

    threshold = args.threshold
    if threshold is None:
        threshold = ckpt.get("best_threshold", None)
    if threshold is None:
        tjson = ckpt_path.parent / "thresholds.json"
        if tjson.exists():
            import json

            data = json.loads(tjson.read_text(encoding="utf-8-sig"))
            threshold = float(data.get("best_threshold", 0.5))
    if threshold is None:
        threshold = 0.5
    threshold = float(threshold)

    masks_dir = out_root / "masks"
    overlays_dir = out_root / "overlays"
    originals_dir = out_root / "originals"
    prob_dir = out_root / "prob_maps"
    for d in (masks_dir, overlays_dir, originals_dir):
        d.mkdir(parents=True, exist_ok=True)
    if bool(args.save_prob_maps):
        prob_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    runtime_mode = str(args.infer_runtime).strip().lower()
    runtime_stats: Dict[str, float] = {}

    def run_direct() -> List[Dict]:
        out_rows: List[Dict] = []
        pbar = tqdm(selected, desc="infer", leave=False)
        for i, img_path in enumerate(pbar, start=1):
            img = load_bgr(img_path)
            prob_np = predict_prob_multi_tta(
                model=model,
                frame_bgr=img,
                device=device,
                tile_sizes=tile_sizes,
                tile_overlaps=tile_overlaps,
                tta_scales=tta_scales,
                tta_hflip=bool(args.tta_hflip),
                use_amp=bool(args.amp),
            )
            pp_ctx = build_guideline_postprocess_context(img)
            mask_bin = postprocess_guideline_mask(
                img,
                prob_np,
                threshold=threshold,
                axis_complete=bool(args.axis_complete),
                axis_complete_max_gap=int(args.axis_complete_max_gap),
                axis_complete_max_extension=int(args.axis_complete_max_extension),
                axis_complete_min_white_ratio=float(args.axis_complete_min_white_ratio),
                axis_complete_min_dark_border_ratio=float(args.axis_complete_min_dark_border_ratio),
                context=pp_ctx,
            )
            mask_u8 = (mask_bin > 0).astype(np.uint8) * 255
            overlay = render_overlay(img, mask_u8, alpha=0.34)
            area = int((mask_u8 > 0).sum())
            base_name = f"{i:04d}_{img_path.stem}.png"
            out_mask = masks_dir / base_name
            out_overlay = overlays_dir / base_name
            out_orig = originals_dir / base_name
            cv2.imwrite(str(out_mask), mask_u8)
            cv2.imwrite(str(out_overlay), overlay)
            cv2.imwrite(str(out_orig), img)
            prob_path = ""
            if bool(args.save_prob_maps):
                prob_u8 = np.clip(prob_np * 255.0, 0, 255).astype(np.uint8)
                out_prob = prob_dir / base_name
                cv2.imwrite(str(out_prob), prob_u8)
                prob_path = str(out_prob)
            out_rows.append(
                {
                    "index": i,
                    "image_path": str(img_path),
                    "mask_path": str(out_mask),
                    "overlay_path": str(out_overlay),
                    "original_path": str(out_orig),
                    "prob_map_path": prob_path,
                    "pred_area": area,
                    "threshold": threshold,
                }
            )
            pbar.set_postfix(area=area)
        return out_rows

    if runtime_mode == "async":
        runtime = AsyncInferenceRuntime(
            model=model,
            device=device,
            max_batch_tiles=int(args.max_batch_tiles),
            max_wait_ms=int(args.max_wait_ms),
            bucket_by_tile=bool(args.bucket_by_tile),
        )
        postprocess_executor = None
        try:
            pbar = tqdm(total=len(selected), desc="infer", leave=False)

            def load_item(idx: int, img_path: Path) -> Dict:
                return {"index": idx, "image_path": img_path, "image": load_bgr(img_path)}

            def build_job(idx: int, loaded: Dict) -> InferenceJob:
                image = loaded["image"]
                return InferenceJob(
                    job_id=int(idx),
                    image_path=str(loaded["image_path"]),
                    image_bgr=image,
                    original_hw=tuple(int(x) for x in image.shape[:2]),
                    tile_sizes=tile_sizes,
                    tile_overlaps=tile_overlaps,
                    tta_scales=tta_scales,
                    tta_hflip=bool(args.tta_hflip),
                    use_amp=bool(args.amp),
                )

            def postprocess_item(idx: int, loaded: Dict, prob_np: np.ndarray) -> Dict:
                img_path = loaded["image_path"]
                img = loaded["image"]
                pp_ctx = build_guideline_postprocess_context(img)
                mask_bin = postprocess_guideline_mask(
                    img,
                    prob_np,
                    threshold=threshold,
                    axis_complete=bool(args.axis_complete),
                    axis_complete_max_gap=int(args.axis_complete_max_gap),
                    axis_complete_max_extension=int(args.axis_complete_max_extension),
                    axis_complete_min_white_ratio=float(args.axis_complete_min_white_ratio),
                    axis_complete_min_dark_border_ratio=float(args.axis_complete_min_dark_border_ratio),
                    context=pp_ctx,
                )
                mask_u8 = (mask_bin > 0).astype(np.uint8) * 255
                overlay = render_overlay(img, mask_u8, alpha=0.34)
                area = int((mask_u8 > 0).sum())
                base_name = f"{int(idx) + 1:04d}_{img_path.stem}.png"
                out_mask = masks_dir / base_name
                out_overlay = overlays_dir / base_name
                out_orig = originals_dir / base_name
                cv2.imwrite(str(out_mask), mask_u8)
                cv2.imwrite(str(out_overlay), overlay)
                cv2.imwrite(str(out_orig), img)
                prob_path = ""
                if bool(args.save_prob_maps):
                    prob_u8 = np.clip(prob_np * 255.0, 0, 255).astype(np.uint8)
                    out_prob = prob_dir / base_name
                    cv2.imwrite(str(out_prob), prob_u8)
                    prob_path = str(out_prob)
                return {
                    "index": int(idx) + 1,
                    "image_path": str(img_path),
                    "mask_path": str(out_mask),
                    "overlay_path": str(out_overlay),
                    "original_path": str(out_orig),
                    "prob_map_path": prob_path,
                    "pred_area": area,
                    "threshold": threshold,
                }

            def submit_process_shared(executor, idx: int, loaded: Dict, prob_np: np.ndarray):
                img_path = loaded["image_path"]
                image_spec, image_shm = share_array(loaded["image"])
                prob_spec, prob_shm = share_array(np.asarray(prob_np, dtype=np.float32))
                base_name = f"{int(idx) + 1:04d}_{img_path.stem}.png"
                out_mask = masks_dir / base_name
                out_overlay = overlays_dir / base_name
                out_orig = originals_dir / base_name
                out_prob = prob_dir / base_name if bool(args.save_prob_maps) else None
                payload = {
                    "index": int(idx) + 1,
                    "image_path": str(img_path),
                    "image_spec": image_spec,
                    "prob_spec": prob_spec,
                    "threshold": float(threshold),
                    "axis_complete": bool(args.axis_complete),
                    "axis_complete_max_gap": int(args.axis_complete_max_gap),
                    "axis_complete_max_extension": int(args.axis_complete_max_extension),
                    "axis_complete_min_white_ratio": float(args.axis_complete_min_white_ratio),
                    "axis_complete_min_dark_border_ratio": float(args.axis_complete_min_dark_border_ratio),
                    "mask_path": str(out_mask),
                    "overlay_path": str(out_overlay),
                    "original_path": str(out_orig),
                    "prob_map_path": str(out_prob) if out_prob is not None else "",
                }
                fut = executor.submit(infer_postprocess_shared, payload)

                def _cleanup(_fut):
                    for shm in (image_shm, prob_shm):
                        try:
                            shm.close()
                        finally:
                            try:
                                shm.unlink()
                            except FileNotFoundError:
                                pass

                fut.add_done_callback(_cleanup)
                return fut

            backend = str(args.postprocess_backend).strip().lower()
            if backend == "process_shared":
                postprocess_executor = build_process_pool(max_workers=int(args.postprocess_workers))

            rows, runtime_stats = run_async_pipeline(
                items=selected,
                load_fn=load_item,
                build_job_fn=build_job,
                postprocess_fn=postprocess_item,
                runtime=runtime,
                loader_workers=int(args.loader_workers),
                postprocess_workers=int(args.postprocess_workers),
                prefetch_items=int(args.prefetch_items),
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
        except Exception:
            runtime.shutdown()
            if postprocess_executor is not None:
                postprocess_executor.shutdown(wait=True)
            if not bool(args.runtime_fallback_direct):
                raise
            print("Infer async runtime failed; retrying in direct mode.")
            runtime_mode = "direct"
            rows = run_direct()
        else:
            runtime.shutdown()
            if postprocess_executor is not None:
                postprocess_executor.shutdown(wait=True)
    else:
        rows = run_direct()

    summary = {
        "timestamp": utc_now_iso(),
        "checkpoint": str(ckpt_path),
        "model_family": str(ckpt.get("model_family", "custom")),
        "model_name": str(ckpt.get("model_name", "")),
        "backbone": str(ckpt.get("backbone", "resnet34")),
        "image_dir": str(image_dir),
        "output_root": str(out_root),
        "num_images_requested": int(args.num_images),
        "num_images_processed": len(rows),
        "tile_config": {"tile_sizes": tile_sizes, "tile_overlaps": tile_overlaps},
        "tta_config": {"tta_scales": tta_scales, "tta_hflip": bool(args.tta_hflip)},
        "benchmark_manifest": str(benchmark_manifest) if benchmark_manifest is not None else "",
        "threshold": threshold,
        "runtime": {
            "infer_runtime": runtime_mode,
            "loader_workers": int(args.loader_workers),
            "postprocess_workers": int(args.postprocess_workers),
            "postprocess_backend": str(args.postprocess_backend if runtime_mode == "async" else "thread"),
            "prefetch_items": int(args.prefetch_items),
            "max_batch_tiles": int(args.max_batch_tiles),
            "max_wait_ms": int(args.max_wait_ms),
            "bucket_by_tile": bool(args.bucket_by_tile),
            "runtime_stats": runtime_stats,
        },
        "rows": rows,
    }
    write_json(out_root / "summary.json", summary)
    if bool(args.save_runtime_stats):
        write_json(
            out_root / "runtime_stats.json",
            {
                "updated_at": utc_now_iso(),
                "runtime_mode": runtime_mode,
                "runtime_stats": runtime_stats,
            },
        )
    print(f"Device: {device}")
    print(f"Images processed: {len(rows)}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Output root: {out_root}")
    print(f"Summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
