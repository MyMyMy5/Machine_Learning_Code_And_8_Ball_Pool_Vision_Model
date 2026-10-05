from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.io_utils import save_mask_png, save_rgb_image
from src.supervised.manifest_inference import run_manifest_inference
from src.supervised.infer import run_supervised_inference
from tools.harvest_common import collect_image_files, load_manifest, normalize_path, output_stem

DEFAULT_NUM_WORKERS = 4
MAX_IN_FLIGHT_FACTOR = 4
_WORKER_STATE: dict[str, Any] = {}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _prediction_destinations(output_root: Path, stem: str, source_path: Path) -> dict[str, Path]:
    return {
        "predicted_overlay": output_root / "Predicted" / f"{stem}.png",
        "predicted_mask": output_root / "Masks_Predicted" / f"{stem}.png",
        "no_prediction_image": output_root / "No_Prediction" / f"{stem}{source_path.suffix.lower()}",
    }


def _clear_previous_outputs(paths: dict[str, Path]) -> None:
    _remove_if_exists(paths["predicted_overlay"])
    _remove_if_exists(paths["predicted_mask"])
    _remove_if_exists(paths["no_prediction_image"])


def _already_classified(paths: dict[str, Path]) -> bool:
    return paths["predicted_overlay"].exists() or paths["no_prediction_image"].exists()


def _log(message: str) -> None:
    print(message, flush=True)


def _build_worker_state(
    *,
    primary_checkpoint: Path,
    primary_reranker_path: Path | None,
    output_root: Path,
    image_size: int,
    detection_min_score: float,
    detection_min_mask_pixels: int,
    crop_batch_size: int,
    amp_enabled: bool,
    cpu_threads_per_worker: int,
    full_manifest: bool,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "manifest": manifest,
        "full_manifest": bool(full_manifest),
        "primary_checkpoint": primary_checkpoint,
        "primary_reranker_path": primary_reranker_path,
        "output_root": output_root,
        "image_size": int(image_size),
        "detection_min_score": float(detection_min_score),
        "detection_min_mask_pixels": int(detection_min_mask_pixels),
        "crop_batch_size": int(crop_batch_size),
        "amp_enabled": bool(amp_enabled),
        "cpu_threads_per_worker": max(1, int(cpu_threads_per_worker)),
    }


def _prepare_jobs(
    *,
    frame_files: list[Path],
    frames_dir: Path,
    output_root: Path,
    skip_existing: bool,
) -> tuple[list[dict[str, object]], int]:
    jobs: list[dict[str, object]] = []
    skipped_existing_count = 0
    for source_index, frame_path in enumerate(frame_files, start=1):
        stem = output_stem(frame_path, frames_dir)
        output_paths = _prediction_destinations(output_root, stem, frame_path)
        if skip_existing and _already_classified(output_paths):
            skipped_existing_count += 1
            continue
        jobs.append(
            {
                "frame_path": str(frame_path),
                "stem": stem,
                "source_index": int(source_index),
            }
        )
    return jobs, skipped_existing_count


def _init_worker(
    primary_checkpoint: str,
    primary_reranker_path: str,
    output_root: str,
    image_size: int,
    detection_min_score: float,
    detection_min_mask_pixels: int,
    crop_batch_size: int,
    amp_enabled: bool,
    cpu_threads_per_worker: int,
    full_manifest: bool,
    manifest: dict[str, Any],
) -> None:
    global _WORKER_STATE
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    cpu_threads_per_worker = max(1, int(cpu_threads_per_worker))
    try:
        import cv2

        cv2.setNumThreads(cpu_threads_per_worker)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(cpu_threads_per_worker)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass
    _WORKER_STATE = _build_worker_state(
        primary_checkpoint=Path(primary_checkpoint),
        primary_reranker_path=Path(primary_reranker_path) if primary_reranker_path else None,
        output_root=Path(output_root),
        image_size=image_size,
        detection_min_score=detection_min_score,
        detection_min_mask_pixels=detection_min_mask_pixels,
        crop_batch_size=crop_batch_size,
        amp_enabled=amp_enabled,
        cpu_threads_per_worker=cpu_threads_per_worker,
        full_manifest=full_manifest,
        manifest=manifest,
    )


def _process_frame_job_with_state(
    job: dict[str, object],
    *,
    worker_state: dict[str, Any],
) -> dict[str, object]:
    frame_path = Path(str(job["frame_path"]))
    stem = str(job["stem"])
    output_root = Path(worker_state["output_root"])
    output_paths = _prediction_destinations(output_root, stem, frame_path)
    started_at = time.perf_counter()

    _clear_previous_outputs(output_paths)
    if bool(worker_state.get("full_manifest")):
        result = run_manifest_inference(
            manifest=worker_state["manifest"],
            input_path=frame_path,
            output_dir=output_root / stem,
            image_size=int(worker_state["image_size"]),
            save_outputs=False,
            save_intermediates=False,
            save_report=False,
            save_recolor_preview=False,
            crop_batch_size=int(worker_state["crop_batch_size"]),
            amp_enabled=bool(worker_state["amp_enabled"]),
        )
    else:
        result = run_supervised_inference(
            checkpoint_path=Path(worker_state["primary_checkpoint"]),
            input_path=frame_path,
            output_dir=output_root / stem,
            image_size=int(worker_state["image_size"]),
            reranker_checkpoint_path=(
                Path(worker_state["primary_reranker_path"])
                if worker_state["primary_reranker_path"] is not None
                else None
            ),
            save_outputs=False,
            save_intermediates=False,
            save_report=False,
            save_recolor_preview=False,
            crop_batch_size=int(worker_state["crop_batch_size"]),
            amp_enabled=bool(worker_state["amp_enabled"]),
        )
    winner = result["winner"]
    final_mask = np.asarray(result["final_mask"], dtype=np.uint8)
    final_overlay = np.asarray(result["final_overlay"], dtype=np.uint8)
    score = float(winner["score"])
    mask_pixels = int(np.count_nonzero(final_mask))
    is_detected = (
        score >= float(worker_state["detection_min_score"])
        and mask_pixels >= int(worker_state["detection_min_mask_pixels"])
    )

    if is_detected:
        save_rgb_image(output_paths["predicted_overlay"], final_overlay)
        save_mask_png(output_paths["predicted_mask"], final_mask)
        decision = "predicted"
        decision_path = output_paths["predicted_overlay"]
        mask_path = output_paths["predicted_mask"]
    else:
        shutil.copy2(frame_path, output_paths["no_prediction_image"])
        decision = "no_prediction"
        decision_path = output_paths["no_prediction_image"]
        mask_path = None

    elapsed_seconds = time.perf_counter() - started_at
    return {
        "source_index": int(job["source_index"]),
        "source_path": str(frame_path),
        "stem": stem,
        "decision": decision,
        "score": score,
        "mask_pixels": mask_pixels,
        "output_path": str(decision_path),
        "mask_output_path": str(mask_path) if mask_path is not None else None,
        "elapsed_seconds": float(elapsed_seconds),
    }


def _process_frame_job(job: dict[str, object]) -> dict[str, object]:
    return _process_frame_job_with_state(job, worker_state=_WORKER_STATE)


def run_harvest(
    *,
    frames_dir: Path,
    output_root: Path,
    manifest_path: Path,
    patterns: list[str],
    recurse: bool,
    image_size: int,
    skip_existing: bool,
    detection_min_score: float,
    detection_min_mask_pixels: int,
    crop_batch_size: int,
    amp_enabled: bool,
    num_workers: int = 1,
    cpu_threads_per_worker: int = 1,
    full_manifest: bool = False,
) -> dict[str, object]:
    frame_files = collect_image_files(frames_dir, patterns, recurse)
    if not frame_files:
        raise SystemExit(f"No input images matched under: {frames_dir}")

    _ensure_dir(output_root / "Predicted")
    _ensure_dir(output_root / "Masks_Predicted")
    _ensure_dir(output_root / "No_Prediction")

    manifest = load_manifest(manifest_path)
    primary_checkpoint = Path(manifest["primary_checkpoint"])
    primary_reranker = manifest.get("primary_reranker")
    primary_reranker_path = Path(primary_reranker) if primary_reranker else None
    worker_state = _build_worker_state(
        primary_checkpoint=primary_checkpoint,
        primary_reranker_path=primary_reranker_path,
        output_root=output_root,
        image_size=image_size,
        detection_min_score=detection_min_score,
        detection_min_mask_pixels=detection_min_mask_pixels,
        crop_batch_size=crop_batch_size,
        amp_enabled=amp_enabled,
        cpu_threads_per_worker=cpu_threads_per_worker,
        full_manifest=full_manifest,
        manifest=manifest,
    )
    jobs, skipped_existing_count = _prepare_jobs(
        frame_files=frame_files,
        frames_dir=frames_dir,
        output_root=output_root,
        skip_existing=skip_existing,
    )

    summary_entries: list[dict[str, object]] = []
    detected_count = 0
    not_detected_count = 0
    total_frames = len(frame_files)
    queued_total = len(jobs)
    num_workers = max(1, int(num_workers))
    started_at = time.perf_counter()

    _log(f"Frames root:        {frames_dir}")
    _log(f"Output root:        {output_root}")
    _log(f"Primary checkpoint: {primary_checkpoint}")
    _log(f"Primary reranker:   {primary_reranker_path if primary_reranker_path is not None else 'disabled'}")
    _log(f"Inference mode:     {'full manifest' if full_manifest else 'fast primary/reranker'}")
    _log(f"Worker processes:   {num_workers}")
    _log(f"CPU threads/worker: {cpu_threads_per_worker}")
    _log(f"Crop batch size:    {crop_batch_size}")
    _log(f"Prediction min score:{detection_min_score}")
    _log(f"Prediction min mask: {detection_min_mask_pixels}")
    _log(f"Skip existing:      {skip_existing}")
    _log(f"Existing skipped:   {skipped_existing_count}")
    _log(f"Frames queued:      {queued_total}")

    if queued_total == 0:
        summary = {
            "frames_dir": str(frames_dir),
            "output_root": str(output_root),
            "manifest": str(manifest_path),
            "primary_checkpoint": str(primary_checkpoint),
            "primary_reranker": str(primary_reranker_path) if primary_reranker_path is not None else None,
            "full_manifest": bool(full_manifest),
            "detection_min_score": float(detection_min_score),
            "detection_min_mask_pixels": int(detection_min_mask_pixels),
            "crop_batch_size": int(crop_batch_size),
            "amp_enabled": bool(amp_enabled),
            "num_workers": int(num_workers),
            "cpu_threads_per_worker": int(cpu_threads_per_worker),
            "total_frame_count": int(total_frames),
            "skipped_existing_count": int(skipped_existing_count),
            "processed_count": 0,
            "predicted_count": 0,
            "no_prediction_count": 0,
            "entries": [],
        }
        summary_path = output_root / "harvest_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _log("Finished. processed=0 predicted=0 no_prediction=0 elapsed=0.0s")
        _log(f"Summary: {summary_path}")
        return summary

    completed = 0
    submitted = 0

    def _record_row(row: dict[str, object]) -> None:
        nonlocal completed, detected_count, not_detected_count
        completed += 1
        if row["decision"] == "predicted":
            detected_count += 1
        else:
            not_detected_count += 1
        summary_entries.append(row)
        elapsed = time.perf_counter() - started_at
        _log(
            f"[done {completed}/{queued_total} | source {row['source_index']}/{total_frames}] "
            f"{row['decision']} score={float(row['score']):.4f} mask_pixels={int(row['mask_pixels'])} "
            f"predicted={detected_count} no_prediction={not_detected_count} "
            f"item_elapsed={float(row['elapsed_seconds']):.1f}s elapsed={elapsed:.1f}s"
        )

    if num_workers == 1:
        for job in jobs:
            submitted += 1
            _log(
                f"[start {submitted}/{queued_total} | source {job['source_index']}/{total_frames}] "
                f"{job['frame_path']}"
            )
            row = _process_frame_job_with_state(job, worker_state=worker_state)
            _record_row(row)
    else:
        max_in_flight = max(num_workers, num_workers * MAX_IN_FLIGHT_FACTOR)
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(
                str(primary_checkpoint),
                str(primary_reranker_path) if primary_reranker_path is not None else "",
                str(output_root),
                int(image_size),
                float(detection_min_score),
                int(detection_min_mask_pixels),
                int(crop_batch_size),
                bool(amp_enabled),
                int(cpu_threads_per_worker),
                bool(full_manifest),
                manifest,
            ),
        ) as executor:
            in_flight: dict[object, dict[str, object]] = {}
            next_job_index = 0

            def _submit_next() -> bool:
                nonlocal submitted, next_job_index
                if next_job_index >= queued_total:
                    return False
                job = jobs[next_job_index]
                next_job_index += 1
                submitted += 1
                _log(
                    f"[start {submitted}/{queued_total} | source {job['source_index']}/{total_frames}] "
                    f"{job['frame_path']}"
                )
                future = executor.submit(_process_frame_job, job)
                in_flight[future] = job
                return True

            while len(in_flight) < min(max_in_flight, queued_total) and _submit_next():
                pass

            while in_flight:
                done, _ = wait(tuple(in_flight.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future)
                    row = future.result()
                    _record_row(row)
                    while len(in_flight) < min(max_in_flight, queued_total) and _submit_next():
                        pass

    summary_entries.sort(key=lambda item: int(item["source_index"]))

    summary = {
        "frames_dir": str(frames_dir),
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "primary_checkpoint": str(primary_checkpoint),
        "primary_reranker": str(primary_reranker_path) if primary_reranker_path is not None else None,
        "full_manifest": bool(full_manifest),
        "detection_min_score": float(detection_min_score),
        "detection_min_mask_pixels": int(detection_min_mask_pixels),
        "crop_batch_size": int(crop_batch_size),
        "amp_enabled": bool(amp_enabled),
        "num_workers": int(num_workers),
        "cpu_threads_per_worker": int(cpu_threads_per_worker),
        "total_frame_count": int(total_frames),
        "skipped_existing_count": int(skipped_existing_count),
        "processed_count": len(summary_entries),
        "predicted_count": detected_count,
        "no_prediction_count": not_detected_count,
        "entries": summary_entries,
    }
    summary_path = output_root / "harvest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started_at
    _log(
        f"Finished. processed={len(summary_entries)} predicted={detected_count} "
        f"no_prediction={not_detected_count} elapsed={elapsed:.1f}s"
    )
    _log(f"Summary: {summary_path}")
    return summary


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "runs" / "supervised_best_manifest.json"))
    parser.add_argument("--patterns", nargs="*", default=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"])
    parser.add_argument("--recurse", action="store_true")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--detection-min-score", type=float, default=-999.0)
    parser.add_argument("--detection-min-mask-pixels", type=int, default=1)
    parser.add_argument("--crop-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1)
    parser.add_argument(
        "--full-manifest",
        action="store_true",
        help="Run the full promoted manifest stack instead of the faster primary/reranker-only path.",
    )
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()

    frames_dir = normalize_path(args.frames_dir)
    output_root = normalize_path(args.output_root)
    manifest_path = normalize_path(args.manifest)
    if not frames_dir.exists():
        raise SystemExit(f"Frames directory not found: {frames_dir}")
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    run_harvest(
        frames_dir=frames_dir,
        output_root=output_root,
        manifest_path=manifest_path,
        patterns=list(args.patterns),
        recurse=bool(args.recurse),
        image_size=args.image_size,
        skip_existing=bool(args.skip_existing),
        detection_min_score=args.detection_min_score,
        detection_min_mask_pixels=args.detection_min_mask_pixels,
        crop_batch_size=args.crop_batch_size,
        amp_enabled=not bool(args.disable_amp),
        num_workers=args.num_workers,
        cpu_threads_per_worker=args.cpu_threads_per_worker,
        full_manifest=bool(args.full_manifest),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
