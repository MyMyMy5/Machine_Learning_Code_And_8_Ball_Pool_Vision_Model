from __future__ import annotations

import argparse
import os
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def resolve_path(path_like: str) -> Path:
    path = Path(str(path_like)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def list_videos(root: Path, recursive: bool) -> List[Path]:
    if not root.exists():
        return []
    globber = root.rglob if recursive else root.glob
    videos = []
    for path in globber("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            videos.append(path.resolve())
    return sorted(videos)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def crop_center(frame: np.ndarray, crop_ratio: float) -> np.ndarray:
    ratio = min(1.0, max(0.1, float(crop_ratio)))
    if ratio >= 0.999:
        return frame
    h, w = frame.shape[:2]
    crop_w = max(1, int(round(w * ratio)))
    crop_h = max(1, int(round(h * ratio)))
    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    return frame[y1 : y1 + crop_h, x1 : x1 + crop_w]


def build_compare_frame(
    frame: np.ndarray,
    crop_ratio: float,
    resize_width: int,
    blur_kernel: int,
) -> np.ndarray:
    cropped = crop_center(frame, crop_ratio)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    target_w = max(32, int(resize_width))
    h, w = gray.shape[:2]
    scale = float(target_w) / float(max(1, w))
    target_h = max(18, int(round(h * scale)))
    small = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    k = max(0, int(blur_kernel))
    if k > 1:
        if k % 2 == 0:
            k += 1
        small = cv2.GaussianBlur(small, (k, k), sigmaX=0.0)
    return small


def compare_frames_mad(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    diff = cv2.absdiff(a, b)
    return float(diff.mean()) / 255.0


def save_frame(
    frame: np.ndarray,
    output_dir: Path,
    stem: str,
    frame_idx: int,
    jpeg_quality: int,
    image_ext: str,
) -> None:
    out_name = f"{stem}_f{frame_idx:08d}{image_ext}"
    out_path = output_dir / out_name
    if image_ext.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        cv2.imwrite(str(out_path), frame)


def extract_sampled_frames(
    cap: cv2.VideoCapture,
    output_dir: Path,
    stem: str,
    frame_step: int,
    jpeg_quality: int,
    image_ext: str,
    dedup_enabled: bool,
    dedup_threshold: float,
    dedup_center_crop_ratio: float,
    dedup_resize_width: int,
    dedup_blur_kernel: int,
    min_gap_frames: int,
) -> Dict[str, float]:
    frame_idx = 0
    saved = 0
    sampled = 0
    skipped_gap = 0
    skipped_dedup = 0
    last_saved_frame_idx: Optional[int] = None
    last_compare_frame: Optional[np.ndarray] = None
    kept_scores: List[float] = []
    skipped_scores: List[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, int(frame_step)) != 0:
            frame_idx += 1
            continue
        sampled += 1
        if last_saved_frame_idx is not None and (frame_idx - last_saved_frame_idx) < max(0, int(min_gap_frames)):
            skipped_gap += 1
            frame_idx += 1
            continue
        compare_frame = build_compare_frame(
            frame=frame,
            crop_ratio=float(dedup_center_crop_ratio),
            resize_width=int(dedup_resize_width),
            blur_kernel=int(dedup_blur_kernel),
        )
        if dedup_enabled and last_compare_frame is not None:
            score = compare_frames_mad(compare_frame, last_compare_frame)
            if score < float(dedup_threshold):
                skipped_dedup += 1
                skipped_scores.append(float(score))
                frame_idx += 1
                continue
            kept_scores.append(float(score))
        save_frame(
            frame=frame,
            output_dir=output_dir,
            stem=stem,
            frame_idx=frame_idx,
            jpeg_quality=jpeg_quality,
            image_ext=image_ext,
        )
        last_saved_frame_idx = int(frame_idx)
        last_compare_frame = compare_frame
        saved += 1
        frame_idx += 1
    return {
        "frames_read": int(frame_idx),
        "frames_sampled": int(sampled),
        "frames_saved": int(saved),
        "frames_skipped_min_gap": int(skipped_gap),
        "frames_skipped_dedup": int(skipped_dedup),
        "avg_kept_diff_score": float(sum(kept_scores) / len(kept_scores)) if kept_scores else 0.0,
        "avg_skipped_diff_score": float(sum(skipped_scores) / len(skipped_scores)) if skipped_scores else 0.0,
    }


def extract_by_fps(
    cap: cv2.VideoCapture,
    output_dir: Path,
    stem: str,
    sample_fps: float,
    jpeg_quality: int,
    image_ext: str,
    dedup_enabled: bool,
    dedup_threshold: float,
    dedup_center_crop_ratio: float,
    dedup_resize_width: int,
    dedup_blur_kernel: int,
    min_gap_frames: int,
) -> Dict[str, int]:
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0:
        raise RuntimeError("Video FPS could not be determined for --sample-fps mode.")
    step = max(1, int(round(source_fps / max(0.01, float(sample_fps)))))
    return extract_sampled_frames(
        cap=cap,
        output_dir=output_dir,
        stem=stem,
        frame_step=step,
        jpeg_quality=jpeg_quality,
        image_ext=image_ext,
        dedup_enabled=dedup_enabled,
        dedup_threshold=dedup_threshold,
        dedup_center_crop_ratio=dedup_center_crop_ratio,
        dedup_resize_width=dedup_resize_width,
        dedup_blur_kernel=dedup_blur_kernel,
        min_gap_frames=min_gap_frames,
    ) | {"source_fps": source_fps, "frame_step": int(step)}


def _jpeg_quality_to_ffmpeg_qscale(jpeg_quality: int) -> int:
    quality = max(2, min(100, int(jpeg_quality)))
    qscale = int(round(np.interp(float(quality), [100.0, 95.0, 90.0, 80.0, 60.0], [2.0, 2.0, 3.0, 5.0, 10.0])))
    return max(2, min(31, qscale))


def _count_extracted_frames(output_dir: Path, stem: str, image_ext: str) -> int:
    pattern = f"{stem}_f*{image_ext}"
    return sum(1 for _ in output_dir.glob(pattern))


def _run_ffmpeg_extract(
    *,
    video_path: Path,
    output_dir: Path,
    stem: str,
    mode: str,
    every_n_frames: int,
    sample_fps: float,
    image_ext: str,
    jpeg_quality: int,
    hwaccel: str,
) -> Dict[str, object]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg was not found on PATH.")

    ensure_dir(output_dir)
    out_pattern = str((output_dir / f"{stem}_f%08d{image_ext}").resolve())
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if hwaccel and hwaccel.lower() not in {"", "none", "auto"}:
        command.extend(["-hwaccel", hwaccel])
    command.extend(["-i", str(video_path.resolve())])

    if mode == "sample_fps":
        vf_expr = f"fps={float(sample_fps):.8f}"
    else:
        step = max(1, int(every_n_frames))
        vf_expr = f"select='not(mod(n\\,{step}))'"
        command.extend(["-vsync", "vfr"])
    command.extend(["-vf", vf_expr])

    if image_ext.lower() in {".jpg", ".jpeg"}:
        command.extend(["-q:v", str(_jpeg_quality_to_ffmpeg_qscale(jpeg_quality))])
    command.append(out_pattern)

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg extraction failed")

    saved = _count_extracted_frames(output_dir, stem, image_ext)
    return {
        "frames_read": -1,
        "frames_sampled": -1,
        "frames_saved": int(saved),
        "frames_skipped_min_gap": 0,
        "frames_skipped_dedup": 0,
        "avg_kept_diff_score": 0.0,
        "avg_skipped_diff_score": 0.0,
        "ffmpeg_backend": True,
        "ffmpeg_hwaccel": hwaccel if hwaccel else "none",
    }


def _extract_single_video(job: Dict[str, object]) -> Dict[str, object]:
    video_path = Path(str(job["video_path"])).resolve()
    output_dir = Path(str(job["output_dir"])).resolve()
    ensure_dir(output_dir)

    backend = str(job.get("backend", "opencv")).lower()
    ffmpeg_hwaccel = str(job.get("ffmpeg_hwaccel", "none"))

    if backend == "ffmpeg":
        try:
            stats = _run_ffmpeg_extract(
                video_path=video_path,
                output_dir=output_dir,
                stem=str(job["stem"]),
                mode=str(job["mode"]),
                every_n_frames=int(job["every_n_frames"]),
                sample_fps=float(job["sample_fps"]),
                image_ext=str(job["image_ext"]),
                jpeg_quality=int(job["jpeg_quality"]),
                hwaccel=ffmpeg_hwaccel,
            )
        except Exception:
            if ffmpeg_hwaccel and ffmpeg_hwaccel.lower() not in {"", "none", "auto"}:
                stats = _run_ffmpeg_extract(
                    video_path=video_path,
                    output_dir=output_dir,
                    stem=str(job["stem"]),
                    mode=str(job["mode"]),
                    every_n_frames=int(job["every_n_frames"]),
                    sample_fps=float(job["sample_fps"]),
                    image_ext=str(job["image_ext"]),
                    jpeg_quality=int(job["jpeg_quality"]),
                    hwaccel="none",
                )
            else:
                raise
        return {
            "video_path": str(video_path),
            "status": "ok",
            "output_dir": str(output_dir),
            **stats,
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video_path": str(video_path),
            "status": "failed_open",
            "output_dir": str(output_dir),
        }

    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        min_gap_seconds = float(job["min_gap_seconds"])
        min_gap_frames = int(round(max(0.0, min_gap_seconds) * source_fps)) if source_fps > 0.0 else 0
        mode = str(job["mode"])
        if mode == "sample_fps":
            stats = extract_by_fps(
                cap=cap,
                output_dir=output_dir,
                stem=str(job["stem"]),
                sample_fps=float(job["sample_fps"]),
                jpeg_quality=int(job["jpeg_quality"]),
                image_ext=str(job["image_ext"]),
                dedup_enabled=bool(job["dedup"]),
                dedup_threshold=float(job["dedup_threshold"]),
                dedup_center_crop_ratio=float(job["dedup_center_crop_ratio"]),
                dedup_resize_width=int(job["dedup_resize_width"]),
                dedup_blur_kernel=int(job["dedup_blur_kernel"]),
                min_gap_frames=int(min_gap_frames),
            )
        else:
            stats = extract_sampled_frames(
                cap=cap,
                output_dir=output_dir,
                stem=str(job["stem"]),
                frame_step=max(1, int(job["every_n_frames"])),
                jpeg_quality=int(job["jpeg_quality"]),
                image_ext=str(job["image_ext"]),
                dedup_enabled=bool(job["dedup"]),
                dedup_threshold=float(job["dedup_threshold"]),
                dedup_center_crop_ratio=float(job["dedup_center_crop_ratio"]),
                dedup_resize_width=int(job["dedup_resize_width"]),
                dedup_blur_kernel=int(job["dedup_blur_kernel"]),
                min_gap_frames=int(min_gap_frames),
            )
        return {
            "video_path": str(video_path),
            "status": "ok",
            "output_dir": str(output_dir),
            **stats,
        }
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from every video in a folder.")
    parser.add_argument("--video-dir", type=str, default="guideline_line/video_drop")
    parser.add_argument("--output-root", type=str, default="guideline_line/video_frames")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--backend", type=str, choices=["opencv", "ffmpeg"], default="opencv")
    parser.add_argument("--ffmpeg-hwaccel", type=str, default="none")
    parser.add_argument("--mode", type=str, choices=["every_n_frames", "sample_fps"], default="sample_fps")
    parser.add_argument("--every-n-frames", type=int, default=30)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--image-ext", type=str, default=".jpg", choices=[".jpg", ".jpeg", ".png"])
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--flat-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dedup-threshold", type=float, default=0.014)
    parser.add_argument("--dedup-center-crop-ratio", type=float, default=0.82)
    parser.add_argument("--dedup-resize-width", type=int, default=224)
    parser.add_argument("--dedup-blur-kernel", type=int, default=5)
    parser.add_argument("--min-gap-seconds", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    video_dir = resolve_path(args.video_dir)
    output_root = resolve_path(args.output_root)
    ensure_dir(video_dir)
    ensure_dir(output_root)

    videos = list_videos(video_dir, recursive=bool(args.recursive))
    if not videos:
        print(f"No videos found in: {video_dir}")
        return

    summary_rows: List[Dict[str, object]] = []
    jobs: List[Dict[str, object]] = []
    for video_path in videos:
        rel_parent = video_path.parent.relative_to(video_dir) if video_path.parent != video_dir else Path()
        target_dir = output_root if bool(args.flat_output) else (output_root / rel_parent / video_path.stem)
        jobs.append(
            {
                "video_path": str(video_path),
                "output_dir": str(target_dir),
                "stem": video_path.stem,
                "backend": str(args.backend),
                "ffmpeg_hwaccel": str(args.ffmpeg_hwaccel),
                "mode": str(args.mode),
                "every_n_frames": int(args.every_n_frames),
                "sample_fps": float(args.sample_fps),
                "jpeg_quality": int(args.jpeg_quality),
                "image_ext": str(args.image_ext),
                "dedup": bool(args.dedup),
                "dedup_threshold": float(args.dedup_threshold),
                "dedup_center_crop_ratio": float(args.dedup_center_crop_ratio),
                "dedup_resize_width": int(args.dedup_resize_width),
                "dedup_blur_kernel": int(args.dedup_blur_kernel),
                "min_gap_seconds": float(args.min_gap_seconds),
            }
        )

    worker_count = max(1, int(args.workers))
    if worker_count == 1 or len(jobs) == 1:
        for job in jobs:
            row = _extract_single_video(job)
            summary_rows.append(row)
            if row.get("status") == "ok":
                print(f"{Path(str(row['video_path'])).name}: saved {row['frames_saved']} frame(s) -> {row['output_dir']}")
            else:
                print(f"{Path(str(row['video_path'])).name}: failed to open")
    else:
        max_workers = min(worker_count, len(jobs), max(1, os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_job = {pool.submit(_extract_single_video, job): job for job in jobs}
            for future in as_completed(future_to_job):
                row = future.result()
                summary_rows.append(row)
                if row.get("status") == "ok":
                    print(f"{Path(str(row['video_path'])).name}: saved {row['frames_saved']} frame(s) -> {row['output_dir']}")
                else:
                    print(f"{Path(str(row['video_path'])).name}: failed to open")

        summary_rows.sort(key=lambda row: str(row.get("video_path", "")).lower())

    summary = {
        "video_dir": str(video_dir),
        "output_root": str(output_root),
        "mode": str(args.mode),
        "backend": str(args.backend),
        "ffmpeg_hwaccel": str(args.ffmpeg_hwaccel),
        "every_n_frames": int(args.every_n_frames),
        "sample_fps": float(args.sample_fps),
        "image_ext": str(args.image_ext),
        "flat_output": bool(args.flat_output),
        "dedup": {
            "enabled": bool(args.dedup),
            "threshold": float(args.dedup_threshold),
            "center_crop_ratio": float(args.dedup_center_crop_ratio),
            "resize_width": int(args.dedup_resize_width),
            "blur_kernel": int(args.dedup_blur_kernel),
            "min_gap_seconds": float(args.min_gap_seconds),
        },
        "workers": int(worker_count),
        "rows": summary_rows,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
