#!/usr/bin/env python3
"""Utility to extract frames from raw pool videos for manual annotation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import cv2 as cv


DEFAULT_INPUT = Path("data/raw_videos")
DEFAULT_OUTPUT = Path("data/manual_edited_images")


def sanitize_stem(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_-]+", "_", name).strip("_")
    return sanitized or "clip"


def iter_videos(directory: Path) -> Iterable[Path]:
    supported = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv"}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in supported:
            yield path


def frame_name(base: str, index: int, ext: str) -> str:
    return f"{base}_{index:06d}{ext}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames for manual cue-line annotation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Directory containing raw videos.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory to store extracted frames.")
    parser.add_argument("--stride", type=int, default=15, help="Keep every N-th frame (default: 15).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum frames saved per video (default: unlimited).")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index offset for saved frames.")
    parser.add_argument("--reset", action="store_true", help="Remove existing files in the output directory before extraction.")
    parser.add_argument("--image-ext", default=".png", choices=[".png", ".jpg", ".jpeg", ".bmp"], help="Image format for saved frames.")
    return parser.parse_args()


def maybe_reset(output_dir: Path, do_reset: bool) -> None:
    if not do_reset or not output_dir.exists():
        return
    removed = 0
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()
            removed += 1
    if removed:
        print(f"[INFO] Cleared {removed} file(s) from {output_dir}.")


def extract_frames(
    video_path: Path,
    output_dir: Path,
    stride: int,
    limit: int | None,
    start_index: int,
    extension: str,
) -> int:
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Unable to open video: {video_path}")
        return 0

    base = sanitize_stem(video_path.stem)
    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        filename = frame_name(base, start_index + saved, extension)
        destination = output_dir / filename
        if destination.exists():
            print(f"[SKIP] {destination} already exists.")
        else:
            if not cv.imwrite(str(destination), frame):
                print(f"[WARN] Failed to write frame to {destination}.")
            else:
                saved += 1

        frame_idx += 1
        if limit is not None and saved >= limit:
            break

    cap.release()
    print(f"[INFO] {video_path.name}: saved {saved} frame(s).")
    return saved


def main() -> None:
    args = parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    maybe_reset(output_dir, args.reset)

    videos = list(iter_videos(args.input))
    if not videos:
        print(f"[INFO] No supported videos found in {args.input}.")
        return

    stride = max(1, args.stride)
    total_saved = 0
    for video_path in videos:
        saved = extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            stride=stride,
            limit=args.limit,
            start_index=args.start_index,
            extension=args.image_ext,
        )
        total_saved += saved

    print(f"[DONE] Saved {total_saved} frame(s) to {output_dir}.")


if __name__ == "__main__":
    main()
