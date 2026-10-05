#!/usr/bin/env python3
"""Record a cropped region from a monitor into a video file."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

try:
    import mss
except ImportError as exc:
    raise ImportError("Install 'mss' to use this recorder: pip install mss") from exc


def load_region(path: Path) -> Tuple[dict[str, int], Optional[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"left", "top", "width", "height"}
    if not required.issubset(data):
        missing = required - data.keys()
        raise ValueError(f"Region JSON missing keys: {', '.join(sorted(missing))}")
    region = {key: int(data[key]) for key in required}
    monitor = int(data["monitor"]) if "monitor" in data else None
    return region, monitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a cropped area from a monitor.")
    parser.add_argument("--monitor", type=int, default=None, help="Monitor index as reported by mss. If omitted, use JSON value or 1.")
    parser.add_argument("--region-json", type=Path, help="JSON file produced by calibrate_region.py.")
    parser.add_argument("--left", type=int, help="Left pixel coordinate of the crop region.")
    parser.add_argument("--top", type=int, help="Top pixel coordinate of the crop region.")
    parser.add_argument("--width", type=int, help="Width of the crop region.")
    parser.add_argument("--height", type=int, help="Height of the crop region.")
    parser.add_argument("--fps", type=float, default=30.0, help="Recording frame rate (default: 30).")
    parser.add_argument("--output", type=Path, default=None, help="Output video path (default: recordings/capture_YYYYmmdd_HHMMSS.mp4).")
    parser.add_argument("--codec", type=str, default="mp4v", help="FourCC codec for VideoWriter (default: mp4v).")
    parser.add_argument("--preview", action="store_true", help="Show a live preview window while recording.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional duration limit in seconds.")
    return parser.parse_args()


def resolve_region(args: argparse.Namespace) -> Tuple[dict[str, int], Optional[int]]:
    region: Optional[dict[str, int]] = None
    monitor_from_json: Optional[int] = None
    if args.region_json is not None:
        region, monitor_from_json = load_region(args.region_json)
    else:
        specified = [args.left, args.top, args.width, args.height]
        if all(value is not None for value in specified):
            region = {"left": args.left, "top": args.top, "width": args.width, "height": args.height}
    if region is None:
        raise SystemExit("Specify region via --region-json or explicit --left/--top/--width/--height.")
    if region["width"] <= 0 or region["height"] <= 0:
        raise SystemExit("Region width/height must be positive.")
    return region, monitor_from_json


def build_writer(path: Path, frame_size: tuple[int, int], fps: float, codec: str) -> cv.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv.VideoWriter_fourcc(*codec)
    writer = cv.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {path} (codec={codec}, fps={fps}).")
    return writer


def determine_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("recordings") / f"capture_{timestamp}.mp4"


def main() -> None:
    args = parse_args()
    region, monitor_override = resolve_region(args)
    monitor_index = args.monitor if args.monitor is not None else (monitor_override if monitor_override is not None else 1)
    output_path = determine_output_path(args)

    fps = max(1.0, float(args.fps))
    frame_interval = 1.0 / fps

    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_index >= len(monitors):
            raise SystemExit(f"Monitor index {monitor_index} out of range (available: 1-{len(monitors) - 1})")

        bbox = {
            "left": region["left"],
            "top": region["top"],
            "width": region["width"],
            "height": region["height"],
        }
        frame_size = (bbox["width"], bbox["height"])

        writer = build_writer(output_path, frame_size, fps, args.codec)
        print(f"[INFO] Recording monitor {monitor_index} region {bbox} to {output_path}")
        print("Press Ctrl+C (or close preview window) to stop.")

        preview_window = "Recorder Preview"
        start_time = time.time()
        next_tick = time.perf_counter()

        try:
            while True:
                frame_raw = sct.grab(bbox)
                frame = np.array(frame_raw)[:, :, :3]
                writer.write(frame)

                if args.preview:
                    cv.imshow(preview_window, frame)
                    if cv.waitKey(1) & 0xFF in (27, ord("q")):
                        print("[INFO] Preview window closed; stopping recording.")
                        break

                if args.max_seconds is not None and (time.time() - start_time) >= args.max_seconds:
                    print("[INFO] Reached max duration; stopping recording.")
                    break

                next_tick += frame_interval
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n[INFO] Recording interrupted by user.")
        finally:
            writer.release()
            if args.preview:
                cv.destroyWindow(preview_window)

    print(f"[DONE] Recording saved to {output_path}")


if __name__ == "__main__":
    main()
