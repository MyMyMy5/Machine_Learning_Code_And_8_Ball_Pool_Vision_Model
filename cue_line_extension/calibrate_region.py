"""
Interactively select a capture region for `live_inference.py`.

Workflow:
1. The script grabs a still frame from the chosen monitor.
2. You left-click the top-left corner of the game area and then the bottom-right corner.
3. It prints the resulting `left,top,width,height` tuple and optionally stores it in JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np

try:
    import mss
except ImportError as exc:
    raise ImportError("Install 'mss' to use this calibration tool: pip install mss") from exc


def capture_monitor(monitor_index: int) -> np.ndarray:
    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_index >= len(monitors):
            raise IndexError(f"Monitor index {monitor_index} out of range (available: 1-{len(monitors) - 1})")
        frame = sct.grab(monitors[monitor_index])
        return np.array(frame)[:, :, :3], monitors[monitor_index]


def compute_region(points: List[Tuple[int, int]], monitor_origin: Tuple[int, int]) -> Tuple[int, int, int, int]:
    (x1, y1), (x2, y2) = points
    left = min(x1, x2) + monitor_origin[0]
    top = min(y1, y2) + monitor_origin[1]
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return left, top, width, height


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive region calibration for live inference")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index as reported by mss (default: 1)")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional JSON file to store the region values")
    args = parser.parse_args()

    frame_bgr, monitor_meta = capture_monitor(args.monitor)
    origin = (monitor_meta["left"], monitor_meta["top"])

    points: List[Tuple[int, int]] = []
    display = frame_bgr.copy()

    instructions = (
        "Calibration: left-click top-left corner, then bottom-right corner. "
        "Press 'r' to reset, 'q' to quit."
    )
    cv.putText(display, instructions, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    window_name = f"Monitor {args.monitor} Calibration"

    def on_mouse(event, x, y, _flags, _param):
        nonlocal display
        if event == cv.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv.circle(display, (x, y), 6, (0, 0, 255), -1)
            if len(points) == 2:
                cv.rectangle(display, points[0], points[1], (0, 255, 255), 2)

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setMouseCallback(window_name, on_mouse)

    while True:
        cv.imshow(window_name, display)
        key = cv.waitKey(10) & 0xFF

        if key == ord("q") or key == 27:
            break
        if key == ord("r"):
            points.clear()
            display = frame_bgr.copy()
            cv.putText(display, instructions, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(points) == 2:
            left, top, width, height = compute_region(points, origin)
            msg = f"Region -> left={left}, top={top}, width={width}, height={height}"
            print(msg)
            if args.save_json is not None:
                data = {
                    "monitor": args.monitor,
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                }
                args.save_json.parent.mkdir(parents=True, exist_ok=True)
                with args.save_json.open("w", encoding="utf-8") as fp:
                    json.dump(data, fp, indent=2)
                print(f"Saved region to {args.save_json}")
            print("Press 'q' or Esc to close, or 'r' to pick again.")
            points.clear()

    cv.destroyWindow(window_name)


if __name__ == "__main__":
    main()
