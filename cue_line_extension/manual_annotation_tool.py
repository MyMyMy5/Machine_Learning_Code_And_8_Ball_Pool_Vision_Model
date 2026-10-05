#!/usr/bin/env python3
"""Interactive tool for manually extending cue lines on pool images."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Iterable, Tuple

import cv2 as cv
import numpy as np


DEFAULT_SOURCE = Path("data/manual_edited_images")
DEFAULT_IMAGES = Path("data/images")
DEFAULT_MASKS = Path("data/masks")
DEFAULT_OVERLAYS = Path("data/overlays")
LINE_THICKNESS = 6
DEFAULT_POINT_RADIUS = 4

Point = Tuple[int, int]
DISCARD_DIR_NAME = "_discarded"


def iter_images(source: Path) -> Iterable[Path]:
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for path in sorted(source.iterdir()):
        if path.suffix.lower() in supported and path.is_file():
            yield path


def extend_line_to_bounds(p1: Point, p2: Point, width: int, height: int) -> tuple[Point, Point] | None:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1.0:
        return None

    ux = dx / length
    uy = dy / length
    scale = max(width, height) * 2.0
    start = (int(round(p1[0] - ux * scale)), int(round(p1[1] - uy * scale)))
    end = (int(round(p1[0] + ux * scale)), int(round(p1[1] + uy * scale)))

    clip_rect = (0, 0, width, height)
    try:
        ok, clipped_start, clipped_end = cv.clipLine(clip_rect, start, end)
    except cv.error:
        ok, clipped_start, clipped_end = cv.clipLine((width, height), start, end)
    if not ok:
        return None
    return clipped_start, clipped_end


def annotate_image(image: np.ndarray, title: str, thickness: int, point_radius: int) -> tuple[str, tuple[Point, Point] | None]:
    window = "Manual Cue Line"
    window_flags = cv.WINDOW_NORMAL
    window_flags |= getattr(cv, "WINDOW_GUI_EXPANDED", 0)
    cv.namedWindow(window, window_flags)
    cv.imshow(window, image)
    points: list[Point] = []

    height, width = image.shape[:2]
    zoom = 1.0
    max_zoom = 8.0
    center_x = width / 2.0
    center_y = height / 2.0
    view = {"x0": 0, "y0": 0, "vw": width, "vh": height}

    def recalc_view() -> None:
        nonlocal center_x, center_y, view
        view_w = max(1, int(round(width / zoom)))
        view_h = max(1, int(round(height / zoom)))
        half_w = view_w / 2.0
        half_h = view_h / 2.0
        center_x = min(max(center_x, half_w), width - half_w)
        center_y = min(max(center_y, half_h), height - half_h)
        x0 = int(round(center_x - half_w))
        y0 = int(round(center_y - half_h))
        x0 = max(0, min(x0, width - view_w))
        y0 = max(0, min(y0, height - view_h))
        view = {"x0": x0, "y0": y0, "vw": view_w, "vh": view_h}

    def display_to_image_coords(dx: int, dy: int) -> tuple[int, int]:
        vx, vy, vw, vh = view["x0"], view["y0"], view["vw"], view["vh"]
        px = vx + dx * vw / width
        py = vy + dy * vh / height
        return int(round(px)), int(round(py))

    def image_to_display_coords(px: int, py: int) -> tuple[int, int] | None:
        vx, vy, vw, vh = view["x0"], view["y0"], view["vw"], view["vh"]
        if px < vx or px >= vx + vw or py < vy or py >= vy + vh:
            return None
        dx = int(round((px - vx) * width / vw))
        dy = int(round((py - vy) * height / vh))
        return dx, dy

    recalc_view()

    def mouse_handler(event: int, x: int, y: int, flags: int, _userdata: object) -> None:
        nonlocal zoom, center_x, center_y
        if event == cv.EVENT_LBUTTONDOWN:
            px, py = display_to_image_coords(x, y)
            if len(points) >= 2:
                points.clear()
            points.append((px, py))
        elif event == cv.EVENT_MOUSEWHEEL:
            prev_zoom = zoom
            if flags > 0:
                zoom = min(max_zoom, zoom * 1.25)
            else:
                zoom = max(1.0, zoom / 1.25)
            if abs(zoom - prev_zoom) > 1e-4:
                px, py = display_to_image_coords(x, y)
                center_x = float(px)
                center_y = float(py)
                recalc_view()

    cv.setMouseCallback(window, mouse_handler)

    while True:
        recalc_view()
        annotated = image.copy()

        extended: tuple[Point, Point] | None = None
        if len(points) == 2:
            extended = extend_line_to_bounds(points[0], points[1], width, height)
            if extended is not None:
                mask = np.zeros((height, width), dtype=np.uint8)
                cv.line(mask, extended[0], extended[1], 255, thickness, cv.LINE_AA)
                color_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                annotated = cv.addWeighted(annotated, 0.7, color_mask, 0.6, 0)

        vx, vy, vw, vh = view["x0"], view["y0"], view["vw"], view["vh"]
        roi = annotated[vy : vy + vh, vx : vx + vw]
        if roi.size == 0:
            roi = annotated
        display = cv.resize(roi, (width, height), interpolation=cv.INTER_LINEAR)

        cv.putText(
            display,
            "Left click: add point | Wheel: zoom | Backspace: undo | R: reset | S: skip | D: discard | Esc: quit | Space/Enter: confirm",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            display,
            title,
            (10, height - 15),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

        for idx, (px, py) in enumerate(points):
            color = (0, 200, 255) if idx == 0 else (255, 200, 0)
            mapped = image_to_display_coords(px, py)
            if mapped is not None:
                cv.circle(display, mapped, point_radius, color, -1, cv.LINE_AA)

        cv.imshow(window, display)
        key = cv.waitKey(20) & 0xFF
        if key == 255:  # no key pressed
            continue
        if key in (27, ord("q"), ord("Q")):
            cv.destroyWindow(window)
            return "exit", None
        if key in (ord("s"), ord("S")):
            cv.destroyWindow(window)
            return "skip", None
        if key in (ord("d"), ord("D")):
            cv.destroyWindow(window)
            return "discard", None
        if key in (ord("r"), ord("R")):
            points.clear()
            continue
        if key in (8, 127):
            if points:
                points.pop()
            continue
        if key in (13, 10, 32):  # enter, return or space
            if extended is not None:
                cv.destroyWindow(window)
                return "ok", extended


def save_mask_and_overlay(image: np.ndarray, seg: tuple[Point, Point], mask_path: Path, overlay_path: Path, thickness: int) -> None:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.line(mask, seg[0], seg[1], 255, thickness, cv.LINE_AA)

    color_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    overlay = cv.addWeighted(image, 0.7, color_mask, 0.6, 0)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    cv.imwrite(str(mask_path), mask)
    cv.imwrite(str(overlay_path), overlay)


def move_or_copy_image(src: Path, dst: Path, copy_only: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_only:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually draw extended cue lines on images.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Directory with images awaiting manual annotation.")
    parser.add_argument("--images-out", type=Path, default=DEFAULT_IMAGES, help="Directory where original images will be stored.")
    parser.add_argument("--masks-out", type=Path, default=DEFAULT_MASKS, help="Directory where generated masks will be stored.")
    parser.add_argument("--overlays-out", type=Path, default=DEFAULT_OVERLAYS, help="Directory where overlay previews will be stored.")
    parser.add_argument("--thickness", type=int, default=LINE_THICKNESS, help="Line thickness for masks and previews.")
    parser.add_argument("--point-radius", type=int, default=DEFAULT_POINT_RADIUS, help="Radius of annotation points (default: 4).")
    parser.add_argument("--copy-source", action="store_true", help="Copy annotated images instead of moving them from the source directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source
    source.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(source))
    if not images:
        print(f"No images found in {source}.")
        return

    print(f"Found {len(images)} image(s) to annotate in {source}.")

    discard_dir = source / DISCARD_DIR_NAME
    discard_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] {image_path.name}")
        image = cv.imread(str(image_path))
        if image is None:
            print(f"  Skipping {image_path.name}: failed to load.")
            continue

        status, result = annotate_image(image, image_path.name, args.thickness, max(1, args.point_radius))
        if status == "exit":
            print("Exiting annotation tool.")
            cv.destroyAllWindows()
            return
        if status == "skip":
            print("  Skipped - image remains in source directory.")
            continue
        if status == "discard":
            target = discard_dir / image_path.name
            if target.exists():
                print(f"  Discard target already exists for {image_path.name}; image remains in source.")
                continue
            shutil.move(str(image_path), str(target))
            print(f"  Discarded image -> {target}")
            continue
        assert status == "ok" and result is not None

        image_dest = args.images_out / image_path.name
        mask_dest = args.masks_out / image_path.name
        overlay_dest = args.overlays_out / image_path.name

        if image_dest.exists() or mask_dest.exists() or overlay_dest.exists():
            print(f"  Output already exists for {image_path.name}; skipping to avoid overwrite.")
            continue

        save_mask_and_overlay(image, result, mask_dest, overlay_dest, args.thickness)
        move_or_copy_image(image_path, image_dest, args.copy_source)
        print(f"  Saved mask -> {mask_dest}")
        print(f"  Saved overlay -> {overlay_dest}")
        print(f"  {'Copied' if args.copy_source else 'Moved'} image -> {image_dest}")

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
