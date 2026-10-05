"""Interactive overlay reviewer: keep or discard pseudo-label frames.

Usage:
    python review_overlays.py --overlays data/overlays --images data/images --masks data/masks

Controls:
    Space / Enter : toggle keep (default keep)
    D             : discard current frame
    A             : previous frame
    S             : skip without decision
    F             : mark for fix (moves to masks_fix folder)
    Q / Esc       : quit and save decisions

If an overlay is discarded or marked for fix, the corresponding image/mask files
are moved to `rejected/` or `fix/` subdirectories under their respective roots.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

import cv2 as cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive review of overlay masks")
    parser.add_argument("--overlays", type=Path, required=True, help="Directory with overlay previews")
    parser.add_argument("--images", type=Path, required=True, help="Directory with original frames")
    parser.add_argument("--masks", type=Path, required=True, help="Directory with mask images")
    parser.add_argument("--output", type=Path, default=Path("data/review_log.txt"), help="File to append review summary")
    return parser.parse_args()


def ensure_dirs(base_paths: List[Path]) -> None:
    for base in base_paths:
        (base / "rejected").mkdir(parents=True, exist_ok=True)
        (base / "fix").mkdir(parents=True, exist_ok=True)


def move_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def review(overlays_dir: Path, images_dir: Path, masks_dir: Path, log_file: Path) -> None:
    overlay_paths = sorted(p for p in overlays_dir.glob("*.png"))
    if not overlay_paths:
        print(f"No overlays found in {overlays_dir}")
        return

    rejected_overlays = overlays_dir / "rejected"
    fix_overlays = overlays_dir / "fix"
    ensure_dirs([overlays_dir, images_dir, masks_dir])

    idx = 0
    total = len(overlay_paths)
    cv.namedWindow("Overlay Review", cv.WINDOW_NORMAL)

    while 0 <= idx < total:
        overlay_path = overlay_paths[idx]
        overlay = cv.imread(str(overlay_path))
        if overlay is None:
            print(f"Failed to read {overlay_path}")
            idx += 1
            continue

        display = overlay.copy()
        cv.putText(display, f"[{idx+1}/{total}] {overlay_path.name}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(display, "Space:Keep  D:Discard  F:Fix  A:Prev  S:Skip  Q:Quit", (20, display.shape[0]-40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow("Overlay Review", display)
        key = cv.waitKey(0) & 0xFF

        stem = overlay_path.stem
        image_path = images_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.png"

        if key in (ord("q"), 27):
            break
        elif key == ord("a"):
            idx = max(0, idx - 1)
            continue
        elif key in (ord("s"), ord(" "), ord("\r")):
            idx += 1
            continue
        elif key == ord("d"):
            move_file(overlay_path, rejected_overlays / overlay_path.name)
            move_file(image_path, images_dir / "rejected" / image_path.name)
            move_file(mask_path, masks_dir / "rejected" / mask_path.name)
            idx += 1
        elif key == ord("f"):
            move_file(overlay_path, fix_overlays / overlay_path.name)
            move_file(image_path, images_dir / "fix" / image_path.name)
            move_file(mask_path, masks_dir / "fix" / mask_path.name)
            idx += 1
        else:
            idx += 1

    cv.destroyWindow("Overlay Review")

    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"Reviewed {total} overlays; rejected saved in subfolders.\n")


def main() -> None:
    args = parse_args()
    review(args.overlays, args.images, args.masks, args.output)


if __name__ == "__main__":
    main()
