from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


WINDOW_NAME = "Negative Image Review"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
RESUME_NAME = "negative_review_resume.json"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review candidate negative images and copy approved ones into a separate folder.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(r"C:\My_Project\8_BALL_POOL\data\images\rejected"),
        help="Source image directory to review.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(r"C:\My_Project\8_BALL_POOL\data\images\negative_selected"),
        help="Folder where approved negative images are copied.",
    )
    parser.add_argument(
        "--resume-last",
        type=str2bool,
        default=True,
        help="Resume from the last visited index if state exists.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Optional zero-based starting index.",
    )
    parser.add_argument(
        "--max-panel-size",
        type=int,
        default=1100,
        help="Maximum display width or height for the review window.",
    )
    return parser


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def list_images(image_dir: Path, save_dir: Path) -> List[Path]:
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    save_resolved = save_dir.resolve()
    files: List[Path] = []
    for p in sorted(image_dir.glob("*"), key=lambda q: str(q).lower()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.resolve().parent == save_resolved:
            continue
        files.append(p.resolve())
    return files


def load_resume_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_resume_state(path: Path, image_dir: Path, save_dir: Path, index: int, image_path: Optional[Path]) -> None:
    payload = {
        "image_dir": str(image_dir.resolve()),
        "save_dir": str(save_dir.resolve()),
        "last_index": int(index),
        "last_image_path": str(image_path.resolve()) if image_path is not None else "",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_start_index(images: List[Path], image_dir: Path, save_dir: Path, start_arg: Optional[int], resume_last: bool) -> int:
    if not images:
        return 0
    if start_arg is not None:
        return max(0, min(len(images) - 1, int(start_arg)))
    if not resume_last:
        return 0
    state_path = save_dir / RESUME_NAME
    st = load_resume_state(state_path)
    if str(st.get("image_dir", "")).lower() != str(image_dir.resolve()).lower():
        return 0
    if str(st.get("save_dir", "")).lower() != str(save_dir.resolve()).lower():
        return 0
    last_path = str(st.get("last_image_path", "")).lower()
    if last_path:
        for i, p in enumerate(images):
            if str(p).lower() == last_path:
                return i
    try:
        return max(0, min(len(images) - 1, int(st.get("last_index", 0))))
    except Exception:
        return 0


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def fit_image(image: np.ndarray, max_panel_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_panel_size:
        return image.copy()
    scale = float(max_panel_size) / float(max(h, w))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)


def selected_copy_path(image_path: Path, save_dir: Path) -> Path:
    return save_dir / image_path.name


def compose_view(
    image: np.ndarray,
    image_path: Path,
    index: int,
    total: int,
    selected_exists: bool,
    saved_count: int,
    max_panel_size: int,
) -> np.ndarray:
    panel = fit_image(image, max_panel_size=max_panel_size)
    header_h = 82
    footer_h = 52
    canvas = np.zeros((panel.shape[0] + header_h + footer_h, panel.shape[1], 3), dtype=np.uint8)
    canvas[header_h : header_h + panel.shape[0], :, :] = panel

    status_text = "APPROVED NEGATIVE" if selected_exists else "NOT APPROVED"
    status_color = (0, 220, 255) if selected_exists else (160, 160, 160)
    cv2.putText(canvas, f"Image {index + 1}/{total}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Approved negatives: {saved_count}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(canvas, status_text, (max(12, canvas.shape[1] - 240), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, status_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, image_path.name, (12, header_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 220, 255), 1, cv2.LINE_AA)

    help_text = "D/Right: next | A/Left: prev | Shift+D/A: jump 25 | S: approve + next | X: unapprove | Q/Esc: quit"
    cv2.putText(canvas, help_text, (12, canvas.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = build_argparser().parse_args()
    image_dir = args.image_dir.resolve()
    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if image_dir == save_dir:
        raise RuntimeError("--image-dir and --save-dir must be different folders.")

    images = list_images(image_dir, save_dir)
    if not images:
        raise RuntimeError(f"No reviewable images found under {image_dir}")

    index = resolve_start_index(images, image_dir, save_dir, args.start, bool(args.resume_last))
    state_path = save_dir / RESUME_NAME

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(1400, int(args.max_panel_size) + 60), min(1100, int(args.max_panel_size) + 160))

    while True:
        image_path = images[index]
        image = read_color(image_path)
        dest_path = selected_copy_path(image_path, save_dir)
        selected_exists = dest_path.exists()
        saved_count = len([p for p in save_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        canvas = compose_view(
            image=image,
            image_path=image_path,
            index=index,
            total=len(images),
            selected_exists=selected_exists,
            saved_count=saved_count,
            max_panel_size=max(300, int(args.max_panel_size)),
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKeyEx(0)
        save_resume_state(state_path, image_dir, save_dir, index, image_path)

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("d"), 2555904):
            index = min(len(images) - 1, index + 1)
            continue
        if key in (ord("a"), 2424832):
            index = max(0, index - 1)
            continue
        if key == ord("D"):
            index = min(len(images) - 1, index + 25)
            continue
        if key == ord("A"):
            index = max(0, index - 25)
            continue
        if key in (ord("s"), ord("S")):
            shutil.copy2(image_path, dest_path)
            index = min(len(images) - 1, index + 1)
            continue
        if key in (ord("x"), ord("X")):
            if dest_path.exists():
                dest_path.unlink()
            continue
        if key == 2359296:  # Home
            index = 0
            continue
        if key == 2293760:  # End
            index = len(images) - 1
            continue

    save_resume_state(state_path, image_dir, save_dir, index, images[index] if images else None)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
