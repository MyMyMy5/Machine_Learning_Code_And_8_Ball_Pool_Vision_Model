from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


WINDOW_NAME = "Extracted Frame Review"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
RESUME_NAME = "frame_review_resume.json"
CATEGORY_SAVE = "Save"
CATEGORY_REJECT = "Reject"
CATEGORY_NO_TABLE = "Table_doesnt_include"


@dataclass
class ReviewImage:
    path: Path
    rel_path: Path


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def resolve_path(path_like: Path | str) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review extracted frames and sort them into Save/Reject/Table_doesnt_include.")
    parser.add_argument("--image-dir", type=Path, default=Path(r"C:\My_Project\SAM_3\guideline_line\video_frames"))
    parser.add_argument("--output-root", type=Path, default=Path(r"C:\My_Project\SAM_3\guideline_line\video_frames_review"))
    parser.add_argument("--recursive", type=str2bool, default=True)
    parser.add_argument("--resume-last", type=str2bool, default=True)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--max-panel-size", type=int, default=1200)
    return parser


def list_images(image_dir: Path, output_root: Path, recursive: bool) -> List[ReviewImage]:
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    globber = image_dir.rglob if recursive else image_dir.glob
    out: List[ReviewImage] = []
    output_root_resolved = output_root.resolve()
    for path in sorted(globber("*"), key=lambda p: str(p).lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if output_root_resolved in path.resolve().parents:
            continue
        out.append(ReviewImage(path=path.resolve(), rel_path=path.resolve().relative_to(image_dir.resolve())))
    return out


def load_resume_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_resume_state(path: Path, image_dir: Path, output_root: Path, index: int, image_path: Optional[Path]) -> None:
    payload = {
        "image_dir": str(image_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "last_index": int(index),
        "last_image_path": str(image_path.resolve()) if image_path is not None else "",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_start_index(images: List[ReviewImage], image_dir: Path, output_root: Path, start_arg: Optional[int], resume_last: bool) -> int:
    if not images:
        return 0
    if start_arg is not None:
        return max(0, min(len(images) - 1, int(start_arg)))
    if not resume_last:
        return 0
    state_path = output_root / RESUME_NAME
    st = load_resume_state(state_path)
    if str(st.get("image_dir", "")).lower() != str(image_dir.resolve()).lower():
        return 0
    if str(st.get("output_root", "")).lower() != str(output_root.resolve()).lower():
        return 0
    last_path = str(st.get("last_image_path", "")).lower()
    if last_path:
        for i, item in enumerate(images):
            if str(item.path).lower() == last_path:
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


def category_path(output_root: Path, category: str, rel_path: Path) -> Path:
    return output_root / category / rel_path


def count_category_images(output_root: Path, category: str) -> int:
    root = output_root / category
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def compose_view(
    image: np.ndarray,
    item: ReviewImage,
    index: int,
    total: int,
    counts: Dict[str, int],
    current_category: str,
    max_panel_size: int,
) -> np.ndarray:
    panel = fit_image(image, max_panel_size=max_panel_size)
    header_h = 108
    footer_h = 60
    canvas = np.zeros((panel.shape[0] + header_h + footer_h, panel.shape[1], 3), dtype=np.uint8)
    canvas[header_h : header_h + panel.shape[0], :, :] = panel

    cv2.putText(canvas, f"Frame {index + 1}/{total}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Save={counts[CATEGORY_SAVE]} | Reject={counts[CATEGORY_REJECT]} | NoTable={counts[CATEGORY_NO_TABLE]}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Current={current_category}", (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 220, 140), 2, cv2.LINE_AA)
    cv2.putText(canvas, item.rel_path.as_posix(), (12, header_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 220, 255), 1, cv2.LINE_AA)

    help_text = "S: Save | R: Reject | D: Table_doesnt_include | Right/Space/N: skip | Left: prev | Shift+Right/Left: jump 25 | Q/Esc: quit"
    cv2.putText(canvas, help_text, (12, canvas.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def move_to_category(item: ReviewImage, image_dir: Path, output_root: Path, category: str) -> Path:
    dest = category_path(output_root, category, item.rel_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(item.path), str(dest))
    return dest


def detect_existing_category(item: ReviewImage, output_root: Path) -> str:
    for category in (CATEGORY_SAVE, CATEGORY_REJECT, CATEGORY_NO_TABLE):
        if category_path(output_root, category, item.rel_path).exists():
            return category
    return "UNSORTED"


def main() -> None:
    args = build_argparser().parse_args()
    image_dir = args.image_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for category in (CATEGORY_SAVE, CATEGORY_REJECT, CATEGORY_NO_TABLE):
        (output_root / category).mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir, output_root, recursive=bool(args.recursive))
    if not images:
        raise RuntimeError(f"No reviewable images found under {image_dir}")

    index = resolve_start_index(images, image_dir, output_root, args.start, bool(args.resume_last))
    state_path = output_root / RESUME_NAME

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(1500, int(args.max_panel_size) + 80), min(1100, int(args.max_panel_size) + 180))

    while True:
        item = images[index]
        image = read_color(item.path)
        counts = {
            CATEGORY_SAVE: count_category_images(output_root, CATEGORY_SAVE),
            CATEGORY_REJECT: count_category_images(output_root, CATEGORY_REJECT),
            CATEGORY_NO_TABLE: count_category_images(output_root, CATEGORY_NO_TABLE),
        }
        current_category = detect_existing_category(item, output_root)
        canvas = compose_view(
            image=image,
            item=item,
            index=index,
            total=len(images),
            counts=counts,
            current_category=current_category,
            max_panel_size=max(300, int(args.max_panel_size)),
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKeyEx(0)
        save_resume_state(state_path, image_dir, output_root, index, item.path)

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (2555904, ord("n"), ord("N"), 32):  # Right arrow, N, Space
            index = min(len(images) - 1, index + 1)
            continue
        if key in (2424832,):  # Left arrow
            index = max(0, index - 1)
            continue
        if key == ord("D"):
            move_to_category(item, image_dir, output_root, CATEGORY_NO_TABLE)
            images.pop(index)
            if not images:
                break
            index = min(index, len(images) - 1)
            continue
        if key == ord("A"):
            index = max(0, index - 25)
            continue
        if key == ord("S"):
            move_to_category(item, image_dir, output_root, CATEGORY_SAVE)
            images.pop(index)
            if not images:
                break
            index = min(index, len(images) - 1)
            continue
        if key in (ord("r"), ord("R")):
            move_to_category(item, image_dir, output_root, CATEGORY_REJECT)
            images.pop(index)
            if not images:
                break
            index = min(index, len(images) - 1)
            continue
        if key == 2621440:  # Shift+Right
            index = min(len(images) - 1, index + 25)
            continue
        if key == 2490368:  # Shift+Left
            index = max(0, index - 25)
            continue
        if key == 2359296:  # Home
            index = 0
            continue
        if key == 2293760:  # End
            index = len(images) - 1
            continue

    save_resume_state(state_path, image_dir, output_root, min(index, len(images) - 1) if images else 0, images[index].path if images else None)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
