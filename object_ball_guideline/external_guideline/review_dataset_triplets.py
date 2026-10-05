from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "ZoomProbe Dataset Review"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class ReviewItem:
    image_path: Path
    mask_path: Path
    preview_path: Path
    sample_id: str
    group: str


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review matching original image, mask, and preview triplets side by side."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"C:\My_Project\SAM_3\guideline_line\data_zoomprobe"),
        help="Dataset root containing images/, masks/, and previews/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest with an images[] list. If provided, review only those items.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based starting index.",
    )
    parser.add_argument(
        "--max-panel-height",
        type=int,
        default=720,
        help="Maximum height used for each panel before concatenation.",
    )
    return parser


def resolve_path(path_like: Path | str) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _group_from_path(image_path: Path) -> str:
    parts = [part.lower() for part in image_path.parts]
    if "quarantine" in parts:
        return "quarantine"
    return "main"


def _triplet_from_image_path(image_path: Path) -> tuple[Path, Path, Path]:
    parent = image_path.parent
    if parent.name.lower() != "images":
        raise RuntimeError(f"Expected image path under an images/ directory: {image_path}")
    dataset_root = parent.parent
    mask_path = dataset_root / "masks" / image_path.name
    preview_path = dataset_root / "previews" / image_path.name
    return mask_path, preview_path, dataset_root


def load_manifest_items(manifest_path: Path) -> tuple[list[ReviewItem], list[str]]:
    path = resolve_path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    images_raw = data.get("images", [])
    rows_raw = data.get("rows", [])
    row_meta: dict[str, dict[str, str]] = {}
    if isinstance(rows_raw, list):
        for raw in rows_raw:
            if not isinstance(raw, dict):
                continue
            image_raw = str(raw.get("image_path", "")).strip()
            if not image_raw:
                continue
            image_path = str(resolve_path(image_raw)).lower()
            row_meta[image_path] = {
                "sample_id": str(raw.get("sample_id", "")).strip(),
            }

    items: list[ReviewItem] = []
    missing: list[str] = []
    for raw_path in images_raw if isinstance(images_raw, list) else []:
        image_path = resolve_path(str(raw_path).strip())
        if not image_path.exists():
            missing.append(f"{image_path.name}: missing image")
            continue
        try:
            mask_path, preview_path, _ = _triplet_from_image_path(image_path)
        except RuntimeError as exc:
            missing.append(str(exc))
            continue
        reasons = []
        if not mask_path.exists():
            reasons.append("mask")
        if not preview_path.exists():
            reasons.append("preview")
        if reasons:
            missing.append(f"{image_path.name}: missing {', '.join(reasons)}")
            continue
        meta = row_meta.get(str(image_path).lower(), {})
        items.append(
            ReviewItem(
                image_path=image_path,
                mask_path=mask_path,
                preview_path=preview_path,
                sample_id=meta.get("sample_id", image_path.stem) or image_path.stem,
                group=_group_from_path(image_path),
            )
        )
    return items, missing


def list_triplets(dataset_root: Path) -> tuple[list[ReviewItem], list[str]]:
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    previews_dir = dataset_root / "previews"

    for directory in (images_dir, masks_dir, previews_dir):
        if not directory.exists():
            raise FileNotFoundError(f"Missing dataset directory: {directory}")

    image_files = sorted([path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS])
    triplets: list[ReviewItem] = []
    missing: list[str] = []

    for image_path in image_files:
        mask_path = masks_dir / image_path.name
        preview_path = previews_dir / image_path.name
        if not mask_path.exists() or not preview_path.exists():
            reasons = []
            if not mask_path.exists():
                reasons.append("mask")
            if not preview_path.exists():
                reasons.append("preview")
            missing.append(f"{image_path.name}: missing {', '.join(reasons)}")
            continue
        triplets.append(
            ReviewItem(
                image_path=image_path,
                mask_path=mask_path,
                preview_path=preview_path,
                sample_id=image_path.stem,
                group="main",
            )
        )

    return triplets, missing


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def fit_panel(image: np.ndarray, max_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= max_height:
        return image.copy()
    scale = float(max_height) / float(max(h, 1))
    out_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (out_w, max_height), interpolation=cv2.INTER_AREA)


def draw_label(panel: np.ndarray, title: str, filename: str, detail_text: str, index_text: str) -> np.ndarray:
    band_h = 92
    canvas = np.zeros((panel.shape[0] + band_h, panel.shape[1], 3), dtype=np.uint8)
    canvas[band_h:, :, :] = panel
    cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, filename, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, detail_text, (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 235, 170), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        index_text,
        (max(12, canvas.shape[1] - 160), 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (120, 255, 120),
        2,
        cv2.LINE_AA,
    )
    return canvas


def compose_view(item: ReviewItem, index: int, total: int, max_panel_height: int) -> np.ndarray:
    panels = []
    items = [
        ("Original", item.image_path),
        ("Mask", item.mask_path),
        ("Preview", item.preview_path),
    ]
    index_text = f"{index + 1}/{total}"
    detail_text = f"id={item.sample_id} | group={item.group}"
    for title, path in items:
        panel = fit_panel(read_color(path), max_panel_height)
        panels.append(draw_label(panel, title, path.name, detail_text, index_text))

    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad_h = max_h - panel.shape[0]
            panel = cv2.copyMakeBorder(panel, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded.append(panel)

    gutter = np.full((max_h, 12, 3), 24, dtype=np.uint8)
    canvas = padded[0]
    for panel in padded[1:]:
        canvas = np.hstack([canvas, gutter, panel])

    footer_h = 38
    footer = np.zeros((footer_h, canvas.shape[1], 3), dtype=np.uint8)
    help_text = "A/Left: previous | D/Right: next | Shift+A/Shift+D: jump 25 | Home/End: ends | Q/Esc: quit"
    cv2.putText(footer, help_text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
    return np.vstack([canvas, footer])


def main() -> None:
    args = build_argparser().parse_args()
    if args.manifest is not None:
        triplets, missing = load_manifest_items(args.manifest)
    else:
        triplets, missing = list_triplets(args.dataset_root)
    if missing:
        print("Missing matches:")
        for line in missing[:20]:
            print(f"  {line}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    if not triplets:
        raise RuntimeError(f"No complete image/mask/preview triplets found under {args.dataset_root}")

    index = max(0, min(len(triplets) - 1, int(args.start)))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        canvas = compose_view(
            item=triplets[index],
            index=index,
            total=len(triplets),
            max_panel_height=max(200, int(args.max_panel_height)),
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKeyEx(0)

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("d"), 2555904):
            index = min(len(triplets) - 1, index + 1)
            continue
        if key in (ord("a"), 2424832):
            index = max(0, index - 1)
            continue
        if key == ord("D"):
            index = min(len(triplets) - 1, index + 25)
            continue
        if key == ord("A"):
            index = max(0, index - 25)
            continue
        if key == 2359296:  # Home
            index = 0
            continue
        if key == 2293760:  # End
            index = len(triplets) - 1
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
