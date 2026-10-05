from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _normalize_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    if os.name != "nt" and path_str.startswith("C:\\"):
        return Path("/mnt/c/" + path_str[3:].replace("\\", "/"))
    if os.name == "nt" and path_str.startswith("/mnt/c/"):
        return Path("C:/" + path_str[len("/mnt/c/"):])
    return Path(path_str)


def _load_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").resize(size)


def _overlay_mask(image: Image.Image, mask_path: Path | None, color: tuple[int, int, int]) -> Image.Image:
    if mask_path is None or not mask_path.exists():
        return image.copy()
    base = image.convert("RGBA")
    mask = Image.open(mask_path).convert("L").resize(image.size)
    overlay = Image.new("RGBA", image.size, color + (0,))
    overlay.putalpha(mask.point(lambda v: 120 if v > 0 else 0))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))


def _row_metric(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None and key == "final_iou":
        value = row.get("iou")
    if value is None and key == "final_dice":
        value = row.get("dice")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def make_contact_sheet(
    *,
    diagnostics_json: Path,
    output_path: Path,
    max_items: int = 50,
    sort_key: str = "final_iou",
    tile_size: tuple[int, int] = (320, 180),
) -> Path:
    payload = json.loads(diagnostics_json.read_text(encoding="utf-8"))
    rows = list(payload.get("rows", []))
    rows.sort(key=lambda row: _row_metric(row, sort_key))
    rows = rows[:max_items]
    cols = 2
    tile_w, tile_h = tile_size
    label_h = 36
    sheet = Image.new("RGB", (cols * tile_w, max(1, len(rows)) * (tile_h + label_h)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(rows):
        image_path = _normalize_path(row.get("winner", {}).get("source_path") or row.get("image_path"))
        if image_path is None or not image_path.exists():
            continue
        gt_path = _normalize_path(row.get("mask_path"))
        pred_path = _normalize_path(str(Path(row["output_dir"]) / "mask_final.png")) if row.get("output_dir") else None
        original = _load_rgb(image_path, tile_size)
        gt = _overlay_mask(original, gt_path, (0, 255, 0))
        pred = _overlay_mask(original, pred_path, (255, 64, 64))
        y = idx * (tile_h + label_h)
        sheet.paste(gt, (0, y))
        sheet.paste(pred, (tile_w, y))
        label = f"{row['id']} {row['failure_type']} iou={_row_metric(row, 'final_iou'):.3f}"
        _text(draw, (4, y + tile_h + 4), label[:90])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--sort-key", default="final_iou")
    args = parser.parse_args()
    print(
        make_contact_sheet(
            diagnostics_json=Path(args.diagnostics_json),
            output_path=Path(args.output),
            max_items=args.max_items,
            sort_key=args.sort_key,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
