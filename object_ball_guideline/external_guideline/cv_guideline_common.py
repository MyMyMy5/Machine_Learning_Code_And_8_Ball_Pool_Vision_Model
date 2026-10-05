from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mask_refinement import component_masks, dark_shell_ratio, principal_axis_stats
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

try:
    from transformers import Mask2FormerForUniversalSegmentation
except Exception:
    Mask2FormerForUniversalSegmentation = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_SEGFORMER_B2 = "nvidia/segformer-b2-finetuned-ade-512-512"
DEFAULT_SEGFORMER_B3 = "nvidia/segformer-b3-finetuned-ade-512-512"
DEFAULT_MASK2FORMER = "facebook/mask2former-swin-small-cityscapes-semantic"


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def parse_csv_floats(raw: str, default: Sequence[float]) -> List[float]:
    s = str(raw or "").strip()
    if not s:
        return [float(x) for x in default]
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out or [float(x) for x in default]


def parse_csv_ints(raw: str, default: Sequence[int]) -> List[int]:
    vals = parse_csv_floats(raw, [float(x) for x in default])
    return [int(round(v)) for v in vals]


def parse_crop_mode_mix(raw: str) -> Dict[str, float]:
    text = str(raw or "").strip()
    if not text:
        text = "tight:0.30,context:0.45,fullframe:0.25"
    weights: Dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid crop mode entry: {part}")
        name, value = part.split(":", 1)
        key = name.strip().lower()
        weights[key] = max(0.0, float(value))
    total = float(sum(weights.values()))
    if total <= 0.0:
        return {"tight": 0.30, "context": 0.45, "fullframe": 0.25}
    return {k: float(v / total) for k, v in weights.items()}


def utc_now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    text = path.read_text(encoding="utf-8-sig")
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def resolve_path(p: str, cwd: Optional[Path] = None) -> Path:
    base = cwd or Path.cwd()
    path = Path(str(p)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def list_images(image_dir: Path, recursive: bool = False) -> List[Path]:
    if not image_dir.exists():
        return []
    globber = image_dir.rglob if recursive else image_dir.glob
    out: List[Path] = []
    for p in globber("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p.resolve())
    return sorted(out)


@dataclass
class AnnotationRecord:
    sample_id: str
    image_path: Path
    mask_path: Path
    prompt: str
    dataset_split: str  # main, quarantine, legacy
    save_mode: str  # normal, force, legacy
    source: str
    sample_weight: float
    mask_area: int
    mask_bbox_w: int
    mask_bbox_h: int
    mask_major_len: float
    refined: bool


@dataclass
class NegativeRecord:
    sample_id: str
    image_path: Path
    source: str


@dataclass
class GuidelinePostprocessContext:
    gray: np.ndarray
    white_like: np.ndarray


@dataclass
class CenterlineReference:
    skeleton: np.ndarray
    dilated: np.ndarray
    count: int
    tol_px: int


def classify_dataset_split(raw: Dict) -> str:
    split = str(raw.get("dataset_split", "")).strip().lower()
    if split == "main":
        return "main"
    if split == "quarantine":
        return "quarantine"
    return "legacy"


def infer_sample_weight(raw: Dict, dataset_split: str) -> float:
    save_mode = str(raw.get("save_mode", "")).strip().lower()
    if dataset_split == "main" and save_mode == "normal":
        return 1.0
    if dataset_split == "quarantine" and save_mode == "force":
        return 0.8
    if dataset_split == "legacy":
        return 0.85
    if dataset_split == "quarantine":
        return 0.8
    return 0.9


def compute_mask_geometry(mask_u8: np.ndarray) -> Dict[str, float]:
    ys, xs = np.where(mask_u8 > 0)
    area = int(len(xs))
    if area <= 0:
        return {
            "area": 0,
            "bbox_w": 0,
            "bbox_h": 0,
            "major_len": 0.0,
        }

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    bbox_w = int(x2 - x1 + 1)
    bbox_h = int(y2 - y1 + 1)

    if area <= 1:
        major_len = float(max(bbox_w, bbox_h))
    else:
        coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        coords_centered = coords - coords.mean(axis=0, keepdims=True)
        cov = np.cov(coords_centered, rowvar=False)
        vals, _ = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0.0)
        major_len = float(math.sqrt(float(vals.max())) * 4.0) if vals.size else float(max(bbox_w, bbox_h))

    return {
        "area": area,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "major_len": major_len,
    }


def load_annotation_records(
    annotations_path: Path,
    include_main: bool = True,
    include_quarantine: bool = True,
    include_legacy: bool = True,
) -> Tuple[List[AnnotationRecord], Dict[str, int]]:
    rows = read_jsonl(annotations_path)
    out: List[AnnotationRecord] = []
    stats = {
        "rows_total": 0,
        "rows_kept": 0,
        "rows_skipped_missing": 0,
        "rows_skipped_split": 0,
        "rows_skipped_badmask": 0,
        "split_main": 0,
        "split_quarantine": 0,
        "split_legacy": 0,
    }
    for raw in rows:
        stats["rows_total"] += 1
        sample_id = str(raw.get("id", "")).strip()
        image_path = resolve_path(str(raw.get("image_path", "")).strip())
        mask_path = resolve_path(str(raw.get("mask_path", "")).strip())
        prompt = str(raw.get("prompt", "white aiming line"))
        dataset_split = classify_dataset_split(raw)
        if dataset_split == "main" and not include_main:
            stats["rows_skipped_split"] += 1
            continue
        if dataset_split == "quarantine" and not include_quarantine:
            stats["rows_skipped_split"] += 1
            continue
        if dataset_split == "legacy" and not include_legacy:
            stats["rows_skipped_split"] += 1
            continue
        if not sample_id or not image_path.exists() or not mask_path.exists():
            stats["rows_skipped_missing"] += 1
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or int((mask > 127).sum()) <= 0:
            stats["rows_skipped_badmask"] += 1
            continue

        save_mode = str(raw.get("save_mode", "legacy")).strip().lower() or "legacy"
        source = str(raw.get("candidate_source", raw.get("inference_source", "legacy"))).strip() or "legacy"
        weight = float(infer_sample_weight(raw, dataset_split))
        geom = compute_mask_geometry((mask > 127).astype(np.uint8))
        refined = bool(raw.get("refined", False)) or source == "polygon_refine"

        out.append(
            AnnotationRecord(
                sample_id=sample_id,
                image_path=image_path,
                mask_path=mask_path,
                prompt=prompt,
                dataset_split=dataset_split,
                save_mode=save_mode,
                source=source,
                sample_weight=weight,
                mask_area=int(geom["area"]),
                mask_bbox_w=int(geom["bbox_w"]),
                mask_bbox_h=int(geom["bbox_h"]),
                mask_major_len=float(geom["major_len"]),
                refined=refined,
            )
        )
        stats["rows_kept"] += 1
        if dataset_split == "main":
            stats["split_main"] += 1
        elif dataset_split == "quarantine":
            stats["split_quarantine"] += 1
        else:
            stats["split_legacy"] += 1
    return out, stats


def load_split_ids(splits_path: Path) -> Tuple[List[str], List[str]]:
    if not splits_path.exists():
        return [], []
    try:
        data = json.loads(splits_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse splits file {splits_path}: {exc}") from exc
    train_ids = [str(x) for x in data.get("train", [])]
    val_ids = [str(x) for x in data.get("val", [])]
    return train_ids, val_ids


def split_by_ids(records: Sequence[AnnotationRecord], train_ids: Sequence[str], val_ids: Sequence[str]) -> Tuple[List[AnnotationRecord], List[AnnotationRecord]]:
    train_set = set(str(x) for x in train_ids)
    val_set = set(str(x) for x in val_ids)
    train: List[AnnotationRecord] = []
    val: List[AnnotationRecord] = []
    for rec in records:
        sid = rec.sample_id
        if sid in train_set:
            train.append(rec)
        elif sid in val_set:
            val.append(rec)
    return train, val


def random_split_ids(ids: Sequence[str], val_ratio: float = 0.15, seed: int = 42) -> Tuple[List[str], List[str]]:
    ids_list = [str(x) for x in ids]
    rng = random.Random(int(seed))
    rng.shuffle(ids_list)
    n_val = max(1, int(round(len(ids_list) * float(val_ratio)))) if len(ids_list) >= 2 else 0
    val_ids = ids_list[:n_val]
    train_ids = ids_list[n_val:]
    return train_ids, val_ids


def stratified_annotation_split(
    records: Sequence[AnnotationRecord],
    val_ratio: float = 0.15,
    seed: int = 42,
    size_thresholds: Sequence[float] = (12.0, 20.0, 40.0),
    stratify_by_source: bool = True,
) -> Tuple[List[str], List[str]]:
    thresholds = sorted(float(x) for x in size_thresholds)

    def size_bucket(major_len: float) -> int:
        for idx, threshold in enumerate(thresholds):
            if float(major_len) < threshold:
                return idx
        return len(thresholds)

    buckets: Dict[Tuple[Any, ...], List[str]] = {}
    for rec in records:
        key: List[Any] = [size_bucket(rec.mask_major_len)]
        if bool(stratify_by_source):
            key.append(str(rec.source))
        buckets.setdefault(tuple(key), []).append(str(rec.sample_id))

    rng = random.Random(int(seed))
    train_ids: List[str] = []
    val_ids: List[str] = []
    for _, bucket_ids in sorted(buckets.items(), key=lambda item: item[0]):
        bucket = list(bucket_ids)
        rng.shuffle(bucket)
        if len(bucket) <= 1:
            train_ids.extend(bucket)
            continue
        n_val = int(round(len(bucket) * float(val_ratio)))
        if float(val_ratio) > 0.0 and n_val <= 0:
            n_val = 1
        n_val = min(max(0, n_val), len(bucket) - 1)
        val_ids.extend(bucket[:n_val])
        train_ids.extend(bucket[n_val:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


def build_negative_records(image_dir: Optional[Path], prefix: str = "NEG") -> List[NegativeRecord]:
    if image_dir is None:
        return []
    imgs = list_images(image_dir, recursive=False)
    out: List[NegativeRecord] = []
    for i, p in enumerate(imgs):
        out.append(
            NegativeRecord(
                sample_id=f"{prefix}_{i:06d}_{p.stem}",
                image_path=p,
                source=prefix,
            )
        )
    return out


def split_negatives(records: Sequence[NegativeRecord], val_ratio: float = 0.15, seed: int = 42) -> Tuple[List[NegativeRecord], List[NegativeRecord]]:
    if not records:
        return [], []
    idxs = list(range(len(records)))
    rng = random.Random(int(seed))
    rng.shuffle(idxs)
    n_val = max(1, int(round(len(idxs) * float(val_ratio)))) if len(idxs) >= 2 else 0
    val_idx = set(idxs[:n_val])
    train: List[NegativeRecord] = []
    val: List[NegativeRecord] = []
    for i, rec in enumerate(records):
        if i in val_idx:
            val.append(rec)
        else:
            train.append(rec)
    return train, val


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_mask_u8(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (m > 127).astype(np.uint8)


def mask_bbox(mask_u8: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask_u8 > 0)
    if xs.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return x1, y1, x2, y2


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def crop_pair(img: np.ndarray, mask: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def resize_pair(img: np.ndarray, mask: np.ndarray, out_size: int) -> Tuple[np.ndarray, np.ndarray]:
    img_r = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return img_r, mask_r


def component_elongation(mask_u8: np.ndarray) -> float:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8.astype(np.uint8), connectivity=8)
    best = 1.0
    for lab in range(1, int(num_labels)):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 5:
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        short = max(1.0, float(min(w, h)))
        long_side = float(max(w, h))
        best = max(best, long_side / short)
    return float(best)


def white_line_distractor_score(img_bgr: np.ndarray) -> float:
    if img_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    white = ((sat <= 85) & (val >= 155)).astype(np.uint8)
    white_px = int(white.sum())
    if white_px <= 0:
        return 0.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    outer = cv2.dilate(white, kernel, iterations=1)
    ring = np.clip(outer - white, 0, 1).astype(np.uint8)
    dark = (gray <= 110).astype(np.uint8)
    dark_ring = int(np.logical_and(ring > 0, dark > 0).sum())
    edge = (cv2.Canny(gray, 60, 140) > 0).astype(np.uint8)
    edge_near = int(np.logical_and(cv2.dilate(white, kernel, iterations=1) > 0, edge > 0).sum())
    elong = component_elongation(white)
    area = float(img_bgr.shape[0] * img_bgr.shape[1])
    white_ratio = float(white_px) / max(1.0, area)
    return float((white_px * 1.0) + (dark_ring * 1.8) + (edge_near * 1.2) + (white_ratio * 200.0) + (elong * 25.0))


def random_positive_crop(
    img: np.ndarray,
    mask_u8: np.ndarray,
    out_size: int,
    rng: random.Random,
    crop_mode_mix: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    bbox = mask_bbox(mask_u8)
    if bbox is None:
        img_r = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        return img_r, np.zeros((out_size, out_size), dtype=np.uint8)

    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    span = float(max(bw, bh))
    mix = crop_mode_mix or {"tight": 0.30, "context": 0.45, "fullframe": 0.25}
    r = rng.random()
    cumulative = 0.0
    mode_name = "context"
    for key in ("tight", "context", "fullframe"):
        cumulative += float(mix.get(key, 0.0))
        if r <= cumulative:
            mode_name = key
            break

    if mode_name == "fullframe":
        return resize_pair(img, mask_u8, out_size)

    if mode_name == "tight":
        target_frac = rng.uniform(0.14, 0.22)
        min_side = max(80, int(round(span * 2.0)))
        jitter_scale = 0.28
        min_context_factor = 1.2
    else:
        target_frac = rng.uniform(0.05, 0.11)
        min_side = max(180, int(round(span * 2.6)))
        jitter_scale = 0.70
        min_context_factor = 1.8

    crop_side = int(round(span / max(target_frac, 1e-3)))
    crop_side = max(crop_side, int(round(span * min_context_factor)), min_side)
    crop_side = min(crop_side, max(h, w))

    jitter = max(jitter_scale * span, 10.0)
    cx = (x1 + x2) * 0.5 + rng.uniform(-jitter, jitter)
    cy = (y1 + y2) * 0.5 + rng.uniform(-jitter, jitter)

    x1c = int(round(cx - crop_side * 0.5))
    y1c = int(round(cy - crop_side * 0.5))
    x2c = x1c + crop_side
    y2c = y1c + crop_side

    if x1c < 0:
        x2c -= x1c
        x1c = 0
    if y1c < 0:
        y2c -= y1c
        y1c = 0
    if x2c > w:
        shift = x2c - w
        x1c -= shift
        x2c = w
    if y2c > h:
        shift = y2c - h
        y1c -= shift
        y2c = h
    x1c, y1c, x2c, y2c = clamp_box(x1c, y1c, x2c, y2c, w, h)
    cimg, cmask = crop_pair(img, mask_u8, (x1c, y1c, x2c, y2c))
    return resize_pair(cimg, cmask, out_size)


def random_background_crop_from_positive(
    img: np.ndarray,
    mask_u8: np.ndarray,
    out_size: int,
    rng: random.Random,
    tries: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    safe = cv2.dilate(mask_u8.astype(np.uint8), np.ones((19, 19), dtype=np.uint8), iterations=1)
    bbox = mask_bbox(mask_u8)
    min_side = max(32, int(0.12 * min(h, w)))
    max_side = max(min_side + 1, int(0.45 * min(h, w)))

    if bbox is not None:
        x1b, y1b, x2b, y2b = bbox
        bw = max(1, x2b - x1b + 1)
        bh = max(1, y2b - y1b + 1)
        span = max(bw, bh)
        cx = 0.5 * (x1b + x2b)
        cy = 0.5 * (y1b + y2b)
        near_half = max(int(round(span * 3.5)), 48)

        # Prefer hard negatives near the target and bias toward nearby white line-like distractors.
        candidates = []
        for _ in range(max(24, int(tries) * 2)):
            side = rng.randint(min_side, max_side)
            side = min(side, h, w)
            if side <= 1:
                break
            jitter_x = rng.uniform(-near_half, near_half)
            jitter_y = rng.uniform(-near_half, near_half)
            x1 = int(round(cx + jitter_x - side * 0.5))
            y1 = int(round(cy + jitter_y - side * 0.5))
            x1 = max(0, min(x1, w - side))
            y1 = max(0, min(y1, h - side))
            x2 = x1 + side
            y2 = y1 + side
            patch_safe = safe[y1:y2, x1:x2]
            if int(patch_safe.sum()) != 0:
                continue
            crop_cx = 0.5 * (x1 + x2)
            crop_cy = 0.5 * (y1 + y2)
            if abs(crop_cx - cx) > near_half and abs(crop_cy - cy) > near_half:
                continue
            crop = img[y1:y2, x1:x2]
            score = white_line_distractor_score(crop)
            candidates.append((float(score), crop))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            crop = candidates[0][1]
            out_img = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            out_mask = np.zeros((out_size, out_size), dtype=np.uint8)
            return out_img, out_mask

    global_candidates = []
    for _ in range(max(16, int(tries))):
        side = rng.randint(min_side, max_side)
        if side >= w or side >= h:
            continue
        x1 = rng.randint(0, max(0, w - side))
        y1 = rng.randint(0, max(0, h - side))
        x2 = x1 + side
        y2 = y1 + side
        patch_safe = safe[y1:y2, x1:x2]
        if int(patch_safe.sum()) == 0:
            crop = img[y1:y2, x1:x2]
            score = white_line_distractor_score(crop)
            global_candidates.append((float(score), crop))
    if global_candidates:
        global_candidates.sort(key=lambda x: x[0], reverse=True)
        crop = global_candidates[0][1]
        out_img = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        out_mask = np.zeros((out_size, out_size), dtype=np.uint8)
        return out_img, out_mask

    inv = (safe == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    cx, cy = int(max_loc[0]), int(max_loc[1])
    side = min(max_side, max(min_side, int(0.22 * min(h, w))))
    x1 = max(0, min(cx - side // 2, w - side))
    y1 = max(0, min(cy - side // 2, h - side))
    x2 = x1 + side
    y2 = y1 + side
    crop = img[y1:y2, x1:x2]
    out_img = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    out_mask = np.zeros((out_size, out_size), dtype=np.uint8)
    return out_img, out_mask


def random_crop_negative_image(img: np.ndarray, out_size: int, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    min_side = max(32, int(0.18 * min(h, w)))
    max_side = max(min_side + 1, int(0.7 * min(h, w)))
    side = rng.randint(min_side, max_side) if max_side > min_side else min_side
    side = min(side, h, w)
    if side <= 1:
        side = min(h, w)
    x1 = rng.randint(0, max(0, w - side)) if w > side else 0
    y1 = rng.randint(0, max(0, h - side)) if h > side else 0
    crop = img[y1 : y1 + side, x1 : x1 + side]
    out_img = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    out_mask = np.zeros((out_size, out_size), dtype=np.uint8)
    return out_img, out_mask


def apply_augment_pair(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    img = img_bgr.copy()
    mask = mask_u8.copy()

    if rng.random() < 0.5:
        alpha = rng.uniform(0.86, 1.16)
        beta = rng.uniform(-14.0, 14.0)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if rng.random() < 0.20:
        sigma = rng.uniform(0.20, 0.90)
        noise = rng.normalvariate(0.0, sigma * 255.0)
        n = np.random.normal(0.0, abs(noise), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

    if rng.random() < 0.25:
        k = 3 if rng.random() < 0.8 else 5
        img = cv2.GaussianBlur(img, (k, k), sigmaX=0.0)

    if rng.random() < 0.25:
        h, w = img.shape[:2]
        angle = rng.uniform(-6.0, 6.0)
        tx = rng.uniform(-0.04, 0.04) * w
        ty = rng.uniform(-0.04, 0.04) * h
        sc = rng.uniform(0.92, 1.08)
        m = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, sc)
        m[:, 2] += [tx, ty]
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    return img, (mask > 0).astype(np.uint8)


def bgr_to_normalized_tensor(
    img_bgr: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> torch.Tensor:
    mean_arr = IMAGENET_MEAN if mean is None else np.asarray(mean, dtype=np.float32)
    std_arr = IMAGENET_STD if std is None else np.asarray(std, dtype=np.float32)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - mean_arr) / std_arr
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).float()


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBNAct(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class GuidelineSegNet(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.backbone = backbone
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=3,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        ch = list(self.encoder.feature_info.channels())
        self.center = ConvBNAct(ch[4], 256)
        self.dec4 = DecoderBlock(256, ch[3], 192)
        self.dec3 = DecoderBlock(192, ch[2], 128)
        self.dec2 = DecoderBlock(128, ch[1], 96)
        self.dec1 = DecoderBlock(96, ch[0], 64)
        self.refine = ConvBNAct(64, 48)
        self.head = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        x0, x1, x2, x3, x4 = feats
        y = self.center(x4)
        y = self.dec4(y, x3)
        y = self.dec3(y, x2)
        y = self.dec2(y, x1)
        y = self.dec1(y, x0)
        y = F.interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=False)
        y = self.refine(y)
        return self.head(y)


def _load_processor_stats(model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    proc = AutoImageProcessor.from_pretrained(model_name)
    mean = np.asarray(getattr(proc, "image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(getattr(proc, "image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    if mean.ndim == 0:
        mean = np.repeat(mean[None], 3)
    if std.ndim == 0:
        std = np.repeat(std[None], 3)
    return mean, std


class SegmentationModel(nn.Module):
    def __init__(
        self,
        model_family: str = "custom",
        backbone: str = "resnet34",
        model_name: str = "",
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_family = str(model_family).strip().lower() or "custom"
        self.backbone = str(backbone).strip() or "resnet34"
        self.model_name = str(model_name).strip()
        if self.model_family == "custom":
            self.net = GuidelineSegNet(backbone=self.backbone, pretrained=pretrained)
            self.input_mean = IMAGENET_MEAN.copy()
            self.input_std = IMAGENET_STD.copy()
        elif self.model_family == "segformer":
            if not self.model_name:
                raise ValueError("SegFormer model_name is required for model_family=segformer")
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                self.model_name,
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
            self.input_mean, self.input_std = _load_processor_stats(self.model_name)
        elif self.model_family == "mask2former":
            if Mask2FormerForUniversalSegmentation is None:
                raise RuntimeError("Mask2Former support is unavailable in this transformers install.")
            raise NotImplementedError("Mask2Former training path is deferred until the SegFormer phase is validated.")
        else:
            raise ValueError(f"Unsupported model_family: {self.model_family}")

    def forward_outputs(self, x: torch.Tensor, output_size: Optional[Tuple[int, int]] = None) -> Dict[str, torch.Tensor]:
        if self.model_family == "custom":
            binary_logits = self.net(x)
            if output_size is not None and tuple(binary_logits.shape[-2:]) != tuple(output_size):
                binary_logits = F.interpolate(binary_logits, size=output_size, mode="bilinear", align_corners=False)
            positive_prob = torch.sigmoid(binary_logits)
            return {
                "binary_logits": binary_logits,
                "class_logits": None,
                "positive_prob": positive_prob,
            }

        if self.model_family == "segformer":
            out = self.net(pixel_values=x, return_dict=True)
            class_logits = out.logits
            if output_size is not None and tuple(class_logits.shape[-2:]) != tuple(output_size):
                class_logits = F.interpolate(class_logits, size=output_size, mode="bilinear", align_corners=False)
            positive_prob = torch.softmax(class_logits, dim=1)[:, 1:2]
            binary_logits = class_logits[:, 1:2] - class_logits[:, 0:1]
            return {
                "binary_logits": binary_logits,
                "class_logits": class_logits,
                "positive_prob": positive_prob,
            }

        raise ValueError(f"Unsupported model_family: {self.model_family}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_outputs(x)["binary_logits"]

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "model_family": self.model_family,
            "backbone": self.backbone,
            "model_name": self.model_name,
            "checkpoint_format_version": 2,
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
        }


def build_segmentation_model(
    model_family: str = "custom",
    backbone: str = "resnet34",
    model_name: str = "",
    pretrained: bool = True,
) -> SegmentationModel:
    family = str(model_family).strip().lower() or "custom"
    if family == "segformer" and not str(model_name).strip():
        model_name = DEFAULT_SEGFORMER_B3
    return SegmentationModel(model_family=family, backbone=backbone, model_name=model_name, pretrained=pretrained)


def checkpoint_model_config(ckpt: Dict[str, Any]) -> Dict[str, str]:
    family = str(ckpt.get("model_family", "custom")).strip().lower() or "custom"
    if family == "custom":
        return {
            "model_family": "custom",
            "backbone": str(ckpt.get("backbone", "resnet34")),
            "model_name": "",
        }
    return {
        "model_family": family,
        "backbone": str(ckpt.get("backbone", "resnet34")),
        "model_name": str(ckpt.get("model_name", "")),
    }


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> Tuple[SegmentationModel, Dict]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    cfg = checkpoint_model_config(ckpt)
    model = build_segmentation_model(
        model_family=cfg["model_family"],
        backbone=cfg["backbone"],
        model_name=cfg["model_name"],
        pretrained=False,
    )
    state = ckpt.get("model_state", {})
    fmt = int(ckpt.get("checkpoint_format_version", 1))
    if fmt <= 1 and cfg["model_family"] == "custom":
        load_result = model.net.load_state_dict(state, strict=True)
    else:
        load_result = model.load_state_dict(state, strict=True)
    if hasattr(load_result, "missing_keys") and (load_result.missing_keys or load_result.unexpected_keys):
        raise RuntimeError(
            f"Checkpoint load mismatch for {checkpoint_path}: "
            f"missing={list(load_result.missing_keys)} unexpected={list(load_result.unexpected_keys)}"
        )
    model.to(device)
    model.eval()
    return model, ckpt


def enumerate_tile_windows(
    frame_shape: Tuple[int, int],
    tile_size: int,
    tile_overlap: int,
) -> List[Tuple[int, int, int, int]]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    tile_size = max(64, int(tile_size))
    tile_overlap = max(0, min(int(tile_overlap), tile_size - 8))
    stride = tile_size - tile_overlap
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if not ys or ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))
    if not xs or xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))
    return [(int(y), int(x), int(min(tile_size, h - y)), int(min(tile_size, w - x))) for y in ys for x in xs]


def extract_padded_tile(
    frame_bgr: np.ndarray,
    y: int,
    x: int,
    tile_size: int,
) -> Tuple[np.ndarray, int, int]:
    tile_size_i = max(64, int(tile_size))
    patch = frame_bgr[int(y) : int(y) + tile_size_i, int(x) : int(x) + tile_size_i]
    ph, pw = patch.shape[:2]
    if ph <= 0 or pw <= 0:
        raise ValueError(f"Invalid tile window at ({y}, {x}) for frame shape={frame_bgr.shape[:2]}")
    if ph != tile_size_i or pw != tile_size_i:
        pad_b = tile_size_i - ph
        pad_r = tile_size_i - pw
        patch = cv2.copyMakeBorder(patch, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT_101)
    return patch, int(ph), int(pw)


def enumerate_tta_variants(
    frame_bgr: np.ndarray,
    tta_scales: Sequence[float],
    tta_hflip: bool,
) -> List[Dict[str, Any]]:
    h, w = frame_bgr.shape[:2]
    variants: List[Dict[str, Any]] = []
    variant_id = 0
    for scale in tta_scales or [1.0]:
        scale_f = max(0.5, float(scale))
        if abs(scale_f - 1.0) < 1e-6:
            scaled = frame_bgr
        else:
            sw = max(32, int(round(w * scale_f)))
            sh = max(32, int(round(h * scale_f)))
            scaled = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
        variants.append(
            {
                "variant_id": int(variant_id),
                "frame_bgr": scaled,
                "flip_variant": False,
                "scale_factor": float(scale_f),
                "original_hw": (int(h), int(w)),
                "scaled_hw": (int(scaled.shape[0]), int(scaled.shape[1])),
            }
        )
        variant_id += 1
        if bool(tta_hflip):
            flipped = cv2.flip(scaled, 1)
            variants.append(
                {
                    "variant_id": int(variant_id),
                    "frame_bgr": flipped,
                    "flip_variant": True,
                    "scale_factor": float(scale_f),
                    "original_hw": (int(h), int(w)),
                    "scaled_hw": (int(flipped.shape[0]), int(flipped.shape[1])),
                }
            )
            variant_id += 1
    return variants


@torch.no_grad()
def run_prob_batch(
    model: nn.Module,
    patches_bgr: Sequence[np.ndarray],
    device: torch.device,
    use_amp: bool = True,
) -> np.ndarray:
    if not patches_bgr:
        return np.zeros((0, 0, 0), dtype=np.float32)
    mean = getattr(model, "input_mean", IMAGENET_MEAN)
    std = getattr(model, "input_std", IMAGENET_STD)
    batch = torch.stack(
        [bgr_to_normalized_tensor(patch, mean=mean, std=std) for patch in patches_bgr],
        dim=0,
    ).to(device)
    tile_hw = tuple(int(x) for x in patches_bgr[0].shape[:2])
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_amp and device.type == "cuda") else nullcontext()
    with amp_ctx:
        if hasattr(model, "forward_outputs"):
            out = model.forward_outputs(batch, output_size=tile_hw)
            prob_t = out["positive_prob"]
        else:
            prob_t = torch.sigmoid(model(batch))
    return prob_t[:, 0].detach().float().cpu().numpy()


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def predict_logits_tiled(
    model: nn.Module,
    frame_bgr: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    tile_overlap: int = 128,
    use_amp: bool = True,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    tile_size = max(64, int(tile_size))
    tile_overlap = max(0, min(int(tile_overlap), tile_size - 8))
    stride = tile_size - tile_overlap
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if not ys or ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))
    if not xs or xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))

    logits_sum = np.zeros((h, w), dtype=np.float32)
    logits_cnt = np.zeros((h, w), dtype=np.float32)

    for y in ys:
        for x in xs:
            patch = frame_bgr[y : y + tile_size, x : x + tile_size]
            ph, pw = patch.shape[:2]
            if ph <= 0 or pw <= 0:
                continue
            if ph != tile_size or pw != tile_size:
                pad_b = tile_size - ph
                pad_r = tile_size - pw
                patch = cv2.copyMakeBorder(patch, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT_101)

            mean = getattr(model, "input_mean", IMAGENET_MEAN)
            std = getattr(model, "input_std", IMAGENET_STD)
            t = bgr_to_normalized_tensor(patch, mean=mean, std=std).unsqueeze(0).to(device)
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_amp and device.type == "cuda") else nullcontext()
            with amp_ctx:
                if hasattr(model, "forward_outputs"):
                    out = model.forward_outputs(t, output_size=(tile_size, tile_size))
                    lg_t = out["binary_logits"]
                else:
                    lg_t = model(t)
            lg = lg_t[0, 0].detach().float().cpu().numpy()
            lg = lg[:ph, :pw]
            logits_sum[y : y + ph, x : x + pw] += lg
            logits_cnt[y : y + ph, x : x + pw] += 1.0

    logits_cnt = np.maximum(logits_cnt, 1e-6)
    return logits_sum / logits_cnt


@torch.no_grad()
def predict_prob_tiled(
    model: nn.Module,
    frame_bgr: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    tile_overlap: int = 128,
    use_amp: bool = True,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    tile_size = max(64, int(tile_size))
    tile_overlap = max(0, min(int(tile_overlap), tile_size - 8))
    stride = tile_size - tile_overlap
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if not ys or ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))
    if not xs or xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))

    prob_sum = np.zeros((h, w), dtype=np.float32)
    prob_cnt = np.zeros((h, w), dtype=np.float32)
    mean = getattr(model, "input_mean", IMAGENET_MEAN)
    std = getattr(model, "input_std", IMAGENET_STD)

    for y in ys:
        for x in xs:
            patch = frame_bgr[y : y + tile_size, x : x + tile_size]
            ph, pw = patch.shape[:2]
            if ph <= 0 or pw <= 0:
                continue
            if ph != tile_size or pw != tile_size:
                pad_b = tile_size - ph
                pad_r = tile_size - pw
                patch = cv2.copyMakeBorder(patch, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT_101)

            t = bgr_to_normalized_tensor(patch, mean=mean, std=std).unsqueeze(0).to(device)
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_amp and device.type == "cuda") else nullcontext()
            with amp_ctx:
                if hasattr(model, "forward_outputs"):
                    out = model.forward_outputs(t, output_size=(tile_size, tile_size))
                    prob_t = out["positive_prob"]
                else:
                    prob_t = torch.sigmoid(model(t))
            pr = prob_t[0, 0].detach().float().cpu().numpy()
            pr = pr[:ph, :pw]
            prob_sum[y : y + ph, x : x + pw] += pr
            prob_cnt[y : y + ph, x : x + pw] += 1.0

    prob_cnt = np.maximum(prob_cnt, 1e-6)
    return prob_sum / prob_cnt


@torch.no_grad()
def predict_prob_multi_tta(
    model: nn.Module,
    frame_bgr: np.ndarray,
    device: torch.device,
    tile_sizes: Sequence[int],
    tile_overlaps: Sequence[int],
    tta_scales: Sequence[float],
    tta_hflip: bool,
    use_amp: bool,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)
    count = 0
    pairs: List[Tuple[int, int]] = []
    if tile_sizes and tile_overlaps and len(tile_sizes) == len(tile_overlaps):
        pairs = [(max(64, int(ts)), max(0, int(to))) for ts, to in zip(tile_sizes, tile_overlaps)]
    else:
        ts = int(tile_sizes[0]) if tile_sizes else 640
        to = int(tile_overlaps[0]) if tile_overlaps else 160
        pairs = [(ts, to)]

    for scale in tta_scales or [1.0]:
        scale_f = max(0.5, float(scale))
        if abs(scale_f - 1.0) < 1e-6:
            scaled = frame_bgr
        else:
            sw = max(32, int(round(w * scale_f)))
            sh = max(32, int(round(h * scale_f)))
            scaled = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)

        variants = [scaled]
        if bool(tta_hflip):
            variants.append(cv2.flip(scaled, 1))

        for v_idx, variant in enumerate(variants):
            for ts, to in pairs:
                prob = predict_prob_tiled(
                    model=model,
                    frame_bgr=variant,
                    device=device,
                    tile_size=ts,
                    tile_overlap=to,
                    use_amp=use_amp,
                )
                if v_idx == 1:
                    prob = prob[:, ::-1].copy()
                if prob.shape[:2] != (h, w):
                    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
                acc += prob.astype(np.float32)
                count += 1

    if count <= 0:
        return np.zeros((h, w), dtype=np.float32)
    return acc / float(count)


def build_sobel_kernels(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    ky = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    return kx, ky


def edge_map_from_prob(prob: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    gx = F.conv2d(prob, kx, padding=1)
    gy = F.conv2d(prob, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def compute_skeleton(mask_u8: np.ndarray) -> np.ndarray:
    img = (mask_u8 > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def build_centerline_reference(gt_u8: np.ndarray, tol_px: int = 2) -> CenterlineReference:
    gt_s = compute_skeleton(gt_u8)
    gt_n = int(gt_s.sum())
    k = max(1, int(tol_px))
    kernel = np.ones((2 * k + 1, 2 * k + 1), dtype=np.uint8)
    if gt_n > 0:
        gt_d = cv2.dilate(gt_s, kernel, iterations=1)
    else:
        gt_d = np.zeros_like(gt_s, dtype=np.uint8)
    return CenterlineReference(
        skeleton=gt_s,
        dilated=gt_d,
        count=gt_n,
        tol_px=k,
    )


def centerline_stats(
    pred_u8: np.ndarray,
    gt_u8: Optional[np.ndarray] = None,
    tol_px: int = 2,
    gt_ref: Optional[CenterlineReference] = None,
) -> Tuple[int, int, int]:
    pred_s = compute_skeleton(pred_u8)
    pred_n = int(pred_s.sum())
    ref = gt_ref if gt_ref is not None else build_centerline_reference(gt_u8 if gt_u8 is not None else np.zeros_like(pred_u8), tol_px=tol_px)
    gt_s = ref.skeleton
    gt_d = ref.dilated
    gt_n = int(ref.count)
    if pred_n <= 0 and gt_n <= 0:
        return 0, 0, 0
    if pred_n <= 0:
        return 0, 0, gt_n
    if gt_n <= 0:
        return 0, pred_n, 0

    k = max(1, int(ref.tol_px if gt_ref is not None else tol_px))
    kernel = np.ones((2 * k + 1, 2 * k + 1), dtype=np.uint8)
    pred_d = cv2.dilate(pred_s, kernel, iterations=1)

    tp = int(np.logical_and(pred_s > 0, gt_d > 0).sum())
    fp = int(np.logical_and(pred_s > 0, gt_d == 0).sum())
    fn = int(np.logical_and(gt_s > 0, pred_d == 0).sum())
    return tp, fp, fn


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    img = img.clamp(0.0, 1.0)
    opened = soft_open(img)
    skel = F.relu(img - opened)
    for _ in range(max(1, int(iters)) - 1):
        img = soft_erode(img)
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(pred_prob: torch.Tensor, target: torch.Tensor, iters: int = 10) -> torch.Tensor:
    pred = pred_prob.clamp(0.0, 1.0)
    tgt = target.clamp(0.0, 1.0)
    skel_pred = soft_skeletonize(pred, iters=iters)
    skel_tgt = soft_skeletonize(tgt, iters=iters)
    tprec = (skel_pred * tgt).flatten(1).sum(dim=1)
    tprec = (tprec + 1e-6) / (skel_pred.flatten(1).sum(dim=1) + 1e-6)
    tsens = (skel_tgt * pred).flatten(1).sum(dim=1)
    tsens = (tsens + 1e-6) / (skel_tgt.flatten(1).sum(dim=1) + 1e-6)
    cl = 1.0 - ((2.0 * tprec * tsens + 1e-6) / (tprec + tsens + 1e-6))
    return cl


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * p * r, p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def build_benchmark_manifest(
    image_dir: Path,
    output_path: Path,
    num_images: int,
    seed: int,
    recursive: bool = False,
    source_summary: Optional[Path] = None,
) -> Dict[str, Any]:
    if source_summary is not None and source_summary.exists():
        summary = json.loads(source_summary.read_text(encoding="utf-8-sig"))
        rows = summary.get("rows", [])
        images = [str(resolve_path(str(r.get("image_path", "")))) for r in rows if str(r.get("image_path", "")).strip()]
    else:
        all_images = list_images(image_dir, recursive=recursive)
        rng = random.Random(int(seed))
        n = max(1, min(int(num_images), len(all_images)))
        images = [str(p) for p in rng.sample(all_images, n)]
    manifest = {
        "created_at": utc_now_iso(),
        "image_dir": str(image_dir),
        "seed": int(seed),
        "num_images": int(len(images)),
        "images": images,
        "source_summary": str(source_summary) if source_summary is not None else "",
    }
    write_json(output_path, manifest)
    return manifest


def load_benchmark_manifest(manifest_path: Path) -> List[Path]:
    data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    images = [resolve_path(str(x)) for x in data.get("images", [])]
    return [p for p in images if p.exists()]


def build_guideline_postprocess_context(frame_bgr: np.ndarray) -> GuidelinePostprocessContext:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    white_like = np.logical_and(sat <= 95, val >= 145).astype(np.uint8)
    return GuidelinePostprocessContext(gray=gray, white_like=white_like)


def _candidate_component_score(
    comp_u8: np.ndarray,
    prob_np: np.ndarray,
    context: GuidelinePostprocessContext,
    min_area: int,
    min_white_ratio: float,
    min_dark_shell_ratio: float,
    min_elongation: float,
) -> float:
    area = int(comp_u8.sum())
    if area < int(min_area):
        return -1e9
    axis = principal_axis_stats(comp_u8)
    elong = float(axis["elongation"]) if axis is not None else 1.0
    if elong < float(min_elongation):
        return -1e9
    white_ratio = float(np.logical_and(comp_u8 > 0, context.white_like > 0).sum()) / float(max(1, area))
    if white_ratio < float(min_white_ratio):
        return -1e9
    shell_ratio = float(
        dark_shell_ratio(
            gray_u8=context.gray,
            mask_u8=comp_u8,
            ring_px=1,
            dark_threshold=125,
            allowed_u8=None,
        )
    )
    if shell_ratio < float(min_dark_shell_ratio):
        return -1e9
    mean_prob = float(prob_np[comp_u8 > 0].mean()) if area > 0 else 0.0
    return float((1.8 * white_ratio) + (1.2 * shell_ratio) + (0.08 * min(elong, 12.0)) + (0.9 * mean_prob))


def axis_complete_mask(
    base_mask_u8: np.ndarray,
    context: GuidelinePostprocessContext,
    axis_complete_max_gap: int = 24,
    axis_complete_max_extension: int = 160,
    axis_complete_min_white_ratio: float = 0.42,
    axis_complete_min_dark_border_ratio: float = 0.05,
) -> np.ndarray:
    comp = (base_mask_u8 > 0).astype(np.uint8)
    if int(comp.sum()) <= 0:
        return comp
    axis = principal_axis_stats(comp)
    if axis is None:
        return comp

    gray = context.gray
    white_like = context.white_like
    h, w = comp.shape[:2]
    dir_x = float(axis.get("vx", 0.0))
    dir_y = float(axis.get("vy", 0.0))
    norm = math.hypot(dir_x, dir_y)
    if norm <= 1e-6:
        return comp
    dir_x /= norm
    dir_y /= norm
    pts = np.column_stack(np.nonzero(comp > 0))
    center_x = float(axis.get("cx", 0.0))
    center_y = float(axis.get("cy", 0.0))
    proj = ((pts[:, 1] - center_x) * dir_x) + ((pts[:, 0] - center_y) * dir_y)
    min_p = float(proj.min())
    max_p = float(proj.max())
    corridor_half = max(2.0, float(axis.get("thickness", 2.0)) * 0.85)

    def try_extend(sign: float, start_proj: float) -> None:
        gap = 0
        nonlocal comp
        for step in range(1, int(axis_complete_max_extension) + 1):
            p = start_proj + (sign * step)
            x = int(round(center_x + dir_x * p))
            y = int(round(center_y + dir_y * p))
            if x < 0 or y < 0 or x >= w or y >= h:
                break
            accepted = False
            rr = int(math.ceil(corridor_half))
            for yy in range(max(0, y - rr), min(h, y + rr + 1)):
                for xx in range(max(0, x - rr), min(w, x + rr + 1)):
                    perp = abs(((xx - center_x) * (-dir_y)) + ((yy - center_y) * dir_x))
                    if perp > corridor_half:
                        continue
                    if white_like[yy, xx] <= 0:
                        continue
                    y0 = max(0, yy - 1)
                    y1 = min(h, yy + 2)
                    x0 = max(0, xx - 1)
                    x1 = min(w, xx + 2)
                    local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                    local_mask[yy - y0, xx - x0] = 1
                    local_shell = float(
                        dark_shell_ratio(
                            gray_u8=gray[y0:y1, x0:x1],
                            mask_u8=local_mask,
                            ring_px=1,
                            dark_threshold=125,
                            allowed_u8=None,
                        )
                    )
                    if local_shell < float(axis_complete_min_dark_border_ratio):
                        continue
                    accepted = True
                    comp[yy, xx] = 1
            if accepted:
                gap = 0
            else:
                gap += 1
                if gap > int(axis_complete_max_gap):
                    break

    # Gate the completion attempt on current evidence quality.
    shell_ratio = float(dark_shell_ratio(gray, comp, ring_px=1, dark_threshold=125, allowed_u8=None))
    white_ratio = float(np.logical_and(comp > 0, white_like > 0).sum()) / float(max(1, int(comp.sum())))
    if white_ratio < float(axis_complete_min_white_ratio) or shell_ratio < float(axis_complete_min_dark_border_ratio):
        return comp

    try_extend(-1.0, min_p)
    try_extend(+1.0, max_p)
    return (comp > 0).astype(np.uint8)


def postprocess_guideline_mask(
    frame_bgr: np.ndarray,
    prob_np: np.ndarray,
    threshold: float,
    min_area: int = 6,
    min_white_ratio: float = 0.34,
    min_dark_shell_ratio: float = 0.08,
    min_elongation: float = 1.25,
    axis_complete: bool = False,
    axis_complete_max_gap: int = 24,
    axis_complete_max_extension: int = 160,
    axis_complete_min_white_ratio: float = 0.42,
    axis_complete_min_dark_border_ratio: float = 0.05,
    context: Optional[GuidelinePostprocessContext] = None,
) -> np.ndarray:
    ctx = context if context is not None else build_guideline_postprocess_context(frame_bgr)
    base = (prob_np >= float(threshold)).astype(np.uint8)
    comps = component_masks(base, min_area=max(1, int(min_area)))
    if not comps:
        return np.zeros_like(base, dtype=np.uint8)

    best_score = -1e9
    best_mask = None
    for comp in comps:
        comp_u8 = (comp > 0).astype(np.uint8)
        if bool(axis_complete):
            comp_u8 = axis_complete_mask(
                base_mask_u8=comp_u8,
                context=ctx,
                axis_complete_max_gap=int(axis_complete_max_gap),
                axis_complete_max_extension=int(axis_complete_max_extension),
                axis_complete_min_white_ratio=float(axis_complete_min_white_ratio),
                axis_complete_min_dark_border_ratio=float(axis_complete_min_dark_border_ratio),
            )
        score = _candidate_component_score(
            comp_u8=comp_u8,
            prob_np=prob_np,
            context=ctx,
            min_area=int(min_area),
            min_white_ratio=float(min_white_ratio),
            min_dark_shell_ratio=float(min_dark_shell_ratio),
            min_elongation=float(min_elongation),
        )
        if score > best_score:
            best_score = score
            best_mask = comp_u8

    if best_mask is None:
        return np.zeros_like(base, dtype=np.uint8)
    return (best_mask > 0).astype(np.uint8)


def sweep_postprocessed_thresholds(
    frame_bgr: np.ndarray,
    prob_np: np.ndarray,
    thresholds: Sequence[float],
    gt_u8: Optional[np.ndarray] = None,
    gt_ref: Optional[CenterlineReference] = None,
    is_negative: bool = False,
    axis_complete: bool = False,
    axis_complete_max_gap: int = 24,
    axis_complete_max_extension: int = 160,
    axis_complete_min_white_ratio: float = 0.42,
    axis_complete_min_dark_border_ratio: float = 0.05,
    context: Optional[GuidelinePostprocessContext] = None,
    preview_threshold: Optional[float] = None,
) -> Tuple[Dict[float, Dict[str, int]], Optional[np.ndarray]]:
    ctx = context if context is not None else build_guideline_postprocess_context(frame_bgr)
    gt = gt_u8 if gt_u8 is not None else np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    local_thr: Dict[float, Dict[str, int]] = {}
    preview_mask: Optional[np.ndarray] = None
    prev_pred: Optional[np.ndarray] = None
    prev_row: Optional[Dict[str, int]] = None

    for thr in thresholds:
        thr_f = float(thr)
        pred = postprocess_guideline_mask(
            frame_bgr,
            prob_np,
            threshold=thr_f,
            axis_complete=bool(axis_complete),
            axis_complete_max_gap=int(axis_complete_max_gap),
            axis_complete_max_extension=int(axis_complete_max_extension),
            axis_complete_min_white_ratio=float(axis_complete_min_white_ratio),
            axis_complete_min_dark_border_ratio=float(axis_complete_min_dark_border_ratio),
            context=ctx,
        )

        row: Dict[str, int]
        if prev_pred is not None and prev_row is not None and np.array_equal(pred, prev_pred):
            row = dict(prev_row)
        else:
            row = {
                "tp": int(np.logical_and(pred > 0, gt > 0).sum()),
                "fp": int(np.logical_and(pred > 0, gt == 0).sum()),
                "fn": int(np.logical_and(pred == 0, gt > 0).sum()),
                "cl_tp": 0,
                "cl_fp": 0,
                "cl_fn": 0,
                "neg_frames": 0,
                "neg_fp_frames": 0,
            }
            if bool(is_negative):
                row["neg_frames"] = 1
                row["neg_fp_frames"] = 1 if int(pred.sum()) > 0 else 0
            else:
                cl_tp, cl_fp, cl_fn = centerline_stats(pred, gt_ref=gt_ref)
                row["cl_tp"] = int(cl_tp)
                row["cl_fp"] = int(cl_fp)
                row["cl_fn"] = int(cl_fn)

        local_thr[thr_f] = row
        if preview_threshold is not None and abs(thr_f - float(preview_threshold)) < 1e-9:
            preview_mask = pred.copy()
        prev_pred = pred
        prev_row = row

    return local_thr, preview_mask


class nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
