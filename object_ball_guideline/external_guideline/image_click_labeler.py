from __future__ import annotations

import argparse
import datetime as dt
import heapq
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from live_label_overlay import (
    DEFAULT_CONFIG_PATH,
    WINDOW_NAME,
    bbox_from_mask,
    detect_candidates,
    detect_white_line_cv_masks,
    ensure_dataset_layout,
    load_config,
    load_sam_model,
    mask_iou,
    pick_click_candidate,
    save_pending_sample,
    str2bool,
    validate_mask,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
POS_LABEL = 1
NEG_LABEL = 0


def list_images(image_dir: Path, recursive: bool) -> List[Path]:
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if recursive:
        files = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        files = [p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: str(p).lower())
    return files


def clamp_idx(idx: int, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(size - 1, int(idx)))


def load_resume_state(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_resume_state(path: Path, image_dir: Path, recursive: bool, index: int, image_path: Optional[Path], saved_total: int) -> None:
    payload = {
        "version": 1,
        "image_dir": str(image_dir.resolve()),
        "recursive": bool(recursive),
        "last_index": int(index),
        "last_image_path": str(image_path.resolve()) if image_path is not None else "",
        "saved_total": int(saved_total),
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_start_index(
    images: List[Path],
    image_dir: Path,
    recursive: bool,
    start_index_arg: Optional[int],
    resume_last: bool,
    resume_state: Dict,
) -> int:
    if start_index_arg is not None:
        return clamp_idx(int(start_index_arg), len(images))
    if not resume_last:
        return 0
    st_dir = str(resume_state.get("image_dir", "")).strip()
    st_recursive = bool(resume_state.get("recursive", False))
    if st_dir.lower() != str(image_dir.resolve()).lower() or st_recursive != bool(recursive):
        return 0
    st_path = str(resume_state.get("last_image_path", "")).strip().lower()
    if st_path:
        for i, p in enumerate(images):
            if str(p.resolve()).lower() == st_path:
                return i
    st_idx = resume_state.get("last_index", 0)
    return clamp_idx(int(st_idx), len(images)) if isinstance(st_idx, int) else 0


def compute_view_rect(frame_w: int, frame_h: int, zoom: float, center_xy: Tuple[float, float]) -> Tuple[int, int, int, int]:
    z = max(1.0, float(zoom))
    view_w = max(1, int(round(frame_w / z)))
    view_h = max(1, int(round(frame_h / z)))
    cx, cy = float(center_xy[0]), float(center_xy[1])
    x1 = int(round(cx - view_w * 0.5))
    y1 = int(round(cy - view_h * 0.5))
    x1 = max(0, min(max(0, frame_w - view_w), x1))
    y1 = max(0, min(max(0, frame_h - view_h), y1))
    return x1, y1, x1 + view_w - 1, y1 + view_h - 1


def display_to_source_view(
    disp_xy: Tuple[int, int],
    display_wh: Tuple[int, int],
    view_rect: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int]]:
    dx, dy = int(disp_xy[0]), int(disp_xy[1])
    disp_w, disp_h = int(display_wh[0]), int(display_wh[1])
    if disp_w <= 0 or disp_h <= 0:
        return None
    if not (0 <= dx < disp_w and 0 <= dy < disp_h):
        return None
    x1, y1, x2, y2 = view_rect
    vw = max(1, x2 - x1 + 1)
    vh = max(1, y2 - y1 + 1)
    fx = float(dx) / float(max(1, disp_w - 1))
    fy = float(dy) / float(max(1, disp_h - 1))
    sx = x1 + int(round(fx * float(vw - 1)))
    sy = y1 + int(round(fy * float(vh - 1)))
    return sx, sy


def build_rescue_profile(profile_cfg: Dict) -> Dict:
    rescue = dict(profile_cfg)
    rescue["post_threshold"] = max(0.05, float(profile_cfg["post_threshold"]) - 0.12)
    rescue["mask_threshold"] = max(0.35, float(profile_cfg["mask_threshold"]) - 0.12)
    rescue["min_score"] = max(0.0, float(profile_cfg["min_score"]) - 0.18)
    rescue["min_area"] = max(6, int(round(float(profile_cfg["min_area"]) * 0.25)))
    rescue["min_aspect_ratio"] = max(1.2, float(profile_cfg["min_aspect_ratio"]) * 0.45)
    rescue["max_fill_ratio"] = min(0.92, float(profile_cfg["max_fill_ratio"]) + 0.22)
    rescue["max_border_touch_ratio"] = min(0.95, float(profile_cfg["max_border_touch_ratio"]) + 0.35)
    return rescue


def build_dual_prompt_profile(profile_cfg: Dict) -> Dict:
    dual = dict(profile_cfg)
    dual["post_threshold"] = max(0.08, float(profile_cfg["post_threshold"]) - 0.14)
    dual["mask_threshold"] = max(0.35, float(profile_cfg["mask_threshold"]) - 0.14)
    dual["min_score"] = max(0.0, float(profile_cfg["min_score"]) - 0.20)
    dual["min_area"] = max(4, int(round(float(profile_cfg["min_area"]) * 0.18)))
    dual["min_aspect_ratio"] = max(1.15, float(profile_cfg["min_aspect_ratio"]) * 0.40)
    dual["max_fill_ratio"] = min(0.92, float(profile_cfg["max_fill_ratio"]) + 0.28)
    dual["max_border_touch_ratio"] = min(0.95, float(profile_cfg["max_border_touch_ratio"]) + 0.35)
    return dual


def map_crop_candidates_to_full(
    crop_candidates: List[Dict],
    crop_xyxy: Tuple[int, int, int, int],
    full_hw: Tuple[int, int],
    scaled_hw: Tuple[int, int],
) -> List[Dict]:
    x1, y1, x2, y2 = crop_xyxy
    full_h, full_w = full_hw
    scaled_h, scaled_w = scaled_hw
    crop_h = max(1, y2 - y1 + 1)
    crop_w = max(1, x2 - x1 + 1)
    out: List[Dict] = []
    for cand in crop_candidates:
        cm = cand["mask"]
        if (scaled_h, scaled_w) != (crop_h, crop_w):
            cm = cv2.resize(cm.astype(np.uint8), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        fm = np.zeros((full_h, full_w), dtype=np.uint8)
        fm[y1 : y2 + 1, x1 : x2 + 1] = (cm > 0).astype(np.uint8)
        cc = dict(cand)
        cc["mask"] = fm
        out.append(cc)
    return out


def detect_click_rescue(
    frame_bgr: np.ndarray,
    click_xy: Tuple[int, int],
    model,
    processor,
    prompt: str,
    profile_cfg: Dict,
    auto_cfg: Dict,
    device: str,
    use_tiles: bool,
    rescue_crop: int,
    rescue_upscale: float,
) -> Optional[Dict]:
    h, w = frame_bgr.shape[:2]
    cx, cy = int(click_xy[0]), int(click_xy[1])
    half = max(32, int(rescue_crop) // 2)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w - 1, cx + half)
    y2 = min(h - 1, cy + half)
    crop = frame_bgr[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        return None
    crop_scaled = cv2.resize(crop, None, fx=float(rescue_upscale), fy=float(rescue_upscale), interpolation=cv2.INTER_CUBIC) if float(rescue_upscale) > 1.0 else crop
    rescue_profile = build_rescue_profile(profile_cfg)
    crop_cands = detect_candidates(
        frame_bgr=crop_scaled,
        model=model,
        processor=processor,
        prompt=prompt,
        profile_cfg=rescue_profile,
        auto_cfg=auto_cfg,
        device=device,
        use_tiles=bool(use_tiles),
        include_cv=True,
    )
    mapped = map_crop_candidates_to_full(crop_cands, (x1, y1, x2, y2), (h, w), crop_scaled.shape[:2])
    picked = pick_click_candidate(mapped, (cx, cy))
    if picked is not None:
        picked["source"] = f"{picked.get('source', 'sam')}_rescue"
        return picked
    cv_cands: List[Dict] = []
    for m in detect_white_line_cv_masks(frame_bgr, auto_cfg["white_cv"]):
        if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and int(m[cy, cx]) > 0:
            valid = validate_mask(m, None, rescue_profile, enforce_score=False)
            if valid is not None:
                valid["source"] = "cv_rescue"
                cv_cands.append(valid)
    return pick_click_candidate(cv_cands, (cx, cy))


def compute_mask_white_support_ratio(frame_bgr: np.ndarray, mask_u8: np.ndarray, white_cfg: Dict) -> float:
    area = int((mask_u8 > 0).sum())
    if area <= 0:
        return 0.0
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(
        hsv,
        (int(white_cfg.get("h_min", 0)), int(white_cfg.get("s_min", 0)), max(0, int(white_cfg.get("v_min", 180)) - 30)),
        (int(white_cfg.get("h_max", 180)), min(255, int(white_cfg.get("s_max", 80)) + 30), int(white_cfg.get("v_max", 255))),
    )
    white_u8 = cv2.dilate((white > 0).astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
    overlap = int(np.logical_and(mask_u8 > 0, white_u8 > 0).sum())
    return float(overlap) / float(max(1, area))


def compute_mask_dark_border_ratio(
    frame_bgr: np.ndarray,
    mask_u8: np.ndarray,
    ring_px: int,
    dark_threshold: int,
) -> float:
    mask = (mask_u8 > 0).astype(np.uint8)
    if int(mask.sum()) <= 0:
        return 0.0
    ring = max(1, int(ring_px))
    kernel = np.ones((3, 3), dtype=np.uint8)
    outer = cv2.dilate(mask, kernel, iterations=ring)
    outer = np.logical_and(outer > 0, mask == 0).astype(np.uint8)
    shell_area = int(outer.sum())
    if shell_area <= 0:
        return 0.0
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray <= int(dark_threshold)).astype(np.uint8)
    dark_shell = int(np.logical_and(outer > 0, dark > 0).sum())
    return float(dark_shell) / float(max(1, shell_area))


def build_relaxed_white_mask(frame_bgr: np.ndarray, white_cfg: Dict, s_relax: int = 32, v_relax: int = 32) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(
        hsv,
        (
            int(white_cfg.get("h_min", 0)),
            max(0, int(white_cfg.get("s_min", 0)) - int(s_relax)),
            max(0, int(white_cfg.get("v_min", 180)) - int(v_relax)),
        ),
        (
            int(white_cfg.get("h_max", 180)),
            min(255, int(white_cfg.get("s_max", 80)) + int(s_relax)),
            int(white_cfg.get("v_max", 255)),
        ),
    )
    return cv2.dilate((white > 0).astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)


def make_segment_corridor_mask(
    shape_hw: Tuple[int, int],
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    corridor_px: float,
    endpoint_margin_px: float,
) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    ys, xs = np.indices((h, w), dtype=np.float32)
    a = np.array([float(p1[0]), float(p1[1])], dtype=np.float32)
    b = np.array([float(p2[0]), float(p2[1])], dtype=np.float32)
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length < 1e-3:
        return np.zeros((h, w), dtype=np.uint8)
    u = vec / length
    v = np.array([-u[1], u[0]], dtype=np.float32)
    rel_x = xs - a[0]
    rel_y = ys - a[1]
    t = rel_x * u[0] + rel_y * u[1]
    n = np.abs(rel_x * v[0] + rel_y * v[1])
    corr = (
        (n <= float(max(1.0, corridor_px)))
        & (t >= -float(max(0.0, endpoint_margin_px)))
        & (t <= length + float(max(0.0, endpoint_margin_px)))
    )
    return corr.astype(np.uint8)


def constrained_edge_expand(mask_u8: np.ndarray, allowed_u8: np.ndarray, iterations: int, kernel_size: int) -> np.ndarray:
    out = (mask_u8 > 0).astype(np.uint8)
    if out.size == 0 or int(out.sum()) <= 0:
        return out
    iters = max(0, int(iterations))
    if iters <= 0:
        return out
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), dtype=np.uint8)
    allowed = (allowed_u8 > 0).astype(np.uint8)
    for _ in range(iters):
        grown = (cv2.dilate(out, kernel, iterations=1) > 0).astype(np.uint8)
        add = np.logical_and(grown > 0, allowed > 0).astype(np.uint8)
        out = np.maximum(out, add)
    return out


def force_connect_endpoints(
    mask_u8: np.ndarray,
    frame_bgr: np.ndarray,
    endpoint_a: Tuple[int, int],
    endpoint_b: Tuple[int, int],
    auto_cfg: Dict,
    args,
) -> np.ndarray:
    out = (mask_u8 > 0).astype(np.uint8)
    h, w = out.shape[:2]
    span = float(math.hypot(float(endpoint_b[0] - endpoint_a[0]), float(endpoint_b[1] - endpoint_a[1])))
    if span < 1.0 or h <= 0 or w <= 0:
        return out

    white_relaxed = build_relaxed_white_mask(
        frame_bgr,
        auto_cfg["white_cv"],
        s_relax=int(args.dual_force_s_relax),
        v_relax=int(args.dual_force_v_relax),
    )
    corridor_px = max(
        2.0,
        min(
            float(args.dual_corridor_px) + float(args.dual_edge_expand_corridor_pad),
            span * float(args.dual_force_connect_corridor_ratio),
        ),
    )
    corridor_u8 = make_segment_corridor_mask(
        shape_hw=(h, w),
        p1=endpoint_a,
        p2=endpoint_b,
        corridor_px=float(corridor_px),
        endpoint_margin_px=float(args.dual_endpoint_margin_px) + 2.0,
    )

    line = np.zeros((h, w), dtype=np.uint8)
    thickness = max(1, int(args.dual_force_connect_thickness))
    cv2.line(line, endpoint_a, endpoint_b, 1, thickness=thickness, lineType=cv2.LINE_AA)

    white_bridge = np.logical_and(line > 0, np.logical_and(corridor_u8 > 0, white_relaxed > 0)).astype(np.uint8)
    min_bridge = max(2, int(round(span * 0.10)))
    if int(white_bridge.sum()) < min_bridge:
        white_bridge = np.logical_and(line > 0, corridor_u8 > 0).astype(np.uint8)

    near_bridge = cv2.dilate(white_bridge, np.ones((3, 3), dtype=np.uint8), iterations=1)
    bridge_edge = np.logical_and(near_bridge > 0, np.logical_and(corridor_u8 > 0, white_relaxed > 0)).astype(np.uint8)

    out = np.logical_or(out > 0, white_bridge > 0).astype(np.uint8)
    out = np.logical_or(out > 0, bridge_edge > 0).astype(np.uint8)
    out = np.logical_and(out > 0, corridor_u8 > 0).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=max(0, int(args.dual_force_close_iter)))
    cv2.circle(out, endpoint_a, max(1, thickness // 2), 1, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(out, endpoint_b, max(1, thickness // 2), 1, thickness=-1, lineType=cv2.LINE_AA)
    return out


def compute_whiteness_score_map(frame_bgr: np.ndarray, white_cfg: Dict, s_relax: int, v_relax: int) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    a_dev = np.abs(lab[:, :, 1] - 128.0) / 127.0
    b_dev = np.abs(lab[:, :, 2] - 128.0) / 127.0
    # Bright + low saturation + near-neutral chroma.
    white_like = np.clip(v, 0.0, 1.0) * np.power(np.clip(1.0 - s, 0.0, 1.0), 1.35) * np.clip(1.0 - 0.75 * (a_dev + b_dev) * 0.5, 0.0, 1.0)
    # Extra boost where relaxed HSV white threshold agrees.
    relaxed = build_relaxed_white_mask(frame_bgr, white_cfg, s_relax=s_relax, v_relax=v_relax).astype(np.float32)
    boosted = np.clip(0.72 * white_like + 0.28 * relaxed, 0.0, 1.0)
    return boosted.astype(np.float32)


def compute_dark_rail_score_map(
    frame_bgr: np.ndarray,
    endpoint_a: Tuple[int, int],
    endpoint_b: Tuple[int, int],
    rail_offset_px: float,
    rail_outer_offset_px: float,
    dark_gamma: float,
) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    dark = np.power(np.clip(1.0 - gray, 0.0, 1.0), float(max(0.1, dark_gamma)))
    h, w = gray.shape[:2]
    ax, ay = float(endpoint_a[0]), float(endpoint_a[1])
    bx, by = float(endpoint_b[0]), float(endpoint_b[1])
    dx = bx - ax
    dy = by - ay
    length = float(math.hypot(dx, dy))
    if length < 1e-3:
        return np.zeros((h, w), dtype=np.float32)
    nx = -dy / length
    ny = dx / length
    ys, xs = np.indices((h, w), dtype=np.float32)

    def sample(offset_px: float) -> Tuple[np.ndarray, np.ndarray]:
        ox = nx * float(offset_px)
        oy = ny * float(offset_px)
        map_x_p = np.clip(xs + ox, 0.0, float(w - 1)).astype(np.float32)
        map_y_p = np.clip(ys + oy, 0.0, float(h - 1)).astype(np.float32)
        map_x_m = np.clip(xs - ox, 0.0, float(w - 1)).astype(np.float32)
        map_y_m = np.clip(ys - oy, 0.0, float(h - 1)).astype(np.float32)
        dark_p = cv2.remap(dark, map_x_p, map_y_p, interpolation=cv2.INTER_LINEAR)
        dark_m = cv2.remap(dark, map_x_m, map_y_m, interpolation=cv2.INTER_LINEAR)
        return dark_p.astype(np.float32), dark_m.astype(np.float32)

    inner = max(1.0, float(rail_offset_px))
    outer = max(inner + 0.3, float(rail_outer_offset_px))
    d1p, d1m = sample(inner)
    d2p, d2m = sample(outer)
    near_pair = np.minimum(d1p, d1m)
    far_pair = np.minimum(d2p, d2m)
    # Border rails should be dark on both sides of a bright centerline.
    rail = np.clip(0.68 * near_pair + 0.32 * far_pair, 0.0, 1.0)
    return rail.astype(np.float32)


def segment_penalty_maps(
    shape_hw: Tuple[int, int],
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    corridor_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    ys, xs = np.indices((h, w), dtype=np.float32)
    a = np.array([float(p1[0]), float(p1[1])], dtype=np.float32)
    b = np.array([float(p2[0]), float(p2[1])], dtype=np.float32)
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length < 1e-3:
        return np.ones((h, w), dtype=np.float32), np.ones((h, w), dtype=np.float32)
    u = vec / length
    v = np.array([-u[1], u[0]], dtype=np.float32)
    rel_x = xs - a[0]
    rel_y = ys - a[1]
    t = rel_x * u[0] + rel_y * u[1]
    n = np.abs(rel_x * v[0] + rel_y * v[1])
    n_norm = np.clip(n / max(1e-3, float(corridor_px)), 0.0, 1.0)
    under = np.clip(-t, 0.0, None)
    over = np.clip(t - length, 0.0, None)
    endpoint = np.clip((under + over) / max(1e-3, length), 0.0, 1.0)
    return n_norm.astype(np.float32), endpoint.astype(np.float32)


def shortest_path_cost(
    cost_map: np.ndarray,
    allowed_u8: np.ndarray,
    start_xy: Tuple[int, int],
    end_xy: Tuple[int, int],
) -> Optional[Tuple[List[Tuple[int, int]], float]]:
    h, w = cost_map.shape[:2]
    sx, sy = int(start_xy[0]), int(start_xy[1])
    ex, ey = int(end_xy[0]), int(end_xy[1])
    if not (0 <= sx < w and 0 <= sy < h and 0 <= ex < w and 0 <= ey < h):
        return None
    allowed = (allowed_u8 > 0).astype(np.uint8)
    allowed[sy, sx] = 1
    allowed[ey, ex] = 1

    n = int(h * w)
    inf = np.float32(1e20)
    dist = np.full((n,), inf, dtype=np.float32)
    prev = np.full((n,), -1, dtype=np.int32)
    s_idx = int(sy * w + sx)
    e_idx = int(ey * w + ex)
    if s_idx == e_idx:
        return [(sx, sy)], 0.0
    dist[s_idx] = np.float32(0.0)

    pq: List[Tuple[float, int]] = [(0.0, s_idx)]
    nbrs = [
        (-1, -1, 1.41421356),
        (0, -1, 1.0),
        (1, -1, 1.41421356),
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (-1, 1, 1.41421356),
        (0, 1, 1.0),
        (1, 1, 1.41421356),
    ]

    while pq:
        d, idx = heapq.heappop(pq)
        if d > float(dist[idx]) + 1e-6:
            continue
        if idx == e_idx:
            break
        y = idx // w
        x = idx - y * w
        for dx, dy, step in nbrs:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if int(allowed[ny, nx]) == 0:
                continue
            nidx = int(ny * w + nx)
            nd = float(d) + float(step) * (1.0 + float(cost_map[ny, nx]))
            if nd + 1e-6 < float(dist[nidx]):
                dist[nidx] = np.float32(nd)
                prev[nidx] = np.int32(idx)
                heapq.heappush(pq, (nd, nidx))

    if not np.isfinite(float(dist[e_idx])) or prev[e_idx] < 0:
        return None
    path: List[Tuple[int, int]] = []
    cur = e_idx
    for _ in range(n):
        y = cur // w
        x = cur - y * w
        path.append((int(x), int(y)))
        if cur == s_idx:
            break
        cur = int(prev[cur])
        if cur < 0:
            break
    if not path or path[-1] != (sx, sy):
        return None
    path.reverse()
    return path, float(dist[e_idx])


def path_polyline_length(path_xy: List[Tuple[int, int]]) -> float:
    if len(path_xy) < 2:
        return 0.0
    total = 0.0
    px, py = path_xy[0]
    for x, y in path_xy[1:]:
        total += math.hypot(float(x - px), float(y - py))
        px, py = x, y
    return float(total)


def draw_path_mask(shape_hw: Tuple[int, int], path_xy: List[Tuple[int, int]], thickness: int) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    out = np.zeros((h, w), dtype=np.uint8)
    if not path_xy:
        return out
    pts = np.array(path_xy, dtype=np.int32).reshape((-1, 1, 2))
    if pts.shape[0] == 1:
        cv2.circle(out, tuple(pts[0, 0].tolist()), max(1, int(thickness) // 2), 1, thickness=-1, lineType=cv2.LINE_AA)
        return out
    cv2.polylines(out, [pts], isClosed=False, color=1, thickness=max(1, int(thickness)), lineType=cv2.LINE_AA)
    return (out > 0).astype(np.uint8)


def detect_dual_path_candidate(
    frame_bgr: np.ndarray,
    endpoint_a: Tuple[int, int],
    endpoint_b: Tuple[int, int],
    negatives: List[Tuple[int, int]],
    auto_cfg: Dict,
    args,
) -> Tuple[Optional[Dict], str]:
    if not bool(args.dual_path_rescue):
        return None, "path_disabled"
    h, w = frame_bgr.shape[:2]
    click_span = float(math.hypot(float(endpoint_b[0] - endpoint_a[0]), float(endpoint_b[1] - endpoint_a[1])))
    ultra_tiny = is_ultra_tiny_span(click_span, args)
    roi_margin_cfg = int(args.ultra_tiny_path_roi_margin) if ultra_tiny else int(args.dual_path_roi_margin)
    roi_margin = max(int(args.dual_roi_margin), roi_margin_cfg)
    rx1, ry1, rx2, ry2 = compute_dual_roi(endpoint_a, endpoint_b, roi_margin, w, h)
    crop = frame_bgr[ry1 : ry2 + 1, rx1 : rx2 + 1]
    if crop.size == 0:
        return None, "path_empty_roi"

    roi_h = ry2 - ry1 + 1
    roi_w = rx2 - rx1 + 1
    path_base_scale = float(args.ultra_tiny_path_upscale) if ultra_tiny else float(args.dual_path_upscale)
    path_max_scale = float(args.ultra_tiny_path_upscale_max) if ultra_tiny else float(args.dual_path_upscale_max)
    scale = adaptive_upscale_for_mode(click_span, path_base_scale, path_max_scale, args, ultra_tiny)
    path_max_side = float(args.ultra_tiny_path_max_side) if ultra_tiny else float(args.dual_path_max_side)
    longest = max(roi_h, roi_w)
    if longest * scale > path_max_side:
        scale = max(1.0, path_max_side / float(max(1, longest)))
    crop_scaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else crop
    sh, sw = crop_scaled.shape[:2]

    ax = max(0, min(sw - 1, int(round((endpoint_a[0] - rx1) * scale))))
    ay = max(0, min(sh - 1, int(round((endpoint_a[1] - ry1) * scale))))
    bx = max(0, min(sw - 1, int(round((endpoint_b[0] - rx1) * scale))))
    by = max(0, min(sh - 1, int(round((endpoint_b[1] - ry1) * scale))))
    span_scaled = max(1.0, math.hypot(float(bx - ax), float(by - ay)))

    corridor_ratio = float(args.ultra_tiny_path_corridor_ratio) if ultra_tiny else float(args.dual_path_corridor_ratio)
    corridor_pad = float(args.ultra_tiny_path_corridor_pad) if ultra_tiny else float(args.dual_path_corridor_pad)
    corridor_px = max(
        2.0,
        min(
            float(args.dual_corridor_px) * scale + corridor_pad,
            span_scaled * corridor_ratio,
        ),
    )
    corridor = make_segment_corridor_mask(
        shape_hw=(sh, sw),
        p1=(ax, ay),
        p2=(bx, by),
        corridor_px=float(corridor_px),
        endpoint_margin_px=float(args.dual_endpoint_margin_px) * scale + 2.0,
    )
    if int(corridor.sum()) <= 0:
        return None, "path_empty_corridor"

    white_score = compute_whiteness_score_map(
        crop_scaled,
        auto_cfg["white_cv"],
        s_relax=int(args.ultra_tiny_path_s_relax) if ultra_tiny else int(args.dual_path_s_relax),
        v_relax=int(args.ultra_tiny_path_v_relax) if ultra_tiny else int(args.dual_path_v_relax),
    )
    rail_offset_px = float(args.ultra_tiny_path_rail_offset_px) if ultra_tiny else float(args.dual_path_rail_offset_px)
    rail_outer_offset_px = float(args.ultra_tiny_path_rail_outer_offset_px) if ultra_tiny else float(args.dual_path_rail_outer_offset_px)
    rail_dark_gamma = float(args.dual_path_rail_dark_gamma)
    rail_score = compute_dark_rail_score_map(
        crop_scaled,
        endpoint_a=(ax, ay),
        endpoint_b=(bx, by),
        rail_offset_px=rail_offset_px,
        rail_outer_offset_px=rail_outer_offset_px,
        dark_gamma=rail_dark_gamma,
    )
    white_seed_thr = float(args.ultra_tiny_path_white_seed_thr) if ultra_tiny else float(args.dual_path_white_seed_thr)
    white_mask_thr = float(args.ultra_tiny_path_white_mask_thr) if ultra_tiny else float(args.dual_path_white_mask_thr)
    white_seed = (white_score >= white_seed_thr).astype(np.uint8)
    dist_to_nonwhite = cv2.distanceTransform((white_seed * 255).astype(np.uint8), cv2.DIST_L2, 3)
    center_pref = np.zeros_like(dist_to_nonwhite, dtype=np.float32)
    m = float(dist_to_nonwhite.max())
    if m > 1e-6:
        center_pref = np.clip(dist_to_nonwhite / m, 0.0, 1.0).astype(np.float32)

    n_pen, endpoint_pen = segment_penalty_maps((sh, sw), (ax, ay), (bx, by), corridor_px=max(1.0, corridor_px))
    cost = (
        float(args.dual_path_w_white) * (1.0 - white_score)
        + (float(args.ultra_tiny_path_w_rail) if ultra_tiny else float(args.dual_path_w_rail)) * (1.0 - rail_score)
        + float(args.dual_path_w_center) * (1.0 - center_pref)
        + float(args.dual_path_w_dist) * n_pen
        + float(args.dual_path_w_endpoint) * endpoint_pen
    ).astype(np.float32)

    allowed = (corridor > 0).astype(np.uint8)
    gray = cv2.cvtColor(crop_scaled, cv2.COLOR_BGR2GRAY)
    min_gray = int(args.ultra_tiny_path_min_gray) if ultra_tiny else int(args.dual_path_min_gray)
    allowed = np.logical_and(allowed > 0, gray >= min_gray).astype(np.uint8)
    for nx_full, ny_full in negatives:
        if not (rx1 <= nx_full <= rx2 and ry1 <= ny_full <= ry2):
            continue
        nx = max(0, min(sw - 1, int(round((nx_full - rx1) * scale))))
        ny = max(0, min(sh - 1, int(round((ny_full - ry1) * scale))))
        neg_rad = float(args.ultra_tiny_path_neg_radius_px) if ultra_tiny else float(args.dual_path_neg_radius_px)
        cv2.circle(allowed, (nx, ny), int(max(1, round(neg_rad * scale))), 0, thickness=-1, lineType=cv2.LINE_AA)
    allowed[ay, ax] = 1
    allowed[by, bx] = 1

    path_res = shortest_path_cost(cost, allowed, (ax, ay), (bx, by))
    if path_res is None:
        return None, "path_not_found"
    path_xy, path_cost = path_res
    path_len = path_polyline_length(path_xy)
    straightness = float(span_scaled / max(1e-6, path_len))
    white_support = float(np.mean([white_score[y, x] for x, y in path_xy])) if path_xy else 0.0
    rail_support = float(np.mean([rail_score[y, x] for x, y in path_xy])) if path_xy else 0.0
    min_len_ratio = float(args.ultra_tiny_path_min_length_ratio) if ultra_tiny else float(args.dual_path_min_length_ratio)
    min_straightness = float(args.ultra_tiny_path_min_straightness) if ultra_tiny else float(args.dual_path_min_straightness)
    min_white_support = float(args.ultra_tiny_path_min_white_support) if ultra_tiny else float(args.dual_path_min_white_support)
    min_rail_support = float(args.ultra_tiny_path_min_rail_support) if ultra_tiny else float(args.dual_path_min_rail_support)
    if path_len < span_scaled * min_len_ratio:
        return None, "path_short"
    if straightness < min_straightness:
        return None, "path_curvy"
    if white_support < min_white_support:
        return None, "path_low_white"
    if rail_support < min_rail_support:
        return None, "path_low_rail"

    white_mask = (white_score >= white_mask_thr).astype(np.uint8)
    dt_white = cv2.distanceTransform((white_mask * 255).astype(np.uint8), cv2.DIST_L2, 3)
    radii = [float(dt_white[y, x]) for x, y in path_xy if 0 <= x < sw and 0 <= y < sh]
    half_w = int(round(np.percentile(np.array(radii, dtype=np.float32), 60.0))) if radii else 1
    min_half = int(args.ultra_tiny_path_min_halfwidth_px) if ultra_tiny else int(args.dual_path_min_halfwidth_px)
    max_half = int(args.ultra_tiny_path_max_halfwidth_px) if ultra_tiny else int(args.dual_path_max_halfwidth_px)
    half_w = max(min_half, min(max_half, int(half_w)))
    thickness = max(1, 2 * half_w + 1)

    if span_scaled >= float(args.dual_path_straighten_min_span_px):
        path_mask = np.zeros((sh, sw), dtype=np.uint8)
        cv2.line(path_mask, (ax, ay), (bx, by), 1, thickness=max(1, int(thickness)), lineType=cv2.LINE_AA)
    else:
        path_mask = draw_path_mask((sh, sw), path_xy, thickness=thickness)
    support = cv2.dilate(white_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    local = np.logical_and(path_mask > 0, np.logical_and(support > 0, corridor > 0)).astype(np.uint8)
    support_frac = float(local.sum()) / float(max(1, int(path_mask.sum())))
    min_support_frac = float(args.ultra_tiny_path_min_support_fraction) if ultra_tiny else float(args.dual_path_min_support_fraction)
    if support_frac < min_support_frac:
        support_wide = cv2.dilate(white_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
        local = np.logical_and(path_mask > 0, np.logical_and(support_wide > 0, corridor > 0)).astype(np.uint8)
    close_iter = int(args.ultra_tiny_path_close_iter) if ultra_tiny else int(args.dual_path_close_iter)
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=max(0, close_iter))
    cv2.circle(local, (ax, ay), max(1, half_w), 1, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(local, (bx, by), max(1, half_w), 1, thickness=-1, lineType=cv2.LINE_AA)
    local = np.logical_and(local > 0, corridor > 0).astype(np.uint8)
    if int(local.sum()) <= 0:
        return None, "path_empty_mask"

    if scale > 1.0:
        local = cv2.resize(local.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((h, w), dtype=np.uint8)
    full[ry1 : ry2 + 1, rx1 : rx2 + 1] = (local > 0).astype(np.uint8)
    cand = {
        "mask": full,
        "score": float((0.65 * white_support + 0.35 * rail_support) * straightness / (1.0 + max(0.0, path_cost) / max(1.0, path_len))),
        "source": "path_dual_roi",
        "scale_used": float(scale),
        "ultra_tiny": bool(ultra_tiny),
        "_path_white_support": float(white_support),
        "_path_rail_support": float(rail_support),
    }
    return cand, "ok"


def compute_mask_fill_and_border(mask_u8: np.ndarray) -> Tuple[float, float, int]:
    mask = (mask_u8 > 0).astype(np.uint8)
    area = int(mask.sum())
    if area <= 0:
        return 1.0, 1.0, 0
    bb = bbox_from_mask(mask)
    if bb is None:
        return 1.0, 1.0, area
    _, _, _, _, bw, bh = bb
    fill = float(area) / float(max(1, int(bw) * int(bh)))
    border = np.zeros_like(mask, dtype=np.uint8)
    border[0, :] = 1
    border[-1, :] = 1
    border[:, 0] = 1
    border[:, -1] = 1
    touched = int(np.logical_and(mask == 1, border == 1).sum())
    border_ratio = float(touched) / float(max(1, area))
    return float(fill), float(border_ratio), int(area)


def rescue_quality_gate(frame_bgr: np.ndarray, candidate: Dict, click_xy: Tuple[int, int], auto_cfg: Dict, args) -> Tuple[bool, str]:
    source = str(candidate.get("source", ""))
    if "rescue" not in source:
        return True, "ok"
    score = candidate.get("score")
    if score is not None and float(score) < float(args.rescue_min_score):
        return False, f"low_score({float(score):.2f}<{float(args.rescue_min_score):.2f})"
    if float(candidate.get("aspect", 0.0)) < float(args.rescue_min_aspect):
        return False, "low_aspect"
    if float(candidate.get("fill_ratio", 1.0)) > float(args.rescue_max_fill_ratio):
        return False, "high_fill"
    white_ratio = compute_mask_white_support_ratio(frame_bgr, candidate["mask"], auto_cfg["white_cv"])
    if white_ratio < float(args.rescue_min_white_ratio):
        return False, "low_white"
    if bool(args.rescue_require_cv_overlap):
        cx, cy = int(click_xy[0]), int(click_xy[1])
        click_cv = []
        for m in detect_white_line_cv_masks(frame_bgr, auto_cfg["white_cv"]):
            if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and int(m[cy, cx]) > 0:
                click_cv.append((m > 0).astype(np.uint8))
        if not click_cv:
            return False, "no_cv_at_click"
        best_iou = max(mask_iou(candidate["mask"], m) for m in click_cv)
        if float(best_iou) < float(args.rescue_min_cv_iou):
            return False, "low_cv_iou"
    return True, "ok"


def point_to_xyxy_box(x: int, y: int, box_px: int, width: int, height: int) -> List[float]:
    half = max(1, int(round(box_px * 0.5)))
    x1 = max(0, int(x) - half)
    y1 = max(0, int(y) - half)
    x2 = min(width - 1, int(x) + half)
    y2 = min(height - 1, int(y) + half)
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return [float(x1), float(y1), float(x2), float(y2)]


def adaptive_prompt_box_sizes(click_span_px: float, base_pos_px: int, base_neg_px: int) -> Tuple[int, int]:
    span = max(1.0, float(click_span_px))
    pos = max(2, int(base_pos_px))
    neg = max(pos, int(base_neg_px))
    if span <= 40.0:
        pos = max(2, min(pos, int(round(span * 0.22))))
        neg = max(pos, min(neg, int(round(span * 0.30))))
    elif span <= 80.0:
        pos = max(3, min(pos, int(round(span * 0.16))))
        neg = max(pos, min(neg, int(round(span * 0.22))))
    return int(pos), int(neg)


def adaptive_upscale_from_span(click_span_px: float, base_scale: float, max_scale: float, args) -> float:
    base = max(1.0, float(base_scale))
    cap = max(base, float(max_scale))
    if not bool(args.dual_adaptive_upscale):
        return base
    span = max(float(args.dual_adaptive_min_span_px), float(click_span_px))
    ref = max(float(args.dual_adaptive_ref_span_px), float(args.dual_adaptive_min_span_px) + 1.0)
    if span >= ref:
        return base
    ratio = ref / max(1e-3, span)
    scale = base * (ratio ** float(args.dual_adaptive_gamma))
    return float(min(cap, max(base, scale)))


def is_ultra_tiny_span(click_span_px: float, args) -> bool:
    return bool(args.ultra_tiny_mode) and float(click_span_px) <= float(args.ultra_tiny_span_px)


def adaptive_upscale_for_mode(click_span_px: float, base_scale: float, max_scale: float, args, ultra_tiny_active: bool) -> float:
    if not ultra_tiny_active:
        return adaptive_upscale_from_span(click_span_px, base_scale, max_scale, args)
    base = max(1.0, float(base_scale))
    cap = max(base, float(max_scale))
    if not bool(args.dual_adaptive_upscale):
        return base
    span = max(float(args.ultra_tiny_adaptive_min_span_px), float(click_span_px))
    ref = max(float(args.ultra_tiny_adaptive_ref_span_px), float(args.ultra_tiny_adaptive_min_span_px) + 1.0)
    if span >= ref:
        return base
    ratio = ref / max(1e-3, span)
    scale = base * (ratio ** float(args.ultra_tiny_adaptive_gamma))
    return float(min(cap, max(base, scale)))


def upscale_prompt_box_px(base_px: int, scale: float, args) -> int:
    box = max(2, int(base_px))
    if not bool(args.dual_roi_prompt_scale):
        return box
    grow = max(1.0, float(scale)) ** float(args.dual_roi_prompt_scale_gamma)
    scaled = int(round(float(box) * grow))
    return int(max(2, min(int(args.dual_roi_prompt_max_px), scaled)))


def run_sam_text_with_boxes(
    frame_bgr: np.ndarray,
    model,
    processor,
    prompt: str,
    prompt_points: List[Dict],
    pos_box_px: int,
    neg_box_px: int,
    profile_cfg: Dict,
    device: str,
    source: str,
) -> List[Dict]:
    h, w = frame_bgr.shape[:2]
    boxes: List[List[float]] = []
    labels: List[int] = []
    for p in prompt_points:
        lbl = int(p["label"])
        boxes.append(point_to_xyxy_box(int(p["x"]), int(p["y"]), int(pos_box_px) if lbl == POS_LABEL else int(neg_box_px), w, h))
        labels.append(1 if lbl == POS_LABEL else 0)
    if not boxes:
        return []
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=prompt, input_boxes=[boxes], input_boxes_labels=[labels], return_tensors="pt").to(device)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()
    with torch.inference_mode(), amp_ctx:
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=float(profile_cfg["post_threshold"]),
        mask_threshold=float(profile_cfg["mask_threshold"]),
        target_sizes=[[h, w]],
    )[0]
    masks = result.get("masks")
    scores = result.get("scores")
    if masks is None or int(masks.shape[0]) == 0:
        return []
    out: List[Dict] = []
    for i in range(int(masks.shape[0])):
        m = masks[i]
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        score_val = None
        if scores is not None and i < len(scores):
            sv = scores[i]
            score_val = float(sv.detach().cpu().item()) if isinstance(sv, torch.Tensor) else float(sv)
        valid = validate_mask((m > 0).astype(np.uint8), score_val, profile_cfg, enforce_score=True)
        if valid is not None:
            valid["source"] = source
            out.append(valid)
    return out


def point_hit_with_tolerance(mask_u8: np.ndarray, x: int, y: int, tol: int = 2) -> bool:
    h, w = mask_u8.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return False
    t = max(0, int(tol))
    x1, y1 = max(0, x - t), max(0, y - t)
    x2, y2 = min(w - 1, x + t), min(h - 1, y + t)
    return int(mask_u8[y1 : y2 + 1, x1 : x2 + 1].max()) > 0


def compute_angle_and_span(mask_u8: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[Optional[float], float]:
    ys, xs = np.where(mask_u8 > 0)
    if ys.size < 8:
        return None, 0.0
    xy = np.column_stack((xs, ys)).astype(np.float32)
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, 0.0
    principal = vh[0]
    vec = np.array([float(p2[0] - p1[0]), float(p2[1] - p1[1])], dtype=np.float32)
    dist = float(np.linalg.norm(vec))
    if dist < 1e-3:
        return None, 0.0
    u = vec / dist
    proj = xy @ u
    span_ratio = float((proj.max() - proj.min()) / max(1e-3, dist))
    theta_a = math.atan2(float(principal[1]), float(principal[0]))
    theta_b = math.atan2(float(u[1]), float(u[0]))
    diff = abs((theta_a - theta_b + math.pi * 0.5) % math.pi - math.pi * 0.5)
    return math.degrees(diff), span_ratio


def compute_segment_corridor_metrics(
    mask_u8: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    corridor_px: float,
    endpoint_margin_px: float,
) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask_u8 > 0)
    if ys.size < 8:
        return None
    xy = np.column_stack((xs, ys)).astype(np.float32)
    a = np.array([float(p1[0]), float(p1[1])], dtype=np.float32)
    b = np.array([float(p2[0]), float(p2[1])], dtype=np.float32)
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length < 1e-3:
        return None
    u = vec / length
    v = np.array([-u[1], u[0]], dtype=np.float32)
    rel = xy - a[None, :]
    t = rel @ u
    n = np.abs(rel @ v)

    corr = (
        (n <= float(max(1.0, corridor_px)))
        & (t >= -float(max(0.0, endpoint_margin_px)))
        & (t <= length + float(max(0.0, endpoint_margin_px)))
    )
    corridor_ratio = float(corr.mean()) if corr.size > 0 else 0.0

    t_min = float(t.min())
    t_max = float(t.max())
    overshoot = max(0.0, -t_min) + max(0.0, t_max - length)
    overshoot_ratio = float(overshoot / max(1e-3, length))
    return corridor_ratio, overshoot_ratio


def rank_dual_candidates(candidates: List[Dict]) -> List[Dict]:
    return sorted(
        candidates,
        key=lambda c: (
            -float(c.get("_span_ratio", 0.0)),
            -(float(c["score"]) if c.get("score") is not None else -1.0),
            -float(c.get("_dark_border_ratio", 0.0)),
            -float(c.get("_corridor_ratio", 0.0)),
            float(c.get("_overshoot_ratio", 0.0)),
            int(c.get("area", 0)),
        ),
    )


def select_dual_candidate(
    frame_bgr: np.ndarray,
    candidates: List[Dict],
    endpoint_a: Tuple[int, int],
    endpoint_b: Tuple[int, int],
    negatives: List[Tuple[int, int]],
    auto_cfg: Dict,
    args,
) -> Tuple[Optional[Dict], str]:
    if not candidates:
        return None, "no_candidate"
    click_span = float(math.hypot(float(endpoint_b[0] - endpoint_a[0]), float(endpoint_b[1] - endpoint_a[1])))
    ultra_tiny = is_ultra_tiny_span(click_span, args)
    short_span = click_span <= float(args.dual_short_span_px)
    long_span = click_span >= float(args.dual_long_span_px)
    if ultra_tiny:
        pos_hit_tol_px = int(args.ultra_tiny_pos_hit_tol_px)
        min_span_ratio = float(args.ultra_tiny_min_span_ratio)
        min_white_ratio = float(args.ultra_tiny_min_white_ratio)
        min_dark_border_ratio = float(args.ultra_tiny_min_dark_border_ratio)
        max_fill_ratio = float(args.ultra_tiny_max_fill_ratio)
        min_corridor_ratio = float(args.ultra_tiny_min_corridor_ratio)
        max_overshoot_ratio = float(args.ultra_tiny_max_overshoot_ratio)
        max_angle_deg = float(args.ultra_tiny_max_angle_deg)
        max_border_touch_ratio = float(args.dual_max_border_touch)
    else:
        if short_span:
            pos_hit_tol_px = int(args.dual_short_pos_hit_tol_px)
            min_span_ratio = float(args.dual_short_min_span_ratio)
            min_white_ratio = float(args.dual_short_min_white_ratio)
            min_dark_border_ratio = float(args.dual_short_min_dark_border_ratio)
            max_fill_ratio = float(args.dual_short_max_fill_ratio)
            min_corridor_ratio = float(args.dual_short_min_corridor_ratio)
            max_overshoot_ratio = float(args.dual_short_max_overshoot_ratio)
            max_angle_deg = float(args.dual_max_angle_deg)
            max_border_touch_ratio = float(args.dual_max_border_touch)
        elif long_span:
            pos_hit_tol_px = int(args.dual_long_pos_hit_tol_px)
            min_span_ratio = float(args.dual_long_min_span_ratio)
            min_white_ratio = float(args.dual_long_min_white_ratio)
            min_dark_border_ratio = float(args.dual_long_min_dark_border_ratio)
            max_fill_ratio = float(args.dual_long_max_fill_ratio)
            min_corridor_ratio = float(args.dual_long_min_corridor_ratio)
            max_overshoot_ratio = float(args.dual_long_max_overshoot_ratio)
            max_angle_deg = float(args.dual_long_max_angle_deg)
            max_border_touch_ratio = float(args.dual_long_max_border_touch)
        else:
            pos_hit_tol_px = int(args.dual_pos_hit_tol_px)
            min_span_ratio = float(args.dual_min_span_ratio)
            min_white_ratio = float(args.dual_min_white_ratio)
            min_dark_border_ratio = float(args.dual_min_dark_border_ratio)
            max_fill_ratio = float(args.dual_max_fill_ratio)
            min_corridor_ratio = float(args.dual_min_corridor_ratio)
            max_overshoot_ratio = float(args.dual_max_overshoot_ratio)
            max_angle_deg = float(args.dual_max_angle_deg)
            max_border_touch_ratio = float(args.dual_max_border_touch)
    white_relaxed_u8 = build_relaxed_white_mask(frame_bgr, auto_cfg["white_cv"])
    corridor_gate_u8 = make_segment_corridor_mask(
        shape_hw=frame_bgr.shape[:2],
        p1=endpoint_a,
        p2=endpoint_b,
        corridor_px=float(args.dual_corridor_px) + float(args.dual_edge_expand_corridor_pad),
        endpoint_margin_px=float(args.dual_endpoint_margin_px) + float(args.dual_edge_expand_corridor_pad),
    )
    expand_allowed_u8 = np.logical_and(white_relaxed_u8 > 0, corridor_gate_u8 > 0).astype(np.uint8)
    passed: List[Dict] = []
    reject_counts: Dict[str, int] = {}
    for cand in candidates:
        mask = (cand["mask"] > 0).astype(np.uint8)
        if bool(args.dual_edge_expand):
            mask = constrained_edge_expand(
                mask_u8=mask,
                allowed_u8=expand_allowed_u8,
                iterations=int(args.dual_edge_expand_iter),
                kernel_size=int(args.dual_edge_expand_kernel),
            )
        if bool(args.dual_force_connect):
            mask = force_connect_endpoints(mask, frame_bgr, endpoint_a, endpoint_b, auto_cfg, args)
        if int(mask.sum()) <= 0:
            reject_counts["empty_mask"] = reject_counts.get("empty_mask", 0) + 1
            continue
        if not point_hit_with_tolerance(mask, endpoint_a[0], endpoint_a[1], pos_hit_tol_px):
            reject_counts["miss_pos"] = reject_counts.get("miss_pos", 0) + 1
            continue
        if not point_hit_with_tolerance(mask, endpoint_b[0], endpoint_b[1], pos_hit_tol_px):
            reject_counts["miss_pos"] = reject_counts.get("miss_pos", 0) + 1
            continue
        if bool(args.dual_require_neg_exclusion):
            if any(point_hit_with_tolerance(mask, nx, ny, int(args.dual_neg_hit_tol_px)) for nx, ny in negatives):
                reject_counts["neg_hit"] = reject_counts.get("neg_hit", 0) + 1
                continue
        fill_ratio, border_touch_ratio, area = compute_mask_fill_and_border(mask)
        if float(fill_ratio) > max_fill_ratio:
            reject_counts["high_fill"] = reject_counts.get("high_fill", 0) + 1
            continue
        if float(border_touch_ratio) > max_border_touch_ratio:
            reject_counts["high_border"] = reject_counts.get("high_border", 0) + 1
            continue
        white_ratio = compute_mask_white_support_ratio(frame_bgr, mask, auto_cfg["white_cv"])
        if white_ratio < min_white_ratio:
            reject_counts["low_white"] = reject_counts.get("low_white", 0) + 1
            continue
        dark_border_ratio = compute_mask_dark_border_ratio(
            frame_bgr,
            mask,
            ring_px=int(args.dual_dark_border_ring_px),
            dark_threshold=int(args.dual_dark_border_threshold),
        )
        if dark_border_ratio < min_dark_border_ratio:
            reject_counts["low_dark_border"] = reject_counts.get("low_dark_border", 0) + 1
            continue
        angle_deg, span_ratio = compute_angle_and_span(mask, endpoint_a, endpoint_b)
        if angle_deg is None:
            reject_counts["low_span"] = reject_counts.get("low_span", 0) + 1
            continue
        if float(angle_deg) > max_angle_deg:
            reject_counts["angle_mismatch"] = reject_counts.get("angle_mismatch", 0) + 1
            continue
        if float(span_ratio) < min_span_ratio:
            reject_counts["low_span"] = reject_counts.get("low_span", 0) + 1
            continue
        corr = compute_segment_corridor_metrics(
            mask_u8=mask,
            p1=endpoint_a,
            p2=endpoint_b,
            corridor_px=float(args.dual_corridor_px),
            endpoint_margin_px=float(args.dual_endpoint_margin_px),
        )
        if corr is None:
            reject_counts["off_axis"] = reject_counts.get("off_axis", 0) + 1
            continue
        corridor_ratio, overshoot_ratio = corr
        if corridor_ratio < min_corridor_ratio:
            reject_counts["off_axis"] = reject_counts.get("off_axis", 0) + 1
            continue
        if overshoot_ratio > max_overshoot_ratio:
            reject_counts["overshoot"] = reject_counts.get("overshoot", 0) + 1
            continue
        enriched = dict(cand)
        enriched["mask"] = mask
        enriched["fill_ratio"] = float(fill_ratio)
        enriched["border_touch_ratio"] = float(border_touch_ratio)
        enriched["area"] = int(area)
        enriched["_span_ratio"] = float(span_ratio)
        enriched["_corridor_ratio"] = float(corridor_ratio)
        enriched["_overshoot_ratio"] = float(overshoot_ratio)
        enriched["_dark_border_ratio"] = float(dark_border_ratio)
        passed.append(enriched)
    if not passed:
        if not reject_counts:
            return None, "strict_reject"
        return None, max(reject_counts.items(), key=lambda kv: kv[1])[0]
    return rank_dual_candidates(passed)[0], "ok"


def compute_dual_roi(endpoint_a: Tuple[int, int], endpoint_b: Tuple[int, int], margin: int, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(endpoint_a[0], endpoint_b[0]) - int(margin))
    y1 = max(0, min(endpoint_a[1], endpoint_b[1]) - int(margin))
    x2 = min(frame_w - 1, max(endpoint_a[0], endpoint_b[0]) + int(margin))
    y2 = min(frame_h - 1, max(endpoint_a[1], endpoint_b[1]) + int(margin))
    if x2 <= x1:
        x2 = min(frame_w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_h - 1, y1 + 1)
    return x1, y1, x2, y2


def remap_points_to_roi(points: List[Dict], roi_xyxy: Tuple[int, int, int, int], scale: float, roi_scaled_w: int, roi_scaled_h: int) -> List[Dict]:
    x1, y1, x2, y2 = roi_xyxy
    out: List[Dict] = []
    for p in points:
        x, y = int(p["x"]), int(p["y"])
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            continue
        sx = int(round((x - x1) * float(scale)))
        sy = int(round((y - y1) * float(scale)))
        sx = max(0, min(roi_scaled_w - 1, sx))
        sy = max(0, min(roi_scaled_h - 1, sy))
        out.append({"x": sx, "y": sy, "label": int(p["label"])})
    return out


def detect_dual_cv_candidates(
    frame_bgr: np.ndarray,
    endpoint_a: Tuple[int, int],
    endpoint_b: Tuple[int, int],
    auto_cfg: Dict,
    profile_cfg: Dict,
    args,
) -> List[Dict]:
    if not bool(args.dual_cv_rescue):
        return []
    h, w = frame_bgr.shape[:2]
    click_span = float(math.hypot(float(endpoint_b[0] - endpoint_a[0]), float(endpoint_b[1] - endpoint_a[1])))
    ultra_tiny = is_ultra_tiny_span(click_span, args)
    roi = compute_dual_roi(endpoint_a, endpoint_b, int(args.dual_roi_margin), w, h)
    rx1, ry1, rx2, ry2 = roi
    crop = frame_bgr[ry1 : ry2 + 1, rx1 : rx2 + 1]
    if crop.size == 0:
        return []
    cv_base_scale = float(args.ultra_tiny_cv_upscale) if ultra_tiny else float(args.dual_cv_upscale)
    cv_max_scale = float(args.ultra_tiny_cv_upscale_max) if ultra_tiny else float(args.dual_cv_upscale_max)
    scale = adaptive_upscale_for_mode(click_span, cv_base_scale, cv_max_scale, args, ultra_tiny)
    crop_scaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else crop
    ch, cw = crop_scaled.shape[:2]
    ax = max(0, min(cw - 1, int(round((endpoint_a[0] - rx1) * scale))))
    ay = max(0, min(ch - 1, int(round((endpoint_a[1] - ry1) * scale))))
    bx = max(0, min(cw - 1, int(round((endpoint_b[0] - rx1) * scale))))
    by = max(0, min(ch - 1, int(round((endpoint_b[1] - ry1) * scale))))

    cv_s_relax = int(args.ultra_tiny_cv_s_relax) if ultra_tiny else int(args.dual_cv_s_relax)
    cv_v_relax = int(args.ultra_tiny_cv_v_relax) if ultra_tiny else int(args.dual_cv_v_relax)
    cv_corridor_ratio = float(args.ultra_tiny_cv_corridor_ratio) if ultra_tiny else float(args.dual_cv_corridor_ratio)
    white_cfg = auto_cfg["white_cv"]
    hsv = cv2.cvtColor(crop_scaled, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(
        hsv,
        (
            int(white_cfg.get("h_min", 0)),
            max(0, int(white_cfg.get("s_min", 0)) - cv_s_relax),
            max(0, int(white_cfg.get("v_min", 180)) - cv_v_relax),
        ),
        (
            int(white_cfg.get("h_max", 180)),
            min(255, int(white_cfg.get("s_max", 80)) + cv_s_relax),
            int(white_cfg.get("v_max", 255)),
        ),
    )
    k = max(1, int(args.dual_cv_kernel))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), dtype=np.uint8)
    if int(args.dual_cv_close_iter) > 0:
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=int(args.dual_cv_close_iter))
    white_u8 = (white > 0).astype(np.uint8)

    span_scaled = max(1.0, click_span * scale)
    corridor_px = max(
        2.0,
        min(
            float(args.dual_corridor_px) * scale,
            span_scaled * cv_corridor_ratio,
        ),
    )
    corridor = make_segment_corridor_mask(
        shape_hw=crop_scaled.shape[:2],
        p1=(ax, ay),
        p2=(bx, by),
        corridor_px=float(corridor_px),
        endpoint_margin_px=float(args.dual_endpoint_margin_px) * scale,
    )
    local = np.logical_and(white_u8 > 0, corridor > 0).astype(np.uint8)
    if int(args.dual_cv_bridge_iter) > 0:
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, kernel, iterations=int(args.dual_cv_bridge_iter))
    if int(local.sum()) <= 0:
        return []

    local_masks: List[np.ndarray] = [local]
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(local, connectivity=8)
    min_area = max(1, int(args.dual_cv_min_area))
    for lid in range(1, int(n_labels)):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == lid).astype(np.uint8)
        local_masks.append(comp)

    rescue_profile = build_dual_prompt_profile(build_rescue_profile(profile_cfg))
    out: List[Dict] = []
    roi_h = ry2 - ry1 + 1
    roi_w = rx2 - rx1 + 1
    for local_mask in local_masks:
        if scale > 1.0:
            local_mask = cv2.resize(local_mask.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((h, w), dtype=np.uint8)
        full[ry1 : ry2 + 1, rx1 : rx2 + 1] = (local_mask > 0).astype(np.uint8)
        valid = validate_mask(full, None, rescue_profile, enforce_score=False)
        if valid is None:
            continue
        valid["source"] = "cv_dual_roi"
        valid["scale_used"] = float(scale)
        out.append(valid)
    return out


def detect_dual_center_rescue_candidates(
    frame_bgr: np.ndarray,
    endpoint_a: Tuple[int, int],
    endpoint_b: Tuple[int, int],
    model,
    processor,
    prompt: str,
    profile_cfg: Dict,
    auto_cfg: Dict,
    device: str,
    args,
) -> List[Dict]:
    if not bool(args.dual_center_rescue):
        return []

    click_span = float(math.hypot(float(endpoint_b[0] - endpoint_a[0]), float(endpoint_b[1] - endpoint_a[1])))
    n_points = max(1, min(7, int(args.dual_center_rescue_points)))

    # Prefer center first, then additional interior points.
    t_values: List[float] = [0.5]
    if n_points > 1:
        extra = np.linspace(0.2, 0.8, n_points - 1)
        for t in extra.tolist():
            if abs(float(t) - 0.5) > 1e-6:
                t_values.append(float(t))

    h, w = frame_bgr.shape[:2]
    crop_auto = int(round(click_span * float(args.dual_center_rescue_crop_span_ratio)))
    rescue_crop = max(int(args.dual_center_rescue_min_crop), crop_auto)
    rescue_crop = min(int(args.dual_center_rescue_max_crop), rescue_crop)
    rescue_crop = max(64, min(max(h, w), rescue_crop))
    rescue_upscale = max(1.0, float(args.dual_center_rescue_upscale))
    use_tiles = bool(args.tile_small_lines)

    out: List[Dict] = []
    seen_points = set()
    for t in t_values:
        x = int(round((1.0 - t) * float(endpoint_a[0]) + t * float(endpoint_b[0])))
        y = int(round((1.0 - t) * float(endpoint_a[1]) + t * float(endpoint_b[1])))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        if (x, y) in seen_points:
            continue
        seen_points.add((x, y))
        cand = detect_click_rescue(
            frame_bgr=frame_bgr,
            click_xy=(x, y),
            model=model,
            processor=processor,
            prompt=prompt,
            profile_cfg=profile_cfg,
            auto_cfg=auto_cfg,
            device=device,
            use_tiles=use_tiles,
            rescue_crop=int(rescue_crop),
            rescue_upscale=float(rescue_upscale),
        )
        if cand is None:
            continue
        c = dict(cand)
        c["source"] = f"{cand.get('source', 'sam')}_dual_center"
        c["center_t"] = float(t)
        out.append(c)
    return out


def detect_dual_prompt_mask(frame_bgr: np.ndarray, prompt_points: List[Dict], model, processor, prompt: str, profile_cfg: Dict, auto_cfg: Dict, device: str, args) -> Tuple[Optional[Dict], str]:
    positives = [p for p in prompt_points if int(p["label"]) == POS_LABEL]
    negatives = [p for p in prompt_points if int(p["label"]) == NEG_LABEL]
    if len(positives) < 2:
        return None, "need_two_positive_clicks"
    p1, p2 = positives[-2], positives[-1]
    endpoint_a = (int(p1["x"]), int(p1["y"]))
    endpoint_b = (int(p2["x"]), int(p2["y"]))
    click_span = float(math.hypot(float(endpoint_b[0] - endpoint_a[0]), float(endpoint_b[1] - endpoint_a[1])))
    ultra_tiny = is_ultra_tiny_span(click_span, args)
    neg_xy = [(int(n["x"]), int(n["y"])) for n in negatives]
    active_prompts = [p1, p2] + negatives
    pos_box_px, neg_box_px = adaptive_prompt_box_sizes(click_span, int(args.prompt_box_pos_px), int(args.prompt_box_neg_px))

    # Ultra tiny mode: prefer endpoint-constrained path first, and optionally skip broad SAM passes.
    path_cand_first = None
    reason_path_first = "path_not_run"
    if ultra_tiny:
        path_cand_first, reason_path_first = detect_dual_path_candidate(frame_bgr, endpoint_a, endpoint_b, neg_xy, auto_cfg, args)
        if path_cand_first is not None:
            picked_path_first, reason_path_sel = select_dual_candidate(frame_bgr, [path_cand_first], endpoint_a, endpoint_b, neg_xy, auto_cfg, args)
            if picked_path_first is not None:
                return picked_path_first, "ok"
            if reason_path_sel:
                reason_path_first = reason_path_sel
        if bool(args.ultra_tiny_path_only):
            return None, reason_path_first

    reason = "no_candidate"
    if not (ultra_tiny and bool(args.ultra_tiny_skip_full_sam)):
        dual_prompt_profile = build_dual_prompt_profile(profile_cfg)
        full_cands = run_sam_text_with_boxes(
            frame_bgr,
            model,
            processor,
            prompt,
            active_prompts,
            pos_box_px,
            neg_box_px,
            dual_prompt_profile,
            device,
            "sam_prompt_full",
        )
        picked, reason = select_dual_candidate(frame_bgr, full_cands, endpoint_a, endpoint_b, neg_xy, auto_cfg, args)
        if picked is not None:
            return picked, "ok"
    if not bool(args.dual_roi_rescue):
        return None, reason_path_first if ultra_tiny and reason_path_first != "path_not_run" else reason

    h, w = frame_bgr.shape[:2]
    roi = compute_dual_roi(endpoint_a, endpoint_b, int(args.dual_roi_margin), w, h)
    rx1, ry1, rx2, ry2 = roi
    crop = frame_bgr[ry1 : ry2 + 1, rx1 : rx2 + 1]
    if crop.size == 0:
        return None, reason
    roi_base_scale = float(args.ultra_tiny_roi_upscale) if ultra_tiny else float(args.dual_roi_upscale)
    roi_max_scale = float(args.ultra_tiny_roi_upscale_max) if ultra_tiny else float(args.dual_roi_upscale_max)
    scale = adaptive_upscale_for_mode(click_span, roi_base_scale, roi_max_scale, args, ultra_tiny)
    crop_scaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else crop
    roi_prompts = remap_points_to_roi(active_prompts, roi, scale, crop_scaled.shape[1], crop_scaled.shape[0])
    if sum(1 for p in roi_prompts if int(p["label"]) == POS_LABEL) < 2:
        return None, reason
    roi_pos_box_px = upscale_prompt_box_px(pos_box_px, scale, args)
    roi_neg_box_px = upscale_prompt_box_px(neg_box_px, scale, args)
    roi_profile = build_dual_prompt_profile(build_rescue_profile(profile_cfg))
    roi_cands_local = run_sam_text_with_boxes(crop_scaled, model, processor, prompt, roi_prompts, roi_pos_box_px, roi_neg_box_px, roi_profile, device, "sam_prompt_roi_local")
    roi_cands = map_crop_candidates_to_full(roi_cands_local, roi, (h, w), crop_scaled.shape[:2])
    for c in roi_cands:
        c["source"] = "sam_prompt_roi"
        c["scale_used"] = float(scale)
    picked_roi, reason_roi = select_dual_candidate(frame_bgr, roi_cands, endpoint_a, endpoint_b, neg_xy, auto_cfg, args)
    if picked_roi is not None:
        return picked_roi, "ok"

    cv_cands = detect_dual_cv_candidates(frame_bgr, endpoint_a, endpoint_b, auto_cfg, profile_cfg, args)
    picked_cv, reason_cv = select_dual_candidate(frame_bgr, cv_cands, endpoint_a, endpoint_b, neg_xy, auto_cfg, args)
    if picked_cv is not None:
        return picked_cv, "ok"

    path_cand, reason_path = detect_dual_path_candidate(frame_bgr, endpoint_a, endpoint_b, neg_xy, auto_cfg, args)
    if path_cand is not None:
        picked_path, reason_path_sel = select_dual_candidate(frame_bgr, [path_cand], endpoint_a, endpoint_b, neg_xy, auto_cfg, args)
        if picked_path is not None:
            return picked_path, "ok"
        if reason_path_sel:
            reason_path = reason_path_sel

    center_cands = detect_dual_center_rescue_candidates(
        frame_bgr=frame_bgr,
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        model=model,
        processor=processor,
        prompt=prompt,
        profile_cfg=profile_cfg,
        auto_cfg=auto_cfg,
        device=device,
        args=args,
    )
    picked_center, reason_center = select_dual_candidate(frame_bgr, center_cands, endpoint_a, endpoint_b, neg_xy, auto_cfg, args)
    if picked_center is not None:
        return picked_center, "ok"

    if reason_path and reason_path not in {"ok", "no_candidate"}:
        return None, reason_path
    if reason_center and reason_center != "no_candidate":
        return None, reason_center
    if reason_cv and reason_cv != "no_candidate":
        return None, reason_cv
    return None, reason_roi if reason_roi else reason


def format_prompt_points_for_meta(prompt_points: List[Dict]) -> List[Dict]:
    return [{"x": int(p["x"]), "y": int(p["y"]), "label": "pos" if int(p["label"]) == POS_LABEL else "neg"} for p in prompt_points]


def render_save_style_preview(frame_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    preview = frame_bgr.copy()
    mask = (mask_u8 > 0).astype(np.uint8)
    color = preview.copy()
    color[mask > 0] = (0, 0, 255)
    preview = cv2.addWeighted(preview, 0.65, color, 0.35, 0.0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (0, 255, 255), 2)
    return preview


def parse_prompt_snapshot(prompt_points_snapshot: List[Dict]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    pos: List[Tuple[int, int]] = []
    neg: List[Tuple[int, int]] = []
    for p in prompt_points_snapshot:
        try:
            x = int(p.get("x", -1))
            y = int(p.get("y", -1))
        except Exception:
            continue
        lbl = str(p.get("label", "")).strip().lower()
        if lbl == "pos":
            pos.append((x, y))
        elif lbl == "neg":
            neg.append((x, y))
    return pos, neg


def evaluate_pending_save_quality(
    pending: Dict,
    auto_cfg: Dict,
    args,
) -> Tuple[bool, str]:
    if not bool(args.save_precision_gate):
        return True, "gate_disabled"
    mask = (pending["mask_u8"] > 0).astype(np.uint8)
    frame_bgr = pending["frame_bgr"]
    h, w = mask.shape[:2]
    if int(mask.sum()) <= 0:
        return False, "empty_mask"

    prompt_snapshot = pending.get("prompt_points_snapshot", [])
    pos, neg = parse_prompt_snapshot(prompt_snapshot if isinstance(prompt_snapshot, list) else [])
    ultra_tiny = False
    if len(pos) >= 2:
        span_px = math.hypot(float(pos[-1][0] - pos[-2][0]), float(pos[-1][1] - pos[-2][1]))
        ultra_tiny = is_ultra_tiny_span(float(span_px), args)

    save_min_white_ratio = float(args.save_min_white_ratio)
    save_max_fill_ratio = float(args.save_max_fill_ratio)
    save_max_area_ratio = float(args.save_max_area_ratio)
    save_max_angle_deg = float(args.save_max_angle_deg)
    save_min_span_ratio = float(args.save_min_span_ratio)
    save_min_corridor_ratio = float(args.save_min_corridor_ratio)
    save_max_overshoot_ratio = float(args.save_max_overshoot_ratio)
    save_min_dark_border_ratio = float(args.save_min_dark_border_ratio)
    if ultra_tiny:
        save_min_white_ratio = max(save_min_white_ratio, float(args.ultra_tiny_save_min_white_ratio))
        save_max_fill_ratio = min(save_max_fill_ratio, float(args.ultra_tiny_save_max_fill_ratio))
        save_max_area_ratio = min(save_max_area_ratio, float(args.ultra_tiny_save_max_area_ratio))
        save_max_angle_deg = min(save_max_angle_deg, float(args.ultra_tiny_save_max_angle_deg))
        save_min_span_ratio = max(save_min_span_ratio, float(args.ultra_tiny_save_min_span_ratio))
        save_min_corridor_ratio = max(save_min_corridor_ratio, float(args.ultra_tiny_save_min_corridor_ratio))
        save_max_overshoot_ratio = min(save_max_overshoot_ratio, float(args.ultra_tiny_save_max_overshoot_ratio))
        save_min_dark_border_ratio = max(save_min_dark_border_ratio, float(args.ultra_tiny_save_min_dark_border_ratio))

    fill_ratio, border_touch_ratio, area = compute_mask_fill_and_border(mask)
    if float(fill_ratio) > save_max_fill_ratio:
        return False, "save_high_fill"
    if float(border_touch_ratio) > float(args.save_max_border_touch):
        return False, "save_high_border"
    area_ratio = float(area) / float(max(1, h * w))
    if area_ratio > save_max_area_ratio:
        return False, "save_large_area"

    white_ratio = compute_mask_white_support_ratio(frame_bgr, mask, auto_cfg["white_cv"])
    if white_ratio < save_min_white_ratio:
        return False, "save_low_white"
    dark_border_ratio = compute_mask_dark_border_ratio(
        frame_bgr,
        mask,
        ring_px=int(args.save_dark_border_ring_px),
        dark_threshold=int(args.save_dark_border_threshold),
    )
    if dark_border_ratio < save_min_dark_border_ratio:
        return False, "save_low_dark_border"

    if len(pos) >= 2:
        endpoint_a, endpoint_b = pos[-2], pos[-1]
        if not point_hit_with_tolerance(mask, endpoint_a[0], endpoint_a[1], int(args.save_pos_hit_tol_px)):
            return False, "save_miss_pos"
        if not point_hit_with_tolerance(mask, endpoint_b[0], endpoint_b[1], int(args.save_pos_hit_tol_px)):
            return False, "save_miss_pos"
        if bool(args.dual_require_neg_exclusion):
            if any(point_hit_with_tolerance(mask, nx, ny, int(args.save_neg_hit_tol_px)) for nx, ny in neg):
                return False, "save_neg_hit"
        angle_deg, span_ratio = compute_angle_and_span(mask, endpoint_a, endpoint_b)
        if angle_deg is None:
            return False, "save_low_span"
        if float(angle_deg) > save_max_angle_deg:
            return False, "save_angle_mismatch"
        if float(span_ratio) < save_min_span_ratio:
            return False, "save_low_span"
        corr = compute_segment_corridor_metrics(
            mask_u8=mask,
            p1=endpoint_a,
            p2=endpoint_b,
            corridor_px=float(args.dual_corridor_px),
            endpoint_margin_px=float(args.dual_endpoint_margin_px),
        )
        if corr is None:
            return False, "save_off_axis"
        corridor_ratio, overshoot_ratio = corr
        if corridor_ratio < save_min_corridor_ratio:
            return False, "save_off_axis"
        if overshoot_ratio > save_max_overshoot_ratio:
            return False, "save_overshoot"
    return True, "ok"


def draw_prompt_points(viz: np.ndarray, prompt_points: List[Dict]) -> None:
    # Keep click markers visually distinct from mask/detection overlays.
    pos_color = (255, 0, 255)   # magenta
    neg_color = (0, 64, 255)    # orange-red
    link_color = (255, 180, 0)  # amber-blue contrast
    for i, p in enumerate(prompt_points):
        x, y, lbl = int(p["x"]), int(p["y"]), int(p["label"])
        color = pos_color if lbl == POS_LABEL else neg_color
        cv2.circle(viz, (x, y), 6, color, 2, cv2.LINE_AA)
        cv2.putText(viz, str(i + 1), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    positives = [p for p in prompt_points if int(p["label"]) == POS_LABEL]
    if len(positives) >= 2:
        p1, p2 = positives[-2], positives[-1]
        cv2.line(viz, (int(p1["x"]), int(p1["y"])), (int(p2["x"]), int(p2["y"])), link_color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label aiming guideline masks from an image folder.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--recursive", type=str2bool, default=False)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--resume-last", type=str2bool, default=True)
    parser.add_argument("--state-path", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--lora-path", type=str, default="none")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--profile", type=str, choices=["balanced", "precision"], default="precision")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["label", "auto", "hybrid"], default="label")
    parser.add_argument("--tile-small-lines", type=str2bool, default=True)
    parser.add_argument("--display-scale", type=float, default=None)
    parser.add_argument("--auto-next-on-save", type=str2bool, default=True)
    parser.add_argument("--click-strategy", type=str, choices=["dual", "single"], default="dual")
    parser.add_argument("--prompt-box-pos-px", type=int, default=6)
    parser.add_argument("--prompt-box-neg-px", type=int, default=8)
    parser.add_argument("--dual-max-angle-deg", type=float, default=18.0)
    parser.add_argument("--dual-min-span-ratio", type=float, default=0.90)
    parser.add_argument("--dual-corridor-px", type=float, default=14.0)
    parser.add_argument("--dual-endpoint-margin-px", type=float, default=8.0)
    parser.add_argument("--dual-min-corridor-ratio", type=float, default=0.72)
    parser.add_argument("--dual-max-overshoot-ratio", type=float, default=0.80)
    parser.add_argument("--dual-edge-expand", type=str2bool, default=False)
    parser.add_argument("--dual-edge-expand-iter", type=int, default=1)
    parser.add_argument("--dual-edge-expand-kernel", type=int, default=3)
    parser.add_argument("--dual-edge-expand-corridor-pad", type=float, default=3.0)
    parser.add_argument("--dual-force-connect", type=str2bool, default=False)
    parser.add_argument("--dual-force-connect-thickness", type=int, default=2)
    parser.add_argument("--dual-force-connect-corridor-ratio", type=float, default=0.42)
    parser.add_argument("--dual-force-s-relax", type=int, default=48)
    parser.add_argument("--dual-force-v-relax", type=int, default=48)
    parser.add_argument("--dual-force-close-iter", type=int, default=1)
    parser.add_argument("--dual-min-white-ratio", type=float, default=0.50)
    parser.add_argument("--dual-min-dark-border-ratio", type=float, default=0.08)
    parser.add_argument("--dual-max-fill-ratio", type=float, default=0.42)
    parser.add_argument("--dual-max-border-touch", type=float, default=0.40)
    parser.add_argument("--dual-dark-border-ring-px", type=int, default=1)
    parser.add_argument("--dual-dark-border-threshold", type=int, default=122)
    parser.add_argument("--dual-pos-hit-tol-px", type=int, default=4)
    parser.add_argument("--dual-neg-hit-tol-px", type=int, default=2)
    parser.add_argument("--dual-short-span-px", type=float, default=42.0)
    parser.add_argument("--dual-short-pos-hit-tol-px", type=int, default=5)
    parser.add_argument("--dual-short-min-span-ratio", type=float, default=0.62)
    parser.add_argument("--dual-short-min-corridor-ratio", type=float, default=0.48)
    parser.add_argument("--dual-short-max-overshoot-ratio", type=float, default=1.00)
    parser.add_argument("--dual-short-min-white-ratio", type=float, default=0.42)
    parser.add_argument("--dual-short-min-dark-border-ratio", type=float, default=0.04)
    parser.add_argument("--dual-short-max-fill-ratio", type=float, default=0.60)
    parser.add_argument("--dual-long-span-px", type=float, default=110.0)
    parser.add_argument("--dual-long-pos-hit-tol-px", type=int, default=5)
    parser.add_argument("--dual-long-min-span-ratio", type=float, default=0.72)
    parser.add_argument("--dual-long-min-corridor-ratio", type=float, default=0.42)
    parser.add_argument("--dual-long-max-overshoot-ratio", type=float, default=1.20)
    parser.add_argument("--dual-long-min-white-ratio", type=float, default=0.40)
    parser.add_argument("--dual-long-min-dark-border-ratio", type=float, default=0.03)
    parser.add_argument("--dual-long-max-fill-ratio", type=float, default=0.68)
    parser.add_argument("--dual-long-max-angle-deg", type=float, default=24.0)
    parser.add_argument("--dual-long-max-border-touch", type=float, default=0.60)
    parser.add_argument("--dual-require-neg-exclusion", type=str2bool, default=True)
    parser.add_argument("--dual-roi-rescue", type=str2bool, default=True)
    parser.add_argument("--dual-roi-margin", type=int, default=96)
    parser.add_argument("--dual-roi-upscale", type=float, default=3.0)
    parser.add_argument("--dual-adaptive-upscale", type=str2bool, default=True)
    parser.add_argument("--dual-adaptive-ref-span-px", type=float, default=56.0)
    parser.add_argument("--dual-adaptive-min-span-px", type=float, default=6.0)
    parser.add_argument("--dual-adaptive-gamma", type=float, default=1.25)
    parser.add_argument("--dual-roi-upscale-max", type=float, default=10.0)
    parser.add_argument("--dual-cv-upscale-max", type=float, default=12.0)
    parser.add_argument("--dual-path-upscale-max", type=float, default=14.0)
    parser.add_argument("--dual-roi-prompt-scale", type=str2bool, default=True)
    parser.add_argument("--dual-roi-prompt-scale-gamma", type=float, default=0.95)
    parser.add_argument("--dual-roi-prompt-max-px", type=int, default=36)
    parser.add_argument("--ultra-tiny-mode", type=str2bool, default=False)
    parser.add_argument("--ultra-tiny-span-px", type=float, default=14.0)
    parser.add_argument("--ultra-tiny-path-only", type=str2bool, default=True)
    parser.add_argument("--ultra-tiny-skip-full-sam", type=str2bool, default=True)
    parser.add_argument("--ultra-tiny-adaptive-ref-span-px", type=float, default=24.0)
    parser.add_argument("--ultra-tiny-adaptive-min-span-px", type=float, default=3.0)
    parser.add_argument("--ultra-tiny-adaptive-gamma", type=float, default=2.0)
    parser.add_argument("--ultra-tiny-roi-upscale", type=float, default=8.0)
    parser.add_argument("--ultra-tiny-roi-upscale-max", type=float, default=18.0)
    parser.add_argument("--ultra-tiny-cv-upscale", type=float, default=10.0)
    parser.add_argument("--ultra-tiny-cv-upscale-max", type=float, default=20.0)
    parser.add_argument("--ultra-tiny-path-upscale", type=float, default=12.0)
    parser.add_argument("--ultra-tiny-path-upscale-max", type=float, default=24.0)
    parser.add_argument("--ultra-tiny-path-max-side", type=int, default=1280)
    parser.add_argument("--ultra-tiny-path-roi-margin", type=int, default=160)
    parser.add_argument("--ultra-tiny-path-corridor-ratio", type=float, default=0.36)
    parser.add_argument("--ultra-tiny-path-corridor-pad", type=float, default=6.0)
    parser.add_argument("--ultra-tiny-path-s-relax", type=int, default=60)
    parser.add_argument("--ultra-tiny-path-v-relax", type=int, default=60)
    parser.add_argument("--ultra-tiny-path-white-seed-thr", type=float, default=0.20)
    parser.add_argument("--ultra-tiny-path-white-mask-thr", type=float, default=0.17)
    parser.add_argument("--ultra-tiny-path-min-white-support", type=float, default=0.26)
    parser.add_argument("--ultra-tiny-path-min-straightness", type=float, default=0.78)
    parser.add_argument("--ultra-tiny-path-min-length-ratio", type=float, default=0.80)
    parser.add_argument("--ultra-tiny-path-min-halfwidth-px", type=int, default=1)
    parser.add_argument("--ultra-tiny-path-max-halfwidth-px", type=int, default=3)
    parser.add_argument("--ultra-tiny-path-min-support-fraction", type=float, default=0.34)
    parser.add_argument("--ultra-tiny-path-close-iter", type=int, default=0)
    parser.add_argument("--ultra-tiny-path-min-gray", type=int, default=24)
    parser.add_argument("--ultra-tiny-path-neg-radius-px", type=float, default=6.0)
    parser.add_argument("--ultra-tiny-cv-s-relax", type=int, default=62)
    parser.add_argument("--ultra-tiny-cv-v-relax", type=int, default=62)
    parser.add_argument("--ultra-tiny-cv-corridor-ratio", type=float, default=0.50)
    parser.add_argument("--ultra-tiny-pos-hit-tol-px", type=int, default=4)
    parser.add_argument("--ultra-tiny-min-span-ratio", type=float, default=0.72)
    parser.add_argument("--ultra-tiny-min-corridor-ratio", type=float, default=0.58)
    parser.add_argument("--ultra-tiny-max-overshoot-ratio", type=float, default=0.90)
    parser.add_argument("--ultra-tiny-min-white-ratio", type=float, default=0.44)
    parser.add_argument("--ultra-tiny-min-dark-border-ratio", type=float, default=0.02)
    parser.add_argument("--ultra-tiny-max-fill-ratio", type=float, default=0.50)
    parser.add_argument("--ultra-tiny-max-angle-deg", type=float, default=16.0)
    parser.add_argument("--dual-cv-rescue", type=str2bool, default=False)
    parser.add_argument("--dual-cv-upscale", type=float, default=4.0)
    parser.add_argument("--dual-cv-s-relax", type=int, default=48)
    parser.add_argument("--dual-cv-v-relax", type=int, default=42)
    parser.add_argument("--dual-cv-corridor-ratio", type=float, default=0.55)
    parser.add_argument("--dual-cv-kernel", type=int, default=3)
    parser.add_argument("--dual-cv-close-iter", type=int, default=1)
    parser.add_argument("--dual-cv-bridge-iter", type=int, default=1)
    parser.add_argument("--dual-cv-min-area", type=int, default=3)
    parser.add_argument("--dual-path-rescue", type=str2bool, default=True)
    parser.add_argument("--dual-path-upscale", type=float, default=6.0)
    parser.add_argument("--dual-path-max-side", type=int, default=1024)
    parser.add_argument("--dual-path-roi-margin", type=int, default=120)
    parser.add_argument("--dual-path-corridor-ratio", type=float, default=0.44)
    parser.add_argument("--dual-path-corridor-pad", type=float, default=5.0)
    parser.add_argument("--dual-path-s-relax", type=int, default=52)
    parser.add_argument("--dual-path-v-relax", type=int, default=52)
    parser.add_argument("--dual-path-rail-offset-px", type=float, default=1.8)
    parser.add_argument("--dual-path-rail-outer-offset-px", type=float, default=2.8)
    parser.add_argument("--dual-path-rail-dark-gamma", type=float, default=1.15)
    parser.add_argument("--dual-path-white-seed-thr", type=float, default=0.26)
    parser.add_argument("--dual-path-white-mask-thr", type=float, default=0.22)
    parser.add_argument("--dual-path-min-white-support", type=float, default=0.30)
    parser.add_argument("--dual-path-min-rail-support", type=float, default=0.10)
    parser.add_argument("--dual-path-min-straightness", type=float, default=0.82)
    parser.add_argument("--dual-path-min-length-ratio", type=float, default=0.86)
    parser.add_argument("--dual-path-min-halfwidth-px", type=int, default=1)
    parser.add_argument("--dual-path-max-halfwidth-px", type=int, default=4)
    parser.add_argument("--dual-path-min-support-fraction", type=float, default=0.45)
    parser.add_argument("--dual-path-straighten-min-span-px", type=float, default=18.0)
    parser.add_argument("--dual-path-close-iter", type=int, default=1)
    parser.add_argument("--dual-path-min-gray", type=int, default=35)
    parser.add_argument("--dual-path-neg-radius-px", type=float, default=5.0)
    parser.add_argument("--dual-path-w-white", type=float, default=1.8)
    parser.add_argument("--dual-path-w-rail", type=float, default=1.3)
    parser.add_argument("--dual-path-w-center", type=float, default=1.2)
    parser.add_argument("--dual-path-w-dist", type=float, default=1.0)
    parser.add_argument("--dual-path-w-endpoint", type=float, default=0.8)
    parser.add_argument("--ultra-tiny-path-rail-offset-px", type=float, default=1.2)
    parser.add_argument("--ultra-tiny-path-rail-outer-offset-px", type=float, default=2.0)
    parser.add_argument("--ultra-tiny-path-min-rail-support", type=float, default=0.06)
    parser.add_argument("--ultra-tiny-path-w-rail", type=float, default=1.0)
    parser.add_argument("--save-precision-gate", type=str2bool, default=True)
    parser.add_argument("--save-min-white-ratio", type=float, default=0.48)
    parser.add_argument("--save-min-dark-border-ratio", type=float, default=0.06)
    parser.add_argument("--save-max-fill-ratio", type=float, default=0.58)
    parser.add_argument("--save-max-border-touch", type=float, default=0.45)
    parser.add_argument("--save-dark-border-ring-px", type=int, default=1)
    parser.add_argument("--save-dark-border-threshold", type=int, default=122)
    parser.add_argument("--save-max-area-ratio", type=float, default=0.020)
    parser.add_argument("--save-pos-hit-tol-px", type=int, default=4)
    parser.add_argument("--save-neg-hit-tol-px", type=int, default=2)
    parser.add_argument("--save-max-angle-deg", type=float, default=18.0)
    parser.add_argument("--save-min-span-ratio", type=float, default=0.74)
    parser.add_argument("--save-min-corridor-ratio", type=float, default=0.50)
    parser.add_argument("--save-max-overshoot-ratio", type=float, default=1.00)
    parser.add_argument("--ultra-tiny-save-min-white-ratio", type=float, default=0.54)
    parser.add_argument("--ultra-tiny-save-max-fill-ratio", type=float, default=0.46)
    parser.add_argument("--ultra-tiny-save-max-area-ratio", type=float, default=0.012)
    parser.add_argument("--ultra-tiny-save-max-angle-deg", type=float, default=15.0)
    parser.add_argument("--ultra-tiny-save-min-span-ratio", type=float, default=0.80)
    parser.add_argument("--ultra-tiny-save-min-corridor-ratio", type=float, default=0.62)
    parser.add_argument("--ultra-tiny-save-max-overshoot-ratio", type=float, default=0.86)
    parser.add_argument("--ultra-tiny-save-min-dark-border-ratio", type=float, default=0.03)
    parser.add_argument("--click-rescue", type=str2bool, default=True)
    parser.add_argument("--rescue-crop", type=int, default=260)
    parser.add_argument("--rescue-upscale", type=float, default=3.0)
    parser.add_argument("--dual-center-rescue", type=str2bool, default=True)
    parser.add_argument("--dual-center-rescue-points", type=int, default=3)
    parser.add_argument("--dual-center-rescue-crop-span-ratio", type=float, default=1.30)
    parser.add_argument("--dual-center-rescue-min-crop", type=int, default=260)
    parser.add_argument("--dual-center-rescue-max-crop", type=int, default=1280)
    parser.add_argument("--dual-center-rescue-upscale", type=float, default=3.0)
    parser.add_argument("--rescue-min-score", type=float, default=0.30)
    parser.add_argument("--rescue-min-aspect", type=float, default=2.20)
    parser.add_argument("--rescue-max-fill-ratio", type=float, default=0.45)
    parser.add_argument("--rescue-min-white-ratio", type=float, default=0.55)
    parser.add_argument("--rescue-require-cv-overlap", type=str2bool, default=True)
    parser.add_argument("--rescue-min-cv-iou", type=float, default=0.02)
    parser.add_argument("--zoom-max", type=float, default=12.0)
    parser.add_argument("--zoom-step", type=float, default=1.18)
    parser.add_argument("--zoom-reset-on-nav", type=str2bool, default=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    model_id = str(args.model_id or config["model_id"])
    prompt = str(args.prompt or config["default_prompt"])
    dataset_root = Path(args.dataset_root or config["paths"]["dataset_root"])
    dataset_dirs = ensure_dataset_layout(dataset_root)
    profile_cfg = config["profiles"][args.profile]
    auto_cfg = config["auto"]
    tiny_cfg = config["tiny_line"]
    display_scale = float(args.display_scale) if args.display_scale is not None else float(config.get("display_scale", 1.0))
    mode = str(args.mode)

    image_dir = Path(args.image_dir)
    images = list_images(image_dir, recursive=bool(args.recursive))
    if not images:
        raise RuntimeError(f"No images found in: {image_dir}")

    state_path = Path(args.state_path) if args.state_path else (dataset_root / "image_click_labeler_state.json")
    resume_state = load_resume_state(state_path)
    idx = resolve_start_index(images, image_dir, bool(args.recursive), args.start_index, bool(args.resume_last), resume_state)
    saved_total = int(resume_state.get("saved_total", 0)) if isinstance(resume_state.get("saved_total", 0), int) else 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_id}")
    model, processor = load_sam_model(model_id, args.lora_path, device)
    print(f"Device: {device}")
    print(f"Prompt: {prompt}")
    print(f"Profile: {args.profile}")
    print(f"Images: {len(images)} from {image_dir}")
    print(f"Start index: {idx + 1}/{len(images)}")
    print(
        "Controls: L-click add/select | R-click clear all | Wheel zoom | N pos/neg | "
        "Backspace undo | T dual/single | U ultra-tiny | V save-preview | S/Enter save | D/A nav | Shift+D/A jump 25 | M mode | 0 zoom reset | Q quit"
    )

    state: Dict = {
        "pending": None,
        "click_request": None,
        "view_zoom": 1.0,
        "view_center": None,
        "view_rect": (0, 0, 0, 0),
        "display_wh": (1, 1),
        "prompt_points": [],
        "click_label_mode": "pos",
        "click_strategy": str(args.click_strategy),
        "prompts_changed": False,
        "last_reject_reason": "",
        "preview_save_style": False,
    }

    cache: Dict = {"idx": None, "mode": None, "detections": []}
    status_msg = ""
    saved_count_session = 0
    mode_cycle = ["label", "hybrid", "auto"]
    last_loaded_idx: Optional[int] = None

    def on_mouse(event, x, y, flags, param):
        nonlocal status_msg
        src = display_to_source_view((int(x), int(y)), state["display_wh"], state["view_rect"])
        if event == cv2.EVENT_LBUTTONDOWN and src is not None:
            if state["click_strategy"] == "dual":
                lbl = POS_LABEL if state["click_label_mode"] == "pos" else NEG_LABEL
                state["prompt_points"].append({"x": int(src[0]), "y": int(src[1]), "label": int(lbl)})
                state["prompts_changed"] = True
                status_msg = f"Added {'POS' if lbl == POS_LABEL else 'NEG'} prompt"
            else:
                state["click_request"] = src
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["pending"] = None
            state["click_request"] = None
            state["prompt_points"] = []
            state["prompts_changed"] = False
            state["last_reject_reason"] = ""
            status_msg = "Selection and prompts cleared"
        elif event == cv2.EVENT_MBUTTONDOWN and src is not None:
            state["view_center"] = (float(src[0]), float(src[1]))
        elif event == cv2.EVENT_MOUSEWHEEL:
            if hasattr(cv2, "getMouseWheelDelta"):
                delta = int(cv2.getMouseWheelDelta(flags))
            else:
                delta = 120 if flags > 0 else -120
            if src is not None:
                state["view_center"] = (float(src[0]), float(src[1]))
            if delta > 0:
                state["view_zoom"] = min(float(args.zoom_max), float(state["view_zoom"]) * float(args.zoom_step))
            elif delta < 0:
                state["view_zoom"] = max(1.0, float(state["view_zoom"]) / float(args.zoom_step))
            status_msg = f"Zoom: x{state['view_zoom']:.2f}"

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        idx = clamp_idx(idx, len(images))
        image_path = images[idx]
        if last_loaded_idx != idx:
            state["pending"] = None
            state["click_request"] = None
            state["prompt_points"] = []
            state["prompts_changed"] = False
            state["last_reject_reason"] = ""
            if bool(args.zoom_reset_on_nav):
                state["view_zoom"] = 1.0
            state["view_center"] = None
            cache["idx"] = None
            cache["mode"] = None
            last_loaded_idx = idx
            save_resume_state(state_path, image_dir, bool(args.recursive), idx, image_path, saved_total)

        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            status_msg = f"Failed to read image: {image_path.name}"
            idx = clamp_idx(idx + 1, len(images))
            continue
        frame_bgr = np.ascontiguousarray(frame_bgr)
        frame_h, frame_w = frame_bgr.shape[:2]
        if state["view_center"] is None:
            state["view_center"] = (frame_w * 0.5, frame_h * 0.5)
        vx1, vy1, vx2, vy2 = compute_view_rect(frame_w, frame_h, float(state["view_zoom"]), (float(state["view_center"][0]), float(state["view_center"][1])))
        state["view_rect"] = (vx1, vy1, vx2, vy2)
        disp_w = max(320, int(round(frame_w * max(0.05, float(display_scale)))))
        disp_h = max(180, int(round(frame_h * max(0.05, float(display_scale)))))
        state["display_wh"] = (disp_w, disp_h)

        auto_enabled = mode in {"auto", "hybrid"}
        use_tiles = bool(args.tile_small_lines)
        if auto_enabled and (cache["idx"] != idx or cache["mode"] != mode):
            cache["detections"] = detect_candidates(
                frame_bgr=frame_bgr,
                model=model,
                processor=processor,
                prompt=prompt,
                profile_cfg=profile_cfg,
                auto_cfg=auto_cfg,
                device=device,
                use_tiles=use_tiles,
                include_cv=True,
            )
            cache["idx"] = idx
            cache["mode"] = mode
        auto_dets: List[Dict] = list(cache["detections"]) if auto_enabled else []

        if state["click_strategy"] == "dual" and bool(state.pop("prompts_changed", False)):
            picked_dual, reject_reason = detect_dual_prompt_mask(
                frame_bgr,
                list(state["prompt_points"]),
                model,
                processor,
                prompt,
                profile_cfg,
                auto_cfg,
                device,
                args,
            )
            if picked_dual is not None:
                state["pending"] = {
                    "mask_u8": picked_dual["mask"].copy(),
                    "frame_bgr": frame_bgr.copy(),
                    "capture_upscale": 1.0,
                    "roi_xyxy": (0, 0, frame_w - 1, frame_h - 1),
                    "score": picked_dual.get("score"),
                    "source": picked_dual.get("source", "sam_prompt"),
                    "refined": False,
                    "prompt_strategy": "dual",
                    "prompt_points_snapshot": format_prompt_points_for_meta(list(state["prompt_points"])),
                }
                state["last_reject_reason"] = ""
                score_txt = picked_dual.get("score")
                score_part = "n/a" if score_txt is None else f"{float(score_txt):.2f}"
                scale_part = ""
                if picked_dual.get("scale_used") is not None:
                    scale_part = f" | local_scale=x{float(picked_dual['scale_used']):.2f}"
                status_msg = f"Mask ready ({picked_dual.get('source', 'sam_prompt')}) | score={score_part}{scale_part}"
            else:
                state["pending"] = None
                state["last_reject_reason"] = str(reject_reason)
                status_msg = f"Strict reject: {reject_reason}"

        click_req = state.pop("click_request", None)
        if click_req is not None and state["click_strategy"] == "single":
            rescue_reject_reason = None
            source_cands = auto_dets if auto_dets else detect_candidates(
                frame_bgr=frame_bgr,
                model=model,
                processor=processor,
                prompt=prompt,
                profile_cfg=profile_cfg,
                auto_cfg=auto_cfg,
                device=device,
                use_tiles=use_tiles,
                include_cv=True,
            )
            picked = pick_click_candidate(source_cands, click_req)
            if picked is None and bool(args.click_rescue):
                rescued = detect_click_rescue(
                    frame_bgr, click_req, model, processor, prompt, profile_cfg, auto_cfg, device, use_tiles, int(args.rescue_crop), float(args.rescue_upscale)
                )
                if rescued is not None:
                    ok, reason = rescue_quality_gate(frame_bgr, rescued, click_req, auto_cfg, args)
                    if ok:
                        picked = rescued
                    else:
                        rescue_reject_reason = reason
            if picked is not None:
                state["pending"] = {
                    "mask_u8": picked["mask"].copy(),
                    "frame_bgr": frame_bgr.copy(),
                    "capture_upscale": 1.0,
                    "roi_xyxy": (0, 0, frame_w - 1, frame_h - 1),
                    "score": picked.get("score"),
                    "source": picked.get("source", "sam"),
                    "refined": False,
                    "prompt_strategy": "single",
                    "prompt_points_snapshot": [],
                }
                state["last_reject_reason"] = ""
                score_txt = picked.get("score")
                score_part = "n/a" if score_txt is None else f"{float(score_txt):.2f}"
                status_msg = f"Mask ready ({picked.get('source', 'sam')}) | score={score_part}"
            else:
                status_msg = f"Rescue rejected ({rescue_reject_reason})" if rescue_reject_reason else "No valid line mask at click"

        pending = state.get("pending")
        if bool(state["preview_save_style"]) and pending is not None:
            viz = render_save_style_preview(frame_bgr, pending["mask_u8"])
        else:
            viz = frame_bgr.copy()
            if auto_dets:
                for det in auto_dets:
                    contours, _ = cv2.findContours(det["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(viz, contours, -1, (255, 0, 255), 2)
            if state["click_strategy"] == "dual" and state["prompt_points"]:
                draw_prompt_points(viz, state["prompt_points"])
            if pending is not None:
                p_mask = (pending["mask_u8"] > 0).astype(np.uint8)
                overlay = viz.copy()
                overlay[p_mask > 0] = (0, 255, 0)
                viz = cv2.addWeighted(viz, 0.62, overlay, 0.38, 0.0)
                contours, _ = cv2.findContours(p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(viz, contours, -1, (0, 255, 255), 2)
                bb = bbox_from_mask(p_mask)
                if bb is not None:
                    x1, y1, x2, y2, _, _ = bb
                    cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 255), 2)

        view = viz[vy1 : vy2 + 1, vx1 : vx2 + 1]
        interp = cv2.INTER_CUBIC if float(state["view_zoom"]) > 1.0 else cv2.INTER_AREA
        show = cv2.resize(view, (disp_w, disp_h), interpolation=interp)

        pos_count = sum(1 for p in state["prompt_points"] if int(p["label"]) == POS_LABEL)
        neg_count = sum(1 for p in state["prompt_points"] if int(p["label"]) == NEG_LABEL)
        dual_span_txt = "n/a"
        ultra_tiny_active_txt = "OFF"
        if pos_count >= 2:
            positives = [p for p in state["prompt_points"] if int(p["label"]) == POS_LABEL]
            pa, pb = positives[-2], positives[-1]
            span_px = math.hypot(float(int(pb["x"]) - int(pa["x"])), float(int(pb["y"]) - int(pa["y"])))
            dual_span_txt = f"{span_px:.1f}px"
            ultra_tiny_active_txt = "ACTIVE" if is_ultra_tiny_span(span_px, args) else "ON"
        elif bool(args.ultra_tiny_mode):
            ultra_tiny_active_txt = "ON"
        hud_lines = [
            f"Image {idx + 1}/{len(images)} | {image_path.name}",
            f"Mode: {mode.upper()} | Profile: {args.profile} | Prompt: {prompt}",
            (
                f"Strategy: {state['click_strategy'].upper()} | Next: {state['click_label_mode'].upper()} | Pos:{pos_count} Neg:{neg_count} | "
                f"Span:{dual_span_txt} | Zoom x{state['view_zoom']:.2f} | View [{vx1}:{vx2}, {vy1}:{vy2}] | Auto detections: {len(auto_dets)} | "
                f"UltraTiny:{ultra_tiny_active_txt} | Save-preview:{'ON' if state['preview_save_style'] else 'OFF'} | Save-gate:{'ON' if bool(args.save_precision_gate) else 'OFF'}"
            ),
            "L-click add/select | R-click clear | Wheel zoom | N pos/neg | Backspace undo | T dual/single | U ultra-tiny | V save-preview | S save | D/A nav | Shift+D/A jump 25 | M mode | 0 reset | Q quit",
        ]
        if state["last_reject_reason"]:
            hud_lines.append(f"Last strict reject: {state['last_reject_reason']}")
        if status_msg:
            hud_lines.append(status_msg)
        y = 24
        for line in hud_lines:
            cv2.putText(show, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
            y += 24

        cv2.imshow(WINDOW_NAME, show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("0"):
            state["view_zoom"] = 1.0
            state["view_center"] = (frame_w * 0.5, frame_h * 0.5)
            status_msg = "Zoom reset"
            continue
        if key == ord("m"):
            cur = mode_cycle.index(mode)
            mode = mode_cycle[(cur + 1) % len(mode_cycle)]
            cache["idx"] = None
            cache["mode"] = None
            status_msg = f"Mode switched to {mode}"
            continue
        if key == ord("n"):
            state["click_label_mode"] = "neg" if state["click_label_mode"] == "pos" else "pos"
            status_msg = f"Click label mode: {state['click_label_mode'].upper()}"
            continue
        if key == ord("t"):
            state["click_strategy"] = "single" if state["click_strategy"] == "dual" else "dual"
            state["pending"] = None
            state["click_request"] = None
            state["prompt_points"] = []
            state["prompts_changed"] = False
            state["last_reject_reason"] = ""
            status_msg = f"Click strategy: {state['click_strategy'].upper()}"
            continue
        if key == ord("u"):
            args.ultra_tiny_mode = not bool(args.ultra_tiny_mode)
            state["pending"] = None
            state["last_reject_reason"] = ""
            state["prompts_changed"] = True if state["prompt_points"] else False
            status_msg = f"Ultra tiny mode: {'ON' if bool(args.ultra_tiny_mode) else 'OFF'}"
            continue
        if key == ord("v"):
            state["preview_save_style"] = not bool(state["preview_save_style"])
            status_msg = f"Save-style preview: {'ON' if state['preview_save_style'] else 'OFF'}"
            continue
        if key in {8, 127}:
            if state["prompt_points"]:
                removed = state["prompt_points"].pop()
                state["prompts_changed"] = True
                status_msg = f"Removed last prompt ({'POS' if int(removed['label']) == POS_LABEL else 'NEG'})"
            else:
                status_msg = "No prompt to undo"
            continue
        if key in {ord("s"), 13}:
            pending = state.get("pending")
            if pending is None:
                status_msg = "No pending mask to save"
                continue
            ok_save, save_reason = evaluate_pending_save_quality(pending, auto_cfg, args)
            if not ok_save:
                status_msg = f"Blocked save ({save_reason})"
                continue
            prompt_points_snapshot = pending.get("prompt_points_snapshot", [])
            pos_saved = sum(1 for p in prompt_points_snapshot if str(p.get("label", "")) == "pos")
            neg_saved = sum(1 for p in prompt_points_snapshot if str(p.get("label", "")) == "neg")
            rec = save_pending_sample(
                pending=pending,
                dataset_dirs=dataset_dirs,
                prompt=prompt,
                monitor_index=-1,
                roi_xyxy=pending["roi_xyxy"],
                tiny_cfg=tiny_cfg,
                extra_meta={
                    "source_image_path": str(image_path.resolve()),
                    "source_index": idx,
                    "source_mode": "image_dir",
                    "prompt_strategy": str(pending.get("prompt_strategy", state["click_strategy"])),
                    "prompt_points": prompt_points_snapshot,
                    "positive_count": int(pos_saved),
                    "negative_count": int(neg_saved),
                    "inference_source": str(pending.get("source", "")),
                },
            )
            state["pending"] = None
            saved_count_session += 1
            saved_total += 1
            status_msg = f"Saved {rec['id']} from image {idx + 1}/{len(images)}"
            if bool(args.auto_next_on_save):
                idx = clamp_idx(idx + 1, len(images))
                save_resume_state(state_path, image_dir, bool(args.recursive), idx, images[idx], saved_total)
            else:
                save_resume_state(state_path, image_dir, bool(args.recursive), idx, image_path, saved_total)
            continue
        if key == 27:
            state["pending"] = None
            status_msg = "Selection canceled"
            continue
        if key == ord("d"):
            idx = clamp_idx(idx + 1, len(images))
            status_msg = f"Image {idx + 1}/{len(images)}"
            continue
        if key == ord("a"):
            idx = clamp_idx(idx - 1, len(images))
            status_msg = f"Image {idx + 1}/{len(images)}"
            continue
        if key == ord("D"):
            idx = clamp_idx(idx + 25, len(images))
            status_msg = f"Jumped to image {idx + 1}/{len(images)}"
            continue
        if key == ord("A"):
            idx = clamp_idx(idx - 25, len(images))
            status_msg = f"Jumped to image {idx + 1}/{len(images)}"
            continue

    cv2.destroyAllWindows()
    save_resume_state(state_path, image_dir, bool(args.recursive), idx, images[clamp_idx(idx, len(images))], saved_total)
    print(f"Saved samples this session: {saved_count_session}")
    print(f"Saved samples total (state): {saved_total}")
    print(f"Resume state: {state_path}")


if __name__ == "__main__":
    main()
