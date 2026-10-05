from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from mss import mss
from PIL import Image
from transformers import Sam3Model, Sam3Processor

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional dependency at runtime
    PeftModel = None


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
WINDOW_NAME = "SAM3 Guideline Labeler"


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def get_cursor_pos() -> Tuple[int, int]:
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return int(pt.x), int(pt.y)


def monitor_contains(monitor: Dict, x: int, y: int) -> bool:
    left = int(monitor["left"])
    top = int(monitor["top"])
    right = left + int(monitor["width"])
    bottom = top + int(monitor["height"])
    return left <= x < right and top <= y < bottom


def choose_monitor(sct_obj, monitor_index: Optional[int]) -> Tuple[int, Dict]:
    monitors = sct_obj.monitors[1:]
    if not monitors:
        return 0, sct_obj.monitors[0]
    if monitor_index is not None and 1 <= monitor_index <= len(monitors):
        return monitor_index, sct_obj.monitors[monitor_index]
    x, y = get_cursor_pos()
    for idx, mon in enumerate(monitors, start=1):
        if monitor_contains(mon, x, y):
            return idx, mon
    return 1, sct_obj.monitors[1]


def str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dataset_layout(data_root: Path) -> Dict[str, Path]:
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    previews_dir = data_root / "previews"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = data_root / "annotations.jsonl"
    if not annotations_path.exists():
        annotations_path.write_text("", encoding="utf-8")
    splits_path = data_root / "splits.json"
    if not splits_path.exists():
        splits_path.write_text(json.dumps({"train": [], "val": []}, indent=2), encoding="utf-8")
    return {
        "images": images_dir,
        "masks": masks_dir,
        "previews": previews_dir,
        "annotations": annotations_path,
        "splits": splits_path,
    }


def compute_capture_upscale(
    width: int,
    height: int,
    cropped: bool,
    enabled: bool,
    factor: float,
    max_side: int,
) -> float:
    if not cropped or not enabled:
        return 1.0
    base = float(max(1.0, factor))
    if base <= 1.0:
        return 1.0
    longest = max(1, int(max(width, height)))
    max_side = max(1, int(max_side))
    if longest * base > max_side:
        base = float(max_side) / float(longest)
    return float(max(1.0, base))


def normalize_roi_xywh(
    x: int,
    y: int,
    w: int,
    h: int,
    max_w: int,
    max_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    if w <= 1 or h <= 1:
        return None
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(max_w - 1, int(x + w - 1))
    y2 = min(max_h - 1, int(y + h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def select_capture_roi_from_frame(
    frame_bgr: np.ndarray,
    display_scale: float,
    window_name: str,
) -> Optional[Tuple[int, int, int, int]]:
    src_h, src_w = frame_bgr.shape[:2]
    if 0 < display_scale < 1.0:
        view = cv2.resize(frame_bgr, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
    else:
        view = frame_bgr.copy()
    hint = view.copy()
    cv2.putText(
        hint,
        "Drag to crop | Enter confirm | C cancel",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(window_name, hint)
    x, y, w, h = cv2.selectROI(window_name, hint, fromCenter=False, showCrosshair=True)
    if w <= 0 or h <= 0:
        return None
    if 0 < display_scale < 1.0:
        x = int(round(float(x) / display_scale))
        y = int(round(float(y) / display_scale))
        w = int(round(float(w) / display_scale))
        h = int(round(float(h) / display_scale))
    return normalize_roi_xywh(int(x), int(y), int(w), int(h), src_w, src_h)


def bbox_from_mask(mask_u8: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    points = cv2.findNonZero(mask_u8.astype(np.uint8))
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return x, y, x + w - 1, y + h - 1, w, h


def border_touch_ratio(mask_u8: np.ndarray) -> float:
    mask = (mask_u8 > 0).astype(np.uint8)
    area = int(mask.sum())
    if area <= 0:
        return 1.0
    border = np.zeros_like(mask, dtype=np.uint8)
    border[0, :] = 1
    border[-1, :] = 1
    border[:, 0] = 1
    border[:, -1] = 1
    touched = int(np.logical_and(mask == 1, border == 1).sum())
    return float(touched) / float(area)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def validate_mask(mask_u8: np.ndarray, score: Optional[float], profile_cfg: Dict, enforce_score: bool) -> Optional[Dict]:
    mask = (mask_u8 > 0).astype(np.uint8)
    area = int(mask.sum())
    if area < int(profile_cfg["min_area"]):
        return None
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2, bw, bh = bbox
    aspect = float(max(float(bw) / max(1.0, float(bh)), float(bh) / max(1.0, float(bw))))
    fill = float(area) / float(max(1, bw * bh))
    border = border_touch_ratio(mask)
    if enforce_score and score is not None and score < float(profile_cfg["min_score"]):
        return None
    if aspect < float(profile_cfg["min_aspect_ratio"]):
        return None
    if fill > float(profile_cfg["max_fill_ratio"]):
        return None
    if border > float(profile_cfg["max_border_touch_ratio"]):
        return None
    return {
        "mask": mask,
        "score": None if score is None else float(score),
        "bbox": (x1, y1, x2, y2),
        "area": area,
        "aspect": aspect,
        "fill_ratio": fill,
        "border_touch_ratio": border,
    }


def dedup_candidates(candidates: List[Dict], iou_threshold: float) -> List[Dict]:
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda c: (
            -1.0 if c.get("score") is None else -float(c.get("score")),
            int(c.get("area", 0)),
        ),
    )
    kept: List[Dict] = []
    for cand in ordered:
        is_dup = False
        for seen in kept:
            if mask_iou(cand["mask"], seen["mask"]) >= float(iou_threshold):
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
    return kept


def run_sam_text(
    frame_bgr: np.ndarray,
    model: Sam3Model,
    processor: Sam3Processor,
    prompt: str,
    profile_cfg: Dict,
    device: str,
) -> List[Dict]:
    h, w = frame_bgr.shape[:2]
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    use_amp = device.startswith("cuda")
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
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
    for idx in range(int(masks.shape[0])):
        m = masks[idx]
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        mask = (m > 0).astype(np.uint8)
        score_val: Optional[float] = None
        if scores is not None and idx < len(scores):
            sv = scores[idx]
            if isinstance(sv, torch.Tensor):
                score_val = float(sv.detach().cpu().item())
            else:
                score_val = float(sv)
        valid = validate_mask(mask, score_val, profile_cfg, enforce_score=True)
        if valid is None:
            continue
        valid["source"] = "sam"
        out.append(valid)
    return out


def detect_white_line_cv_masks(frame_bgr: np.ndarray, cv_cfg: Dict) -> List[np.ndarray]:
    if not bool(cv_cfg.get("enabled", False)):
        return []
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(
        hsv,
        (int(cv_cfg["h_min"]), int(cv_cfg["s_min"]), int(cv_cfg["v_min"])),
        (int(cv_cfg["h_max"]), int(cv_cfg["s_max"]), int(cv_cfg["v_max"])),
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel, iterations=1)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_len = float(cv_cfg.get("min_length", 8))
    min_area = float(cv_cfg.get("min_area", 8))
    masks: List[np.ndarray] = []
    for cnt in contours:
        length = float(cv2.arcLength(cnt, closed=False))
        area = float(cv2.contourArea(cnt))
        if length < min_len or area < min_area:
            continue
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)
        masks.append(mask)
    return masks


def generate_tiles(width: int, height: int, tile_size: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    if tile_size <= 0:
        return [(0, 0, width, height)]
    step = int(round(tile_size * (1.0 - overlap)))
    step = max(32, min(step, tile_size))
    xs = list(range(0, max(1, width - tile_size + 1), step))
    ys = list(range(0, max(1, height - tile_size + 1), step))
    if not xs or xs[-1] != max(0, width - tile_size):
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] != max(0, height - tile_size):
        ys.append(max(0, height - tile_size))
    tiles: List[Tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            x2 = min(width, x + tile_size)
            y2 = min(height, y + tile_size)
            tiles.append((x, y, x2, y2))
    return tiles


def detect_candidates(
    frame_bgr: np.ndarray,
    model: Sam3Model,
    processor: Sam3Processor,
    prompt: str,
    profile_cfg: Dict,
    auto_cfg: Dict,
    device: str,
    use_tiles: bool,
    include_cv: bool,
) -> List[Dict]:
    h, w = frame_bgr.shape[:2]
    sam_candidates: List[Dict] = []
    if use_tiles:
        tile_cfg = auto_cfg["tile"]
        tiles = generate_tiles(w, h, int(tile_cfg["size"]), float(tile_cfg["overlap"]))
        for x1, y1, x2, y2 in tiles:
            tile = frame_bgr[y1:y2, x1:x2]
            tile_cands = run_sam_text(tile, model, processor, prompt, profile_cfg, device)
            for cand in tile_cands:
                local_mask = cand["mask"]
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = local_mask
                valid = validate_mask(full_mask, cand.get("score"), profile_cfg, enforce_score=True)
                if valid is None:
                    continue
                valid["source"] = "sam_tile"
                sam_candidates.append(valid)
    else:
        sam_candidates = run_sam_text(frame_bgr, model, processor, prompt, profile_cfg, device)

    out = sam_candidates[:]
    if include_cv:
        cv_masks = detect_white_line_cv_masks(frame_bgr, auto_cfg["white_cv"])
        for m in cv_masks:
            valid = validate_mask(m, None, profile_cfg, enforce_score=False)
            if valid is None:
                continue
            valid["source"] = "cv"
            out.append(valid)

    out = dedup_candidates(out, float(auto_cfg["dedup_iou"]))
    max_masks = int(auto_cfg["max_masks"])
    return out[:max_masks]


def pick_click_candidate(candidates: List[Dict], click_xy: Tuple[int, int]) -> Optional[Dict]:
    cx, cy = click_xy
    hits: List[Dict] = []
    for cand in candidates:
        mask = cand["mask"]
        h, w = mask.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        if int(mask[cy, cx]) == 0:
            continue
        hits.append(cand)
    if not hits:
        return None
    hits.sort(
        key=lambda c: (
            int(c["area"]),
            0.0 if c.get("score") is None else -float(c["score"]),
        )
    )
    return hits[0]


def display_to_source(
    disp_x: int,
    disp_y: int,
    display_scale: float,
    frame_shape: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    if 0 < display_scale < 1.0:
        src_x = int(round(float(disp_x) / display_scale))
        src_y = int(round(float(disp_y) / display_scale))
    else:
        src_x, src_y = int(disp_x), int(disp_y)
    h, w = frame_shape
    if not (0 <= src_x < w and 0 <= src_y < h):
        return None
    return src_x, src_y


def draw_brush(mask_u8: np.ndarray, p0: Tuple[int, int], p1: Tuple[int, int], radius: int, value: int) -> None:
    thickness = max(1, int(radius) * 2)
    cv2.line(mask_u8, p0, p1, int(value), thickness=thickness, lineType=cv2.LINE_AA)
    cv2.circle(mask_u8, p1, int(radius), int(value), thickness=-1, lineType=cv2.LINE_AA)


def tiny_line_flag(mask_u8: np.ndarray, tiny_cfg: Dict) -> bool:
    area = int((mask_u8 > 0).sum())
    if area <= 0:
        return False
    h, w = mask_u8.shape[:2]
    bbox = bbox_from_mask(mask_u8)
    if bbox is None:
        return False
    _, _, _, _, bw, bh = bbox
    area_ratio = float(area) / float(max(1, h * w))
    max_side = max(int(bw), int(bh))
    return (
        area_ratio <= float(tiny_cfg["max_area_ratio"])
        or max_side <= int(tiny_cfg["max_bbox_side"])
    )


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def append_jsonl(path: Path, obj: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True) + "\n")


def save_pending_sample(
    pending: Dict,
    dataset_dirs: Dict[str, Path],
    prompt: str,
    monitor_index: int,
    roi_xyxy: Tuple[int, int, int, int],
    tiny_cfg: Dict,
    extra_meta: Optional[Dict] = None,
) -> Dict:
    sample_id = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    image_path = (dataset_dirs["images"] / f"{sample_id}.png").resolve()
    mask_path = (dataset_dirs["masks"] / f"{sample_id}.png").resolve()
    preview_path = (dataset_dirs["previews"] / f"{sample_id}.png").resolve()

    frame = np.ascontiguousarray(pending["frame_bgr"])
    mask = (pending["mask_u8"] > 0).astype(np.uint8)
    cv2.imwrite(str(image_path), frame)
    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

    preview = frame.copy()
    color = preview.copy()
    color[mask > 0] = (0, 0, 255)
    preview = cv2.addWeighted(preview, 0.65, color, 0.35, 0.0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (0, 255, 255), 2)
    cv2.imwrite(str(preview_path), preview)

    rec = {
        "id": sample_id,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "prompt": str(prompt),
        "monitor_index": int(monitor_index),
        "roi_xyxy": [int(v) for v in roi_xyxy],
        "capture_upscale": float(pending["capture_upscale"]),
        "tiny_line": bool(tiny_line_flag(mask, tiny_cfg)),
        "refined": bool(pending.get("refined", False)),
        "created_at": utc_now_iso(),
    }
    if extra_meta:
        for k, v in extra_meta.items():
            if k not in rec:
                rec[k] = v
    append_jsonl(dataset_dirs["annotations"], rec)
    return rec


def load_sam_model(model_id: str, lora_path: Optional[str], device: str) -> Tuple[Sam3Model, Sam3Processor]:
    base = Sam3Model.from_pretrained(model_id)
    model: Sam3Model
    if lora_path and lora_path.strip().lower() not in {"", "none"}:
        if PeftModel is None:
            raise RuntimeError("`peft` is required when --lora-path is set.")
        model = PeftModel.from_pretrained(base, lora_path)
    else:
        model = base
    model = model.to(device).eval()
    processor = Sam3Processor.from_pretrained(model_id)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    return model, processor


def cycle_mode(mode: str) -> str:
    if mode == "label":
        return "hybrid"
    if mode == "hybrid":
        return "auto"
    return "label"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live SAM3 overlay and line-only label exporter.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--monitor-index", type=int, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--lora-path", type=str, default="none")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--profile", type=str, choices=["balanced", "precision"], default="precision")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["label", "auto", "hybrid"], default="hybrid")
    parser.add_argument("--crop-upscale-factor", type=float, default=None)
    parser.add_argument("--tile-small-lines", type=str2bool, default=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    model_id = str(args.model_id or config["model_id"])
    prompt = str(args.prompt or config["default_prompt"])
    dataset_root = Path(args.dataset_root or config["paths"]["dataset_root"])
    dataset_dirs = ensure_dataset_layout(dataset_root)
    profile_cfg = config["profiles"][args.profile]
    auto_cfg = config["auto"]
    tiny_cfg = config["tiny_line"]

    crop_cfg = config["crop_upscale"]
    crop_upscale_factor = float(args.crop_upscale_factor) if args.crop_upscale_factor is not None else float(crop_cfg["factor"])
    display_scale = float(config.get("display_scale", 1.0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_id}")
    model, processor = load_sam_model(model_id, args.lora_path, device)
    print(f"Device: {device}")
    print(f"Prompt: {prompt}")
    print(f"Mode: {args.mode} | Profile: {args.profile}")
    print("Controls: L-click select | L-drag add | R-drag erase | [ ] brush | S/Enter save | Esc cancel | A mode | R crop | F full | Q quit")

    state: Dict = {
        "pending": None,
        "click_request": None,
        "brush_active": False,
        "brush_mode": None,
        "brush_last": None,
        "brush_size": 7,
        "frame_shape": (1, 1),
        "display_scale": display_scale,
    }

    def on_mouse(event, x, y, flags, param):
        frame_shape = state.get("frame_shape", (1, 1))
        src = display_to_source(int(x), int(y), float(state.get("display_scale", 1.0)), frame_shape)
        pending = state.get("pending")

        if event == cv2.EVENT_LBUTTONDOWN:
            if pending is None:
                if src is not None:
                    state["click_request"] = src
            else:
                if src is not None:
                    state["brush_active"] = True
                    state["brush_mode"] = "add"
                    state["brush_last"] = src
                    draw_brush(pending["mask_u8"], src, src, int(state["brush_size"]), 1)
                    pending["refined"] = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            if pending is not None and src is not None:
                state["brush_active"] = True
                state["brush_mode"] = "erase"
                state["brush_last"] = src
                draw_brush(pending["mask_u8"], src, src, int(state["brush_size"]), 0)
                pending["refined"] = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if state.get("brush_active", False) and pending is not None and src is not None:
                last = state.get("brush_last", src)
                value = 1 if state.get("brush_mode") == "add" else 0
                draw_brush(pending["mask_u8"], last, src, int(state["brush_size"]), value)
                state["brush_last"] = src
                pending["refined"] = True

        elif event in {cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP}:
            state["brush_active"] = False
            state["brush_mode"] = None
            state["brush_last"] = None

    mode = args.mode
    capture_roi: Optional[Tuple[int, int, int, int]] = None
    status_msg = ""

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    with mss() as sct:
        selected_idx, selected_monitor = choose_monitor(sct, args.monitor_index)
        monitor = {
            "left": int(selected_monitor["left"]),
            "top": int(selected_monitor["top"]),
            "width": int(selected_monitor["width"]),
            "height": int(selected_monitor["height"]),
        }
        print(f"Using monitor {selected_idx}: {monitor}")

        while True:
            if capture_roi is None:
                active_capture = monitor
                roi_offset_x = 0
                roi_offset_y = 0
                capture_upscale = 1.0
                roi_meta = (0, 0, monitor["width"] - 1, monitor["height"] - 1)
            else:
                rx1, ry1, rx2, ry2 = capture_roi
                active_capture = {
                    "left": int(monitor["left"] + rx1),
                    "top": int(monitor["top"] + ry1),
                    "width": int(rx2 - rx1 + 1),
                    "height": int(ry2 - ry1 + 1),
                }
                roi_offset_x = int(rx1)
                roi_offset_y = int(ry1)
                capture_upscale = compute_capture_upscale(
                    int(active_capture["width"]),
                    int(active_capture["height"]),
                    cropped=True,
                    enabled=bool(crop_cfg["enabled"]),
                    factor=crop_upscale_factor,
                    max_side=int(crop_cfg["max_side"]),
                )
                roi_meta = (rx1, ry1, rx2, ry2)

            raw = np.asarray(sct.grab(active_capture), dtype=np.uint8)
            frame = np.ascontiguousarray(raw[:, :, :3])
            if capture_upscale > 1.0:
                frame = cv2.resize(frame, None, fx=capture_upscale, fy=capture_upscale, interpolation=cv2.INTER_CUBIC)
            frame_bgr = np.ascontiguousarray(frame)
            frame_h, frame_w = frame_bgr.shape[:2]
            state["frame_shape"] = (frame_h, frame_w)
            state["display_scale"] = display_scale

            auto_enabled = mode in {"auto", "hybrid"}
            use_tiles = bool(args.tile_small_lines) and (capture_roi is not None) and bool(auto_cfg["tile"]["enabled_crop_only"])
            if bool(args.tile_small_lines) and not bool(auto_cfg["tile"]["enabled_crop_only"]):
                use_tiles = True

            auto_dets: List[Dict] = []
            if auto_enabled:
                auto_dets = detect_candidates(
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

            click_req = state.pop("click_request", None)
            if click_req is not None:
                source_cands = auto_dets
                if not source_cands:
                    source_cands = detect_candidates(
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
                if picked is not None:
                    state["pending"] = {
                        "mask_u8": picked["mask"].copy(),
                        "frame_bgr": frame_bgr.copy(),
                        "capture_upscale": float(capture_upscale),
                        "roi_xyxy": tuple(int(v) for v in roi_meta),
                        "score": picked.get("score"),
                        "source": picked.get("source", "sam"),
                        "refined": False,
                    }
                    score_txt = picked.get("score")
                    score_part = "n/a" if score_txt is None else f"{score_txt:.2f}"
                    status_msg = (
                        f"Mask ready ({picked.get('source', 'sam')}) | score={score_part} "
                        "Press S/Enter to save, Esc to cancel"
                    )
                else:
                    status_msg = "No valid line mask at click."

            viz = frame_bgr.copy()

            if auto_dets:
                for det in auto_dets:
                    contours, _ = cv2.findContours(det["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(viz, contours, -1, (255, 0, 255), 2)
                    x1, y1, _, _ = det["bbox"]
                    s = det.get("score")
                    label = det.get("source", "det")
                    if s is not None:
                        label = f"{label}:{s:.2f}"
                    cv2.putText(viz, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            pending = state.get("pending")
            if pending is not None:
                p_mask = (pending["mask_u8"] > 0).astype(np.uint8)
                color = viz.copy()
                color[p_mask > 0] = (0, 255, 0)
                viz = cv2.addWeighted(viz, 0.60, color, 0.40, 0.0)
                contours, _ = cv2.findContours(p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(viz, contours, -1, (0, 255, 255), 2)
                bbox = bbox_from_mask(p_mask)
                if bbox is not None:
                    x1, y1, x2, y2, _, _ = bbox
                    cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 255), 2)

            hud_lines = [
                f"Mode: {mode.upper()} | Profile: {args.profile} | Prompt: {prompt}",
                (
                    f"Auto detections: {len(auto_dets)} | Capture: "
                    + (f"crop {roi_meta[2]-roi_meta[0]+1}x{roi_meta[3]-roi_meta[1]+1}" if capture_roi else "full")
                    + f" | upscale x{capture_upscale:.2f} | brush {state['brush_size']}"
                ),
                "L-click select, L-drag add, R-drag erase, [ ] brush, S/Enter save, Esc cancel, A mode, R crop, F full, Q quit",
            ]
            if status_msg:
                hud_lines.append(status_msg)

            y = 24
            for line in hud_lines:
                cv2.putText(viz, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
                y += 24

            if 0 < display_scale < 1.0:
                show = cv2.resize(viz, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
            else:
                show = viz

            cv2.imshow(WINDOW_NAME, show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("a"):
                mode = cycle_mode(mode)
                status_msg = f"Mode switched to {mode}"
                continue

            if key in {ord("s"), 13}:
                pending = state.get("pending")
                if pending is None:
                    status_msg = "No pending mask to save."
                    continue
                rec = save_pending_sample(
                    pending=pending,
                    dataset_dirs=dataset_dirs,
                    prompt=prompt,
                    monitor_index=selected_idx,
                    roi_xyxy=pending["roi_xyxy"],
                    tiny_cfg=tiny_cfg,
                )
                state["pending"] = None
                status_msg = (
                    f"Saved {rec['id']} | tiny_line={rec['tiny_line']} | refined={rec['refined']}"
                )
                continue

            if key == 27:
                state["pending"] = None
                status_msg = "Pending mask canceled."
                continue

            if key == ord("["):
                state["brush_size"] = max(1, int(state["brush_size"]) - 1)
                status_msg = f"Brush size: {state['brush_size']}"
                continue

            if key == ord("]"):
                state["brush_size"] = min(64, int(state["brush_size"]) + 1)
                status_msg = f"Brush size: {state['brush_size']}"
                continue

            if key == ord("r"):
                state["pending"] = None
                local_roi = select_capture_roi_from_frame(frame_bgr, display_scale, WINDOW_NAME)
                cv2.setMouseCallback(WINDOW_NAME, on_mouse)
                if local_roi is None:
                    status_msg = "Crop canceled."
                    continue
                lx1, ly1, lx2, ly2 = local_roi
                if capture_upscale > 1.0:
                    lx1 = int(round(lx1 / capture_upscale))
                    ly1 = int(round(ly1 / capture_upscale))
                    lx2 = int(round(lx2 / capture_upscale))
                    ly2 = int(round(ly2 / capture_upscale))
                rx1 = max(0, min(monitor["width"] - 1, int(roi_offset_x + lx1)))
                ry1 = max(0, min(monitor["height"] - 1, int(roi_offset_y + ly1)))
                rx2 = max(0, min(monitor["width"] - 1, int(roi_offset_x + lx2)))
                ry2 = max(0, min(monitor["height"] - 1, int(roi_offset_y + ly2)))
                if rx2 <= rx1 or ry2 <= ry1:
                    status_msg = "Invalid crop. Keeping current capture."
                    continue
                capture_roi = (rx1, ry1, rx2, ry2)
                status_msg = f"Crop set: {rx2-rx1+1}x{ry2-ry1+1}"
                continue

            if key == ord("f"):
                capture_roi = None
                state["pending"] = None
                status_msg = "Full capture restored."
                continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
