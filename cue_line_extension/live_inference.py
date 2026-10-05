"""
Run the trained cue-line segmentation model on live screen captures.

The script mirrors the preprocessing used during training, converts each grab
into a segmentation mask, refines it into a straight cue line, and overlays the
result back onto the captured frame for visual feedback.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

try:
    import mss
except ImportError as exc:
    raise ImportError("Install 'mss' to enable live capture: pip install mss") from exc

from ML import UNet
from preview import postprocess_mask


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    if "model" not in state:
        raise KeyError("Checkpoint does not contain a 'model' state dict")

    model = UNet().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def to_tensor(frame_bgr: np.ndarray, image_size: tuple[int, int], device: torch.device) -> torch.Tensor:
    resized = cv.resize(frame_bgr, image_size, interpolation=cv.INTER_LINEAR)
    rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(device, non_blocking=True)


def draw_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    mask_uint8 = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
    colored = np.zeros_like(frame)
    colored[mask_uint8 > 64] = (0, 255, 255)
    return cv.addWeighted(frame, 1.0, colored, alpha, 0.0)


def parse_region(region: str | None, monitor: dict[str, int]) -> dict[str, int]:
    if region is None:
        return {
            "top": monitor["top"],
            "left": monitor["left"],
            "width": monitor["width"],
            "height": monitor["height"],
        }

    parts = region.split(",")
    if len(parts) != 4:
        raise ValueError("Region must be formatted as 'left,top,width,height'")
    left, top, width, height = (int(p.strip()) for p in parts)
    return {"top": top, "left": left, "width": width, "height": height}


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live cue-line inference using a trained UNet checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint_epoch_XXX_best.pth")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--monitor", type=int, default=None, help="Monitor index as reported by mss (default: JSON value or 1)")
    parser.add_argument("--region", type=str, default=None, help="Optional capture region: left,top,width,height (overrides JSON)")
    parser.add_argument("--region-json", type=Path, default=Path("config/region.json"), help="JSON file with monitor/region calibration (default: config/region.json)")
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512), metavar=("W", "H"))
    parser.add_argument("--threshold", type=float, default=0.35, help="Mask probability threshold before fitting")
    parser.add_argument("--line-thickness", type=int, default=6, help="Line thickness used during refinement")
    parser.add_argument("--min-points", type=int, default=50, help="Minimum mask pixels needed before fitting")
    parser.add_argument("--alpha", type=float, default=0.7, help="Overlay strength (0=no overlay, 1=full)")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Resize factor for the display window")
    parser.add_argument("--amp", action="store_true", help="Enable torch.autocast when running on CUDA")
    parser.add_argument("--quit-key", type=str, default="q", help="Key to quit the preview window")
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_model(args.checkpoint, device)
    enable_amp = args.amp and device.type == "cuda"

    region_data: dict[str, int] | None = None
    if args.region_json is not None and Path(args.region_json).exists():
        try:
            region_data = json.loads(Path(args.region_json).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse region JSON at {args.region_json}: {exc}") from exc

    with mss.mss() as sct:
        monitors = sct.monitors
        default_monitor_index = 1
        if region_data is not None and "monitor" in region_data:
            default_monitor_index = int(region_data["monitor"])
        monitor_index = args.monitor if args.monitor is not None else default_monitor_index
        if monitor_index >= len(monitors):
            raise IndexError(f"Monitor index {monitor_index} out of range (available: 1-{len(monitors) - 1})")

        json_region = None
        if region_data is not None and all(key in region_data for key in ("left", "top", "width", "height")):
            json_region = f"{region_data['left']},{region_data['top']},{region_data['width']},{region_data['height']}"

        region_arg = args.region if args.region is not None else json_region
        capture_region = parse_region(region_arg, monitors[monitor_index])

        window_name = f"Cue Line Live Preview ({args.checkpoint.name})"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        fps_avg = 0.0
        frame_count = 0
        last_time = time.perf_counter()

        try:
            while True:
                grab = sct.grab(capture_region)
                frame_bgr = np.array(grab)[:, :, :3]

                input_tensor = to_tensor(frame_bgr, tuple(args.image_size), device)

                with torch.no_grad():
                    if enable_amp:
                        with torch.autocast(device_type=device.type, enabled=True):
                            logits = model(input_tensor)
                    else:
                        logits = model(input_tensor)
                    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

                refined = postprocess_mask(probs, args.threshold, args.line_thickness, args.min_points)
                refined_resized = cv.resize(refined, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv.INTER_LINEAR)
                display_frame = draw_overlay(frame_bgr, refined_resized, args.alpha)

                now = time.perf_counter()
                dt = now - last_time
                last_time = now
                fps = 1.0 / dt if dt > 0 else 0.0
                frame_count += 1
                fps_avg = fps if frame_count == 1 else fps_avg * 0.9 + fps * 0.1
                cv.putText(display_frame, f"FPS: {fps_avg:.1f}", (12, 26), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if args.display_scale != 1.0:
                    display_frame = cv.resize(
                        display_frame,
                        None,
                        fx=args.display_scale,
                        fy=args.display_scale,
                        interpolation=cv.INTER_LINEAR,
                    )

                cv.imshow(window_name, display_frame)
                key = cv.waitKey(1) & 0xFF
                if key == 27 or key == ord(args.quit_key.lower()):
                    break
        finally:
            cv.destroyWindow(window_name)


if __name__ == "__main__":
    main()
