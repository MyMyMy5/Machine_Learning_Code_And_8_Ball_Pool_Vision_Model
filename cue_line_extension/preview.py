# preview.py
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import cv2 as cv

from ML import UNet  # reuse the model definition

def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = UNet()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model

def infer_single(model, image_path: Path, device: torch.device, image_size=(512, 512)):
    img = Image.open(image_path).convert("RGB").resize(image_size)
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0]
    return probs.cpu().numpy()

def save_overlay(base_image: Path, mask_array, out_path: Path):
    img = Image.open(base_image).convert("RGB")
    mask = np.clip(mask_array, 0.0, 1.0)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize(img.size, resample=Image.BILINEAR)
    img.putalpha(255)
    tint = Image.new("RGBA", img.size, (255, 255, 0, 120))
    overlay = Image.composite(tint, Image.new("RGBA", img.size, (0, 0, 0, 0)), mask_img)
    blended = Image.alpha_composite(img, overlay)
    blended.convert("RGB").save(out_path)



def postprocess_mask(mask_array: np.ndarray, threshold: float, thickness: int, min_points: int = 50) -> np.ndarray:
    mask = np.clip(mask_array, 0.0, 1.0)
    binary = mask > threshold
    if binary.sum() < min_points:
        return mask

    coords = np.column_stack(np.nonzero(binary)).astype(np.float32)
    points = np.column_stack((coords[:, 1], coords[:, 0]))
    line = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    height, width = mask.shape
    eps = 1e-6
    intersections = []

    if abs(vx) > eps:
        for x in (0.0, width - 1.0):
            t = (x - x0) / vx
            y = y0 + vy * t
            if -2.0 <= y <= height + 1.0:
                intersections.append((int(round(x)), int(round(np.clip(y, 0, height - 1)))))
    if abs(vy) > eps:
        for y in (0.0, height - 1.0):
            t = (y - y0) / vy
            x = x0 + vx * t
            if -2.0 <= x <= width + 1.0:
                intersections.append((int(round(np.clip(x, 0, width - 1))), int(round(y))))

    if len(intersections) < 2:
        return mask

    best_pair = None
    best_dist = -1.0
    for i in range(len(intersections)):
        for j in range(i + 1, len(intersections)):
            (x1, y1), (x2, y2) = intersections[i], intersections[j]
            dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if dist > best_dist:
                best_dist = dist
                best_pair = ((x1, y1), (x2, y2))

    if best_pair is None or best_dist <= 0:
        return mask

    refined = np.zeros_like(mask, dtype=np.uint8)
    cv.line(refined, best_pair[0], best_pair[1], 255, thickness, cv.LINE_AA)
    refined = refined.astype(np.float32) / 255.0
    return np.maximum(mask, refined)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--line-thickness", type=int, default=6)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)

    for image_path in sorted(args.images.iterdir()):
        mask_array = infer_single(model, image_path, device)
        refined = postprocess_mask(mask_array, args.threshold, args.line_thickness)
        save_overlay(image_path, refined, args.output / image_path.name)

if __name__ == "__main__":
    main()
