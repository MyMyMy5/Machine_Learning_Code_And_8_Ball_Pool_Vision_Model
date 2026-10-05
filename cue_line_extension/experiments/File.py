# File.py
# Extend the 8-ball aim line automatically (green or blue felt).
# Python 3.9+  |  pip install opencv-python numpy

import sys
import math
import argparse
from pathlib import Path
import cv2
import numpy as np


# ---------------- geometry ----------------

def extend_to_rect(point, direction, rect):
    """Extend a line in both directions from point along direction to the rectangle rect=(xmin,ymin,xmax,ymax)."""
    x0, y0 = point
    dx, dy = direction
    xmin, ymin, xmax, ymax = rect

    n = math.hypot(dx, dy)
    if n == 0:
        return (x0, y0, x0, y0)
    dx, dy = dx / n, dy / n

    ts = []
    # (x,y) = (x0,y0) + t*(dx,dy)
    if dx != 0:
        for x in (xmin, xmax):
            t = (x - x0) / dx
            y = y0 + t * dy
            if ymin <= y <= ymax:
                ts.append(t)
    if dy != 0:
        for y in (ymin, ymax):
            t = (y - y0) / dy
            x = x0 + t * dx
            if xmin <= x <= xmax:
                ts.append(t)

    if len(ts) < 2:
        return (x0, y0, x0, y0)

    a, b = min(ts), max(ts)
    xa, ya = int(round(x0 + a * dx)), int(round(y0 + a * dy))
    xb, yb = int(round(x0 + b * dx)), int(round(y0 + b * dy))
    return (xa, ya, xb, yb)


def shrink_rect(rect, m):
    xmin, ymin, xmax, ymax = rect
    return (xmin + m, ymin + m, xmax - m, ymax - m)


def rect_mask(shape, rect, val=255):
    xmin, ymin, xmax, ymax = rect
    m = np.zeros(shape[:2], np.uint8)
    m[ymin:ymax + 1, xmin:xmax + 1] = val
    return m


# ---------------- table & masks ----------------

def table_rect_auto(hsv_img):
    """Union of green + blue felt; return table rectangle and mask."""
    # green
    mask_g = cv2.inRange(hsv_img, np.array([35, 30, 40]), np.array([95, 255, 255]))
    # blue
    mask_b = cv2.inRange(hsv_img, np.array([85, 40, 40]), np.array([140, 255, 255]))
    mask = cv2.bitwise_or(mask_g, mask_b)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8), 2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num > 1:
        k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == k).astype(np.uint8) * 255

    ys, xs = np.where(mask > 0)
    rect = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return rect, mask


def white_mask(hsv_img, table_mask, s_max=150, v_min=110):
    """Low-S / high-V pixels restricted to felt (potential aim pixels)."""
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]
    m = cv2.inRange(np.dstack([s, v]),
                    np.array([0, s_max], dtype=np.uint8),
                    np.array([255, 255], dtype=np.uint8))
    m = cv2.bitwise_and(m, table_mask)
    return cv2.medianBlur(m, 5)


# ---------------- cue ball detection ----------------

def detect_cue_ball_strict(bgr, hsv, rect, s_thresh=70, v_thresh=180, rail_margin=45):
    """
    HoughCircles candidates -> keep only bright/low-S circles away from rails.
    Returns (x,y) or None.
    """
    xmin, ymin, xmax, ymax = rect
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    wb = cv2.GaussianBlur(gray, (9, 9), 1.2)
    circles = cv2.HoughCircles(wb, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=120, param2=18, minRadius=9, maxRadius=28)
    if circles is None:
        return None

    H, W = gray.shape[:2]
    best = None
    best_score = -1e9
    for (x, y, r) in np.uint16(np.around(circles[0])):
        x, y, r = int(x), int(y), int(r)
        if not (xmin < x < xmax and ymin < y < ymax):
            continue

        # reject near rails/pockets
        if (x - xmin) < rail_margin or (y - ymin) < rail_margin or \
           (xmax - x) < rail_margin or (ymax - y) < rail_margin:
            continue

        # circular mask for stats
        cmask = np.zeros((H, W), np.uint8)
        cv2.circle(cmask, (x, y), int(r * 0.95), 255, -1)

        # HSV means inside circle
        s_mean = cv2.mean(hsv[:, :, 1], mask=cmask)[0]
        v_mean = cv2.mean(hsv[:, :, 2], mask=cmask)[0]

        # cue ball should be low S, high V
        if s_mean > s_thresh or v_mean < v_thresh:
            continue

        score = (255 - s_mean) + 1.5 * v_mean  # whiteness score
        if score > best_score:
            best_score = score
            best = (x, y)

    return best


# ---------------- fit direction ----------------

def ransac_direction(points, iters=600, tol=2.5):
    """RANSAC + PCA to get a unit direction vector through sparse pixels."""
    if len(points) < 2:
        return None
    P = np.asarray(points, np.float32)
    rng = np.random.default_rng(0)
    best_dir, best_inl = None, -1
    for _ in range(iters):
        i, j = rng.integers(0, len(P), 2)
        if i == j:
            continue
        v = P[j] - P[i]
        n = np.hypot(v[0], v[1])
        if n < 1e-3:
            continue
        v /= n
        d = np.abs((P[:, 0] - P[i, 0]) * v[1] - (P[:, 1] - P[i, 1]) * v[0])
        inl = int(np.sum(d <= tol))
        if inl > best_inl:
            best_inl, best_dir = inl, v.copy()
    if best_dir is None:
        return None
    # refine via PCA on all points
    m = P.mean(axis=0)
    W = P - m
    cov = W.T @ W
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    n = np.hypot(v[0], v[1])
    return (float(v[0] / n), float(v[1] / n))


def fallback_hough_dir(region_mask):
    """If RANSAC fails, try Hough lines inside the donut mask."""
    edges = cv2.Canny(region_mask, 40, 100, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=18,
                            minLineLength=20, maxLineGap=8)
    if lines is None:
        return None
    # take longest segment
    L = max(lines[:, 0, :], key=lambda p: math.hypot(p[2] - p[0], p[3] - p[1]))
    dx, dy = L[2] - L[0], L[3] - L[1]
    n = math.hypot(dx, dy)
    if n == 0:
        return None
    return (dx / n, dy / n)


# ---------------- main pipeline ----------------

def extend_aim_line(bgr, *, rail_margin=45, s_max=150, v_min=110,
                    donut_inner=8, donut_outer=60, prefer_left=True, thickness=3):
    """Return RGB image with extended aim line drawn."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 1) table rect + safe inner rect (avoid rails)
    table_rect, table_mask = table_rect_auto(hsv)
    inner_rect = shrink_rect(table_rect, rail_margin)  # used only for detection
    inner_mask = rect_mask(bgr.shape, inner_rect)

    # 2) white pixels INSIDE inner felt only
    wmask = white_mask(hsv, cv2.bitwise_and(table_mask, inner_mask),
                       s_max=s_max, v_min=v_min)

    # 3) strict cue ball detection (also inside inner felt)
    cue = detect_cue_ball_strict(bgr, hsv, inner_rect,
                                 s_thresh=70, v_thresh=180,
                                 rail_margin=rail_margin)
    overlay = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if cue is None:
        # Could not find a confident cue ball; return original overlay.
        return overlay

    # 4) donut region around cue (still clipped to inner felt)
    H, W = wmask.shape[:2]
    Y, X = np.ogrid[:H, :W]
    dist = np.hypot(X - cue[0], Y - cue[1])
    donut = ((dist >= donut_inner) & (dist <= donut_outer)).astype(np.uint8) * 255
    region = cv2.bitwise_and(wmask, donut)

    ys, xs = np.where(region > 0)
    pts = list(zip(xs.tolist(), ys.tolist()))

    direction = None
    if len(pts) >= 60:
        direction = ransac_direction(pts, iters=600, tol=2.5)
    if direction is None:
        direction = fallback_hough_dir(region)

    if direction is None:
        # nothing found; return original overlay with cue marker for debugging
        cv2.circle(overlay, (cue[0], cue[1]), 5, (0, 255, 0), -1)
        return overlay

    # 5) extend to FULL table rect (not inner rect)
    xa, ya, xb, yb = extend_to_rect((cue[0], cue[1]), direction, table_rect)
    cv2.line(overlay, (xa, ya), (xb, yb), (255, 255, 0), thickness)
    cv2.circle(overlay, (cue[0], cue[1]), 5, (0, 255, 0), -1)
    return overlay


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Extend 8-ball aim line from a screenshot.")
    ap.add_argument("image", help="path to screenshot (png/jpg/jpeg)")
    ap.add_argument("--out", default="extend.png", help="output file")
    ap.add_argument("--resize", type=float, default=1.0, help="scale image before processing")
    ap.add_argument("--rail_margin", type=int, default=45, help="pixels shrunk from table rect to avoid rails")
    ap.add_argument("--inner_r", type=int, default=8, help="donut inner radius around cue")
    ap.add_argument("--outer_r", type=int, default=60, help="donut outer radius around cue")
    ap.add_argument("--s_max", type=int, default=150, help="max S for white mask")
    ap.add_argument("--v_min", type=int, default=110, help="min V for white mask")
    ap.add_argument("--prefer_right", action="store_true", help="cue likely on right half")
    ap.add_argument("--thickness", type=int, default=3, help="drawn line thickness")
    args = ap.parse_args()

    # Resolve and read
    img_path = Path(args.image).expanduser()
    if not img_path.is_absolute():
        img_path = (Path.cwd() / img_path).resolve()
    if not img_path.exists():
        sys.exit(f"[error] file not found: {img_path}")

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        sys.exit("[error] OpenCV failed to decode the image. Re-save the file as PNG/JPG or check the path.")

    if args.resize and args.resize != 1.0:
        bgr = cv2.resize(bgr, (0, 0), fx=args.resize, fy=args.resize)

    overlay = extend_aim_line(
        bgr,
        rail_margin=args.rail_margin,
        s_max=args.s_max,
        v_min=args.v_min,
        donut_inner=args.inner_r,
        donut_outer=args.outer_r,
        prefer_left=not args.prefer_right,
        thickness=args.thickness,
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        sys.exit(f"[error] failed to write: {out_path}")
    print(f"[ok] wroteasdad: {out_path}")


if __name__ == "__main__":
    main()
