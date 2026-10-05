import math
import time

import cv2 as cv
import numpy as np

try:
    import mss
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "Study.py requires the 'mss' package for screen capture. Install it with 'pip install mss'."
    ) from exc

LOW_WHITE = np.array([0, 0, 245], dtype=np.uint8)
HIGH_WHITE = np.array([180, 30, 255], dtype=np.uint8)
BRIGHT_MIN, BRIGHT_MAX = 200, 255
STRUCTURING_KERNEL = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
HOUGH_PARAMS = dict(rho=1, theta=np.pi / 180, threshold=40, minLineLength=30, maxLineGap=15)
MIN_LINE_LENGTH = 1
MAX_LINE_LENGTH = 150
MIN_LINE_ANGLE = 15
MAX_LINE_ANGLE = 85
LINE_EXTENSION_THICKNESS = 6
WINDOW_NAME = "Line Detector Overlay"
MONITOR_INDEX = 1  # 0 = entire virtual desktop, 1 = primary monitor


def extend_segment_along_mask(x1, y1, x2, y2, mask, thickness=LINE_EXTENSION_THICKNESS, margin_factor=1.0, min_points=20):
    """Extend a detected line segment until it reaches the frame borders."""
    height, width = mask.shape[:2]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return x1, y1, x2, y2

    dir_x = dx / length
    dir_y = dy / length
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    margin = int(max(10, length * margin_factor))
    x_min = max(0, min(x1, x2) - margin)
    x_max = min(width, max(x1, x2) + margin)
    y_min = max(0, min(y1, y2) - margin)
    y_max = min(height, max(y1, y2) + margin)
    if x_min >= x_max or y_min >= y_max:
        return x1, y1, x2, y2

    region = mask[y_min:y_max, x_min:x_max]
    ys, xs = np.where(region > 0)
    if ys.size < min_points:
        return x1, y1, x2, y2

    xs = xs.astype(np.float32) + x_min
    ys = ys.astype(np.float32) + y_min
    px = xs - cx
    py = ys - cy

    perp = np.abs(px * (-dir_y) + py * dir_x)
    line_points_mask = perp <= thickness
    if not np.any(line_points_mask):
        return x1, y1, x2, y2

    proj = px[line_points_mask] * dir_x + py[line_points_mask] * dir_y
    if proj.size == 0:
        return x1, y1, x2, y2

    start_proj = proj.min()
    end_proj = proj.max()
    if end_proj - start_proj < length * 0.25:
        return x1, y1, x2, y2

    start_point = (cx + start_proj * dir_x, cy + start_proj * dir_y)
    end_point = (cx + end_proj * dir_x, cy + end_proj * dir_y)

    def clamp_point(px_val, py_val):
        xi = int(round(px_val))
        yi = int(round(py_val))
        xi = max(0, min(width - 1, xi))
        yi = max(0, min(height - 1, yi))
        return xi, yi

    def project_to_border(x0, y0, vx, vy):
        eps = 1e-6
        candidates = []
        if abs(vx) > eps:
            for xb in (0, width - 1):
                t = (xb - x0) / vx
                y = y0 + t * vy
                if t >= 0 and 0 <= y <= height - 1:
                    candidates.append((t, xb, y))
        if abs(vy) > eps:
            for yb in (0, height - 1):
                t = (yb - y0) / vy
                x = x0 + t * vx
                if t >= 0 and 0 <= x <= width - 1:
                    candidates.append((t, x, yb))
        if not candidates:
            return clamp_point(x0, y0)
        t, x, y = min(candidates, key=lambda c: c[0])
        return clamp_point(x, y)

    start_x, start_y = project_to_border(start_point[0], start_point[1], -dir_x, -dir_y)
    end_x, end_y = project_to_border(end_point[0], end_point[1], dir_x, dir_y)

    return start_x, start_y, end_x, end_y


def build_white_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Return a cleaned mask of bright white pixels."""
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    mask_white = cv.inRange(hsv, LOW_WHITE, HIGH_WHITE)

    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    mask_bright = cv.inRange(gray, BRIGHT_MIN, BRIGHT_MAX)
    mask_white = cv.bitwise_and(mask_white, mask_bright)

    mask_closed = cv.morphologyEx(mask_white, cv.MORPH_CLOSE, STRUCTURING_KERNEL, iterations=1)
    return mask_closed


def detect_lines_and_overlay(frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """Detect lines on the frame and return an overlay frame plus count."""
    mask = build_white_mask(frame_bgr)
    edges = cv.Canny(mask, 50, 150)
    line_segments = cv.HoughLinesP(edges, **HOUGH_PARAMS)

    overlay = np.zeros_like(frame_bgr)
    count = 0

    if line_segments is not None:
        for (x1, y1, x2, y2) in line_segments[:, 0]:
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            angle = abs(math.degrees(math.atan2(dy, dx)))
            angle = angle if angle <= 90 else 180 - angle

            if not (MIN_LINE_LENGTH <= length <= MAX_LINE_LENGTH):
                continue
            if not (MIN_LINE_ANGLE <= angle <= MAX_LINE_ANGLE):
                continue

            extended = extend_segment_along_mask(x1, y1, x2, y2, mask)
            cv.line(overlay, (extended[0], extended[1]), (extended[2], extended[3]), (0, 255, 0), 3, cv.LINE_AA)
            count += 1

    if count:
        combined = cv.addWeighted(frame_bgr, 0.75, overlay, 1.0, 0.0)
    else:
        combined = frame_bgr.copy()

    return combined, count


def configure_window(monitor: dict) -> None:
    """Create a full-screen always-on-top window for the overlay."""
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    try:
        cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)
    except cv.error:
        pass  # Not all builds expose WND_PROP_TOPMOST
    cv.moveWindow(WINDOW_NAME, monitor.get("left", 0), monitor.get("top", 0))
    cv.resizeWindow(WINDOW_NAME, monitor["width"], monitor["height"])


def main() -> None:
    with mss.mss() as sct:
        if MONITOR_INDEX >= len(sct.monitors):
            raise ValueError(f"Requested monitor index {MONITOR_INDEX} but only {len(sct.monitors) - 1} monitors available")
        monitor = sct.monitors[MONITOR_INDEX]
        configure_window(monitor)

        frame_counter = 0
        fps_timer = time.perf_counter()
        fps_value = 0.0

        try:
            while True:
                sct_frame = sct.grab(monitor)
                frame_bgra = np.array(sct_frame, dtype=np.uint8)
                frame_bgr = cv.cvtColor(frame_bgra, cv.COLOR_BGRA2BGR)

                overlay_frame, line_count = detect_lines_and_overlay(frame_bgr)

                frame_counter += 1
                if frame_counter >= 30:
                    now = time.perf_counter()
                    fps_value = frame_counter / (now - fps_timer)
                    fps_timer = now
                    frame_counter = 0

                status_color = (0, 255, 0) if line_count else (0, 0, 255)
                cv.putText(overlay_frame, f"Lines: {line_count}", (24, 48), cv.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2, cv.LINE_AA)
                cv.putText(overlay_frame, f"FPS: {fps_value:.1f}", (24, 96), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
                cv.putText(overlay_frame, "Press Q or Esc to quit", (24, overlay_frame.shape[0] - 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)

                cv.imshow(WINDOW_NAME, overlay_frame)

                key = cv.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv.destroyWindow(WINDOW_NAME)


if __name__ == "__main__":
    main()
