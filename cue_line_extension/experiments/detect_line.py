"""
Utility script to detect the straight guide line emerging from the cue ball in a pool table
scene.  This implementation uses colour segmentation to isolate the bright white and
green guide line, computes the predominant line orientation, and then applies a
probabilistic Hough transform to extract candidate line segments.  Only the longest
segment whose orientation is consistent with the dominant direction is retained.

The approach avoids any hard‑coded pixel coordinates: it works by analysing the
geometric properties of the segmented mask, so it adapts to different cue positions
and orientations on the table.
"""

import math
from typing import List, Tuple, Optional

import cv2 as cv
import numpy as np


def _build_mask(hsv: np.ndarray) -> np.ndarray:
    """Return a binary mask selecting the bright white and bright green parts of the guide line.

    Parameters
    ----------
    hsv : numpy.ndarray
        The input image in HSV colour space.

    Returns
    -------
    numpy.ndarray
        A binary mask where pixels corresponding to the guide line are set to 255.
    """
    # white portions: low saturation, high value
    low_white = np.array([0, 0, 200], dtype=np.uint8)
    high_white = np.array([180, 40, 255], dtype=np.uint8)
    mask_white = cv.inRange(hsv, low_white, high_white)

    # green portions: hue around 60°, high saturation and value
    low_green = np.array([35, 80, 100], dtype=np.uint8)
    high_green = np.array([85, 255, 255], dtype=np.uint8)
    mask_green = cv.inRange(hsv, low_green, high_green)

    # combine both colour masks
    mask = cv.bitwise_or(mask_white, mask_green)

    # close small gaps to produce continuous line segments
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _predominant_orientation(mask: np.ndarray) -> float:
    """Estimate the predominant orientation of non‑zero pixels in the mask.

    The direction vector is computed using a least‑squares fit (cv.fitLine) over the
    coordinates of the mask, which is more robust than simply averaging Hough
    parameters.  The returned angle is in degrees and ranges from −180 to 180.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask from which to extract orientation.

    Returns
    -------
    float
        The dominant angle in degrees of the guide line.  If the mask is empty,
        a default angle of 0° is returned.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return 0.0
    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    vx, vy, _, _ = cv.fitLine(coords, cv.DIST_L2, 0, 0.01, 0.01)
    angle = math.degrees(math.atan2(float(vy), float(vx)))
    return angle


def _filter_and_select_line(
    lines: Optional[np.ndarray],
    orientation: float,
    angle_thresh: float = 20.0,
    min_length: float = 50.0,
) -> Optional[Tuple[int, int, int, int]]:
    """Filter Hough lines by orientation and length, then return the longest.

    Parameters
    ----------
    lines : numpy.ndarray or None
        Output of cv.HoughLinesP (an array of line segments).
    orientation : float
        Dominant angle computed from the mask.
    angle_thresh : float, optional
        Allowed deviation in degrees from the dominant orientation for a segment to
        be considered part of the guide line.  Defaults to 20°.
    min_length : float, optional
        Minimum length, in pixels, for a candidate segment.  Defaults to 50.

    Returns
    -------
    tuple or None
        The endpoints (x1, y1, x2, y2) of the longest qualifying segment.  If no
        suitable segment is found, returns None.
    """
    if lines is None:
        return None
    candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for (x1, y1, x2, y2) in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_length:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        diff = (angle - orientation + 180.0) % 360.0 - 180.0
        if abs(diff) < angle_thresh:
            candidates.append((length, (int(x1), int(y1), int(x2), int(y2))))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def detect_guide_line(image_path: str) -> np.ndarray:
    """Detect and overlay the straight guide line emerging from the cue ball.

    The function reads the supplied image, segments the guide line via HSV
    thresholding, estimates its predominant orientation, performs a probabilistic
    Hough transform and selects the most representative segment.  The chosen line
    is drawn in green on a copy of the original RGB image and returned.

    Parameters
    ----------
    image_path : str
        Path to the input image file.

    Returns
    -------
    numpy.ndarray
        A copy of the input image in RGB space with the detected guide line drawn.
    """
    img_bgr = cv.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    mask = _build_mask(hsv)
    orientation = _predominant_orientation(mask)
    edges = cv.Canny(mask, 50, 150)
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=30,
        minLineLength=30,
        maxLineGap=20,
    )
    best = _filter_and_select_line(lines, orientation, angle_thresh=20.0, min_length=50.0)
    result = img_rgb.copy()
    if best is not None:
        x1, y1, x2, y2 = best
        cv.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return result


if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(
        description="Detect the straight guide line in a pool table scene and save the result."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "output",
        help="Path to save the output image with the detected guide line overlaid",
    )
    args = parser.parse_args()
    output_img = detect_guide_line(args.image)
    # convert back to BGR for saving via OpenCV
    cv.imwrite(args.output, cv.cvtColor(output_img, cv.COLOR_RGB2BGR))