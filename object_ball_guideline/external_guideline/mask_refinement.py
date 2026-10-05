from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def polygon_to_mask(shape_hw: Tuple[int, int], points: List[Tuple[int, int]]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    out = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 3:
        return out
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(out, [pts], 1)
    return out


def _neighbor_counts(mask_u8: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D((mask_u8 > 0).astype(np.uint8), cv2.CV_16U, kernel, borderType=cv2.BORDER_CONSTANT)


def principal_axis_stats(mask_u8: np.ndarray) -> Optional[Dict]:
    ys, xs = np.where(mask_u8 > 0)
    n = int(xs.shape[0])
    if n < 3:
        return None
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    center = pts.mean(axis=0)
    centered = pts - center[None, :]
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2):
        return None
    vals, vecs = np.linalg.eigh(cov)
    ord_idx = np.argsort(vals)[::-1]
    vals = vals[ord_idx]
    vecs = vecs[:, ord_idx]
    major = vecs[:, 0]
    minor = vecs[:, 1]
    major_var = float(max(1e-8, vals[0]))
    minor_var = float(max(1e-8, vals[1]))
    elong = float((major_var + 1e-8) / (minor_var + 1e-8))
    major_std = float(np.sqrt(major_var))
    minor_std = float(np.sqrt(minor_var))
    return {
        "center_xy": (float(center[0]), float(center[1])),
        "major_vec": (float(major[0]), float(major[1])),
        "minor_vec": (float(minor[0]), float(minor[1])),
        "major_var": major_var,
        "minor_var": minor_var,
        "major_std": major_std,
        "minor_std": minor_std,
        "elongation": elong,
        "count": n,
    }


def component_masks(mask_u8: np.ndarray, min_area: int) -> List[np.ndarray]:
    src = (mask_u8 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    out: List[np.ndarray] = []
    min_a = max(1, int(min_area))
    for i in range(1, int(n)):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_a:
            continue
        out.append((labels == i).astype(np.uint8))
    out.sort(key=lambda m: int(m.sum()), reverse=True)
    return out


def prune_axis_spurs(mask_u8: np.ndarray, max_iters: int = 3) -> Tuple[np.ndarray, Dict]:
    current = (mask_u8 > 0).astype(np.uint8)
    debug: Dict = {"spur_iters": 0, "spur_removed_px": 0}
    if int(current.sum()) <= 0:
        return current, debug

    for _ in range(max(0, int(max_iters))):
        axis = principal_axis_stats(current)
        if axis is None:
            break

        ys, xs = np.where(current > 0)
        if xs.size <= 0:
            break

        cx, cy = axis["center_xy"]
        ux, uy = axis["major_vec"]
        rel_x = xs.astype(np.float32) - float(cx)
        rel_y = ys.astype(np.float32) - float(cy)
        normal_dist = np.abs(rel_x * float(-uy) + rel_y * float(ux))

        # Keep the main body intact and only trim obvious tiny off-axis nubs.
        band_q = float(np.percentile(normal_dist, 82.0))
        spur_thresh = band_q + max(0.45, 0.22 * max(1.0, band_q))

        neighbors = _neighbor_counts(current)
        remove = np.zeros_like(current, dtype=np.uint8)
        for x, y, nd in zip(xs.tolist(), ys.tolist(), normal_dist.tolist()):
            ncount = int(neighbors[int(y), int(x)])
            if ncount <= 1 and float(nd) > max(0.75, band_q + 0.10):
                remove[int(y), int(x)] = 1
            elif ncount <= 2 and float(nd) > float(spur_thresh):
                remove[int(y), int(x)] = 1

        removed_px = int(remove.sum())
        if removed_px <= 0:
            break

        # Refuse aggressive cleanup; this pass is only for 1-2 px spur artifacts.
        if removed_px > max(3, int(round(0.08 * float(current.sum())))):
            break

        next_mask = np.logical_and(current > 0, remove == 0).astype(np.uint8)
        comps = component_masks(next_mask, min_area=1)
        if not comps:
            break
        next_mask = comps[0]
        if int(next_mask.sum()) <= 0:
            break

        debug["spur_iters"] = int(debug["spur_iters"]) + 1
        debug["spur_removed_px"] = int(debug["spur_removed_px"]) + removed_px
        current = next_mask

    return current, debug


def dark_shell_ratio(
    gray_u8: np.ndarray,
    mask_u8: np.ndarray,
    ring_px: int,
    dark_threshold: int,
    allowed_u8: Optional[np.ndarray] = None,
) -> float:
    mask = (mask_u8 > 0).astype(np.uint8)
    if int(mask.sum()) <= 0:
        return 0.0
    ring = max(1, int(ring_px))
    shell = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=ring)
    shell = np.logical_and(shell > 0, mask == 0).astype(np.uint8)
    if allowed_u8 is not None:
        shell = np.logical_and(shell > 0, allowed_u8 > 0).astype(np.uint8)
    shell_area = int(shell.sum())
    if shell_area <= 0:
        return 0.0
    dark = (gray_u8 <= int(dark_threshold)).astype(np.uint8)
    dark_count = int(np.logical_and(shell > 0, dark > 0).sum())
    return float(dark_count) / float(max(1, shell_area))


def refine_line_mask_from_polygon(
    frame_bgr: np.ndarray,
    polygon_mask_u8: np.ndarray,
    min_area: int,
    min_elongation: float,
    dark_threshold: int,
    ring_px: int,
) -> Tuple[np.ndarray, Dict]:
    poly = (polygon_mask_u8 > 0).astype(np.uint8)
    poly_area = int(poly.sum())
    info: Dict = {"mode": "refine_line_v1", "poly_area": poly_area}
    if poly_area <= 0:
        info["reason"] = "empty_polygon"
        return np.zeros_like(poly, dtype=np.uint8), info

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    m = poly > 0
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    s_vals = s[m]
    v_vals = v[m]
    a_dev = np.abs(a - 128.0)
    b_dev = np.abs(b - 128.0)
    chroma_dev = a_dev + b_dev
    chroma_vals = chroma_dev[m]
    if s_vals.size <= 0 or v_vals.size <= 0:
        info["reason"] = "empty_polygon_values"
        return np.zeros_like(poly, dtype=np.uint8), info

    v_seed = float(max(145.0, np.percentile(v_vals, 58.0)))
    s_seed = float(min(125.0, max(10.0, np.percentile(s_vals, 45.0))))
    c_seed = float(min(70.0, max(10.0, np.percentile(chroma_vals, 65.0))))
    white_strict = np.logical_and.reduce(
        (
            poly > 0,
            v >= v_seed,
            s <= s_seed,
            chroma_dev <= c_seed,
        )
    ).astype(np.uint8)

    min_seed_area = max(int(min_area), int(round(0.003 * float(poly_area))))
    if int(white_strict.sum()) < min_seed_area:
        v_fallback = float(max(138.0, np.percentile(v_vals, 70.0)))
        s_fallback = float(min(145.0, np.percentile(s_vals, 62.0) + 18.0))
        c_fallback = float(min(90.0, np.percentile(chroma_vals, 80.0) + 18.0))
        white_strict = np.logical_and.reduce(
            (
                poly > 0,
                v >= v_fallback,
                s <= s_fallback,
                chroma_dev <= c_fallback,
            )
        ).astype(np.uint8)

    s_relaxed = min(255.0, s_seed + 50.0)
    v_relaxed = max(90.0, v_seed - 40.0)
    c_relaxed = min(140.0, c_seed + 38.0)
    white_relaxed = np.logical_and.reduce(
        (
            poly > 0,
            v >= v_relaxed,
            s <= s_relaxed,
            chroma_dev <= c_relaxed,
        )
    ).astype(np.uint8)
    white_relaxed = cv2.morphologyEx(white_relaxed, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    comps = component_masks(white_strict, min_area=max(2, int(min_area)))
    if not comps:
        comps = component_masks(white_relaxed, min_area=max(2, int(min_area)))
    if not comps:
        info["reason"] = "no_white_component"
        return np.zeros_like(poly, dtype=np.uint8), info

    best_score = -1e9
    best_mask = np.zeros_like(poly, dtype=np.uint8)
    best_debug: Dict = {}

    ys_poly, xs_poly = np.where(np.logical_and(poly > 0, white_relaxed > 0))
    if xs_poly.size > 0:
        poly_xy = np.column_stack((xs_poly, ys_poly)).astype(np.float32)
    else:
        poly_xy = np.zeros((0, 2), dtype=np.float32)

    for comp in comps:
        area = int(comp.sum())
        if area < int(min_area):
            continue
        axis = principal_axis_stats(comp)
        if axis is None:
            continue
        elong = float(axis["elongation"])
        if elong < float(min_elongation) and area >= max(10, int(min_area) * 2):
            continue

        comp_u8 = (comp > 0).astype(np.uint8)
        ys, xs = np.where(comp_u8 > 0)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bw = max(1, x2 - x1 + 1)
        bh = max(1, y2 - y1 + 1)
        fill = float(area) / float(max(1, bw * bh))
        white_support = float(np.logical_and(comp_u8 > 0, white_relaxed > 0).sum()) / float(max(1, area))
        dark_ratio = dark_shell_ratio(
            gray_u8=gray,
            mask_u8=comp_u8,
            ring_px=int(ring_px),
            dark_threshold=int(dark_threshold),
            allowed_u8=poly,
        )

        grown = comp_u8.copy()
        if poly_xy.shape[0] > 0:
            cx, cy = axis["center_xy"]
            ux, uy = axis["major_vec"]
            rel = np.column_stack((poly_xy[:, 0] - cx, poly_xy[:, 1] - cy)).astype(np.float32)
            t = rel[:, 0] * float(ux) + rel[:, 1] * float(uy)
            n = np.abs(rel[:, 0] * float(-uy) + rel[:, 1] * float(ux))
            rel_c = np.column_stack((xs.astype(np.float32) - cx, ys.astype(np.float32) - cy))
            t_c = rel_c[:, 0] * float(ux) + rel_c[:, 1] * float(uy)
            t_min = float(t_c.min()) - max(2.0, float(axis["major_std"]) * 0.18 + 1.2)
            t_max = float(t_c.max()) + max(2.0, float(axis["major_std"]) * 0.18 + 1.2)
            half_width = float(max(1.6, axis["minor_std"] * 3.0 + 1.2))

            near_axis = n <= half_width
            if np.any(near_axis):
                t_near = t[near_axis]
                if t_near.size > 0:
                    t_min = min(t_min, float(np.percentile(t_near, 1.0)) - 1.0)
                    t_max = max(t_max, float(np.percentile(t_near, 99.0)) + 1.0)

            keep = near_axis & (t >= t_min) & (t <= t_max)
            if np.any(keep):
                xk = poly_xy[keep, 0].astype(np.int32)
                yk = poly_xy[keep, 1].astype(np.int32)
                allowed = np.zeros_like(poly, dtype=np.uint8)
                allowed[yk, xk] = 1
                allowed = cv2.dilate(allowed, np.ones((3, 3), dtype=np.uint8), iterations=1)
                allowed = np.logical_and(allowed > 0, poly > 0).astype(np.uint8)
                allowed = np.logical_and(allowed > 0, white_relaxed > 0).astype(np.uint8)

                seed = np.logical_and(comp_u8 > 0, allowed > 0).astype(np.uint8)
                if int(seed.sum()) > 0:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    growth_span = max(1.0, float(t_max - t_min))
                    max_grow_iters = int(max(20.0, min(256.0, np.ceil(growth_span + 6.0))))
                    for _ in range(max_grow_iters):
                        dil = cv2.dilate(seed, kernel, iterations=1)
                        nxt = np.logical_and(dil > 0, allowed > 0).astype(np.uint8)
                        merged = np.logical_or(seed > 0, nxt > 0).astype(np.uint8)
                        if int(np.logical_xor(merged > 0, seed > 0).sum()) <= 0:
                            break
                        seed = merged
                    grown = seed
                grown = np.logical_and(grown > 0, poly > 0).astype(np.uint8)
                grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
                keep_comps = component_masks(grown, min_area=max(2, int(min_area)))
                if keep_comps:
                    keep_comps = sorted(keep_comps, key=lambda m_: int(np.logical_and(m_ > 0, comp_u8 > 0).sum()), reverse=True)
                    grown = keep_comps[0]

        grown_area = int(grown.sum())
        grown_white = float(np.logical_and(grown > 0, white_relaxed > 0).sum()) / float(max(1, grown_area))
        grown_dark = dark_shell_ratio(
            gray_u8=gray,
            mask_u8=grown,
            ring_px=int(ring_px),
            dark_threshold=int(dark_threshold),
            allowed_u8=poly,
        )
        score = (
            1.55 * grown_white
            + 1.10 * grown_dark
            + 0.75 * min(10.0, elong) / 10.0
            - 0.45 * fill
        )
        if score > best_score:
            best_score = float(score)
            best_mask = grown.astype(np.uint8)
            best_debug = {
                "component_area": int(area),
                "grown_area": int(grown_area),
                "elongation": float(elong),
                "fill_ratio": float(fill),
                "white_support": float(grown_white),
                "dark_border_ratio": float(grown_dark),
                "score": float(score),
            }

    if int(best_mask.sum()) <= 0:
        info["reason"] = "no_component_passed"
        return np.zeros_like(poly, dtype=np.uint8), info

    best_mask = np.logical_and(best_mask > 0, poly > 0).astype(np.uint8)
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    comps_final = component_masks(best_mask, min_area=max(2, int(min_area)))
    if comps_final:
        best_mask = max(comps_final, key=lambda m_: int(m_.sum()))

    spur_debug: Dict = {"spur_iters": 0, "spur_removed_px": 0}
    cleaned_mask, spur_debug = prune_axis_spurs(best_mask, max_iters=3)
    if int(cleaned_mask.sum()) > 0:
        best_mask = cleaned_mask

    info.update(
        {
            "reason": "ok",
            "seed_area": int(white_strict.sum()),
            "relaxed_white_area": int(white_relaxed.sum()),
            "best": best_debug,
            "final_area": int(best_mask.sum()),
            "spur_cleanup": spur_debug,
            "v_seed": float(v_seed),
            "s_seed": float(s_seed),
            "chroma_seed": float(c_seed),
        }
    )
    return (best_mask > 0).astype(np.uint8), info
