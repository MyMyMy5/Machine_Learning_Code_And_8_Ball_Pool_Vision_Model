from __future__ import annotations

from dataclasses import asdict, dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from cv_guideline_common import (
    build_centerline_reference,
    build_guideline_postprocess_context,
    postprocess_guideline_mask,
    sweep_postprocessed_thresholds,
)


@dataclass
class SharedArraySpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str


def share_array(array: np.ndarray) -> tuple[Dict[str, Any], shared_memory.SharedMemory]:
    arr = np.ascontiguousarray(array)
    shm = shared_memory.SharedMemory(create=True, size=int(arr.nbytes))
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[...] = arr
    spec = SharedArraySpec(name=shm.name, shape=tuple(int(x) for x in arr.shape), dtype=str(arr.dtype))
    return asdict(spec), shm


def _attach_array(spec_dict: Dict[str, Any]) -> tuple[np.ndarray, shared_memory.SharedMemory]:
    spec = SharedArraySpec(
        name=str(spec_dict["name"]),
        shape=tuple(int(x) for x in spec_dict["shape"]),
        dtype=str(spec_dict["dtype"]),
    )
    shm = shared_memory.SharedMemory(name=spec.name)
    arr = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=shm.buf)
    return arr, shm


def _render_overlay(frame_bgr: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.32) -> np.ndarray:
    out = frame_bgr.copy()
    m = mask_u8 > 0
    if np.any(m):
        color = np.array([0, 255, 255], dtype=np.float32)
        out[m] = np.clip(out[m].astype(np.float32) * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def train_val_postprocess_shared(payload: Dict[str, Any]) -> Dict[str, Any]:
    image, image_shm = _attach_array(payload["image_spec"])
    gt, gt_shm = _attach_array(payload["gt_spec"])
    prob, prob_shm = _attach_array(payload["prob_spec"])
    try:
        image_np = np.ascontiguousarray(image)
        gt_np = np.ascontiguousarray(gt)
        prob_np = np.ascontiguousarray(prob)
        gt_ref = None if bool(payload["is_negative"]) else build_centerline_reference(gt_np, tol_px=2)
        pp_ctx = build_guideline_postprocess_context(image_np)
        gt_f = gt_np.astype(np.float32)
        prob_clip = np.clip(prob_np, 1e-5, 1.0 - 1e-5)
        bce = float(-(gt_f * np.log(prob_clip) + (1.0 - gt_f) * np.log(1.0 - prob_clip)).mean())
        inter = float((prob_np * gt_f).sum())
        den = float(prob_np.sum() + gt_f.sum())
        dice = 1.0 - ((2.0 * inter + 1e-6) / (den + 1e-6))
        local_thr, _ = sweep_postprocessed_thresholds(
            frame_bgr=image_np,
            prob_np=prob_np,
            thresholds=payload["thresholds"],
            gt_u8=gt_np,
            gt_ref=gt_ref,
            is_negative=bool(payload["is_negative"]),
            axis_complete=bool(payload["axis_complete"]),
            axis_complete_max_gap=int(payload["axis_complete_max_gap"]),
            axis_complete_max_extension=int(payload["axis_complete_max_extension"]),
            axis_complete_min_white_ratio=float(payload["axis_complete_min_white_ratio"]),
            axis_complete_min_dark_border_ratio=float(payload["axis_complete_min_dark_border_ratio"]),
            context=pp_ctx,
        )
        return {
            "index": int(payload["index"]),
            "val_loss": float(0.5 * bce + 0.5 * dice),
            "per_thr": local_thr,
        }
    finally:
        image_shm.close()
        gt_shm.close()
        prob_shm.close()


def eval_postprocess_shared(payload: Dict[str, Any]) -> Dict[str, Any]:
    image, image_shm = _attach_array(payload["image_spec"])
    gt, gt_shm = _attach_array(payload["gt_spec"])
    prob, prob_shm = _attach_array(payload["prob_spec"])
    try:
        image_np = np.ascontiguousarray(image)
        gt_np = np.ascontiguousarray(gt)
        prob_np = np.ascontiguousarray(prob)
        gt_ref = None if bool(payload["is_negative"]) else build_centerline_reference(gt_np, tol_px=2)
        pp_ctx = build_guideline_postprocess_context(image_np)
        preview_thr = payload.get("preview_threshold", None)
        local_thr, preview_mask = sweep_postprocessed_thresholds(
            frame_bgr=image_np,
            prob_np=prob_np,
            thresholds=payload["thresholds"],
            gt_u8=gt_np,
            gt_ref=gt_ref,
            is_negative=bool(payload["is_negative"]),
            axis_complete=bool(payload["axis_complete"]),
            axis_complete_max_gap=int(payload["axis_complete_max_gap"]),
            axis_complete_max_extension=int(payload["axis_complete_max_extension"]),
            axis_complete_min_white_ratio=float(payload["axis_complete_min_white_ratio"]),
            axis_complete_min_dark_border_ratio=float(payload["axis_complete_min_dark_border_ratio"]),
            context=pp_ctx,
            preview_threshold=(float(preview_thr) if preview_thr is not None else None),
        )
        preview_written = False
        preview_path = str(payload.get("preview_path", "")).strip()
        if preview_path and preview_mask is not None:
            t_prev = float(preview_thr) if preview_thr is not None else float(payload["thresholds"][len(payload["thresholds"]) // 2])
            ov_pred = _render_overlay(image_np, preview_mask, alpha=0.32)
            ov_gt = _render_overlay(image_np, gt_np, alpha=0.22)
            canvas = np.hstack([image_np, ov_gt, ov_pred])
            cv2.putText(
                canvas,
                f"{payload['sample_id']} | kind={payload['kind']} | thr={t_prev:.2f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(preview_path, canvas)
            preview_written = True
        return {
            "index": int(payload["index"]),
            "sample_id": str(payload["sample_id"]),
            "kind": str(payload["kind"]),
            "per_thr": local_thr,
            "preview_written": bool(preview_written),
        }
    finally:
        image_shm.close()
        gt_shm.close()
        prob_shm.close()


def infer_postprocess_shared(payload: Dict[str, Any]) -> Dict[str, Any]:
    image, image_shm = _attach_array(payload["image_spec"])
    prob, prob_shm = _attach_array(payload["prob_spec"])
    try:
        image_np = np.ascontiguousarray(image)
        prob_np = np.ascontiguousarray(prob)
        pp_ctx = build_guideline_postprocess_context(image_np)
        mask_bin = postprocess_guideline_mask(
            image_np,
            prob_np,
            threshold=float(payload["threshold"]),
            axis_complete=bool(payload["axis_complete"]),
            axis_complete_max_gap=int(payload["axis_complete_max_gap"]),
            axis_complete_max_extension=int(payload["axis_complete_max_extension"]),
            axis_complete_min_white_ratio=float(payload["axis_complete_min_white_ratio"]),
            axis_complete_min_dark_border_ratio=float(payload["axis_complete_min_dark_border_ratio"]),
            context=pp_ctx,
        )
        mask_u8 = (mask_bin > 0).astype(np.uint8) * 255
        overlay = _render_overlay(image_np, mask_u8, alpha=0.34)
        mask_path = Path(payload["mask_path"])
        overlay_path = Path(payload["overlay_path"])
        original_path = Path(payload["original_path"])
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        original_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), mask_u8)
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(original_path), image_np)
        prob_map_path = str(payload.get("prob_map_path", "")).strip()
        if prob_map_path:
            prob_out = np.clip(prob_np * 255.0, 0, 255).astype(np.uint8)
            out_path = Path(prob_map_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), prob_out)
        return {
            "index": int(payload["index"]),
            "image_path": str(payload["image_path"]),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "original_path": str(original_path),
            "prob_map_path": prob_map_path,
            "pred_area": int((mask_u8 > 0).sum()),
            "threshold": float(payload["threshold"]),
        }
    finally:
        image_shm.close()
        prob_shm.close()
