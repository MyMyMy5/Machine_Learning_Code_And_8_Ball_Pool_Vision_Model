from __future__ import annotations

import argparse
import gc
import json
import math
import random
from concurrent.futures import Executor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from inference_runtime import AsyncInferenceRuntime, InferenceJob, build_process_pool, run_async_pipeline
from shared_runtime_postprocess import share_array, train_val_postprocess_shared

from cv_guideline_common import (
    AnnotationRecord,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NegativeRecord,
    append_jsonl,
    apply_augment_pair,
    bgr_to_normalized_tensor,
    build_centerline_reference,
    build_guideline_postprocess_context,
    build_segmentation_model,
    build_negative_records,
    cldice_loss,
    build_sobel_kernels,
    edge_map_from_prob,
    load_annotation_records,
    load_bgr,
    load_mask_u8,
    load_split_ids,
    load_checkpoint_model,
    parse_csv_floats,
    parse_csv_ints,
    parse_crop_mode_mix,
    precision_recall_f1,
    predict_prob_multi_tta,
    random_background_crop_from_positive,
    random_crop_negative_image,
    random_positive_crop,
    random_split_ids,
    resolve_path,
    safe_div,
    split_by_ids,
    split_negatives,
    stratified_annotation_split,
    str2bool,
    sweep_postprocessed_thresholds,
    utc_now_iso,
    write_json,
)


DEFAULT_ANNOTATIONS = Path("guideline_line/data_zoomprobe_merged/annotations.jsonl")
DEFAULT_SPLITS = Path("guideline_line/data_zoomprobe_merged/splits.json")


def threshold_score(
    pixel_precision: float,
    pixel_recall: float,
    pixel_f1: float,
    centerline_f1: float,
    fp_per_frame: float,
    mode: str,
) -> float:
    mode_s = str(mode).strip().lower()
    if mode_s == "precision":
        return float((0.50 * pixel_precision) + (0.30 * pixel_f1) + (0.20 * centerline_f1) - (0.20 * fp_per_frame))
    if mode_s == "centerline":
        return float(
            (0.10 * pixel_precision)
            + (0.20 * pixel_recall)
            + (0.10 * pixel_f1)
            + (0.60 * centerline_f1)
            - (0.20 * fp_per_frame)
        )
    return float(
        (0.20 * pixel_precision)
        + (0.25 * pixel_recall)
        + (0.20 * pixel_f1)
        + (0.35 * centerline_f1)
        - (0.20 * fp_per_frame)
    )


def build_run_dir(path_arg: Optional[str]) -> Path:
    if path_arg:
        return resolve_path(path_arg)
    import datetime as dt

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    return resolve_path(f"guideline_line/runs_cv/run_{ts}")


def size_bucket_label(major_len: float, thresholds: Sequence[float]) -> str:
    bounds = sorted(float(x) for x in thresholds)
    for threshold in bounds:
        if float(major_len) < threshold:
            return f"<{threshold:g}"
    if not bounds:
        return "all"
    return f">={bounds[-1]:g}"


def summarize_positive_records(records: Sequence[AnnotationRecord], size_thresholds: Sequence[float]) -> Dict[str, object]:
    source_counts: Dict[str, int] = {}
    size_counts: Dict[str, int] = {}
    refined_counts = {"refined": 0, "not_refined": 0}
    major_lengths: List[float] = []
    for rec in records:
        source_counts[rec.source] = int(source_counts.get(rec.source, 0)) + 1
        bucket = size_bucket_label(rec.mask_major_len, size_thresholds)
        size_counts[bucket] = int(size_counts.get(bucket, 0)) + 1
        if bool(rec.refined):
            refined_counts["refined"] += 1
        else:
            refined_counts["not_refined"] += 1
        major_lengths.append(float(rec.mask_major_len))

    major_stats: Dict[str, float] = {}
    if major_lengths:
        arr = np.asarray(major_lengths, dtype=np.float32)
        major_stats = {
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    ordered_size_counts: Dict[str, int] = {}
    for upper in sorted(float(x) for x in size_thresholds):
        label = f"<{upper:g}"
        ordered_size_counts[label] = int(size_counts.get(label, 0))
    tail_label = f">={sorted(float(x) for x in size_thresholds)[-1]:g}" if size_thresholds else "all"
    ordered_size_counts[tail_label] = int(size_counts.get(tail_label, 0))

    return {
        "count": int(len(records)),
        "source_counts": dict(sorted(source_counts.items(), key=lambda item: item[0])),
        "size_bucket_counts": ordered_size_counts,
        "refined_counts": refined_counts,
        "major_len_stats": major_stats,
    }


def canonical_path_key(path_like: Path | str) -> str:
    path = resolve_path(str(path_like))
    return str(path).replace("/", "\\").lower()


def load_manifest_payload(manifest_path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse manifest JSON {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Manifest must be a JSON object: {manifest_path}")
    return payload


def manifest_positive_sample_ids(payload: Dict[str, object]) -> Set[str]:
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        return set()
    raw_ids = meta.get("positive_sample_ids", [])
    if not isinstance(raw_ids, list):
        return set()
    return {str(x).strip() for x in raw_ids if str(x).strip()}


def manifest_image_keys(payload: Dict[str, object]) -> Set[str]:
    raw_images = payload.get("images", [])
    if not isinstance(raw_images, list):
        return set()
    out: Set[str] = set()
    for raw_path in raw_images:
        text = str(raw_path).strip()
        if not text:
            continue
        out.add(canonical_path_key(text))
    return out


def resolve_hardcase_positive_ids(
    payload: Dict[str, object],
    train_pos_records: Sequence[AnnotationRecord],
) -> Set[str]:
    requested_ids = manifest_positive_sample_ids(payload)
    requested_image_keys = manifest_image_keys(payload)
    matched: Set[str] = set()
    for rec in train_pos_records:
        image_key = canonical_path_key(rec.image_path)
        if rec.sample_id in requested_ids or image_key in requested_image_keys:
            matched.add(str(rec.sample_id))
    return matched


def resolve_hardcase_negative_image_keys(
    payload: Dict[str, object],
    train_neg_records: Sequence[NegativeRecord],
) -> Set[str]:
    requested_image_keys = manifest_image_keys(payload)
    matched: Set[str] = set()
    for rec in train_neg_records:
        image_key = canonical_path_key(rec.image_path)
        if image_key in requested_image_keys:
            matched.add(image_key)
    return matched


def build_loss_target_mask(
    mask_u8: np.ndarray,
    mode: str = "binary",
    soft_distance_scale: float = 1.0,
) -> np.ndarray:
    hard = (mask_u8 > 0).astype(np.float32)
    mode_s = str(mode).strip().lower()
    if mode_s == "binary":
        return hard
    if mode_s != "soft_distance":
        raise ValueError(f"Unsupported target mode: {mode}")
    if int(hard.sum()) <= 0:
        return hard

    fg = (hard > 0).astype(np.uint8)
    bg = (fg == 0).astype(np.uint8)
    dist_in = cv2.distanceTransform(fg, cv2.DIST_L2, 3).astype(np.float32)
    dist_out = cv2.distanceTransform(bg, cv2.DIST_L2, 3).astype(np.float32)
    signed = dist_in - dist_out
    scale = max(0.25, float(soft_distance_scale))
    logits = np.clip(signed / scale, -20.0, 20.0)
    soft = 1.0 / (1.0 + np.exp(-logits))
    soft = np.maximum(soft, hard * 0.85)
    return np.clip(soft.astype(np.float32), 0.0, 1.0)


class TrainSampleDataset(Dataset):
    def __init__(
        self,
        pos_records: Sequence[AnnotationRecord],
        ext_neg_records: Sequence[NegativeRecord],
        img_size: int,
        neg_per_pos: float,
        augment: bool,
        seed: int,
        crop_mode_mix: Optional[Dict[str, float]] = None,
        crop_mode_mix_small: Optional[Dict[str, float]] = None,
        crop_mode_mix_ultra_tiny: Optional[Dict[str, float]] = None,
        small_line_major_thresh: float = 0.0,
        ultra_tiny_major_thresh: float = 0.0,
        small_line_weight_mult: float = 1.0,
        ultra_tiny_weight_mult: float = 1.0,
        refined_weight_mult: float = 1.0,
        hardcase_positive_ids: Optional[Set[str]] = None,
        hardcase_negative_image_keys: Optional[Set[str]] = None,
        hardcase_positive_weight_mult: float = 1.0,
        hardcase_negative_weight_mult: float = 1.0,
        input_mean: Optional[np.ndarray] = None,
        input_std: Optional[np.ndarray] = None,
        target_mode: str = "binary",
        target_soft_distance_scale: float = 1.0,
    ):
        self.pos_records = list(pos_records)
        self.ext_neg_records = list(ext_neg_records)
        self.img_size = int(img_size)
        self.neg_per_pos = max(0.0, float(neg_per_pos))
        self.augment = bool(augment)
        self.seed = int(seed)
        self.crop_mode_mix = dict(crop_mode_mix or {"tight": 0.30, "context": 0.45, "fullframe": 0.25})
        self.crop_mode_mix_small = dict(crop_mode_mix_small or self.crop_mode_mix)
        self.crop_mode_mix_ultra_tiny = dict(crop_mode_mix_ultra_tiny or self.crop_mode_mix_small)
        self.small_line_major_thresh = max(0.0, float(small_line_major_thresh))
        self.ultra_tiny_major_thresh = max(0.0, float(ultra_tiny_major_thresh))
        self.small_line_weight_mult = max(1.0, float(small_line_weight_mult))
        self.ultra_tiny_weight_mult = max(1.0, float(ultra_tiny_weight_mult))
        self.refined_weight_mult = max(1.0, float(refined_weight_mult))
        self.hardcase_positive_ids = {str(x) for x in (hardcase_positive_ids or set()) if str(x)}
        self.hardcase_negative_image_keys = {str(x) for x in (hardcase_negative_image_keys or set()) if str(x)}
        self.hardcase_positive_weight_mult = max(1.0, float(hardcase_positive_weight_mult))
        self.hardcase_negative_weight_mult = max(1.0, float(hardcase_negative_weight_mult))
        self.input_mean = IMAGENET_MEAN if input_mean is None else np.asarray(input_mean, dtype=np.float32)
        self.input_std = IMAGENET_STD if input_std is None else np.asarray(input_std, dtype=np.float32)
        self.target_mode = str(target_mode).strip().lower() or "binary"
        self.target_soft_distance_scale = max(0.25, float(target_soft_distance_scale))

        self.pos_count = len(self.pos_records)
        self.neg_target_total = int(round(self.pos_count * self.neg_per_pos))
        if self.ext_neg_records:
            self.neg_intra_target = int(math.ceil(self.neg_target_total * 0.5))
        else:
            self.neg_intra_target = int(self.neg_target_total)
        self.neg_ext_target = max(0, self.neg_target_total - self.neg_intra_target)
        self.total_count = self.pos_count + self.neg_intra_target + self.neg_ext_target

    def __len__(self) -> int:
        return max(1, self.total_count)

    def _rng(self, idx: int) -> random.Random:
        return random.Random((self.seed * 1000003) + int(idx))

    def _crop_mix_for_record(self, rec: AnnotationRecord) -> Dict[str, float]:
        major = float(rec.mask_major_len)
        if self.ultra_tiny_major_thresh > 0.0 and major > 0.0 and major < self.ultra_tiny_major_thresh:
            return self.crop_mode_mix_ultra_tiny
        if self.small_line_major_thresh > 0.0 and major > 0.0 and major < self.small_line_major_thresh:
            return self.crop_mode_mix_small
        return self.crop_mode_mix

    def _positive_sample_weight(self, rec: AnnotationRecord) -> float:
        weight = float(rec.sample_weight)
        major = float(rec.mask_major_len)
        if self.ultra_tiny_major_thresh > 0.0 and major > 0.0 and major < self.ultra_tiny_major_thresh:
            weight *= self.ultra_tiny_weight_mult
        elif self.small_line_major_thresh > 0.0 and major > 0.0 and major < self.small_line_major_thresh:
            weight *= self.small_line_weight_mult
        if bool(rec.refined):
            weight *= self.refined_weight_mult
        if rec.sample_id in self.hardcase_positive_ids:
            weight *= self.hardcase_positive_weight_mult
        return float(weight)

    def _load_positive(self, rec: AnnotationRecord, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
        img = load_bgr(rec.image_path)
        mask = load_mask_u8(rec.mask_path)
        img_c, mask_c = random_positive_crop(img, mask, self.img_size, rng, crop_mode_mix=self._crop_mix_for_record(rec))
        if self.augment:
            img_c, mask_c = apply_augment_pair(img_c, mask_c, rng)
        return img_c, mask_c

    def _load_intra_negative(self, rec: AnnotationRecord, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
        img = load_bgr(rec.image_path)
        mask = load_mask_u8(rec.mask_path)
        img_c, mask_c = random_background_crop_from_positive(img, mask, self.img_size, rng)
        if self.augment:
            img_c, mask_c = apply_augment_pair(img_c, mask_c, rng)
        return img_c, mask_c

    def _load_external_negative(self, rec: NegativeRecord, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
        img = load_bgr(rec.image_path)
        img_c, mask_c = random_crop_negative_image(img, self.img_size, rng)
        if self.augment:
            img_c, mask_c = apply_augment_pair(img_c, mask_c, rng)
        return img_c, mask_c

    def __getitem__(self, idx: int):
        idx = int(idx)
        rng = self._rng(idx)
        if self.pos_count <= 0:
            raise RuntimeError("No positive records available for training.")

        if idx < self.pos_count:
            rec = self.pos_records[idx % self.pos_count]
            img_c, mask_c = self._load_positive(rec, rng)
            weight = self._positive_sample_weight(rec)
            sample_type = "positive"
            sid = rec.sample_id
            source = rec.source
            is_negative = 0
        elif idx < self.pos_count + self.neg_intra_target:
            pos_idx = (idx - self.pos_count) % self.pos_count
            rec = self.pos_records[pos_idx]
            img_c, mask_c = self._load_intra_negative(rec, rng)
            weight = float(max(0.75, rec.sample_weight * 0.95))
            sample_type = "intra_negative"
            sid = rec.sample_id
            source = "intra_negative_patch"
            is_negative = 1
        else:
            if not self.ext_neg_records:
                pos_idx = (idx - self.pos_count) % self.pos_count
                rec = self.pos_records[pos_idx]
                img_c, mask_c = self._load_intra_negative(rec, rng)
                weight = float(max(0.75, rec.sample_weight * 0.95))
                sample_type = "intra_negative_fallback"
                sid = rec.sample_id
                source = "intra_negative_patch"
            else:
                n_idx = (idx - self.pos_count - self.neg_intra_target) % len(self.ext_neg_records)
                nrec = self.ext_neg_records[n_idx]
                img_c, mask_c = self._load_external_negative(nrec, rng)
                weight = 1.0
                if canonical_path_key(nrec.image_path) in self.hardcase_negative_image_keys:
                    weight *= self.hardcase_negative_weight_mult
                sample_type = "external_negative"
                sid = nrec.sample_id
                source = nrec.source
            is_negative = 1

        image_t = bgr_to_normalized_tensor(img_c, mean=self.input_mean, std=self.input_std)
        mask_hard_np = (mask_c > 0).astype(np.float32)
        mask_loss_np = build_loss_target_mask(
            mask_c,
            mode=self.target_mode,
            soft_distance_scale=self.target_soft_distance_scale,
        )
        mask_t = torch.from_numpy(mask_hard_np).unsqueeze(0)
        mask_loss_t = torch.from_numpy(mask_loss_np.astype(np.float32)).unsqueeze(0)
        return {
            "image": image_t,
            "mask": mask_t,
            "mask_loss": mask_loss_t,
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
            "is_negative": torch.tensor(is_negative, dtype=torch.long),
            "sample_type": sample_type,
            "sample_id": sid,
            "source": source,
        }


def collate_train(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    masks_loss = torch.stack([b["mask_loss"] for b in batch], dim=0)
    weights = torch.stack([b["sample_weight"] for b in batch], dim=0)
    is_neg = torch.stack([b["is_negative"] for b in batch], dim=0)
    sample_type = [b["sample_type"] for b in batch]
    sample_id = [b["sample_id"] for b in batch]
    source = [b["source"] for b in batch]
    return images, masks, masks_loss, weights, is_neg, sample_type, sample_id, source


def tensor_losses_per_sample(
    outputs: Dict[str, Optional[torch.Tensor]],
    target_hard: torch.Tensor,
    target_loss: Optional[torch.Tensor],
    model_family: str,
    pos_weight: float,
    sobel_kx: torch.Tensor,
    sobel_ky: torch.Tensor,
    loss_ce_weight: float,
    loss_dice_weight: float,
    loss_cldice_weight: float,
    loss_edge_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    family = str(model_family).strip().lower()
    positive_prob = outputs["positive_prob"]
    if positive_prob is None:
        raise RuntimeError("Model outputs missing positive_prob.")
    target_for_loss = target_hard if target_loss is None else target_loss

    if family == "custom":
        logits = outputs["binary_logits"]
        ce_raw = F.binary_cross_entropy_with_logits(
            logits,
            target_for_loss,
            reduction="none",
            pos_weight=torch.tensor([float(pos_weight)], dtype=logits.dtype, device=logits.device),
        )
        ce = ce_raw.flatten(1).mean(dim=1)
    else:
        class_logits = outputs["class_logits"]
        if class_logits is None:
            raise RuntimeError(f"Model family {family} requires class_logits for CE loss.")
        labels = target_hard[:, 0].long()
        ce_raw = F.cross_entropy(class_logits, labels, reduction="none")
        ce = ce_raw.flatten(1).mean(dim=1)

    inter = (positive_prob * target_for_loss).flatten(1).sum(dim=1)
    den = positive_prob.flatten(1).sum(dim=1) + target_for_loss.flatten(1).sum(dim=1)
    dice = 1.0 - (2.0 * inter + 1e-6) / (den + 1e-6)
    cld = cldice_loss(positive_prob, target_hard, iters=10)
    pred_edge = edge_map_from_prob(positive_prob, sobel_kx, sobel_ky)
    tgt_edge = edge_map_from_prob(target_for_loss, sobel_kx, sobel_ky)
    edge = (pred_edge - tgt_edge).abs().flatten(1).mean(dim=1)
    total = (
        (float(loss_ce_weight) * ce)
        + (float(loss_dice_weight) * dice)
        + (float(loss_cldice_weight) * cld)
        + (float(loss_edge_weight) * edge)
    )
    return total, ce, dice, cld, edge


def evaluate_model(
    model: nn.Module,
    val_pos_records: Sequence[AnnotationRecord],
    val_neg_records: Sequence[NegativeRecord],
    device: torch.device,
    tile_sizes: Sequence[int],
    tile_overlaps: Sequence[int],
    tta_scales: Sequence[float],
    tta_hflip: bool,
    use_amp: bool,
    thresholds: Sequence[float],
    selection_metric: str,
    axis_complete: bool,
    axis_complete_max_gap: int,
    axis_complete_max_extension: int,
    axis_complete_min_white_ratio: float,
    axis_complete_min_dark_border_ratio: float,
    val_runtime: str = "direct",
    val_loader_workers: int = 4,
    val_postprocess_workers: int = 4,
    val_prefetch_items: int = 16,
    val_max_batch_tiles: int = 32,
    val_max_wait_ms: int = 4,
    val_bucket_by_tile: bool = True,
    val_postprocess_backend: str = "thread",
    val_items_cache: Optional[Sequence[Dict]] = None,
    val_postprocess_executor: Optional[Executor] = None,
) -> Dict:
    model.eval()
    per_thr = {
        float(t): {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "cl_tp": 0,
            "cl_fp": 0,
            "cl_fn": 0,
            "neg_frames": 0,
            "neg_fp_frames": 0,
        }
        for t in thresholds
    }
    val_loss_terms: List[float] = []
    samples_total = 0

    if val_items_cache is not None:
        all_items = list(val_items_cache)
    else:
        all_items: List[Dict] = build_validation_items(val_pos_records, val_neg_records)
    runtime_stats: Optional[Dict[str, float]] = None
    runtime_mode = str(val_runtime).strip().lower()

    if runtime_mode == "async":
        runtime = AsyncInferenceRuntime(
            model=model,
            device=device,
            max_batch_tiles=int(val_max_batch_tiles),
            max_wait_ms=int(val_max_wait_ms),
            bucket_by_tile=bool(val_bucket_by_tile),
        )
        postprocess_executor = val_postprocess_executor
        owns_postprocess_executor = False
        try:
            pbar = tqdm(total=len(all_items), desc="val", leave=False)

            def load_item(idx: int, item: Dict) -> Dict:
                return item

            def build_job(idx: int, loaded: Dict) -> InferenceJob:
                img = loaded["image"]
                return InferenceJob(
                    job_id=int(idx),
                    image_path=f"val_{idx}",
                    image_bgr=img,
                    original_hw=tuple(int(x) for x in img.shape[:2]),
                    tile_sizes=tile_sizes,
                    tile_overlaps=tile_overlaps,
                    tta_scales=tta_scales,
                    tta_hflip=bool(tta_hflip),
                    use_amp=bool(use_amp),
                )

            def postprocess_item(idx: int, loaded: Dict, prob_np: np.ndarray) -> Dict:
                img = loaded["image"]
                gt = loaded["gt"]
                gt_ref = loaded.get("gt_ref")
                is_negative = bool(loaded["is_negative"])
                pp_ctx = build_guideline_postprocess_context(img)
                gt_f = gt.astype(np.float32)
                prob_clip = np.clip(prob_np, 1e-5, 1.0 - 1e-5)
                bce = float(-(gt_f * np.log(prob_clip) + (1.0 - gt_f) * np.log(1.0 - prob_clip)).mean())
                inter = float((prob_np * gt_f).sum())
                den = float(prob_np.sum() + gt_f.sum())
                dice = 1.0 - ((2.0 * inter + 1e-6) / (den + 1e-6))
                local_thr, _ = sweep_postprocessed_thresholds(
                    frame_bgr=img,
                    prob_np=prob_np,
                    thresholds=thresholds,
                    gt_u8=gt,
                    gt_ref=gt_ref,
                    is_negative=is_negative,
                    axis_complete=bool(axis_complete),
                    axis_complete_max_gap=int(axis_complete_max_gap),
                    axis_complete_max_extension=int(axis_complete_max_extension),
                    axis_complete_min_white_ratio=float(axis_complete_min_white_ratio),
                    axis_complete_min_dark_border_ratio=float(axis_complete_min_dark_border_ratio),
                    context=pp_ctx,
                )
                return {
                    "index": int(idx),
                    "val_loss": float(0.5 * bce + 0.5 * dice),
                    "per_thr": local_thr,
                }

            def submit_process_shared(executor, idx: int, loaded: Dict, prob_np: np.ndarray):
                image_spec, image_shm = share_array(loaded["image"])
                gt_spec, gt_shm = share_array(loaded["gt"])
                prob_spec, prob_shm = share_array(np.asarray(prob_np, dtype=np.float32))
                payload = {
                    "index": int(idx),
                    "image_spec": image_spec,
                    "gt_spec": gt_spec,
                    "prob_spec": prob_spec,
                    "thresholds": [float(t) for t in thresholds],
                    "is_negative": bool(loaded["is_negative"]),
                    "axis_complete": bool(axis_complete),
                    "axis_complete_max_gap": int(axis_complete_max_gap),
                    "axis_complete_max_extension": int(axis_complete_max_extension),
                    "axis_complete_min_white_ratio": float(axis_complete_min_white_ratio),
                    "axis_complete_min_dark_border_ratio": float(axis_complete_min_dark_border_ratio),
                }
                fut = executor.submit(train_val_postprocess_shared, payload)

                def _cleanup(_fut):
                    for shm in (image_shm, gt_shm, prob_shm):
                        try:
                            shm.close()
                        finally:
                            try:
                                shm.unlink()
                            except FileNotFoundError:
                                pass

                fut.add_done_callback(_cleanup)
                return fut

            backend = str(val_postprocess_backend).strip().lower()
            if backend == "process_shared" and postprocess_executor is None:
                postprocess_executor = build_process_pool(max_workers=int(val_postprocess_workers))
                owns_postprocess_executor = True

            results, runtime_stats = run_async_pipeline(
                items=all_items,
                load_fn=load_item,
                build_job_fn=build_job,
                postprocess_fn=postprocess_item,
                runtime=runtime,
                loader_workers=int(val_loader_workers),
                postprocess_workers=int(val_postprocess_workers),
                prefetch_items=int(val_prefetch_items),
                postprocess_executor=postprocess_executor,
                postprocess_submit_fn=submit_process_shared if backend == "process_shared" else None,
                progress_callback=lambda done, total, stats: (
                    pbar.update(max(0, int(done) - int(pbar.n))),
                    pbar.set_postfix(
                        img=f"{done}/{total}",
                        ips=f"{stats.get('images_per_sec', 0.0):.2f}",
                        tps=f"{stats.get('tiles_per_sec', 0.0):.1f}",
                        b=f"{stats.get('avg_tile_batch', 0.0):.1f}",
                    ),
                ),
            )
            pbar.close()
        finally:
            runtime.shutdown()
            if owns_postprocess_executor and postprocess_executor is not None:
                postprocess_executor.shutdown(wait=True)

        for row in results:
            val_loss_terms.append(float(row["val_loss"]))
            for thr in thresholds:
                thr_f = float(thr)
                src = row["per_thr"][thr_f]
                st = per_thr[thr_f]
                st["tp"] += int(src["tp"])
                st["fp"] += int(src["fp"])
                st["fn"] += int(src["fn"])
                st["cl_tp"] += int(src["cl_tp"])
                st["cl_fp"] += int(src["cl_fp"])
                st["cl_fn"] += int(src["cl_fn"])
                st["neg_frames"] += int(src["neg_frames"])
                st["neg_fp_frames"] += int(src["neg_fp_frames"])
            samples_total += 1
    else:
        pbar = tqdm(all_items, desc="val", leave=False)
        for loaded in pbar:
            img = loaded["image"]
            gt = loaded["gt"]
            gt_ref = loaded.get("gt_ref")
            is_negative = bool(loaded["is_negative"])

            prob_np = predict_prob_multi_tta(
                model=model,
                frame_bgr=img,
                device=device,
                tile_sizes=tile_sizes,
                tile_overlaps=tile_overlaps,
                tta_scales=tta_scales,
                tta_hflip=tta_hflip,
                use_amp=use_amp,
            )
            pp_ctx = build_guideline_postprocess_context(img)
            gt_f = gt.astype(np.float32)
            prob_clip = np.clip(prob_np, 1e-5, 1.0 - 1e-5)
            bce = float(-(gt_f * np.log(prob_clip) + (1.0 - gt_f) * np.log(1.0 - prob_clip)).mean())
            inter = float((prob_np * gt_f).sum())
            den = float(prob_np.sum() + gt_f.sum())
            dice = 1.0 - ((2.0 * inter + 1e-6) / (den + 1e-6))
            val_loss_terms.append(0.5 * bce + 0.5 * dice)
            local_thr, _ = sweep_postprocessed_thresholds(
                frame_bgr=img,
                prob_np=prob_np,
                thresholds=thresholds,
                gt_u8=gt,
                gt_ref=gt_ref,
                is_negative=is_negative,
                axis_complete=bool(axis_complete),
                axis_complete_max_gap=int(axis_complete_max_gap),
                axis_complete_max_extension=int(axis_complete_max_extension),
                axis_complete_min_white_ratio=float(axis_complete_min_white_ratio),
                axis_complete_min_dark_border_ratio=float(axis_complete_min_dark_border_ratio),
                context=pp_ctx,
            )
            for thr in thresholds:
                thr_f = float(thr)
                st = per_thr[thr_f]
                src = local_thr[thr_f]
                st["tp"] += int(src["tp"])
                st["fp"] += int(src["fp"])
                st["fn"] += int(src["fn"])
                st["cl_tp"] += int(src["cl_tp"])
                st["cl_fp"] += int(src["cl_fp"])
                st["cl_fn"] += int(src["cl_fn"])
                st["neg_frames"] += int(src["neg_frames"])
                st["neg_fp_frames"] += int(src["neg_fp_frames"])
            samples_total += 1

    best_thr = None
    best_score = -1e9
    best_metrics = {}
    by_threshold = {}

    for thr in thresholds:
        thr_f = float(thr)
        st = per_thr[thr_f]
        p, r, f1 = precision_recall_f1(st["tp"], st["fp"], st["fn"])
        cp, cr, cf1 = precision_recall_f1(st["cl_tp"], st["cl_fp"], st["cl_fn"])
        fp_per_frame = safe_div(st["neg_fp_frames"], st["neg_frames"]) if st["neg_frames"] > 0 else 0.0
        composite = threshold_score(
            pixel_precision=float(p),
            pixel_recall=float(r),
            pixel_f1=float(f1),
            centerline_f1=float(cf1),
            fp_per_frame=float(fp_per_frame),
            mode=selection_metric,
        )
        by_threshold[str(round(thr_f, 4))] = {
            "pixel_precision": float(p),
            "pixel_recall": float(r),
            "pixel_f1": float(f1),
            "centerline_precision": float(cp),
            "centerline_recall": float(cr),
            "centerline_f1": float(cf1),
            "fp_per_frame": float(fp_per_frame),
            "tp": int(st["tp"]),
            "fp": int(st["fp"]),
            "fn": int(st["fn"]),
            "neg_frames": int(st["neg_frames"]),
            "neg_fp_frames": int(st["neg_fp_frames"]),
            "composite_score": float(composite),
        }
        if (composite > best_score) or (
            abs(composite - best_score) < 1e-9 and cf1 > float(best_metrics.get("centerline_f1", -1.0))
        ) or (
            abs(composite - best_score) < 1e-9
            and abs(cf1 - float(best_metrics.get("centerline_f1", -1.0))) < 1e-9
            and p > float(best_metrics.get("pixel_precision", -1.0))
        ):
            best_score = float(composite)
            best_thr = float(thr_f)
            best_metrics = by_threshold[str(round(thr_f, 4))]

    out = {
        "sample_count": int(samples_total),
        "val_loss": float(np.mean(val_loss_terms)) if val_loss_terms else 0.0,
        "best_threshold": float(best_thr if best_thr is not None else thresholds[0]),
        "best": best_metrics,
        "by_threshold": by_threshold,
        "runtime_mode": runtime_mode,
        "runtime_stats": runtime_stats or {},
        "runtime_postprocess_backend": str(val_postprocess_backend).strip().lower() if runtime_mode == "async" else "thread",
    }
    return out


def make_manifest_rows_train(
    train_pos_records: Sequence[AnnotationRecord],
    train_neg_records: Sequence[NegativeRecord],
    neg_per_pos: float,
) -> List[Dict]:
    rows: List[Dict] = []
    for rec in train_pos_records:
        rows.append(
            {
                "sample_id": rec.sample_id,
                "kind": "positive",
                "image_path": str(rec.image_path),
                "mask_path": str(rec.mask_path),
                "dataset_split": rec.dataset_split,
                "save_mode": rec.save_mode,
                "sample_weight": float(rec.sample_weight),
                "source": rec.source,
            }
        )

    neg_target_total = int(round(len(train_pos_records) * max(0.0, float(neg_per_pos))))
    intra_target = int(math.ceil(neg_target_total * 0.5)) if train_neg_records else neg_target_total
    ext_target = max(0, neg_target_total - intra_target)
    for i in range(intra_target):
        if not train_pos_records:
            break
        src = train_pos_records[i % len(train_pos_records)]
        rows.append(
            {
                "sample_id": f"INTRA_{i:06d}",
                "kind": "intra_negative",
                "image_path": str(src.image_path),
                "mask_path": None,
                "dataset_split": src.dataset_split,
                "save_mode": src.save_mode,
                "sample_weight": float(max(0.75, src.sample_weight * 0.95)),
                "source": "intra_negative_patch",
            }
        )
    for i in range(ext_target):
        if not train_neg_records:
            break
        src = train_neg_records[i % len(train_neg_records)]
        rows.append(
            {
                "sample_id": src.sample_id,
                "kind": "external_negative",
                "image_path": str(src.image_path),
                "mask_path": None,
                "dataset_split": "negative_external",
                "save_mode": "none",
                "sample_weight": 1.0,
                "source": src.source,
            }
        )
    return rows


def make_manifest_rows_val(
    val_pos_records: Sequence[AnnotationRecord],
    val_neg_records: Sequence[NegativeRecord],
) -> List[Dict]:
    rows: List[Dict] = []
    for rec in val_pos_records:
        rows.append(
            {
                "sample_id": rec.sample_id,
                "kind": "positive",
                "image_path": str(rec.image_path),
                "mask_path": str(rec.mask_path),
                "dataset_split": rec.dataset_split,
                "save_mode": rec.save_mode,
                "sample_weight": float(rec.sample_weight),
                "source": rec.source,
            }
        )
    for rec in val_neg_records:
        rows.append(
            {
                "sample_id": rec.sample_id,
                "kind": "external_negative",
                "image_path": str(rec.image_path),
                "mask_path": None,
                "dataset_split": "negative_external",
                "save_mode": "none",
                "sample_weight": 1.0,
                "source": rec.source,
            }
        )
    return rows


def set_encoder_trainability(model: nn.Module, trainable: bool) -> None:
    target = None
    if hasattr(model, "model_family"):
        family = str(getattr(model, "model_family", "")).strip().lower()
        if family == "custom":
            target = getattr(getattr(model, "net", None), "encoder", None)
        elif family == "segformer":
            target = getattr(getattr(model, "net", None), "segformer", None)
    if target is None:
        return
    for p in target.parameters():
        p.requires_grad = bool(trainable)


def build_validation_items(
    val_pos_records: Sequence[AnnotationRecord],
    val_neg_records: Sequence[NegativeRecord],
) -> List[Dict]:
    items: List[Dict] = []
    for idx, rec in enumerate(val_pos_records):
        img = load_bgr(rec.image_path)
        gt = load_mask_u8(rec.mask_path)
        items.append(
            {
                "index": int(idx),
                "kind": "pos",
                "image": img,
                "gt": gt,
                "gt_ref": build_centerline_reference(gt, tol_px=2),
                "is_negative": False,
                "sample_id": rec.sample_id,
            }
        )
    base_idx = len(items)
    for off, rec in enumerate(val_neg_records):
        img = load_bgr(rec.image_path)
        items.append(
            {
                "index": int(base_idx + off),
                "kind": "neg",
                "image": img,
                "gt": np.zeros(img.shape[:2], dtype=np.uint8),
                "gt_ref": None,
                "is_negative": True,
                "sample_id": rec.sample_id,
            }
        )
    return items


def get_encoder_module(model: nn.Module) -> Optional[nn.Module]:
    if not hasattr(model, "model_family"):
        return None
    family = str(getattr(model, "model_family", "")).strip().lower()
    if family == "custom":
        return getattr(getattr(model, "net", None), "encoder", None)
    if family == "segformer":
        return getattr(getattr(model, "net", None), "segformer", None)
    if family == "mask2former":
        base = getattr(model, "net", None)
        return getattr(base, "model", None) if base is not None else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train non-LoRA cue-line segmentation model (precision-first).")
    parser.add_argument("--annotations", type=str, default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--splits", type=str, default=str(DEFAULT_SPLITS))
    parser.add_argument("--negative-image-dir", type=str, default="")
    parser.add_argument("--include-main", type=str2bool, default=True)
    parser.add_argument("--include-quarantine", type=str2bool, default=True)
    parser.add_argument("--include-legacy", type=str2bool, default=True)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--neg-per-pos", type=float, default=0.5)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--early-stop-patience", type=int, default=18)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio-fallback", type=float, default=0.15)
    parser.add_argument("--split-mode", type=str, default="existing", choices=["existing", "random", "stratified"])
    parser.add_argument("--split-size-thresholds", type=str, default="12,20,40")
    parser.add_argument("--split-stratify-source", type=str2bool, default=True)
    parser.add_argument("--model-family", type=str, default="custom", choices=["custom", "segformer", "mask2former"])
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--backbone", type=str, default="tf_efficientnetv2_s")
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--tile-sizes", type=str, default="")
    parser.add_argument("--tile-overlaps", type=str, default="")
    parser.add_argument("--tta-scales", type=str, default="1.0,1.15")
    parser.add_argument("--tta-hflip", type=str2bool, default=True)
    parser.add_argument("--loss-ce-weight", type=float, default=0.45)
    parser.add_argument("--loss-dice-weight", type=float, default=0.25)
    parser.add_argument("--loss-cldice-weight", type=float, default=0.20)
    parser.add_argument("--loss-edge-weight", type=float, default=0.10)
    parser.add_argument("--target-mode", type=str, default="binary", choices=["binary", "soft_distance"])
    parser.add_argument("--target-soft-distance-scale", type=float, default=1.0)
    parser.add_argument("--crop-mode-mix", type=str, default="tight:0.30,context:0.45,fullframe:0.25")
    parser.add_argument("--crop-mode-mix-small", type=str, default="tight:0.65,context:0.35,fullframe:0.00")
    parser.add_argument("--crop-mode-mix-ultra-tiny", type=str, default="tight:0.85,context:0.15,fullframe:0.00")
    parser.add_argument("--small-line-major-thresh", type=float, default=20.0)
    parser.add_argument("--ultra-tiny-major-thresh", type=float, default=12.0)
    parser.add_argument("--small-line-weight-mult", type=float, default=1.35)
    parser.add_argument("--ultra-tiny-weight-mult", type=float, default=1.75)
    parser.add_argument("--refined-weight-mult", type=float, default=1.05)
    parser.add_argument("--hardcase-positive-manifest", type=str, default="")
    parser.add_argument("--hardcase-negative-manifest", type=str, default="")
    parser.add_argument("--hardcase-positive-weight-mult", type=float, default=1.0)
    parser.add_argument("--hardcase-negative-weight-mult", type=float, default=1.0)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--encoder-lr-mult", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "onecycle"])
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--axis-complete", type=str2bool, default=True)
    parser.add_argument("--axis-complete-max-gap", type=int, default=24)
    parser.add_argument("--axis-complete-max-extension", type=int, default=160)
    parser.add_argument("--axis-complete-min-white-ratio", type=float, default=0.42)
    parser.add_argument("--axis-complete-min-dark-border-ratio", type=float, default=0.05)
    parser.add_argument("--selection-metric", type=str, default="balanced", choices=["precision", "balanced", "centerline"])
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--val-runtime", type=str, default="direct", choices=["direct", "async"])
    parser.add_argument("--val-loader-workers", type=int, default=4)
    parser.add_argument("--val-postprocess-workers", type=int, default=4)
    parser.add_argument("--val-postprocess-backend", type=str, default="thread", choices=["thread", "process_shared"])
    parser.add_argument("--val-prefetch-items", type=int, default=16)
    parser.add_argument("--val-max-batch-tiles", type=int, default=32)
    parser.add_argument("--val-max-wait-ms", type=int, default=4)
    parser.add_argument("--val-bucket-by-tile", type=str2bool, default=True)
    parser.add_argument("--val-save-runtime-stats", type=str2bool, default=True)
    parser.add_argument("--val-runtime-fallback-direct", type=str2bool, default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    annotations_path = resolve_path(args.annotations)
    splits_path = resolve_path(args.splits)
    run_dir = build_run_dir(args.save_dir if args.save_dir else None)
    run_dir.mkdir(parents=True, exist_ok=True)
    crop_mode_mix = parse_crop_mode_mix(args.crop_mode_mix)
    crop_mode_mix_small = parse_crop_mode_mix(args.crop_mode_mix_small)
    crop_mode_mix_ultra_tiny = parse_crop_mode_mix(args.crop_mode_mix_ultra_tiny)
    split_size_thresholds = parse_csv_floats(args.split_size_thresholds, [12.0, 20.0, 40.0])
    tile_sizes = parse_csv_ints(args.tile_sizes, [int(args.tile_size), max(int(args.tile_size), 896)])
    tile_overlaps = parse_csv_ints(args.tile_overlaps, [int(args.tile_overlap), max(int(args.tile_overlap), 224)])
    if len(tile_overlaps) < len(tile_sizes):
        tile_overlaps.extend([tile_overlaps[-1] if tile_overlaps else 160] * (len(tile_sizes) - len(tile_overlaps)))
    elif len(tile_overlaps) > len(tile_sizes):
        tile_overlaps = tile_overlaps[: len(tile_sizes)]
    tta_scales = parse_csv_floats(args.tta_scales, [1.0, 1.15])

    neg_dir = resolve_path(args.negative_image_dir) if str(args.negative_image_dir).strip() else None
    if neg_dir is not None and not neg_dir.exists():
        raise FileNotFoundError(f"Negative image dir not found: {neg_dir}")

    records, load_stats = load_annotation_records(
        annotations_path=annotations_path,
        include_main=bool(args.include_main),
        include_quarantine=bool(args.include_quarantine),
        include_legacy=bool(args.include_legacy),
    )
    if not records:
        raise RuntimeError("No valid positive records found after filtering.")

    split_mode = str(args.split_mode).strip().lower()
    train_ids: List[str] = []
    val_ids: List[str] = []
    split_reason = ""
    if split_mode == "stratified":
        train_ids, val_ids = stratified_annotation_split(
            records,
            val_ratio=float(args.val_ratio_fallback),
            seed=int(args.seed),
            size_thresholds=split_size_thresholds,
            stratify_by_source=bool(args.split_stratify_source),
        )
        split_reason = "generated_stratified"
        write_json(
            run_dir / "splits_stratified.json",
            {
                "train": train_ids,
                "val": val_ids,
                "created_at": utc_now_iso(),
                "mode": "stratified",
                "val_ratio": float(args.val_ratio_fallback),
                "size_thresholds": split_size_thresholds,
                "stratify_by_source": bool(args.split_stratify_source),
            },
        )
    elif split_mode == "random":
        fallback_train, fallback_val = random_split_ids(
            [r.sample_id for r in records], val_ratio=float(args.val_ratio_fallback), seed=int(args.seed)
        )
        train_ids = fallback_train
        val_ids = fallback_val
        split_reason = "generated_random"
        write_json(
            run_dir / "splits_random.json",
            {
                "train": train_ids,
                "val": val_ids,
                "created_at": utc_now_iso(),
                "mode": "random",
                "val_ratio": float(args.val_ratio_fallback),
            },
        )
    else:
        train_ids, val_ids = load_split_ids(splits_path)
        if train_ids and val_ids:
            split_reason = "loaded_existing"
        else:
            fallback_train, fallback_val = random_split_ids(
                [r.sample_id for r in records], val_ratio=float(args.val_ratio_fallback), seed=int(args.seed)
            )
            train_ids = fallback_train
            val_ids = fallback_val
            split_reason = "fallback_random_missing_existing"
            write_json(
                run_dir / "splits_fallback.json",
                {
                    "train": train_ids,
                    "val": val_ids,
                    "created_at": utc_now_iso(),
                    "mode": "random",
                    "reason": "splits file missing or empty",
                    "val_ratio": float(args.val_ratio_fallback),
                },
            )

    train_pos_records, val_pos_records = split_by_ids(records, train_ids, val_ids)
    if not train_pos_records:
        raise RuntimeError("No training positive records matched split IDs.")
    if not val_pos_records:
        raise RuntimeError("No validation positive records matched split IDs.")
    train_pos_summary = summarize_positive_records(train_pos_records, split_size_thresholds)
    val_pos_summary = summarize_positive_records(val_pos_records, split_size_thresholds)
    write_json(
        run_dir / "split_summary.json",
        {
            "created_at": utc_now_iso(),
            "split_mode_requested": split_mode,
            "split_mode_used": split_reason,
            "size_thresholds": split_size_thresholds,
            "train_positive_summary": train_pos_summary,
            "val_positive_summary": val_pos_summary,
        },
    )

    neg_records_all = build_negative_records(neg_dir, prefix="NEG_EXT") if neg_dir is not None else []
    train_neg_records, val_neg_records = split_negatives(
        neg_records_all, val_ratio=float(args.val_ratio_fallback), seed=int(args.seed)
    )
    hardcase_positive_manifest_path = (
        resolve_path(str(args.hardcase_positive_manifest).strip()) if str(args.hardcase_positive_manifest).strip() else None
    )
    hardcase_negative_manifest_path = (
        resolve_path(str(args.hardcase_negative_manifest).strip()) if str(args.hardcase_negative_manifest).strip() else None
    )
    hardcase_positive_ids: Set[str] = set()
    hardcase_negative_image_keys: Set[str] = set()
    hardcase_replay_summary: Dict[str, object] = {
        "positive_manifest": str(hardcase_positive_manifest_path) if hardcase_positive_manifest_path is not None else "",
        "negative_manifest": str(hardcase_negative_manifest_path) if hardcase_negative_manifest_path is not None else "",
        "positive_weight_mult": float(args.hardcase_positive_weight_mult),
        "negative_weight_mult": float(args.hardcase_negative_weight_mult),
        "matched_train_positive_count": 0,
        "matched_train_negative_count": 0,
    }
    if hardcase_positive_manifest_path is not None:
        if not hardcase_positive_manifest_path.exists():
            raise FileNotFoundError(f"Hardcase positive manifest not found: {hardcase_positive_manifest_path}")
        hardcase_positive_payload = load_manifest_payload(hardcase_positive_manifest_path)
        requested_positive_ids = manifest_positive_sample_ids(hardcase_positive_payload)
        requested_positive_images = manifest_image_keys(hardcase_positive_payload)
        hardcase_positive_ids = resolve_hardcase_positive_ids(hardcase_positive_payload, train_pos_records)
        hardcase_replay_summary["positive_manifest_meta"] = hardcase_positive_payload.get("meta", {})
        hardcase_replay_summary["requested_positive_ids"] = int(len(requested_positive_ids))
        hardcase_replay_summary["requested_positive_images"] = int(len(requested_positive_images))
        hardcase_replay_summary["matched_train_positive_count"] = int(len(hardcase_positive_ids))
        hardcase_replay_summary["matched_train_positive_ids"] = sorted(hardcase_positive_ids)
    if hardcase_negative_manifest_path is not None:
        if not hardcase_negative_manifest_path.exists():
            raise FileNotFoundError(f"Hardcase negative manifest not found: {hardcase_negative_manifest_path}")
        hardcase_negative_payload = load_manifest_payload(hardcase_negative_manifest_path)
        requested_negative_images = manifest_image_keys(hardcase_negative_payload)
        hardcase_negative_image_keys = resolve_hardcase_negative_image_keys(hardcase_negative_payload, train_neg_records)
        matched_negative_paths = sorted(
            str(rec.image_path)
            for rec in train_neg_records
            if canonical_path_key(rec.image_path) in hardcase_negative_image_keys
        )
        hardcase_replay_summary["negative_manifest_meta"] = hardcase_negative_payload.get("meta", {})
        hardcase_replay_summary["requested_negative_images"] = int(len(requested_negative_images))
        hardcase_replay_summary["matched_train_negative_count"] = int(len(hardcase_negative_image_keys))
        hardcase_replay_summary["matched_train_negative_paths"] = matched_negative_paths
    write_json(run_dir / "hardcase_replay_summary.json", hardcase_replay_summary)
    val_items_cache = build_validation_items(
        val_pos_records=val_pos_records,
        val_neg_records=val_neg_records,
    )

    train_manifest_rows = make_manifest_rows_train(
        train_pos_records=train_pos_records,
        train_neg_records=train_neg_records,
        neg_per_pos=float(args.neg_per_pos),
    )
    val_manifest_rows = make_manifest_rows_val(
        val_pos_records=val_pos_records,
        val_neg_records=val_neg_records,
    )
    append_jsonl(run_dir / "train_manifest.jsonl", train_manifest_rows)
    append_jsonl(run_dir / "val_manifest.jsonl", val_manifest_rows)

    init_ckpt_path: Optional[Path] = None
    init_ckpt_payload: Optional[Dict] = None
    if str(args.init_checkpoint).strip():
        init_ckpt_path = resolve_path(str(args.init_checkpoint).strip())
        if not init_ckpt_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {init_ckpt_path}")
        init_ckpt_payload = torch.load(str(init_ckpt_path), map_location="cpu")
        if not isinstance(init_ckpt_payload, dict) or "model_state" not in init_ckpt_payload:
            raise RuntimeError(f"Checkpoint does not contain model_state: {init_ckpt_path}")
        ckpt_family = str(init_ckpt_payload.get("model_family", "custom")).strip().lower() or "custom"
        ckpt_model_name = str(init_ckpt_payload.get("model_name", "")).strip()
        ckpt_backbone = str(init_ckpt_payload.get("backbone", "")).strip()
        if ckpt_family and ckpt_family != str(args.model_family).strip().lower():
            print(f"Init checkpoint family '{ckpt_family}' overrides requested family '{args.model_family}'")
            args.model_family = ckpt_family
        if ckpt_model_name and ckpt_model_name != str(args.model_name).strip():
            print(f"Init checkpoint model '{ckpt_model_name}' overrides requested model_name '{args.model_name}'")
            args.model_name = ckpt_model_name
        if ckpt_backbone and ckpt_backbone != str(args.backbone).strip():
            print(f"Init checkpoint backbone '{ckpt_backbone}' overrides requested backbone '{args.backbone}'")
            args.backbone = ckpt_backbone

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmentation_model(
        model_family=str(args.model_family),
        backbone=str(args.backbone),
        model_name=str(args.model_name),
        pretrained=True,
    ).to(device)
    if init_ckpt_payload is not None:
        fmt = int(init_ckpt_payload.get("checkpoint_format_version", 1))
        if fmt <= 1 and str(args.model_family).strip().lower() == "custom":
            load_result = model.net.load_state_dict(init_ckpt_payload["model_state"], strict=True)
        else:
            load_result = model.load_state_dict(init_ckpt_payload["model_state"], strict=False)
            missing = list(getattr(load_result, "missing_keys", []))
            unexpected = list(getattr(load_result, "unexpected_keys", []))
            if missing or unexpected:
                raise RuntimeError(
                    f"Init checkpoint load mismatch for {init_ckpt_path}: missing={missing}, unexpected={unexpected}"
                )
        print(
            f"Loaded init checkpoint: {init_ckpt_path} | "
            f"epoch={int(init_ckpt_payload.get('epoch', 0))} | "
            f"family={str(init_ckpt_payload.get('model_family', args.model_family))} | "
            f"backbone={str(init_ckpt_payload.get('backbone', args.backbone))}"
        )
    encoder_module = get_encoder_module(model)
    encoder_param_ids = set()
    if encoder_module is not None:
        encoder_param_ids = {id(p) for p in encoder_module.parameters()}
    encoder_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in encoder_param_ids:
            encoder_params.append(p)
        else:
            other_params.append(p)
    enc_lr_mult = max(0.0, float(args.encoder_lr_mult))
    optimizer_groups = []
    if encoder_params:
        optimizer_groups.append(
            {
                "params": encoder_params,
                "lr": float(args.lr) * enc_lr_mult,
                "weight_decay": 1e-4,
            }
        )
    if other_params:
        optimizer_groups.append(
            {
                "params": other_params,
                "lr": float(args.lr),
                "weight_decay": 1e-4,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups if optimizer_groups else model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(bool(args.amp) and device.type == "cuda"))
    sobel_kx, sobel_ky = build_sobel_kernels(device)

    train_dataset = TrainSampleDataset(
        pos_records=train_pos_records,
        ext_neg_records=train_neg_records,
        img_size=int(args.img_size),
        neg_per_pos=float(args.neg_per_pos),
        augment=True,
        seed=int(args.seed),
        crop_mode_mix=crop_mode_mix,
        crop_mode_mix_small=crop_mode_mix_small,
        crop_mode_mix_ultra_tiny=crop_mode_mix_ultra_tiny,
        small_line_major_thresh=float(args.small_line_major_thresh),
        ultra_tiny_major_thresh=float(args.ultra_tiny_major_thresh),
        small_line_weight_mult=float(args.small_line_weight_mult),
        ultra_tiny_weight_mult=float(args.ultra_tiny_weight_mult),
        refined_weight_mult=float(args.refined_weight_mult),
        hardcase_positive_ids=hardcase_positive_ids,
        hardcase_negative_image_keys=hardcase_negative_image_keys,
        hardcase_positive_weight_mult=float(args.hardcase_positive_weight_mult),
        hardcase_negative_weight_mult=float(args.hardcase_negative_weight_mult),
        input_mean=getattr(model, "input_mean", IMAGENET_MEAN),
        input_std=getattr(model, "input_std", IMAGENET_STD),
        target_mode=str(args.target_mode),
        target_soft_distance_scale=float(args.target_soft_distance_scale),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=max(0, int(args.workers)),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_train,
        drop_last=False,
        persistent_workers=(int(args.workers) > 0),
    )
    optimizer_steps_per_epoch = max(1, int(math.ceil(len(train_loader) / float(max(1, int(args.grad_accum_steps))))))
    total_steps = max(1, optimizer_steps_per_epoch * int(args.epochs))
    warmup_steps = max(0, int(round(optimizer_steps_per_epoch * float(args.warmup_epochs))))
    if str(args.scheduler).strip().lower() == "onecycle":
        max_lrs = [group.get("lr", float(args.lr)) for group in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs if len(max_lrs) > 1 else max_lrs[0],
            total_steps=total_steps,
            pct_start=max(0.01, min(0.4, float(args.warmup_epochs) / max(1.0, float(args.epochs)))),
            anneal_strategy="cos",
        )
        step_scheduler_per_batch = True
    else:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return max(1e-3, float(step + 1) / float(warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        step_scheduler_per_batch = True

    print(f"Device: {device}")
    print(f"Model family: {args.model_family}")
    if str(args.model_name).strip():
        print(f"Model name: {args.model_name}")
    print(f"Backbone: {args.backbone}")
    print(f"Target mode: {str(args.target_mode).strip().lower()} | Soft distance scale: {float(args.target_soft_distance_scale):.2f}")
    print(f"Encoder LR mult: {float(args.encoder_lr_mult):.4f} | Grad clip: {float(args.grad_clip_norm):.4f}")
    print(f"Optimizer steps/epoch: {optimizer_steps_per_epoch} | Grad accum: {int(args.grad_accum_steps)}")
    print(f"Split mode: {split_reason}")
    print(f"Train positives: {len(train_pos_records)} | Val positives: {len(val_pos_records)}")
    print(f"Train negatives(ext): {len(train_neg_records)} | Val negatives(ext): {len(val_neg_records)}")
    print(f"Train rows total (epoch): {len(train_dataset)}")
    print(
        "Hardcase replay: "
        f"pos={int(hardcase_replay_summary.get('matched_train_positive_count', 0))} "
        f"(x{float(args.hardcase_positive_weight_mult):.2f}) | "
        f"neg={int(hardcase_replay_summary.get('matched_train_negative_count', 0))} "
        f"(x{float(args.hardcase_negative_weight_mult):.2f})"
    )
    print(f"Save dir: {run_dir}")
    print(f"Data load stats: {json.dumps(load_stats)}")
    print(f"Train positive summary: {json.dumps(train_pos_summary)}")
    print(f"Val positive summary: {json.dumps(val_pos_summary)}")

    thresholds = [round(x, 2) for x in np.arange(0.30, 0.8001, 0.05)]
    history: List[Dict] = []
    runtime_history: List[Dict] = []
    best_score = -1e9
    best_epoch = 0
    best_threshold = float(thresholds[0])
    epochs_no_improve = 0
    best_metrics_snapshot = {}
    best_centerline_score = -1e9
    best_centerline_epoch = 0
    persistent_val_postprocess_executor: Optional[Executor] = None
    if str(args.val_runtime).strip().lower() == "async" and str(args.val_postprocess_backend).strip().lower() == "process_shared":
        persistent_val_postprocess_executor = build_process_pool(max_workers=int(args.val_postprocess_workers))

    try:
        for epoch in range(1, int(args.epochs) + 1):
            encoder_trainable = int(epoch) > int(args.freeze_encoder_epochs)
            set_encoder_trainability(model, encoder_trainable)
            model.train()

            running_loss = 0.0
            batch_count = 0
            pbar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False)
            optimizer.zero_grad(set_to_none=True)
            for images, masks, masks_loss, sample_w, _is_neg, _stype, _sid, _src in pbar:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                masks_loss = masks_loss.to(device, non_blocking=True)
                sample_w = sample_w.to(device, non_blocking=True).view(-1)
                amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (bool(args.amp) and device.type == "cuda") else torch.enable_grad()
                with amp_ctx:
                    outputs = model.forward_outputs(images, output_size=tuple(masks.shape[-2:]))
                    per_sample, _ce, _dice, _cld, _edge = tensor_losses_per_sample(
                        outputs=outputs,
                        target_hard=masks,
                        target_loss=masks_loss,
                        model_family=str(args.model_family),
                        pos_weight=float(args.pos_weight),
                        sobel_kx=sobel_kx,
                        sobel_ky=sobel_ky,
                        loss_ce_weight=float(args.loss_ce_weight),
                        loss_dice_weight=float(args.loss_dice_weight),
                        loss_cldice_weight=float(args.loss_cldice_weight),
                        loss_edge_weight=float(args.loss_edge_weight),
                    )
                    loss = (per_sample * sample_w).mean() / float(max(1, int(args.grad_accum_steps)))

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if ((batch_count + 1) % max(1, int(args.grad_accum_steps)) == 0) or ((batch_count + 1) == len(train_loader)):
                        if float(args.grad_clip_norm) > 0.0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
                        prev_scale = float(scaler.get_scale())
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        # With AMP, scaler.step may skip optimizer.step on overflow.
                        # Step the scheduler only when optimizer actually stepped.
                        optimizer_stepped = float(scaler.get_scale()) >= prev_scale
                        if step_scheduler_per_batch and optimizer_stepped:
                            scheduler.step()
                else:
                    loss.backward()
                    if ((batch_count + 1) % max(1, int(args.grad_accum_steps)) == 0) or ((batch_count + 1) == len(train_loader)):
                        if float(args.grad_clip_norm) > 0.0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if step_scheduler_per_batch:
                            scheduler.step()

                running_loss += float(loss.detach().item()) * float(max(1, int(args.grad_accum_steps)))
                batch_count += 1
                pbar.set_postfix(loss=f"{float(loss.detach().item()) * float(max(1, int(args.grad_accum_steps))):.4f}")

            train_loss = running_loss / max(1, batch_count)
            try:
                val = evaluate_model(
                    model=model,
                    val_pos_records=val_pos_records,
                    val_neg_records=val_neg_records,
                    device=device,
                    tile_sizes=tile_sizes,
                    tile_overlaps=tile_overlaps,
                    tta_scales=tta_scales,
                    tta_hflip=bool(args.tta_hflip),
                    use_amp=bool(args.amp),
                    thresholds=thresholds,
                    selection_metric=str(args.selection_metric),
                    axis_complete=bool(args.axis_complete),
                    axis_complete_max_gap=int(args.axis_complete_max_gap),
                    axis_complete_max_extension=int(args.axis_complete_max_extension),
                    axis_complete_min_white_ratio=float(args.axis_complete_min_white_ratio),
                    axis_complete_min_dark_border_ratio=float(args.axis_complete_min_dark_border_ratio),
                    val_runtime=str(args.val_runtime),
                    val_loader_workers=int(args.val_loader_workers),
                    val_postprocess_workers=int(args.val_postprocess_workers),
                    val_postprocess_backend=str(args.val_postprocess_backend),
                    val_prefetch_items=int(args.val_prefetch_items),
                    val_max_batch_tiles=int(args.val_max_batch_tiles),
                    val_max_wait_ms=int(args.val_max_wait_ms),
                    val_bucket_by_tile=bool(args.val_bucket_by_tile),
                    val_items_cache=val_items_cache,
                    val_postprocess_executor=persistent_val_postprocess_executor,
                )
            except Exception:
                if not bool(args.val_runtime_fallback_direct) or str(args.val_runtime).strip().lower() != "async":
                    raise
                print("Validation async runtime failed; retrying in direct mode.")
                val = evaluate_model(
                    model=model,
                    val_pos_records=val_pos_records,
                    val_neg_records=val_neg_records,
                    device=device,
                    tile_sizes=tile_sizes,
                    tile_overlaps=tile_overlaps,
                    tta_scales=tta_scales,
                    tta_hflip=bool(args.tta_hflip),
                    use_amp=bool(args.amp),
                    thresholds=thresholds,
                    selection_metric=str(args.selection_metric),
                    axis_complete=bool(args.axis_complete),
                    axis_complete_max_gap=int(args.axis_complete_max_gap),
                    axis_complete_max_extension=int(args.axis_complete_max_extension),
                    axis_complete_min_white_ratio=float(args.axis_complete_min_white_ratio),
                    axis_complete_min_dark_border_ratio=float(args.axis_complete_min_dark_border_ratio),
                    val_runtime="direct",
                    val_loader_workers=int(args.val_loader_workers),
                    val_postprocess_workers=int(args.val_postprocess_workers),
                    val_postprocess_backend="thread",
                    val_prefetch_items=int(args.val_prefetch_items),
                    val_max_batch_tiles=int(args.val_max_batch_tiles),
                    val_max_wait_ms=int(args.val_max_wait_ms),
                    val_bucket_by_tile=bool(args.val_bucket_by_tile),
                    val_items_cache=val_items_cache,
                    val_postprocess_executor=None,
                )
            runtime_history.append(
                {
                    "epoch": int(epoch),
                    "runtime_mode": str(val.get("runtime_mode", "direct")),
                    "postprocess_backend": str(val.get("runtime_postprocess_backend", "thread")),
                    "runtime_stats": dict(val.get("runtime_stats", {})),
                }
            )
            best_val = val["best"]
            composite = float(best_val.get("composite_score", 0.0))
            val_loss = float(val.get("val_loss", 0.0))
            epoch_row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "best_threshold": float(val["best_threshold"]),
                "pixel_precision": float(best_val.get("pixel_precision", 0.0)),
                "pixel_recall": float(best_val.get("pixel_recall", 0.0)),
                "pixel_f1": float(best_val.get("pixel_f1", 0.0)),
                "centerline_f1": float(best_val.get("centerline_f1", 0.0)),
                "fp_per_frame": float(best_val.get("fp_per_frame", 0.0)),
                "composite_score": float(composite),
            }
            history.append(epoch_row)
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"P={epoch_row['pixel_precision']:.4f} R={epoch_row['pixel_recall']:.4f} "
                f"F1={epoch_row['pixel_f1']:.4f} cF1={epoch_row['centerline_f1']:.4f} "
                f"FP/frame={epoch_row['fp_per_frame']:.4f} thr={epoch_row['best_threshold']:.2f}"
            )

            improved = composite > (best_score + float(args.early_stop_min_delta))
            if improved:
                best_score = float(composite)
                best_epoch = int(epoch)
                best_threshold = float(val["best_threshold"])
                epochs_no_improve = 0
                best_metrics_snapshot = dict(best_val)
                best_ckpt = {
                    "epoch": int(epoch),
                    "model_state": model.state_dict(),
                    **model.checkpoint_metadata(),
                    "backbone": str(args.backbone),
                    "img_size": int(args.img_size),
                    "best_threshold": float(best_threshold),
                    "args": vars(args),
                    "best_metrics": best_metrics_snapshot,
                    "loss_config": {
                        "ce": float(args.loss_ce_weight),
                        "dice": float(args.loss_dice_weight),
                        "cldice": float(args.loss_cldice_weight),
                        "edge": float(args.loss_edge_weight),
                    },
                    "target_config": {
                        "mode": str(args.target_mode),
                        "soft_distance_scale": float(args.target_soft_distance_scale),
                    },
                    "crop_mode_mix": crop_mode_mix,
                    "crop_mode_mix_small": crop_mode_mix_small,
                    "crop_mode_mix_ultra_tiny": crop_mode_mix_ultra_tiny,
                    "tile_config": {"tile_sizes": tile_sizes, "tile_overlaps": tile_overlaps},
                    "tta_config": {"tta_scales": tta_scales, "tta_hflip": bool(args.tta_hflip)},
                }
                torch.save(best_ckpt, run_dir / "best.pt")
                write_json(
                    run_dir / "thresholds.json",
                    {
                        "best_threshold": float(best_threshold),
                        "best_epoch": int(best_epoch),
                        "best_composite_score": float(best_score),
                        "selection_metric": str(args.selection_metric),
                        "by_threshold": val["by_threshold"],
                        "updated_at": utc_now_iso(),
                    },
                )
                print(f"Saved best checkpoint: {run_dir / 'best.pt'}")
            else:
                epochs_no_improve += 1

            cf1_now = float(best_val.get("centerline_f1", 0.0))
            if cf1_now > (best_centerline_score + float(args.early_stop_min_delta)):
                best_centerline_score = cf1_now
                best_centerline_epoch = int(epoch)
                best_center_ckpt = {
                    "epoch": int(epoch),
                    "model_state": model.state_dict(),
                    **model.checkpoint_metadata(),
                    "backbone": str(args.backbone),
                    "img_size": int(args.img_size),
                    "best_threshold": float(val["best_threshold"]),
                    "args": vars(args),
                    "best_metrics": dict(best_val),
                    "checkpoint_kind": "best_centerline",
                    "loss_config": {
                        "ce": float(args.loss_ce_weight),
                        "dice": float(args.loss_dice_weight),
                        "cldice": float(args.loss_cldice_weight),
                        "edge": float(args.loss_edge_weight),
                    },
                    "target_config": {
                        "mode": str(args.target_mode),
                        "soft_distance_scale": float(args.target_soft_distance_scale),
                    },
                    "crop_mode_mix": crop_mode_mix,
                    "crop_mode_mix_small": crop_mode_mix_small,
                    "crop_mode_mix_ultra_tiny": crop_mode_mix_ultra_tiny,
                    "tile_config": {"tile_sizes": tile_sizes, "tile_overlaps": tile_overlaps},
                    "tta_config": {"tta_scales": tta_scales, "tta_hflip": bool(args.tta_hflip)},
                }
                torch.save(best_center_ckpt, run_dir / "best_centerline.pt")
                print(f"Saved best centerline checkpoint: {run_dir / 'best_centerline.pt'}")

            last_ckpt = {
                "epoch": int(epoch),
                "model_state": model.state_dict(),
                **model.checkpoint_metadata(),
                "backbone": str(args.backbone),
                "img_size": int(args.img_size),
                "best_threshold": float(best_threshold),
                "args": vars(args),
                "loss_config": {
                    "ce": float(args.loss_ce_weight),
                    "dice": float(args.loss_dice_weight),
                    "cldice": float(args.loss_cldice_weight),
                    "edge": float(args.loss_edge_weight),
                },
                "target_config": {
                    "mode": str(args.target_mode),
                    "soft_distance_scale": float(args.target_soft_distance_scale),
                },
                "crop_mode_mix": crop_mode_mix,
                "crop_mode_mix_small": crop_mode_mix_small,
                "crop_mode_mix_ultra_tiny": crop_mode_mix_ultra_tiny,
                "tile_config": {"tile_sizes": tile_sizes, "tile_overlaps": tile_overlaps},
                "tta_config": {"tta_scales": tta_scales, "tta_hflip": bool(args.tta_hflip)},
            }
            torch.save(last_ckpt, run_dir / "last.pt")

            metrics_obj = {
                "created_at": utc_now_iso(),
                "model_family": str(args.model_family),
                "model_name": str(args.model_name),
                "checkpoint_format_version": 2,
                "annotations": str(annotations_path),
                "splits": str(splits_path),
                "negative_image_dir": str(neg_dir) if neg_dir is not None else "",
                "init_checkpoint": str(init_ckpt_path) if init_ckpt_path is not None else "",
                "train_samples_epoch": int(len(train_dataset)),
                "train_positive_records": int(len(train_pos_records)),
                "val_positive_records": int(len(val_pos_records)),
                "train_negative_records": int(len(train_neg_records)),
                "val_negative_records": int(len(val_neg_records)),
                "best_epoch": int(best_epoch),
                "best_composite_score": float(best_score),
                "best_threshold": float(best_threshold),
                "split_mode_requested": split_mode,
                "split_mode_used": split_reason,
                "split_size_thresholds": split_size_thresholds,
                "train_positive_summary": train_pos_summary,
                "val_positive_summary": val_pos_summary,
                "selection_metric": str(args.selection_metric),
                "best_centerline_epoch": int(best_centerline_epoch),
                "best_centerline_score": float(best_centerline_score),
                "hardcase_replay": hardcase_replay_summary,
                "loss_config": {
                    "ce": float(args.loss_ce_weight),
                    "dice": float(args.loss_dice_weight),
                    "cldice": float(args.loss_cldice_weight),
                    "edge": float(args.loss_edge_weight),
                },
                "target_config": {
                    "mode": str(args.target_mode),
                    "soft_distance_scale": float(args.target_soft_distance_scale),
                },
                "crop_mode_mix": crop_mode_mix,
                "crop_mode_mix_small": crop_mode_mix_small,
                "crop_mode_mix_ultra_tiny": crop_mode_mix_ultra_tiny,
                "tile_config": {"tile_sizes": tile_sizes, "tile_overlaps": tile_overlaps},
                "tta_config": {"tta_scales": tta_scales, "tta_hflip": bool(args.tta_hflip)},
                "history": history,
                "runtime_history": runtime_history,
            }
            write_json(run_dir / "metrics.json", metrics_obj)
            if bool(args.val_save_runtime_stats):
                write_json(
                    run_dir / "runtime_stats.json",
                    {
                        "updated_at": utc_now_iso(),
                        "val_runtime": str(args.val_runtime),
                        "history": runtime_history,
                    },
                )

            if epochs_no_improve >= int(args.early_stop_patience):
                print(
                    f"Early stopping: no composite improvement (delta>{float(args.early_stop_min_delta):.6f}) "
                    f"for {int(args.early_stop_patience)} epoch(s)."
                )
                break
    finally:
        if persistent_val_postprocess_executor is not None:
            persistent_val_postprocess_executor.shutdown(wait=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(
        run_dir / "config_used.json",
        {
            "args": vars(args),
            "resolved_paths": {
                "annotations": str(annotations_path),
                "splits": str(splits_path),
                "negative_image_dir": str(neg_dir) if neg_dir is not None else "",
                "save_dir": str(run_dir),
                "init_checkpoint": str(init_ckpt_path) if init_ckpt_path is not None else "",
            },
            "model_family": str(args.model_family),
            "model_name": str(args.model_name),
            "checkpoint_format_version": 2,
            "split_mode_requested": split_mode,
            "split_mode_used": split_reason,
            "split_size_thresholds": split_size_thresholds,
            "hardcase_replay": hardcase_replay_summary,
            "loss_config": {
                "ce": float(args.loss_ce_weight),
                "dice": float(args.loss_dice_weight),
                "cldice": float(args.loss_cldice_weight),
                "edge": float(args.loss_edge_weight),
            },
            "target_config": {
                "mode": str(args.target_mode),
                "soft_distance_scale": float(args.target_soft_distance_scale),
            },
            "crop_mode_mix": crop_mode_mix,
            "crop_mode_mix_small": crop_mode_mix_small,
            "crop_mode_mix_ultra_tiny": crop_mode_mix_ultra_tiny,
            "tile_config": {"tile_sizes": tile_sizes, "tile_overlaps": tile_overlaps},
            "tta_config": {"tta_scales": tta_scales, "tta_hflip": bool(args.tta_hflip)},
            "data_load_stats": load_stats,
            "train_positive_summary": train_pos_summary,
            "val_positive_summary": val_pos_summary,
            "train_pos_count": len(train_pos_records),
            "val_pos_count": len(val_pos_records),
            "train_neg_count": len(train_neg_records),
            "val_neg_count": len(val_neg_records),
            "runtime": {
                "val_runtime": str(args.val_runtime),
                "val_loader_workers": int(args.val_loader_workers),
                "val_postprocess_workers": int(args.val_postprocess_workers),
                "val_prefetch_items": int(args.val_prefetch_items),
                "val_max_batch_tiles": int(args.val_max_batch_tiles),
                "val_max_wait_ms": int(args.val_max_wait_ms),
                "val_bucket_by_tile": bool(args.val_bucket_by_tile),
                "val_runtime_fallback_direct": bool(args.val_runtime_fallback_direct),
            },
            "created_at": utc_now_iso(),
        },
    )

    print(f"Training complete. Best epoch: {best_epoch} | best composite: {best_score:.6f}")
    print(f"Best centerline epoch: {best_centerline_epoch} | best centerline F1: {best_centerline_score:.6f}")
    print(f"Best checkpoint: {run_dir / 'best.pt'}")
    print(f"Best centerline checkpoint: {run_dir / 'best_centerline.pt'}")
    print(f"Last checkpoint: {run_dir / 'last.pt'}")
    print(f"Metrics: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
