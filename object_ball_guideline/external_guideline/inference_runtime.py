from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import cv2
import numpy as np
import torch

from cv_guideline_common import (
    enumerate_tile_windows,
    enumerate_tta_variants,
    extract_padded_tile,
    predict_prob_multi_tta,
    run_prob_batch,
)


@dataclass
class InferenceJob:
    job_id: int
    image_path: str
    image_bgr: np.ndarray
    original_hw: Tuple[int, int]
    tile_sizes: Sequence[int]
    tile_overlaps: Sequence[int]
    tta_scales: Sequence[float]
    tta_hflip: bool
    use_amp: bool


@dataclass
class TileTask:
    job_id: int
    variant_id: int
    tile_size: int
    y: int
    x: int
    valid_h: int
    valid_w: int
    flip_variant: bool
    scale_factor: float
    use_amp: bool = True
    enqueued_at: float = 0.0
    enqueue_seq: int = 0


@dataclass
class TileResult:
    job_id: int
    variant_id: int
    tile_size: int
    y: int
    x: int
    valid_h: int
    valid_w: int
    prob_tile: np.ndarray


@dataclass
class JobAccumulator:
    prob_sum: np.ndarray
    prob_cnt: np.ndarray
    outstanding_tile_count: int
    completed_variants: int
    future: Future
    original_hw: Tuple[int, int]
    variant_frames: Dict[int, np.ndarray] = field(default_factory=dict)
    variant_flips: Dict[int, bool] = field(default_factory=dict)
    variant_prob_sum: Dict[int, np.ndarray] = field(default_factory=dict)
    variant_prob_cnt: Dict[int, np.ndarray] = field(default_factory=dict)
    variant_remaining: Dict[int, int] = field(default_factory=dict)
    subrun_count: int = 0
    image_path: str = ""


@dataclass
class RuntimeStats:
    images_submitted: int = 0
    images_completed: int = 0
    tiles_submitted: int = 0
    tiles_completed: int = 0
    queue_wait_ms_sum: float = 0.0
    forward_ms_sum: float = 0.0
    merge_ms_sum: float = 0.0
    batch_count: int = 0
    batch_size_sum: int = 0
    max_tile_batch: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    def as_dict(self) -> Dict[str, float]:
        elapsed = max(1e-6, time.perf_counter() - float(self.started_at))
        batch_count = max(1, int(self.batch_count))
        return {
            "images_submitted": int(self.images_submitted),
            "images_completed": int(self.images_completed),
            "tiles_submitted": int(self.tiles_submitted),
            "tiles_completed": int(self.tiles_completed),
            "avg_tile_batch": float(self.batch_size_sum) / float(batch_count),
            "max_tile_batch": int(self.max_tile_batch),
            "avg_queue_wait_ms": float(self.queue_wait_ms_sum) / float(batch_count),
            "avg_forward_ms": float(self.forward_ms_sum) / float(batch_count),
            "avg_merge_ms": float(self.merge_ms_sum) / float(max(1, int(self.tiles_completed))),
            "images_per_sec": float(self.images_completed) / elapsed,
            "tiles_per_sec": float(self.tiles_completed) / elapsed,
            "elapsed_sec": float(elapsed),
        }


class AsyncInferenceRuntime:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        max_batch_tiles: int = 32,
        max_wait_ms: int = 4,
        bucket_by_tile: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.max_batch_tiles = max(1, int(max_batch_tiles))
        self.max_wait_ms = max(1, int(max_wait_ms))
        self.bucket_by_tile = bool(bucket_by_tile)
        self._jobs: Dict[int, JobAccumulator] = {}
        self._jobs_lock = threading.Lock()
        self._task_queue: "queue.Queue[Optional[TileTask]]" = queue.Queue()
        self._result_queue: "queue.Queue[Optional[TileResult]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._stats = RuntimeStats()
        self._stats_lock = threading.Lock()
        self._gpu_thread = threading.Thread(target=self._run_gpu_worker, daemon=True, name="guideline-gpu-runtime")
        self._merge_thread = threading.Thread(target=self._run_merge_worker, daemon=True, name="guideline-merge-runtime")
        self._gpu_thread.start()
        self._merge_thread.start()

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return int(self._seq)

    def submit_job(self, job: InferenceJob) -> Future:
        h, w = int(job.original_hw[0]), int(job.original_hw[1])
        future: Future = Future()
        accumulator = JobAccumulator(
            prob_sum=np.zeros((h, w), dtype=np.float32),
            prob_cnt=np.zeros((h, w), dtype=np.float32),
            outstanding_tile_count=0,
            completed_variants=0,
            future=future,
            original_hw=(h, w),
            image_path=str(job.image_path),
        )
        pairs: List[Tuple[int, int]] = []
        if job.tile_sizes and job.tile_overlaps and len(job.tile_sizes) == len(job.tile_overlaps):
            pairs = [(max(64, int(ts)), max(0, int(to))) for ts, to in zip(job.tile_sizes, job.tile_overlaps)]
        else:
            ts = int(job.tile_sizes[0]) if job.tile_sizes else 640
            to = int(job.tile_overlaps[0]) if job.tile_overlaps else 160
            pairs = [(ts, to)]

        variants = enumerate_tta_variants(job.image_bgr, job.tta_scales, job.tta_hflip)
        with self._jobs_lock:
            self._jobs[int(job.job_id)] = accumulator
        subrun_id = 0
        for variant in variants:
            variant_frame = variant["frame_bgr"]
            vh, vw = variant_frame.shape[:2]
            for ts, to in pairs:
                windows = enumerate_tile_windows((vh, vw), tile_size=ts, tile_overlap=to)
                accumulator.variant_frames[subrun_id] = variant_frame
                accumulator.variant_flips[subrun_id] = bool(variant["flip_variant"])
                accumulator.variant_prob_sum[subrun_id] = np.zeros((vh, vw), dtype=np.float32)
                accumulator.variant_prob_cnt[subrun_id] = np.zeros((vh, vw), dtype=np.float32)
                accumulator.variant_remaining[subrun_id] = len(windows)
                accumulator.subrun_count += 1
                accumulator.outstanding_tile_count += len(windows)
                for y, x, valid_h, valid_w in windows:
                    self._task_queue.put(
                        TileTask(
                            job_id=int(job.job_id),
                            variant_id=int(subrun_id),
                            tile_size=int(ts),
                            y=int(y),
                            x=int(x),
                            valid_h=int(valid_h),
                            valid_w=int(valid_w),
                            flip_variant=bool(variant["flip_variant"]),
                            scale_factor=float(variant["scale_factor"]),
                            use_amp=bool(job.use_amp),
                            enqueued_at=time.perf_counter(),
                            enqueue_seq=self._next_seq(),
                        )
                    )
                subrun_id += 1

        if accumulator.outstanding_tile_count <= 0:
            with self._jobs_lock:
                self._jobs.pop(int(job.job_id), None)
            future.set_result(np.zeros((h, w), dtype=np.float32))
            return future
        with self._stats_lock:
            self._stats.images_submitted += 1
            self._stats.tiles_submitted += int(accumulator.outstanding_tile_count)
        return future

    def shutdown(self) -> None:
        self._stop_event.set()
        self._task_queue.put(None)
        self._result_queue.put(None)
        self._gpu_thread.join(timeout=5.0)
        self._merge_thread.join(timeout=5.0)

    def get_stats(self) -> Dict[str, float]:
        with self._stats_lock:
            return self._stats.as_dict()

    def _take_task_batch(self, pending_by_tile: Dict[int, List[TileTask]]) -> List[TileTask]:
        if not pending_by_tile:
            return []
        if not self.bucket_by_tile:
            tile_size = sorted(pending_by_tile.keys())[0]
        else:
            ranked = []
            for tile_size, items in pending_by_tile.items():
                if not items:
                    continue
                ranked.append((len(items), -int(items[0].enqueue_seq), int(tile_size)))
            ranked.sort(reverse=True)
            tile_size = int(ranked[0][2])
        batch = pending_by_tile[tile_size][: self.max_batch_tiles]
        pending_by_tile[tile_size] = pending_by_tile[tile_size][self.max_batch_tiles :]
        if not pending_by_tile[tile_size]:
            pending_by_tile.pop(tile_size, None)
        return batch

    def _run_gpu_worker(self) -> None:
        pending_by_tile: Dict[int, List[TileTask]] = {}
        stop_seen = False
        while True:
            if not pending_by_tile:
                try:
                    item = self._task_queue.get(timeout=0.1)
                except queue.Empty:
                    if stop_seen or self._stop_event.is_set():
                        break
                    continue
                if item is None:
                    stop_seen = True
                    continue
                pending_by_tile.setdefault(int(item.tile_size), []).append(item)

            deadline = time.perf_counter() + (float(self.max_wait_ms) / 1000.0)
            while sum(len(v) for v in pending_by_tile.values()) < int(self.max_batch_tiles):
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    nxt = self._task_queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if nxt is None:
                    stop_seen = True
                    continue
                pending_by_tile.setdefault(int(nxt.tile_size), []).append(nxt)

            batch = self._take_task_batch(pending_by_tile)
            if not batch:
                if stop_seen or self._stop_event.is_set():
                    break
                continue

            batch_now = time.perf_counter()
            queue_wait_ms = float(
                np.mean([max(0.0, (batch_now - float(task.enqueued_at)) * 1000.0) for task in batch], dtype=np.float64)
            )
            patches: List[np.ndarray] = []
            meta: List[TileTask] = []
            for task in batch:
                with self._jobs_lock:
                    acc = self._jobs.get(int(task.job_id))
                if acc is None:
                    continue
                frame_bgr = acc.variant_frames.get(int(task.variant_id))
                if frame_bgr is None:
                    continue
                patch, _, _ = extract_padded_tile(frame_bgr, int(task.y), int(task.x), int(task.tile_size))
                patches.append(patch)
                meta.append(task)
            if not patches:
                continue

            forward_start = time.perf_counter()
            prob_batch = run_prob_batch(self.model, patches, device=self.device, use_amp=bool(meta[0].use_amp))
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
            for task, prob_tile in zip(meta, prob_batch):
                self._result_queue.put(
                    TileResult(
                        job_id=int(task.job_id),
                        variant_id=int(task.variant_id),
                        tile_size=int(task.tile_size),
                        y=int(task.y),
                        x=int(task.x),
                        valid_h=int(task.valid_h),
                        valid_w=int(task.valid_w),
                        prob_tile=np.ascontiguousarray(prob_tile[: int(task.valid_h), : int(task.valid_w)], dtype=np.float32),
                    )
                )
            with self._stats_lock:
                self._stats.batch_count += 1
                self._stats.batch_size_sum += len(meta)
                self._stats.max_tile_batch = max(int(self._stats.max_tile_batch), len(meta))
                self._stats.queue_wait_ms_sum += float(queue_wait_ms)
                self._stats.forward_ms_sum += float(forward_ms)
                self._stats.tiles_completed += int(len(meta))

        self._result_queue.put(None)

    def _run_merge_worker(self) -> None:
        stop_seen = False
        while True:
            try:
                item = self._result_queue.get(timeout=0.1)
            except queue.Empty:
                with self._jobs_lock:
                    jobs_left = bool(self._jobs)
                if (stop_seen or self._stop_event.is_set()) and not jobs_left:
                    break
                continue
            if item is None:
                stop_seen = True
                with self._jobs_lock:
                    jobs_left = bool(self._jobs)
                if not jobs_left:
                    break
                continue

            merge_start = time.perf_counter()
            with self._jobs_lock:
                acc = self._jobs.get(int(item.job_id))
            if acc is None:
                continue

            sub_sum = acc.variant_prob_sum[int(item.variant_id)]
            sub_cnt = acc.variant_prob_cnt[int(item.variant_id)]
            y0 = int(item.y)
            x0 = int(item.x)
            y1 = y0 + int(item.valid_h)
            x1 = x0 + int(item.valid_w)
            sub_sum[y0:y1, x0:x1] += item.prob_tile
            sub_cnt[y0:y1, x0:x1] += 1.0
            acc.variant_remaining[int(item.variant_id)] -= 1
            acc.outstanding_tile_count -= 1

            if acc.variant_remaining[int(item.variant_id)] <= 0:
                prob_variant = acc.variant_prob_sum[int(item.variant_id)] / np.maximum(
                    acc.variant_prob_cnt[int(item.variant_id)], 1e-6
                )
                if bool(acc.variant_flips[int(item.variant_id)]):
                    prob_variant = prob_variant[:, ::-1].copy()
                oh, ow = acc.original_hw
                if prob_variant.shape[:2] != (oh, ow):
                    prob_variant = cv2.resize(prob_variant, (ow, oh), interpolation=cv2.INTER_LINEAR)
                acc.prob_sum += prob_variant.astype(np.float32)
                acc.prob_cnt += 1.0
                acc.completed_variants += 1
                acc.variant_prob_sum.pop(int(item.variant_id), None)
                acc.variant_prob_cnt.pop(int(item.variant_id), None)
                acc.variant_frames.pop(int(item.variant_id), None)
                acc.variant_flips.pop(int(item.variant_id), None)
                acc.variant_remaining.pop(int(item.variant_id), None)

            if acc.outstanding_tile_count <= 0 and acc.completed_variants >= acc.subrun_count:
                final_prob = acc.prob_sum / np.maximum(acc.prob_cnt, 1e-6)
                acc.future.set_result(final_prob.astype(np.float32))
                with self._jobs_lock:
                    self._jobs.pop(int(item.job_id), None)
                with self._stats_lock:
                    self._stats.images_completed += 1

            merge_ms = (time.perf_counter() - merge_start) * 1000.0
            with self._stats_lock:
                self._stats.merge_ms_sum += float(merge_ms)


LoadedT = TypeVar("LoadedT")
ResultT = TypeVar("ResultT")
ItemT = TypeVar("ItemT")


def run_async_pipeline(
    items: Sequence[ItemT],
    load_fn: Callable[[int, ItemT], LoadedT],
    build_job_fn: Callable[[int, LoadedT], InferenceJob],
    postprocess_fn: Callable[[int, LoadedT, np.ndarray], ResultT],
    runtime: AsyncInferenceRuntime,
    loader_workers: int = 4,
    postprocess_workers: int = 4,
    prefetch_items: int = 16,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    postprocess_executor: Optional[Executor] = None,
    postprocess_submit_fn: Optional[Callable[[Executor, int, LoadedT, np.ndarray], Future]] = None,
) -> Tuple[List[ResultT], Dict[str, float]]:
    total = len(items)
    if total <= 0:
        return [], runtime.get_stats()

    prefetch = max(1, int(prefetch_items))
    next_submit = 0
    results: Dict[int, ResultT] = {}
    pending_load: Dict[Future, int] = {}
    pending_infer: Dict[Future, Tuple[int, LoadedT]] = {}
    pending_post: Dict[Future, int] = {}

    owns_post_executor = postprocess_executor is None
    post_executor: Optional[Executor] = postprocess_executor
    if post_executor is None:
        post_executor = ThreadPoolExecutor(max_workers=max(1, int(postprocess_workers)))

    with ThreadPoolExecutor(max_workers=max(1, int(loader_workers))) as load_pool:
        try:
            assert post_executor is not None

            def submit_postprocess(idx: int, loaded: LoadedT, prob: np.ndarray) -> Future:
                if postprocess_submit_fn is not None:
                    return postprocess_submit_fn(post_executor, idx, loaded, prob)
                return post_executor.submit(postprocess_fn, idx, loaded, prob)

            def top_up() -> None:
                nonlocal next_submit
                while next_submit < total and (len(pending_load) + len(pending_infer) + len(pending_post)) < prefetch:
                    idx = int(next_submit)
                    fut = load_pool.submit(load_fn, idx, items[idx])
                    pending_load[fut] = idx
                    next_submit += 1

            top_up()
            while len(results) < total:
                progressed = False

                for fut in list(pending_post.keys()):
                    if fut.done():
                        idx = pending_post.pop(fut)
                        results[idx] = fut.result()
                        if progress_callback is not None:
                            progress_callback(len(results), total, runtime.get_stats())
                        progressed = True

                for fut in list(pending_infer.keys()):
                    if fut.done():
                        idx, loaded = pending_infer.pop(fut)
                        prob = fut.result()
                        pf = submit_postprocess(idx, loaded, prob)
                        pending_post[pf] = idx
                        progressed = True

                for fut in list(pending_load.keys()):
                    if fut.done():
                        idx = pending_load.pop(fut)
                        loaded = fut.result()
                        job = build_job_fn(idx, loaded)
                        inf_fut = runtime.submit_job(job)
                        pending_infer[inf_fut] = (idx, loaded)
                        progressed = True

                top_up()
                if progressed:
                    continue

                waitables: List[Future] = list(pending_post.keys()) + list(pending_infer.keys()) + list(pending_load.keys())
                if not waitables:
                    break
                wait(waitables, timeout=0.05, return_when=FIRST_COMPLETED)
        finally:
            if owns_post_executor and post_executor is not None:
                post_executor.shutdown(wait=True)

    ordered = [results[i] for i in range(total)]
    if progress_callback is not None:
        progress_callback(len(ordered), total, runtime.get_stats())
    return ordered, runtime.get_stats()


def build_process_pool(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max(1, int(max_workers)),
        mp_context=mp.get_context("spawn"),
    )


def compare_runtime_equivalence(
    model: torch.nn.Module,
    frames_bgr: Sequence[np.ndarray],
    device: torch.device,
    tile_sizes: Sequence[int],
    tile_overlaps: Sequence[int],
    tta_scales: Sequence[float],
    tta_hflip: bool,
    use_amp: bool,
    max_batch_tiles: int = 32,
    max_wait_ms: int = 4,
    bucket_by_tile: bool = True,
) -> Dict[str, float]:
    direct_probs = [
        predict_prob_multi_tta(
            model=model,
            frame_bgr=frame,
            device=device,
            tile_sizes=tile_sizes,
            tile_overlaps=tile_overlaps,
            tta_scales=tta_scales,
            tta_hflip=tta_hflip,
            use_amp=use_amp,
        )
        for frame in frames_bgr
    ]

    runtime = AsyncInferenceRuntime(
        model=model,
        device=device,
        max_batch_tiles=max_batch_tiles,
        max_wait_ms=max_wait_ms,
        bucket_by_tile=bucket_by_tile,
    )
    try:
        jobs = [
            InferenceJob(
                job_id=i,
                image_path="",
                image_bgr=frame,
                original_hw=tuple(int(x) for x in frame.shape[:2]),
                tile_sizes=tile_sizes,
                tile_overlaps=tile_overlaps,
                tta_scales=tta_scales,
                tta_hflip=tta_hflip,
                use_amp=use_amp,
            )
            for i, frame in enumerate(frames_bgr)
        ]
        futures = [runtime.submit_job(job) for job in jobs]
        async_probs = [f.result() for f in futures]
    finally:
        runtime.shutdown()

    max_abs = 0.0
    mean_abs = 0.0
    total = 0
    for direct, async_prob in zip(direct_probs, async_probs):
        diff = np.abs(direct.astype(np.float32) - async_prob.astype(np.float32))
        max_abs = max(max_abs, float(diff.max()) if diff.size > 0 else 0.0)
        mean_abs += float(diff.mean()) if diff.size > 0 else 0.0
        total += 1
    return {
        "items": int(total),
        "max_abs_diff": float(max_abs),
        "mean_abs_diff": float(mean_abs / max(1, total)),
    }
