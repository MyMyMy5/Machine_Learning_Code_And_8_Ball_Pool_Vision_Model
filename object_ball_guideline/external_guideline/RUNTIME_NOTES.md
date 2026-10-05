# Runtime Notes

Updated: 2026-03-02

## Purpose

This file documents the accuracy-preserving async inference runtime added for `guideline_line`.

The runtime exists to speed up:
- training-time validation
- standalone evaluation
- large-batch inference

without changing:
- model outputs
- tile geometry
- TTA behavior
- postprocess logic
- promotion metrics

## Why This Is Not a Direct `block_blaster` Port

`block_blaster` uses a process-oriented inference server pattern that is well suited for:
- many CPU worker processes
- small repeated GPU requests
- long-lived static model weights

`guideline_line` has a different bottleneck:
- sequential image loading
- sequential tiled/TTA inference
- CPU postprocess between GPU bursts

Training-time validation also needs the live in-memory model each epoch. On Windows, a separate CUDA process server would add complexity and synchronization risk without being the highest-confidence first step.

So V1 uses:
- in-process threaded runtime
- queue-based tile submission
- batched GPU tile execution
- CPU loader/postprocess overlap

## Runtime Modes

Supported modes:
- `direct`
- `async`

`direct`:
- original sequential execution
- reference path for correctness checks

`async`:
- CPU loader workers prepare jobs
- GPU worker batches tiles by tile size
- CPU postprocess workers reconstruct and score outputs

## Tuning Knobs

Training validation:
- `--val-runtime`
- `--val-loader-workers`
- `--val-postprocess-workers`
- `--val-prefetch-items`
- `--val-max-batch-tiles`
- `--val-max-wait-ms`
- `--val-bucket-by-tile`

Standalone eval/infer:
- `--infer-runtime`
- `--loader-workers`
- `--postprocess-workers`
- `--prefetch-items`
- `--max-batch-tiles`
- `--max-wait-ms`
- `--bucket-by-tile`

Recommended starting values:
- loader workers: `4`
- postprocess workers: `4`
- prefetch items: `16`
- max batch tiles: `32`
- max wait ms: `4`
- bucket by tile: `true`

## Accuracy Guardrail

The async runtime is intended to preserve numerical behavior.

Reference helper:
- `compare_runtime_equivalence(...)` in `guideline_line/inference_runtime.py`

Expected tolerance:
- `max_abs_diff <= 1e-4`
- `mean_abs_diff <= 1e-6`

If async output diverges materially from direct output, treat that as a correctness bug and use `direct`.

## Fallback Behavior

Training:
- default expectation is correctness first
- async validation can be forced to fail loudly
- optional fallback to direct exists via `--val-runtime-fallback-direct`

Eval / infer:
- async can fall back to direct with:
  - `--runtime-fallback-direct true`

## Future V2

Deferred on purpose:
- shared-memory transport
- external always-on daemon
- separate CUDA process server

V2 is only justified if profiling shows queue transport itself is a real bottleneck after CPU/GPU overlap is already working well.
