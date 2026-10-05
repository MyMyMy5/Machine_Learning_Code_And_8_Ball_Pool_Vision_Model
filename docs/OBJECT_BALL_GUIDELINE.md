# Object-ball guideline model package

This package documents the promoted inference stack from `Test_New_Approach`.
It is a separate model task from the cue-line extension described in
`cue_line_extension`: its positive label is the short white guideline leaving
the object ball after contact. The project contract explicitly treats the
cue-ball aiming line as negative material (see
the [canonical progress record](../object_ball_guideline/docs/experiments/supervised_target_line_progress.md)).

## Promoted stack

The active manifest is `object_ball_guideline/runs/supervised_best_manifest.json`
with `candidate_name` `v47b_roi_expand_selector_probe`. The primary segmenter
and reranker are still the v2 checkpoints. v47b changed line-endpoint proposal
and arbitration rules; it did not produce a new primary segmentation
checkpoint. Keep the manifest policy intact when packaging its checkpoints.

The manifest refers to these 15 physical checkpoint files. Repeated references
to the same model, and Windows/WSL spellings of one path, are listed once.

| Role in the promoted manifest | Checkpoint path under `object_ball_guideline/` |
| --- | --- |
| Primary segmenter and reranker | `runs/supervised_train_v2/guideline_unet_best.pt`; `runs/supervised_train_v2/guideline_reranker_best.pt` |
| Fallback segmenter and reranker | `runs/supervised_train_v5_negft/guideline_unet_best.pt`; `runs/supervised_train_v5_negft/guideline_reranker_best.pt` |
| Rescue segmenter and reranker | `runs/supervised_train_v6_residual/guideline_unet_best.pt`; `runs/supervised_train_v6_residual/guideline_reranker_best.pt` |
| Fallback and reticle specialist | `runs/supervised_train_v7_bestfinal5_residual/guideline_unet_best.pt`; `runs/supervised_train_v7_bestfinal5_residual/guideline_reranker_b.pt` |
| Micro-line specialist | `runs/supervised_train_v10_repair_lr2e-4_pw18/guideline_unet_best.pt` |
| Thin-line specialist pools | `runs/supervised_train_v14_thinline_repair_lw2_nw05_abs02_lr1e-4_pw18/guideline_unet_best.pt`; `runs/supervised_train_v18_thinline512_v7init_lw25_nw075_abs01_lr5e-5_pw18/guideline_unet_best.pt` |
| Image veto models | `runs/final_veto_v3_image_focus/image_veto_global_s42.pt`; `runs/supplemental_v7_validation/image_veto_v4_external_reticle/image_veto_v4_global_s42.pt` |
| External main and tiny-line rescue models | `external_guideline/runs_cv/ft_after_clean2000_softdistance_v1/best.pt`; `external_guideline/runs_cv/ft_balanced_ultratiny_replay_v1/best.pt` |

The latest `.pt` file by timestamp in the source repository is the v46
specialist checkpoint, written 2026-05-22 04:34:25 PDT. The experiment record
rejects v46 as a primary model, and the v47b manifest does not reference it. It
can be included as a clearly labelled experiment artifact, but it must not
replace a v47b dependency.
There is no rescue-arbiter checkpoint referenced by the active manifest; the
separate `rescue_arbiter_v1_s42.pt` file belongs to a rejected experiment.

## External runtime files

`src/supervised/infer.py` loads the external rescue models through
`_load_external_guideline_runtime`. It sets the runtime root to
`checkpoint_path.parents[2]`, adds that directory to `sys.path`, then imports
`cv_guideline_common`. Preserve this layout so the root remains
`external_guideline`:

```text
external_guideline/
  cv_guideline_common.py
  mask_refinement.py
  runs_cv/
    ft_after_clean2000_softdistance_v1/best.pt
    ft_balanced_ultratiny_replay_v1/best.pt
```

`cv_guideline_common.py` contains the external checkpoint model, loader,
multi-scale/TTA predictor, and postprocessing functions used by this project.
Its only local Python import is `mask_refinement.py`. The imported runtime also
requires `cv2`, `numpy`, `torch`, `timm`, and `transformers`; the source
repository's historical training requirements did not list `timm` or
`transformers`; `requirements-publication.txt` includes them. The two bundled
rescue checkpoints use the local custom segmentation architecture. Their
runtime does not require a SAM foundation-model checkpoint. Related local
training and dataset annotation scripts are preserved beside the runtime.

## Portable paths and execution

The source manifest contains absolute paths rooted at `C:\My_Project\...` and
`/mnt/c/My_Project/...`. In a publication copy, change only checkpoint path
values to package-relative paths such as `runs/...` and
`external_guideline/runs_cv/...`; preserve all thresholds, gates, and rescue
rules. `manifest_inference.normalize_path` converts Windows and WSL absolute
paths, but leaves relative paths relative to the process working directory.
The publication wrapper resolves all checkpoint paths against its own package
directory, preserving the numerical policy. It can be invoked from any working
directory with correctly specified input and output paths.

The current `scripts/run_supervised_best_wsl.sh` passes inference parameters
directly to `src.supervised.infer`; it does not load every setting from the
manifest. It also has absolute checkpoint paths for external models and
absolute paths embedded in its micro-line, colored-blob, and image-veto JSON
settings. Updating the manifest alone is insufficient for a portable launch.
Those historical launchers are preserved unchanged for provenance. Use the new
manifest-driven wrapper for a portable launch.

Install a compatible PyTorch/torchvision build and the publication dependencies,
then run the full promoted policy:

```powershell
python -m pip install -r object_ball_guideline/requirements-publication.txt
cd object_ball_guideline
python predict.py --input ../assets/original_frame.png --output runs/demo_prediction
```

The output directory contains `mask_final.png`, `overlay_final.png`, and a
candidate report. `--disable-amp` disables mixed-precision inference. The
underlying runtime selects CUDA when available and otherwise uses CPU.

The direct training entry point is `python -m src.supervised.train`, backed by
`src/supervised/data.py`. Historical `scripts/train_supervised_wsl.sh` uses the
original local dataset path. Training constructs its
crop index from an external `data_zoomprobe` annotation dataset and selected
negative-image folders. Those datasets are not included in this inference
checkpoint package, so the code and weights alone do not reproduce training.
For a newly collected dataset matching the zoomprobe schema, run from this
package directory and specify your own data and output roots:

```powershell
python -m src.supervised.train --data-root PATH_TO_DATA_ZOOMPROBE --output-root runs/new_training_run --epochs 40
```

Inspect `python -m src.supervised.train --help` for hard-negative folders,
model choice, loss weights, cropping, exclusion IDs, and resume options. Existing
dataset schemas and split rules are in `src/supervised/data.py`. The original
project's long-line masks cannot be substituted directly for these localized
outgoing-guideline labels.

## Source-of-truth records

For the target definition and promotion history, see
[the canonical progress record](../object_ball_guideline/docs/experiments/supervised_target_line_progress.md) and
[the handoff](../object_ball_guideline/docs/experiments/HANDOFF.md). The handoff identifies
v47b as the current promoted manifest and says the change is an
inference/proposal repair, not a stronger primary segmentation checkpoint.
