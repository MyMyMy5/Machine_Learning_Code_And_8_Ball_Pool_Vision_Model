# Publication validation

The publication copy was checked on **October 5, 2026**. These checks establish
artifact integrity, runtime compatibility, and execution on one example frame.
They are separate from the historical model-quality benchmarks.

## Results

| Check | Result | Evidence |
|---|---|---|
| Model artifact integrity | PASS — 19 sizes and SHA-256 hashes match the original files | `tools/verify_artifacts.py`, `MODEL_REGISTRY.json`, `CHECKSUMS.sha256` |
| Promoted inference dependencies | PASS — all 15 distinct dependency files are present inside the package | Portable manifest and artifact verifier |
| Source parsing | PASS — 106 distributed Python source files parse | `python tools/validate_models.py sources` |
| Promoted policy preservation | PASS — only checkpoint paths changed; all other JSON values match the historical manifest | Source validation mode |
| Copied regression suite | PASS — 90 tests passed in 43.02 seconds | Existing `object_ball_guideline/tests/` suite |
| Original checkpoint loading | PASS — epoch 241 best, epoch 247 fine-tuned best, and epoch 250 final load strictly into `ML.UNet` | [Forward-pass records](validation_records/original_forward_passes.json) |
| Original single-frame inference | PASS for runtime execution — each checkpoint produces a finite 512 × 512 probability mask and a saved overlay on CUDA; the epoch-247 example has a known extension bug | Same forward-pass records and known failure below |
| Later checkpoint loading | PASS — all 16 later checkpoint files, including both external rescue models and rejected v46, load through their packaged runtime loaders | [Loader records](validation_records/later_checkpoint_loads.json) |
| Full promoted single-frame inference | PASS — v47b writes its mask, overlay and candidate report | [Prediction summary](validation_records/later_prediction_summary.json) |
| Publication guides and asset links | PASS — checked relative links in the new guides, README pages, and checkpoint catalog | Publication packaging audit |

The later example output has shape **1050 × 1576**, contains **107 nonzero mask
pixels**, and selects the `external_main_reticle_rescue` branch. The overlay was
visually reviewed: the short guideline at the purple object ball is highlighted.
The original extended prediction has an incorrect geometric extension and stray
predicted fragments, as identified by the owner. These observations apply to
this example only; successful execution does not establish visual correctness.

## Known original extension failure

The [original epoch-247 prediction](../assets/original_model_prediction.png)
contains an incorrect fitted extension and extra yellow segments. It remains in
the repository as failure evidence. Compare it with the existing
[correct manual label](../assets/manual_overlay.png), which contains one straight
white extended line through the selected guideline.

The README now shows that manual label as the intended output and explicitly
identifies it as an annotation. This documentation correction does not fix the
model or its geometric postprocessing. No checkpoint, label file, training code,
or inference policy was changed.

The new portable wrapper initially supplied strings where the historical
manifest runtime expected `Path` objects. Its adapter now uses the existing
manifest loader and resolves both top-level `Path` values and nested checkpoint
strings against the package directory. The full inference command passed after
that correction. The original model and inference source were not edited.

## Validation environment

GPU checks used the existing WSL Ubuntu training environment. The original
source's historical Windows environment is recorded separately in
`cue_line_extension/requirements-historical.txt`.

| Dependency | Version used for publication validation |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| timm | 1.0.26 |
| transformers | 5.4.0 |
| NumPy | 2.4.3 |
| OpenCV headless | 4.13.0.92 |
| Pillow | 12.1.1 |
| pytest | 9.1.1, installed into an isolated validation directory |

## Reproduce the checks

After installing the appropriate dependencies and pulling the LFS weights, run
from the repository root:

```powershell
python tools/verify_artifacts.py
python tools/validate_models.py sources
python tools/validate_models.py original
python tools/validate_models.py weights

cd object_ball_guideline
python -m pip install pytest
python -m pytest tests -q
python predict.py --input ../assets/original_frame.png --output validation_outputs/example
```

The validator's `original` mode writes example overlays under
`validation_outputs/original/`; `weights` writes a later loader summary. These
generated outputs are ignored by Git. The full original datasets, exact
historical data snapshots, and every historical evaluation image are not bundled,
so the archived benchmark results are preserved as recorded evidence rather
than claimed as newly reproduced measurements.

Live screen capture, interactive annotation windows, full retraining, gameplay
latency, and fresh independent accuracy benchmarks were not rerun for this
publication. No checkpoint was retrained or promoted during packaging.
