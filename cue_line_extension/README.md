# Original line-extension model

Read the [complete guide](../docs/CUE_LINE_EXTENSION.md) for dataset creation, two-point annotation, training, offline predictions, and live capture.

The latest surviving fine-tuned best is `checkpoints/checkpoint_epoch_247_best.pth`. The historical live-inference commands selected `checkpoints/checkpoint_epoch_241_best.pth`; both match `ML.UNet` with 38 base channels. `preview.postprocess_mask` fits and extends the predicted line to image boundaries.

All original Python source in this folder is copied unchanged. The sample capture coordinates in `config/region.example.json` are historical; use `calibrate_region.py` to create your own `config/region.json`.
