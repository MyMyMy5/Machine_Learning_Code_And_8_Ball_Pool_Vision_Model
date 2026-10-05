# Project history and artifact provenance

This page records what can be inferred from the surviving 8 Ball Pool project files, run logs, and file timestamps. File timestamps establish when an artifact was present or last modified in this archive; they do not, by themselves, establish when the project began or prove what ran successfully. The owner recalls beginning the work in late 2024, but that recollection is not supported by a contemporaneous record in the files described here.

## Original full-frame guideline project

| Date (Pacific) | Surviving evidence | What it supports |
| --- | --- | --- |
| Late 2024 | Owner recollection | A useful account of when work may have begun, not a timestamp verified by this archive. |
| 2025-09-25 | Earliest surviving source-file modification time visible in the original workspace | A source file in the current archive carries this date. It does not establish the project start date. |
| 2025-10-05 03:15:43 | First surviving checkpoint timestamp | Training artifacts existed by this time. Other archived training activity appears on October 5, 6, 8, 9, 16, and 17. |
| 2025-10-16 21:10:12 | `checkpoint_epoch_241_best.pth` | Original run’s saved best checkpoint. The trainer selects this tag by validation loss. |
| 2025-10-16 21:26:37 | `checkpoint_epoch_250_final.pth` | Final checkpoint recorded for the 250-epoch original run. The periodic `checkpoint_epoch_250.pth` is a separate file. |
| 2025-10-17 06:42:59 | `checkpoint_epoch_247_best.pth` in a later fine-tune archive | Latest saved best fine-tune checkpoint present in the publication artifact set. |
| 2025-10-18 01:45:07 | Fine-tune metrics log timestamp; records continue through epoch 487 | The log contains later metric rows than the latest saved best checkpoint. No later checkpoint was present in the artifact set used for this history. |

The checked-in baseline metrics log records these original-run values:

| Original baseline artifact | Validation loss | Validation Dice |
| --- | ---: | ---: |
| Epoch 241 best | 0.6570720593 | 0.4476531157 |
| Epoch 250 final | 0.6572360559 | 0.4473790514 |

The separate fine-tune log records epoch 247 at validation loss `0.6533210361` and validation Dice `0.4510562549`. The baseline log also has its own epoch-247 row (`0.6578926465` loss, `0.4468931389` Dice); that row is not the fine-tune artifact. These are historical run metrics, not new measurements. The data snapshots and exact split behind the later fine-tune values are not bundled here, so the values should not be compared as if they came from a verified common test set. In `ML.py`, Dice is calculated after a 0.5 probability cutoff, and the best checkpoint is chosen by validation loss rather than Dice.

The historical baseline pipeline extracted video frames, made heuristic masks with `prepare_dataset.py`, and later added a two-point manual annotation path for thin or small selected guidelines. The manual tool extends its line through the full image; it does not encode an arrow direction. The original project was described as extending the cue-ball aiming line, but at least one inspected manual label follows the short outgoing guideline at a purple object ball rather than the long cue-ball-to-reticle guide. Dataset masks therefore define the archived target semantics more precisely than the project name does.

Some prototype files appear in the current working folder with October 15, 2025 creation dates. Those dates can reflect copying into this folder and are not reliable evidence of when the prototypes were first written.

## Separate object-ball guideline project

The later `Test_New_Approach` project has a different target and should not be described as a continuation checkpoint of the original full-frame model. Its masks target a localized outgoing guideline at an object ball; the original project instead labels an extended line across the full screenshot according to each source mask.

| Date (Pacific) | Separate-project artifact | Interpretation |
| --- | --- | --- |
| 2026-03-26 22:34:45 | Primary `v2` checkpoint | Earliest primary checkpoint identified in that project’s artifact history. |
| 2026-05-22 04:34:25 | `v46` checkpoint | Present in the archive but marked rejected. |
| 2026-05-22 10:05 | Promoted `v47b` update | Changes inference policy; it does not replace the primary segmentation weights. |

These projects use different label scope and inference conventions. The localized object-ball output should not be presented as a general full-frame extension of the original cue-line target.

## Historical environment and verification boundary

The archived local environment was recorded as PyTorch `2.6.0+cu124`, torchvision `0.21.0`, NumPy `2.1.2`, OpenCV `4.12.0.88`, Pillow `11`, and MSS `10.1`. These versions describe that workstation, not a tested compatibility guarantee for every Windows or CUDA setup.

The original checkpoint format stores model weights under a `model` key, and the inference scripts load that state into the `UNet` defined in `ML.py` with its default `base_channels=38`. A strict state-dictionary load checks model-shape compatibility only. It does not validate segmentation quality, thin-line recall, screen capture alignment, live speed, or in-game outcomes. No claim of reliable gameplay performance is supported by the archived validation numbers alone.
