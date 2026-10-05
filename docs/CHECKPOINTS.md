# Checkpoint catalog

All weights are the original serialized checkpoint files, copied without conversion or retraining and tracked with Git LFS. The total is 19 files, 847,700,453 bytes (about 808 MiB).

## Which weights to use

- **Original selected-line extension:** `cue_line_extension/checkpoints/checkpoint_epoch_247_best.pth` is the latest surviving fine-tuned best. Epoch 241 is the checkpoint selected in the recorded live commands. Both require the architecture and preprocessing in `cue_line_extension/ML.py` and `preview.py`.
- **Later object-ball guideline detector:** use `object_ball_guideline/predict.py` and its promoted v47b manifest. The v2 primary model is only one part of this stack; all 15 dependency files are included.
- **Latest rejected experiment:** v46 is preserved for historical completeness. It is not a promoted replacement and is not referenced by v47b.

## Complete inventory

Times below are the source files' last-modified timestamps in UTC, preserved during copying. They establish the surviving artifact timestamps, not the start of the project. The history page also presents the important dates in Pacific time.

| Artifact | Size (MiB) | Source modified (UTC) | Role |
|---|---:|---|---|
| [original/checkpoints/checkpoint_epoch_241_best.pth](../cue_line_extension/checkpoints/checkpoint_epoch_241_best.pth) | 125.38 | 2025-10-17 04:10:12 | Historical best used by recorded live-inference commands |
| [original/checkpoints/checkpoint_epoch_247_best.pth](../cue_line_extension/checkpoints/checkpoint_epoch_247_best.pth) | 125.38 | 2025-10-17 13:42:59 | Latest surviving fine-tuned best; minimum recorded fine-tune validation loss |
| [original/checkpoints/checkpoint_epoch_250_final.pth](../cue_line_extension/checkpoints/checkpoint_epoch_250_final.pth) | 125.38 | 2025-10-17 04:26:37 | Final checkpoint of the original 250-epoch run |
| [later/runs/supervised_train_v2/guideline_unet_best.pt](../object_ball_guideline/runs/supervised_train_v2/guideline_unet_best.pt) | 29.68 | 2026-03-27 05:34:45 | Referenced by the promoted v47b manifest |
| [later/runs/supervised_train_v2/guideline_reranker_best.pt](../object_ball_guideline/runs/supervised_train_v2/guideline_reranker_best.pt) | 0.02 | 2026-03-27 16:47:10 | Referenced by the promoted v47b manifest |
| [later/external_guideline/runs_cv/ft_after_clean2000_softdistance_v1/best.pt](../object_ball_guideline/external_guideline/runs_cv/ft_after_clean2000_softdistance_v1/best.pt) | 96.36 | 2026-03-06 16:36:02 | Referenced by the promoted v47b manifest |
| [later/external_guideline/runs_cv/ft_balanced_ultratiny_replay_v1/best.pt](../object_ball_guideline/external_guideline/runs_cv/ft_balanced_ultratiny_replay_v1/best.pt) | 96.36 | 2026-03-04 12:24:50 | Referenced by the promoted v47b manifest |
| [later/runs/supervised_train_v5_negft/guideline_unet_best.pt](../object_ball_guideline/runs/supervised_train_v5_negft/guideline_unet_best.pt) | 29.68 | 2026-03-27 21:10:54 | Referenced by the promoted v47b manifest |
| [later/runs/supervised_train_v5_negft/guideline_reranker_best.pt](../object_ball_guideline/runs/supervised_train_v5_negft/guideline_reranker_best.pt) | 0.02 | 2026-03-27 21:28:15 | Referenced by the promoted v47b manifest |
| [later/runs/supervised_train_v7_bestfinal5_residual/guideline_unet_best.pt](../object_ball_guideline/runs/supervised_train_v7_bestfinal5_residual/guideline_unet_best.pt) | 29.68 | 2026-03-28 14:26:28 | Referenced by the promoted v47b manifest |
| [later/runs/supervised_train_v6_residual/guideline_reranker_best.pt](../object_ball_guideline/runs/supervised_train_v6_residual/guideline_reranker_best.pt) | 0.02 | 2026-03-27 23:22:50 | Referenced by the promoted v47b manifest |
| [later/runs/final_veto_v3_image_focus/image_veto_global_s42.pt](../object_ball_guideline/runs/final_veto_v3_image_focus/image_veto_global_s42.pt) | 1.01 | 2026-04-28 18:53:20 | Referenced by the promoted v47b manifest |
| [later/runs/supplemental_v7_validation/image_veto_v4_external_reticle/image_veto_v4_global_s42.pt](../object_ball_guideline/runs/supplemental_v7_validation/image_veto_v4_external_reticle/image_veto_v4_global_s42.pt) | 1.01 | 2026-04-30 00:36:33 | Referenced by the promoted v47b manifest |
| [later/runs/supervised_train_v10_repair_lr2e-4_pw18/guideline_unet_best.pt](../object_ball_guideline/runs/supervised_train_v10_repair_lr2e-4_pw18/guideline_unet_best.pt) | 29.68 | 2026-04-25 01:18:25 | Referenced by the promoted v47b manifest |
| [later/runs/supervised_train_v6_residual/guideline_unet_best.pt](../object_ball_guideline/runs/supervised_train_v6_residual/guideline_unet_best.pt) | 29.68 | 2026-03-27 23:01:33 | Referenced by the promoted v47b manifest |
| [later/runs/supervised_train_v7_bestfinal5_residual/guideline_reranker_b.pt](../object_ball_guideline/runs/supervised_train_v7_bestfinal5_residual/guideline_reranker_b.pt) | 0.02 | 2026-03-28 14:54:53 | Referenced by the promoted v47b manifest |
| [later/runs/supervised_train_v14_thinline_repair_lw2_nw05_abs02_lr1e-4_pw18/guideline_unet_best.pt](../object_ball_guideline/runs/supervised_train_v14_thinline_repair_lw2_nw05_abs02_lr1e-4_pw18/guideline_unet_best.pt) | 29.68 | 2026-04-30 08:38:59 | Referenced by the promoted v47b manifest |
| [later/runs/supervised_train_v18_thinline512_v7init_lw25_nw075_abs01_lr5e-5_pw18/guideline_unet_best.pt](../object_ball_guideline/runs/supervised_train_v18_thinline512_v7init_lw25_nw075_abs01_lr5e-5_pw18/guideline_unet_best.pt) | 29.71 | 2026-05-06 07:31:50 | Referenced by the promoted v47b manifest |
| [later/runs/supervised_train_v46_seg_specialist_v10_512_lw3_nw1_abs02_lr5e-5_pw24_ctx/guideline_unet_best.pt](../object_ball_guideline/runs/supervised_train_v46_seg_specialist_v10_512_lw3_nw1_abs02_lr5e-5_pw24_ctx/guideline_unet_best.pt) | 29.68 | 2026-05-22 11:34:25 | Latest surviving experimental checkpoint; rejected, not promoted |

## Integrity and provenance

Run `python tools/verify_artifacts.py` from the repository root after cloning and pulling LFS objects. It checks the size and SHA-256 of each model and confirms that every promoted manifest dependency exists inside the package.

[MODEL_REGISTRY.json](../MODEL_REGISTRY.json) records source paths, checksums, file sizes, timestamps, roles, and the path-only manifest relocation map. [CHECKSUMS.sha256](../CHECKSUMS.sha256) is a compact model checksum list.

The original checkpoint payloads include model and training state. The later checkpoint payloads preserve their model configuration, and selected history/config files are copied beside the weights. Historical evaluation summaries are preserved under [benchmark_records](../object_ball_guideline/benchmark_records/). Complete training datasets and every intermediate experimental artifact are not bundled.
