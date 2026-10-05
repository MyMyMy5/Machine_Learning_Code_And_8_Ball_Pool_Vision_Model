"""Run the preserved promoted policy with package-relative checkpoint paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def resolve_checkpoint_paths(value):
    if isinstance(value, dict):
        return {key: resolve_checkpoint_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_checkpoint_paths(item) for item in value]
    if isinstance(value, Path):
        path = value if value.is_absolute() else PACKAGE_ROOT / value
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint (run git lfs pull): {path}")
        return path.resolve()
    if isinstance(value, str) and value.lower().endswith((".pt", ".pth")):
        path = Path(value)
        if not path.is_absolute():
            path = PACKAGE_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint (run git lfs pull): {path}")
        return str(path.resolve())
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=PACKAGE_ROOT / "runs/supervised_best_manifest.json")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-batch-size", type=int, default=16)
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input image does not exist: {args.input}")
    # Only checkpoint strings are relocated; every numerical and selection rule
    # is kept exactly as in the historical promoted manifest.
    from src.supervised.manifest_inference import load_manifest, run_manifest_inference
    manifest = resolve_checkpoint_paths(load_manifest(args.manifest))
    result = run_manifest_inference(
        manifest=manifest,
        input_path=args.input.resolve(),
        output_dir=args.output.resolve(),
        image_size=args.image_size,
        crop_batch_size=args.crop_batch_size,
        amp_enabled=not args.disable_amp,
        return_overlay=False,
    )
    summary = {
        "candidate_name": manifest.get("candidate_name"),
        "output_dir": str(args.output.resolve()),
        "pred_pixels": int((result["final_mask"] > 0).sum()),
        "mask_shape": list(result["final_mask"].shape),
        "winner_source": result.get("winner", {}).get("candidate_source"),
        "mask_path": str(args.output.resolve() / "mask_final.png"),
    }
    (args.output.resolve() / "prediction_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
