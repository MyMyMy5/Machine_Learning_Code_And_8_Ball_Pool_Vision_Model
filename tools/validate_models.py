"""Validate the packaged sources and checkpoint loaders without training."""
import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def original():
    import torch
    from PIL import Image
    sys.path.insert(0, str(ROOT / "cue_line_extension"))
    from ML import UNet
    from preview import infer_single, postprocess_mask, save_overlay
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = ROOT / "validation_outputs/original"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("checkpoint_epoch_241_best.pth", "checkpoint_epoch_247_best.pth", "checkpoint_epoch_250_final.pth"):
        state = torch.load(ROOT / "cue_line_extension/checkpoints" / name, map_location="cpu", weights_only=True)
        model = UNet()
        matched = model.load_state_dict(state["model"], strict=True)
        model.eval().to(device)
        mask = infer_single(model, ROOT / "assets/original_frame.png", device)
        refined = postprocess_mask(mask, 0.35, 6)
        save_overlay(ROOT / "assets/original_frame.png", refined, output / f"{name}.png")
        records.append({"checkpoint": name, "epoch": state.get("epoch"), "strict_load": not (matched.missing_keys or matched.unexpected_keys), "output_shape": list(mask.shape), "finite": bool(torch.from_numpy(mask).isfinite().all()), "device": str(device), "min_probability": float(mask.min()), "max_probability": float(mask.max())})
        del model, state
    (output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


def validate_sources():
    count = 0
    for path in ROOT.rglob("*.py"):
        if path.name.startswith(".validation") or any(part in {".git", ".validation_deps", "__pycache__"} for part in path.parts):
            continue
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        count += 1
    source = json.loads((ROOT / "object_ball_guideline/docs/historical_supervised_best_manifest.json").read_text())
    portable = json.loads((ROOT / "object_ball_guideline/runs/supervised_best_manifest.json").read_text())
    def without_paths(value):
        if isinstance(value, dict):
            return {k: without_paths(v) for k, v in value.items()}
        if isinstance(value, list):
            return [without_paths(v) for v in value]
        if isinstance(value, str) and value.lower().endswith((".pt", ".pth")):
            return "CHECKPOINT_PATH"
        return value
    assert without_paths(source) == without_paths(portable), "Manifest policy changed beyond path relocation"
    print(f"PASS: {count} Python sources parse; promoted policy unchanged beyond checkpoint path relocation")


def validate_later_weights():
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / "object_ball_guideline"))
    from src.supervised.infer import _load_guideline_model, _load_reranker_cached, _load_external_guideline_runtime
    from src.supervised.image_final_veto import load_image_final_veto_checkpoint
    registry = json.loads((ROOT / "MODEL_REGISTRY.json").read_text())
    output = []
    for item in registry["models"]:
        if not item["path"].startswith("object_ball_guideline/"):
            continue
        path = ROOT / item["path"]
        if "/external_guideline/" in item["path"]:
            model = _load_external_guideline_runtime(path)["model"]
        elif "image_veto" in path.name:
            model = load_image_final_veto_checkpoint(path, device=torch.device("cpu"))[0]
        elif "reranker" in path.name:
            model = _load_reranker_cached(path, torch.device("cpu"))
        else:
            model = _load_guideline_model(path, torch.device("cpu"))
        output.append({"checkpoint": item["path"], "loader_completed": True, "type": type(model).__name__})
        print(f"PASS: {item['path']}", flush=True)
    target = ROOT / "validation_outputs/later_checkpoint_loads.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"PASS: {len(output)} later checkpoints loaded by their packaged runtime loaders")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["original", "sources", "weights"])
    args = parser.parse_args()
    if args.mode == "original":
        original()
    elif args.mode == "weights":
        validate_later_weights()
    else:
        validate_sources()
