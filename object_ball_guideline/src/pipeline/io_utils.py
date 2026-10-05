from __future__ import annotations

import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def stable_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}, got {type(data)!r}")
    return data


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_path(value: str | None, repo_root: Path = REPO_ROOT) -> str | None:
    if value in (None, "", "auto"):
        return None
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return str(path)


def resolve_env_paths(
    env_config: dict[str, Any], repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    resolved = deepcopy(env_config)
    defaults = resolved.setdefault("defaults", {})
    defaults["repo_root"] = str(repo_root)
    defaults["windows_repo_root"] = str(repo_root)
    defaults.setdefault("wsl_repo_root", None)
    for _, env_spec in resolved.get("environments", {}).items():
        for key in (
            "python",
            "working_dir",
            "repo_path",
            "checkpoint_path",
            "bpe_path",
        ):
            if key in env_spec:
                env_spec[key] = resolve_path(env_spec.get(key), repo_root=repo_root)
        if "extra_pythonpath" in env_spec:
            env_spec["extra_pythonpath"] = [
                resolve_path(item, repo_root=repo_root) or item
                for item in env_spec.get("extra_pythonpath", [])
            ]
    return resolved


def load_runtime_configs(
    config_override_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    default_config = load_yaml_file(REPO_ROOT / "configs" / "default.yaml")
    prompts_config = load_yaml_file(REPO_ROOT / "configs" / "prompts.yaml")
    env_config = load_yaml_file(REPO_ROOT / "configs" / "envs.yaml")
    if config_override_path:
        override = load_yaml_file(config_override_path)
        default_config = merge_dicts(default_config, override)
    env_config = resolve_env_paths(env_config, REPO_ROOT)
    return default_config, prompts_config, env_config


def apply_cli_overrides(
    config: dict[str, Any],
    prompt_set_override: str | None,
    recolor_hex: str | None,
    crop_scale: float | None,
    max_ball_candidates: int | None,
    debug: bool,
    save_intermediates: bool,
    no_recolor: bool,
) -> dict[str, Any]:
    result = deepcopy(config)
    if prompt_set_override:
        result.setdefault("sam3", {})["prompt_set"] = prompt_set_override
    if recolor_hex:
        result.setdefault("recolor", {})["default_color"] = recolor_hex
    if crop_scale is not None:
        result.setdefault("candidate_generation", {})["crop_scale"] = crop_scale
    if max_ball_candidates is not None:
        result.setdefault("candidate_generation", {})[
            "max_ball_candidates"
        ] = max_ball_candidates
    if debug:
        result.setdefault("runtime", {})["debug"] = True
    if save_intermediates:
        result.setdefault("runtime", {})["save_intermediates"] = True
    if no_recolor:
        result.setdefault("recolor", {})["enabled"] = False
    return result


def load_rgb_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def save_rgb_image(path: str | Path, image: np.ndarray) -> None:
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(path)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def expand_box_xyxy(
    box: list[float] | tuple[float, float, float, float],
    margin: int,
    image_shape: tuple[int, int],
) -> list[int]:
    height, width = image_shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, int(round(x0 - margin)))
    y0 = max(0, int(round(y0 - margin)))
    x1 = min(width, int(round(x1 + margin)))
    y1 = min(height, int(round(y1 + margin)))
    return [x0, y0, x1, y1]


def mask_to_box_xyxy(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def draw_box(
    image: np.ndarray,
    box: list[int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    output = image.copy()
    x0, y0, x1, y1 = [int(v) for v in box]
    cv2.rectangle(output, (x0, y0), (x1, y1), color, thickness)
    return output
