from __future__ import annotations

import json
import os
from pathlib import Path


def normalize_path(path_str: str) -> Path:
    if os.name != "nt" and path_str.startswith("C:\\"):
        return Path("/mnt/c/" + path_str[3:].replace("\\", "/"))
    if os.name == "nt" and path_str.startswith("/mnt/c/"):
        return Path("C:/" + path_str[len("/mnt/c/"):])
    return Path(path_str)


def _normalize_nested_config_paths(value: dict[str, object]) -> dict[str, object]:
    normalized = dict(value)
    for nested_key in ("checkpoint", "checkpoint_path", "reranker_checkpoint", "reranker_checkpoint_path"):
        nested_value = normalized.get(nested_key)
        if nested_value:
            normalized[nested_key] = str(normalize_path(str(nested_value)))
    return normalized


def load_manifest(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized: dict[str, object] = {}
    for key, value in manifest.items():
        if key in {"metrics", "documentation", "review_bundle", "benchmark"}:
            normalized[key] = value
        elif key in {"micro_line_rescue", "colored_blob_rescue", "selector_candidate_pool_rescue", "image_final_veto"} and isinstance(value, dict):
            normalized[key] = _normalize_nested_config_paths(value)
        elif key in {"colored_blob_rescue", "selector_candidate_pool_rescue", "image_final_veto"} and isinstance(value, list):
            normalized[key] = [
                _normalize_nested_config_paths(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif key.endswith("_checkpoint") or key.endswith("_reranker"):
            normalized[key] = normalize_path(str(value))
        else:
            normalized[key] = value
    return normalized


def collect_image_files(input_dir: Path, patterns: list[str], recurse: bool) -> list[Path]:
    items = []
    for pattern in patterns:
        if recurse:
            items.extend(input_dir.rglob(pattern))
        else:
            items.extend(input_dir.glob(pattern))
    files = sorted(path for path in items if path.is_file())
    unique: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def output_stem(path: Path, root_dir: Path) -> str:
    try:
        relative = path.relative_to(root_dir)
    except ValueError:
        return path.stem
    stem_path = relative.with_suffix("")
    return "__".join(stem_path.parts)
