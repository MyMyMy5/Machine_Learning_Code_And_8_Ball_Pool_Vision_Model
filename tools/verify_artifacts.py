"""Verify all packaged checkpoint hashes and promoted manifest dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from checkpoint_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from checkpoint_values(item)
    elif isinstance(value, str) and value.lower().endswith((".pt", ".pth")):
        yield value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    registry = json.loads((root / "MODEL_REGISTRY.json").read_text(encoding="utf-8"))
    failed = []
    for item in registry["models"]:
        path = root / item["path"]
        if not path.is_file():
            failed.append(f"Missing: {item['path']}")
            continue
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != item["bytes"] or actual != item["sha256"]:
            failed.append(f"Size/hash mismatch (check Git LFS): {item['path']}")
    package = root / "object_ball_guideline"
    manifest = json.loads((package / "runs/supervised_best_manifest.json").read_text(encoding="utf-8"))
    dependencies = set(checkpoint_values(manifest))
    for value in dependencies:
        path = Path(value)
        if path.is_absolute() or ":" in value or not (package / path).is_file():
            failed.append(f"Non-portable or missing manifest dependency: {value}")
        elif not (package / path).resolve().is_relative_to(package.resolve()):
            failed.append(f"Manifest dependency outside package: {value}")
    if failed:
        print("\n".join(failed))
        raise SystemExit(1)
    print(f"PASS: {len(registry['models'])} checkpoint hashes; {len(dependencies)} promoted dependencies; candidate={manifest['candidate_name']}")


if __name__ == "__main__":
    main()
