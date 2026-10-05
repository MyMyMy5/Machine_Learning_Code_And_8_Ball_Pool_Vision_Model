#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="$REPO_ROOT/.wsl_envs/guideline_train"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

if command -v micromamba >/dev/null 2>&1; then
  MAMBA_BIN="micromamba"
elif command -v conda >/dev/null 2>&1; then
  MAMBA_BIN=""
else
  MAMBA_BIN="$(bash "$REPO_ROOT/tools/bootstrap_micromamba_wsl.sh")"
fi

if [[ -n "${MAMBA_BIN:-}" ]]; then
  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    "$MAMBA_BIN" create -y -p "$ENV_PREFIX" python=3.12
  fi
  "$MAMBA_BIN" run -p "$ENV_PREFIX" python -m pip install --upgrade pip
  "$MAMBA_BIN" run -p "$ENV_PREFIX" python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url "$TORCH_INDEX"
  "$MAMBA_BIN" run -p "$ENV_PREFIX" python -m pip install -r "$REPO_ROOT/requirements-train.txt"
else
  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    conda create -y -p "$ENV_PREFIX" python=3.12
  fi
  conda run -p "$ENV_PREFIX" python -m pip install --upgrade pip
  conda run -p "$ENV_PREFIX" python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url "$TORCH_INDEX"
  conda run -p "$ENV_PREFIX" python -m pip install -r "$REPO_ROOT/requirements-train.txt"
fi

echo "Supervised training env ready at $ENV_PREFIX"
