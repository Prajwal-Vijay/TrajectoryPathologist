#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_ROOT="${VL_EXTERNAL_ROOT:-/media/helen/8TB_VOL3/TrajectoryPathologist_VL}"
ENV_PREFIX="$EXTERNAL_ROOT/env"
MODEL_DIR="$EXTERNAL_ROOT/models/Qwen3-VL-8B-Instruct"

export CONDA_PKGS_DIRS="$EXTERNAL_ROOT/conda-pkgs"
export PIP_CACHE_DIR="$EXTERNAL_ROOT/pip-cache"
export PIP_CONFIG_FILE=/dev/null
export HF_HOME="$EXTERNAL_ROOT/huggingface"
export TMPDIR="$EXTERNAL_ROOT/tmp"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p \
  "$CONDA_PKGS_DIRS" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$TMPDIR" \
  "$EXTERNAL_ROOT/models" \
  "$EXTERNAL_ROOT/mosaics" \
  "$EXTERNAL_ROOT/outputs"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.12 pip
fi

"$ENV_PREFIX/bin/python" -m pip install --upgrade pip
"$ENV_PREFIX/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1 torchvision==0.24.1
"$ENV_PREFIX/bin/python" -m pip install \
  transformers==5.18.0 \
  accelerate \
  av==15.1.0 \
  huggingface_hub \
  mcap \
  mcap-protobuf-support \
  numpy \
  opencv-python-headless \
  pytest \
  qwen-vl-utils==0.0.14

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  "$ENV_PREFIX/bin/python" - "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    local_dir=sys.argv[1],
)
PY
fi

ln -sfn "$ENV_PREFIX" "$REPO_ROOT/VL_implemenation/env"
ln -sfn "$MODEL_DIR" "$REPO_ROOT/VL_implemenation/model"
ln -sfn "$EXTERNAL_ROOT/mosaics" "$REPO_ROOT/VL_implemenation/mosaics"
ln -sfn "$EXTERNAL_ROOT/outputs" "$REPO_ROOT/VL_implemenation/outputs"

PYTHONPATH="$REPO_ROOT" "$ENV_PREFIX/bin/python" - <<'PY'
import torch
import transformers

assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("gpu:", torch.cuda.get_device_name(0))
print("bf16:", torch.cuda.is_bf16_supported())
PY
