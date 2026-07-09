#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MOSS_VENV="${MOSS_VENV:-$ROOT_DIR/.venv-moss}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
HF_HOME="${HF_HOME:-/store/store5/data/acp21rjf/hf-cache}"
MOSS_SRC_DIR="${MOSS_SRC_DIR:-$ROOT_DIR/scratch/deps/MOSS-Audio}"
MOSS_REPO="${MOSS_REPO:-https://github.com/OpenMOSS/MOSS-Audio.git}"
MOSS_COMMIT="${MOSS_COMMIT:-5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883}"

export UV_CACHE_DIR
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT_DIR/.cache/pip}"
export TMPDIR="${TMPDIR:-$ROOT_DIR/scratch/tmp}"
export TEMP="${TEMP:-$TMPDIR}"
export TMP="${TMP:-$TMPDIR}"

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HUGGINGFACE_HUB_CACHE" "$(dirname "$MOSS_SRC_DIR")"

uv venv --seed --python "$PYTHON_BIN" "$MOSS_VENV"
# shellcheck disable=SC1091
source "$MOSS_VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel packaging ninja

# Driver 535 advertises CUDA 12.2 on this server, so use PyTorch CUDA 11.8
# wheels rather than CUDA 12.8 wheels that require newer drivers.
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.6.0+cu118 torchaudio==2.6.0+cu118

python -m pip install \
  --upgrade-strategy only-if-needed \
  transformers==4.57.1 accelerate "numpy>=2.0" safetensors soundfile \
  tiktoken einops scipy tqdm packaging requests huggingface_hub librosa

if [[ ! -d "$MOSS_SRC_DIR/.git" ]]; then
  git clone "$MOSS_REPO" "$MOSS_SRC_DIR"
fi
git -C "$MOSS_SRC_DIR" fetch --depth 1 origin "$MOSS_COMMIT"
git -C "$MOSS_SRC_DIR" checkout --detach "$MOSS_COMMIT"

python -m pip install --no-deps -e "$MOSS_SRC_DIR"

python - <<'PY'
import torch
import transformers
from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("MOSS imports ok", MossAudioModel, MossAudioProcessor)
PY
