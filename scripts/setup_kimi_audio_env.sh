#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

KIMI_VENV="${KIMI_VENV:-$ROOT_DIR/.venv-kimi}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
HF_HOME="${HF_HOME:-/store/store5/acp21rjf/hf-cache}"

export UV_CACHE_DIR
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT_DIR/.cache/pip}"
export TMPDIR="${TMPDIR:-$ROOT_DIR/scratch/tmp}"

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HUGGINGFACE_HUB_CACHE"

uv venv --seed --python "$PYTHON_BIN" "$KIMI_VENV"
# shellcheck disable=SC1091
source "$KIMI_VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel packaging ninja

# Driver 535 advertises CUDA 12.2 on this server, so use PyTorch CUDA 11.8
# wheels rather than newer CUDA 12.4/12.6 wheels that require newer drivers.
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.6.0+cu118 torchaudio==2.6.0+cu118

python -m pip install \
  transformers accelerate soundfile librosa tqdm loguru huggingface_hub \
  conformer diffusers tiktoken timm torchdyn omegaconf hyperpyyaml \
  sentencepiece easydict fire ujson immutabledict rich wget gdown \
  datasets jsonlines pandas validators sty sox six blobfile sacrebleu \
  decord pillow cairosvg openai-whisper aiohttp colorama

python -m pip install flash_attn==2.7.4.post1 --no-build-isolation
python -m pip install --no-deps \
  "git+https://github.com/MoonshotAI/Kimi-Audio.git@349251e1d8f4f98d58fda59246381faecd7392e0"

python - <<'PY'
import torch
import transformers
from kimia_infer.api.kimia import KimiAudio

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("KimiAudio import ok", KimiAudio)
PY
