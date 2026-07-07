# Resume Todo: Kimi Audio EmoNet Smoke Test

Last updated: 2026-07-07 UTC

## Goal

Get `moonshotai/Kimi-Audio-7B-Instruct` running on a tiny EmoNet smoke test, with reusable repo scripts and storage kept out of `/home/acp21rjf`.

## User Constraints

- Do not create meaningful files under `/home/acp21rjf`; it is space constrained.
- Do not modify shared Python, Conda, CUDA, or NVIDIA installs.
- Use only project-local environments created for this repo.
- Datasets should live under `/store/store4` or `/store/store5` under `data`; `store5` has more free space and is being used.
- Large model/cache paths are under `/store/store5/acp21rjf`.
- Keep the repo updated/pushed as work progresses.

## Current Repo

- Repo: `/exp/exp4/acp21rjf/SLLM-understanding`
- Remote: `https://github.com/robflynnyh/SLLM-Understanding`
- Branch: `main`
- Latest pushed commits before this note:
  - `5c8fecf Add EmoNet data preparation scaffold`
  - `b6561ee Add Kimi Audio evaluation setup`
  - `0b4c7c5 Add EmoNet smoke and quick split builder`
  - `55dbbfb Keep Kimi pip cache repo-local`
  - `ec3803b Seed pip in Kimi venv setup`
  - `e19306a Route Kimi setup temp files to repo scratch`
  - `a932992 Pin Kimi torchvision to CUDA 11.8 torch stack`

## Storage Paths

- Data root: `/store/store5/acp21rjf/data/emonet-voice-bench`
- Model root: `/store/store5/acp21rjf/models/Kimi-Audio-7B-Instruct`
- HF cache: `/store/store5/acp21rjf/hf-cache`
- Repo venv: `/exp/exp4/acp21rjf/SLLM-understanding/.venv-kimi`
- Repo pip cache: `/exp/exp4/acp21rjf/SLLM-understanding/.cache/pip`
- Repo scratch/tmp: `/exp/exp4/acp21rjf/SLLM-understanding/scratch/tmp`

Approx sizes after prep/download:

- Kimi text-only model snapshot: about `22G`
- EmoNet data root: about `2.1G`
- HF cache: about `1.3M`
- `/store/store5`: about `229G` free
- `/tmp` root partition is tight, avoid it

## Completed

- Project-local `.venv-kimi` was created.
- `torch==2.6.0+cu118`, `torchaudio==2.6.0+cu118`, and `torchvision==0.21.0+cu118` installed.
- `flash_attn==2.7.4.post1` built successfully from source.
  - Build used repo-local scratch and cache, not `/home/acp21rjf` or `/tmp`.
  - Wheel cached under `.cache/pip`.
- Kimi package installed from:
  - `git+https://github.com/MoonshotAI/Kimi-Audio.git@349251e1d8f4f98d58fda59246381faecd7392e0`
- Import test passed:
  - `torch 2.6.0+cu118`
  - CUDA runtime `11.8`
  - `torch.cuda.is_available() == True`
  - `transformers 5.13.0`
  - `from kimia_infer.api.kimia import KimiAudio` works
- Kimi text-only snapshot downloaded successfully:
  - Command: `. .venv-kimi/bin/activate && python scripts/download_kimi_audio.py`
  - Output path: `/store/store5/acp21rjf/models/Kimi-Audio-7B-Instruct`
  - Mode: text-only allow patterns, includes main model and `whisper-large-v3`, excludes audio detokenizer/vocoder.
- EmoNet data prep completed successfully:
  - Command: `python scripts/prepare_emonet.py all --data-root /store/store5/acp21rjf/data/emonet-voice-bench`
  - Manifest: `/store/store5/acp21rjf/data/emonet-voice-bench/manifests/train.jsonl`
  - Rows: `12600`
  - Extracted data root size: about `2.1G`

## Active Process At Handoff

No Kimi setup, model download, or EmoNet prep process was still running when this note was updated.

## Next Todo

1. Build deterministic smoke/quick splits:

   ```bash
   cd /exp/exp4/acp21rjf/SLLM-understanding
   . .venv-kimi/bin/activate
   python scripts/build_emonet_splits.py \
     --data-root /store/store5/acp21rjf/data/emonet-voice-bench \
     --overwrite
   ```

2. Build a tiny Kimi request file first, not the full 1600-request smoke:

   ```bash
   python scripts/build_emonet_requests.py \
     --data-root /store/store5/acp21rjf/data/emonet-voice-bench \
     --manifest /store/store5/acp21rjf/data/emonet-voice-bench/manifests/smoke.jsonl \
     --output runs/emonet_smoke_requests.jsonl \
     --emotion-set official40
   ```

3. Run only a few Kimi requests first:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/run_kimi_emonet_requests.py \
     --model-path /store/store5/acp21rjf/models/Kimi-Audio-7B-Instruct \
     --requests runs/emonet_smoke_requests.jsonl \
     --output runs/kimi_smoke_raw.jsonl \
     --limit 5 \
     --overwrite
   ```

4. If the model OOMs on a 20GB RTX A4500, inspect/patch the Kimi loader.
   - Upstream `KimiAudio.__init__` loads `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16, trust_remote_code=True)` and then calls `.to(torch.cuda.current_device())`.
   - There is no `device_map` in that path, so single-GPU OOM is plausible.
   - Likely next options: add a repo-side patched loader using `device_map="auto"`/multi-GPU, or test lower-memory loading if supported by the remote model code.
5. If smoke succeeds, commit any new code/docs and push remaining changes.

## Important Script Files

- `scripts/setup_kimi_audio_env.sh`
- `scripts/download_kimi_audio.py`
- `scripts/prepare_emonet.py`
- `scripts/build_emonet_splits.py`
- `scripts/build_emonet_requests.py`
- `scripts/run_kimi_emonet_requests.py`
- `configs/kimi_audio.json`
- `configs/emonet_data.json`
- `configs/emonet_eval.json`
- `README.md`

## Eval Setup Notes

- User wants raw scores kept so calibration can be explored later.
- Prompt form:

  ```text
  Score this audio from 1-10 based on the presence of the following emotion: {emotion}
  ```

- Current request builder creates one request per `(audio, emotion)`.
- Official label set is 40 emotions.
- Dataset also has `Authenticity` and `Arousal`; use `--emotion-set all42` only if needed.
- Human labels in the dataset are one target emotion per row, raw `0/1/2`, mapped in prep to `0/5/10`.

## Useful Checks

```bash
cd /exp/exp4/acp21rjf/SLLM-understanding
git status --short --branch

. .venv-kimi/bin/activate
python - <<'PY'
import torch, transformers, flash_attn
from kimia_infer.api.kimia import KimiAudio
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(transformers.__version__)
print(flash_attn.__version__)
print(KimiAudio)
PY

du -sh /store/store5/acp21rjf/models/Kimi-Audio-7B-Instruct
du -sh /store/store5/acp21rjf/data/emonet-voice-bench
df -h /store/store5 /exp/exp4 /tmp
```
