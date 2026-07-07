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
- Latest pushed commits before the 2026-07-07 smoke continuation:
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
  - `transformers 4.47.1`
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
- Deterministic split build completed:
  - Command: `python scripts/build_emonet_splits.py --data-root /store/store5/acp21rjf/data/emonet-voice-bench --overwrite`
  - Smoke manifest: `/store/store5/acp21rjf/data/emonet-voice-bench/manifests/smoke.jsonl`
  - Quick manifest: `/store/store5/acp21rjf/data/emonet-voice-bench/manifests/quick.jsonl`
  - Local prepared official-label manifest currently has 36 labels, so smoke rows are `36` and quick rows are `360`.
- Request build completed:
  - Command: `python scripts/build_emonet_requests.py --data-root /store/store5/acp21rjf/data/emonet-voice-bench --manifest /store/store5/acp21rjf/data/emonet-voice-bench/manifests/smoke.jsonl --output runs/emonet_smoke_requests.jsonl --emotion-set official40`
  - Output: `runs/emonet_smoke_requests.jsonl`
  - Rows: `36`; emotions per row: `40`; total requests: `1440`.
- Tiny Kimi smoke completed successfully:
  - Output: `runs/kimi_smoke_raw.jsonl`
  - Rows: `5`
  - Raw responses were parseable integer scores.
  - `runs/` is gitignored, so these outputs are local artifacts.
- All-at-once request build and tiny smoke completed:
  - Request command: `python scripts/build_emonet_requests.py --data-root /store/store5/acp21rjf/data/emonet-voice-bench --manifest /store/store5/acp21rjf/data/emonet-voice-bench/manifests/smoke.jsonl --output runs/emonet_smoke_all_at_once_requests.jsonl --emotion-set official40 --mode all_at_once --limit 2`
  - Kimi output: `runs/kimi_smoke_all_at_once_raw.jsonl`
  - Rows run: `1`
  - The response parsed cleanly as one JSON object with all `40` expected emotions, no missing/extra/invalid score keys.
  - The first raw all-at-once response assigned `0` to every emotion, so format is validated but score quality needs a larger check.
- Human-rubric one-by-one mode added:
  - Builder mode: `--mode one_by_one_human_rubric`
  - Prompt asks for the presence of one emotion on the human-style `0/1/2` scale.
  - Rubric: `0` absent, `1` weakly or ambiguously present, `2` clearly or strongly present.
  - Runner parses this mode into `raw_parsed_score` and `raw_parsed_score_0_2`, while preserving raw text.
  - A full 40-emotion row-level smoke for row `537` wrote `runs/kimi_check_human_rubric_row537_all_raw.jsonl`.
  - Target emotion `Embarrassment` parsed as `0`; the row's human mean raw score is `0.3333333333333333`.
  - Nonzero model rubric scores were all `1`: `Contentment`, `Hope`, `Triumph`, `Pride`, `Interest`, `Concentration`, `Contemplation`, `Doubt`, `Confusion`, `Contempt`, `Fatigue`, and `Emotional Numbness`.

## Active Process At Handoff

No Kimi setup, model download, EmoNet prep, or Kimi smoke process was still running when this note was updated.

## Next Todo

1. Run a larger local one-by-one smoke, still before the full request set:

   ```bash
   cd /exp/exp4/acp21rjf/SLLM-understanding
   . .venv-kimi/bin/activate
   TMPDIR=$PWD/scratch/tmp \
   TEMP=$PWD/scratch/tmp \
   TMP=$PWD/scratch/tmp \
   XDG_CACHE_HOME=$PWD/.cache \
   PIP_CACHE_DIR=$PWD/.cache/pip \
   HF_HOME=/store/store5/acp21rjf/hf-cache \
   HUGGINGFACE_HUB_CACHE=/store/store5/acp21rjf/hf-cache/hub \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   python scripts/run_kimi_emonet_requests.py \
     --model-path /store/store5/acp21rjf/models/Kimi-Audio-7B-Instruct \
     --requests runs/emonet_smoke_requests.jsonl \
     --output runs/kimi_smoke_raw_50.jsonl \
     --limit 50 \
     --overwrite \
     --device-map auto \
     --max-primary-gpu-memory 10GiB \
     --max-gpu-memory 18GiB \
     --max-cpu-memory 96GiB
   ```

2. Run a larger all-at-once smoke to check whether the all-zero first response is isolated or systematic:

   ```bash
   cd /exp/exp4/acp21rjf/SLLM-understanding
   . .venv-kimi/bin/activate
   TMPDIR=$PWD/scratch/tmp \
   TEMP=$PWD/scratch/tmp \
   TMP=$PWD/scratch/tmp \
   XDG_CACHE_HOME=$PWD/.cache \
   PIP_CACHE_DIR=$PWD/.cache/pip \
   HF_HOME=/store/store5/acp21rjf/hf-cache \
   HUGGINGFACE_HUB_CACHE=/store/store5/acp21rjf/hf-cache/hub \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   python scripts/run_kimi_emonet_requests.py \
     --model-path /store/store5/acp21rjf/models/Kimi-Audio-7B-Instruct \
     --requests runs/emonet_smoke_all_at_once_requests.jsonl \
     --output runs/kimi_smoke_all_at_once_raw_10.jsonl \
     --limit 10 \
     --overwrite \
     --max-new-tokens 768 \
     --device-map auto \
     --max-primary-gpu-memory 10GiB \
     --max-gpu-memory 18GiB \
     --max-cpu-memory 96GiB
   ```

3. If the larger smokes are stable, run the full `runs/emonet_smoke_requests.jsonl` file or build a manifest-sized request subset for calibration experiments.
4. Build and smoke the human-rubric request file:

   ```bash
   cd /exp/exp4/acp21rjf/SLLM-understanding
   . .venv-kimi/bin/activate
   python scripts/build_emonet_requests.py \
     --data-root /store/store5/acp21rjf/data/emonet-voice-bench \
     --manifest /store/store5/acp21rjf/data/emonet-voice-bench/manifests/smoke.jsonl \
     --output runs/emonet_smoke_human_rubric_requests.jsonl \
     --emotion-set official40 \
     --mode one_by_one_human_rubric
   ```

5. Add scoring/aggregation scripts only after enough raw Kimi outputs exist to validate the format.

## Loader Notes

- The Kimi repo dependency metadata does not pin `transformers`; the installed package only requires bare `transformers`.
- The model config says it was saved with `transformers_version: 4.44.1`, but local testing found `4.47.1` keeps the required Kimi/Qwen2 API shape while avoiding the 5.x API drift.
- The local `whisper-large-v3/model.safetensors` has no safetensors metadata. The runner patches Transformers' safetensors reader in-process to treat metadata-less files as PyTorch format without editing the 3GB checkpoint.
- Single-GPU loading OOMed on a 20GB RTX A4500. The runner now supports `--device-map auto` plus `--max-primary-gpu-memory` to leave room on GPU 0 for the audio tokenizer and Whisper sidecar.

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
