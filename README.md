# SLLM Understanding

Reusable code for preparing and evaluating speech-language model experiments on
EmoNet-Voice Bench.

## Storage

Keep large datasets outside this repo. On this server the default dataset root is:

```bash
/store/store5/acp21rjf/data/emonet-voice-bench
```

On another server, pass `--data-root /path/to/emonet-voice-bench` or set:

```bash
export EMONET_DATA_ROOT=/path/to/emonet-voice-bench
```

Repo-local folders such as `outputs/`, `runs/`, and `scratch/` are for small
generated outputs and are gitignored.

## Dataset Preparation

The dataset is about 1.1 GB as Parquet shards. The downloader uses only the
Python standard library. The preparation step requires `pyarrow` to read Parquet
and extract embedded audio bytes.

Download the raw Parquet shards:

```bash
python scripts/prepare_emonet.py download
```

Prepare extracted audio plus JSONL manifests:

```bash
python scripts/prepare_emonet.py prepare
```

Or run both:

```bash
python scripts/prepare_emonet.py all
```

Useful options:

```bash
python scripts/prepare_emonet.py all --data-root /store/store5/acp21rjf/data/emonet-voice-bench
python scripts/prepare_emonet.py prepare --max-rows 50
python scripts/prepare_emonet.py prepare --no-audio
```

The main manifest is written to:

```bash
$EMONET_DATA_ROOT/manifests/train.jsonl
```

Each manifest row contains the audio path, target emotion, raw human votes on
the 0-2 scale, mapped votes on the 0-10 scale, and aggregate scores.

Build smaller deterministic manifests:

```bash
python scripts/build_emonet_splits.py \
  --data-root /store/store5/acp21rjf/data/emonet-voice-bench
```

Defaults:

- `smoke.jsonl`: 1 clip per official label, 40 clips total
- `quick.jsonl`: 10 clips per official label, 400 clips total

Use `--label-set all42` if you want to include the two extra HF labels
`Authenticity` and `Arousal`.

## Model Requests

For the one-by-one evaluation variant, build request JSONL from a prepared
manifest:

```bash
python scripts/build_emonet_requests.py \
  --data-root /store/store5/acp21rjf/data/emonet-voice-bench \
  --output runs/emonet_one_by_one_requests.jsonl
```

The prompt template is:

```text
Score this audio from 1-10 based on the presence of the following emotion: {emotion}
```

This emits one request per audio/emotion pair. Raw model outputs and parsed raw
scores should always be preserved. Calibration methods should read those raw
scores and write separate derived fields or files, never overwrite the original
model scores.

To use the human-style 0-2 presence rubric instead:

```bash
python scripts/build_emonet_requests.py \
  --data-root /store/store5/acp21rjf/data/emonet-voice-bench \
  --output runs/emonet_human_rubric_requests.jsonl \
  --mode one_by_one_human_rubric
```

To save compute by scoring only each row's target emotion, add
`--emotion-set target`:

```bash
python scripts/build_emonet_requests.py \
  --data-root /store/store5/acp21rjf/data/emonet-voice-bench \
  --output runs/emonet_target_human_rubric_requests.jsonl \
  --mode one_by_one_human_rubric \
  --emotion-set target
```

The rubric prompt asks for a single score:

```text
0 = the emotion is not present in the audio.
1 = the emotion is weakly or ambiguously present in the audio.
2 = the emotion is clearly or strongly present in the audio.
```

To ask the model directly for the paper-comparable 0-10 score:

```bash
python scripts/build_emonet_requests.py \
  --data-root /store/store5/acp21rjf/data/emonet-voice-bench \
  --output runs/emonet_target_paper_0_10_requests.jsonl \
  --mode one_by_one_paper_0_10 \
  --emotion-set target
```

The paper-scale prompt defines `0` as absent, `5` as mildly or ambiguously
present, and `10` as strongly present.

There is also a contrastive 0-2 rubric that asks for both the emotion and an
explicit opposite or contrast state:

```bash
python scripts/build_emonet_requests.py \
  --data-root /store/store5/acp21rjf/data/emonet-voice-bench \
  --output runs/emonet_target_contrastive_rubric_requests.jsonl \
  --mode one_by_one_contrastive_rubric \
  --emotion-set target
```

The required response format is:

```json
{"emotion_score": 0, "opposite_score": 0}
```

For the all-at-once evaluation variant, emit one request per audio. The prompt
requires the model to return a single JSON object with one numeric score for
every emotion:

```bash
python scripts/build_emonet_requests.py \
  --data-root /store/store5/acp21rjf/data/emonet-voice-bench \
  --output runs/emonet_all_at_once_requests.jsonl \
  --mode all_at_once
```

The required response format is:

```json
{"scores": {"Emotion Name": 0}}
```

All-at-once runs need a larger generation cap than one-by-one runs; use
`--max-new-tokens 768` or higher for the official 40-emotion prompt.

Summarize prediction metrics on the paper's 0-10 error scale:

```bash
python scripts/summarize_emonet_predictions.py \
  --predictions runs/moss_train100_target_human_rubric_raw.jsonl
```

This leaves Pearson and Spearman unchanged from the raw 0-2 scale, because
positive linear rescaling does not change correlation. It only rescales MAE and
RMSE from 0-2 units to 0-10 units.

## Kimi-Audio

The first open audio-language model target is
`moonshotai/Kimi-Audio-7B-Instruct`. The current server driver is CUDA 12.2, so
the setup script uses PyTorch 2.6 CUDA 11.8 wheels in a project-local venv
rather than touching any shared conda environment.

Create the env:

```bash
scripts/setup_kimi_audio_env.sh
```

Download the text-output subset of the Kimi checkpoint to store5:

```bash
. .venv-kimi/bin/activate
python scripts/download_kimi_audio.py
```

This intentionally skips the large audio detokenizer/vocoder files. For SER
scoring we call `KimiAudio(..., load_detokenizer=False)`, because we only need
text outputs. The full HF snapshot is about 42.6GB; the text-output subset is
still roughly 23GB plus any secondary tokenizer downloads.

Run a small request file:

```bash
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
  --requests runs/emonet_one_by_one_requests.jsonl \
  --output runs/kimi_predictions_raw.jsonl \
  --limit 10 \
  --max-new-tokens 16 \
  --device-map auto \
  --max-primary-gpu-memory 10GiB \
  --max-gpu-memory 18GiB \
  --max-cpu-memory 96GiB
```

## MOSS-Audio

MOSS-Audio is the model family from `OpenMOSS/MOSS-Audio`, released as 4B/8B
Instruct and Thinking variants. The default local target here is
`OpenMOSS-Team/MOSS-Audio-4B-Instruct` because it is the smallest direct
instruction-following checkpoint.

Create the separate MOSS env:

```bash
scripts/setup_moss_audio_env.sh
```

Download the checkpoint to store5:

```bash
. .venv-moss/bin/activate
python scripts/download_moss_audio.py
```

Run the target-only human 0-2 rubric requests:

```bash
TMPDIR=$PWD/scratch/tmp \
TEMP=$PWD/scratch/tmp \
TMP=$PWD/scratch/tmp \
XDG_CACHE_HOME=$PWD/.cache \
PIP_CACHE_DIR=$PWD/.cache/pip \
HF_HOME=/store/store5/acp21rjf/hf-cache \
HUGGINGFACE_HUB_CACHE=/store/store5/acp21rjf/hf-cache/hub \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_moss_emonet_requests.py \
  --model-path /store/store5/acp21rjf/models/MOSS-Audio-4B-Instruct \
  --requests runs/emonet_train100_target_human_rubric_requests.jsonl \
  --output runs/moss_train100_target_human_rubric_raw.jsonl \
  --limit 100 \
  --overwrite \
  --max-new-tokens 16
```

Keep the 4B MOSS runner on one visible GPU. `--device-map auto` can split the
audio and language paths onto different GPUs and fail inside the model's audio
embedding scatter.

## Label Scales

The released human labels are discrete:

- `0`: emotion not present
- `1`: emotion weakly present
- `2`: emotion strongly present

For comparison with model outputs, the paper maps them to 0-10:

```text
0 -> 0
1 -> 5
2 -> 10
```
