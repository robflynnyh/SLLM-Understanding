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
  --device-map auto \
  --max-primary-gpu-memory 10GiB \
  --max-gpu-memory 18GiB \
  --max-cpu-memory 96GiB
```

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
