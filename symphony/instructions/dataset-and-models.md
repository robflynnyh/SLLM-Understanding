# Dataset And Models

Canonical EmoNet-Voice Bench dataset root:

```text
/store/store5/data/acp21rjf/data/emonet-voice-bench
```

Canonical manifest:

```text
/store/store5/data/acp21rjf/data/emonet-voice-bench/manifests/train.jsonl
```

Treat this dataset root as read-only. Do not edit, move, delete, or overwrite
raw audio, raw parquet shards, extracted audio, or existing manifests under the
store path unless a human explicitly asks for that exact mutation.

Dataset identity:

```text
t1a5anu-anon/emonet-voice-bench
```

The released human labels are raw `0/1/2` scores:

```text
0 = emotion not present
1 = emotion weakly present
2 = emotion strongly present
```

For paper-style reporting, map human labels to the `0-10` scale:

```text
0 -> 0
1 -> 5
2 -> 10
```

Default Symphony evaluation setup unless the issue says otherwise:

```text
model: OpenMOSS-Team/MOSS-Audio-4B-Instruct
mode: one_by_one_paper_0_10
emotion_set: target
scoring: direct 0-10 paper scale
requests: one request per audio, scoring only the row's target_label
```

Build target-only paper-scale requests:

```bash
python scripts/build_emonet_requests.py \
  --data-root /store/store5/data/acp21rjf/data/emonet-voice-bench \
  --manifest /store/store5/data/acp21rjf/data/emonet-voice-bench/manifests/train.jsonl \
  --output runs/emonet_target_paper_0_10_requests.jsonl \
  --mode one_by_one_paper_0_10 \
  --emotion-set target
```

Run MOSS on one visible GPU. Do not use `--device-map auto` for the 4B MOSS
runner; it can split audio and language tensors onto different GPUs and fail in
the audio embedding scatter.

```bash
TMPDIR=$PWD/scratch/tmp \
TEMP=$PWD/scratch/tmp \
TMP=$PWD/scratch/tmp \
XDG_CACHE_HOME=$PWD/.cache \
PIP_CACHE_DIR=$PWD/.cache/pip \
HF_HOME=/store/store5/data/acp21rjf/hf-cache \
HUGGINGFACE_HUB_CACHE=/store/store5/data/acp21rjf/hf-cache/hub \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_moss_emonet_requests.py \
  --model-path /store/store5/data/acp21rjf/models/MOSS-Audio-4B-Instruct \
  --requests runs/emonet_target_paper_0_10_requests.jsonl \
  --output runs/moss_target_paper_0_10_raw.jsonl \
  --overwrite \
  --max-new-tokens 16
```

Summarize metrics on the paper's `0-10` error scale:

```bash
python scripts/summarize_emonet_predictions.py \
  --predictions runs/moss_target_paper_0_10_raw.jsonl
```

Known local model roots:

```text
/store/store5/data/acp21rjf/models/MOSS-Audio-4B-Instruct
/store/store5/data/acp21rjf/models/Kimi-Audio-7B-Instruct
```

Known local cache roots:

```text
/store/store5/data/acp21rjf/hf-cache
```

MOSS environment:

```bash
scripts/setup_moss_audio_env.sh
. .venv-moss/bin/activate
python scripts/download_moss_audio.py
```

Kimi environment:

```bash
scripts/setup_kimi_audio_env.sh
. .venv-kimi/bin/activate
python scripts/download_kimi_audio.py
```

For comparisons already run on the first 100 train rows:

- MOSS direct `0-10`, target-only: `runs/moss_train100_target_paper_0_10_raw.jsonl`
- MOSS raw `0-2`, target-only: `runs/moss_train100_target_human_rubric_raw.jsonl`
- Kimi raw `0-2`, target-only: `runs/kimi_train100_target_human_rubric_raw.jsonl`

Treat `runs/` as local generated evidence; it is ignored and should not be
committed.
