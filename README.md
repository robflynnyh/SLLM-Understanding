# SLLM Understanding

Reusable code for preparing and evaluating speech-language model experiments
across audio-understanding and speech-judge benchmarks. The repo currently
supports EmoNet-Voice Bench, VoiceMOS 2022, SOMOS, in-context ASR, and a
TED-LIUM real-vs-synthetic speech benchmark, and is intended to grow to cover
additional conversational-agent judge benchmarks.

## Storage

Keep large datasets outside this repo. On this server the default EmoNet dataset
root is:

```bash
/store/store5/data/acp21rjf/data/emonet-voice-bench
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
python scripts/prepare_emonet.py all --data-root /store/store5/data/acp21rjf/data/emonet-voice-bench
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
  --data-root /store/store5/data/acp21rjf/data/emonet-voice-bench
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
  --data-root /store/store5/data/acp21rjf/data/emonet-voice-bench \
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
  --data-root /store/store5/data/acp21rjf/data/emonet-voice-bench \
  --output runs/emonet_human_rubric_requests.jsonl \
  --mode one_by_one_human_rubric
```

To save compute by scoring only each row's target emotion, add
`--emotion-set target`:

```bash
python scripts/build_emonet_requests.py \
  --data-root /store/store5/data/acp21rjf/data/emonet-voice-bench \
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
  --data-root /store/store5/data/acp21rjf/data/emonet-voice-bench \
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
  --data-root /store/store5/data/acp21rjf/data/emonet-voice-bench \
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
  --data-root /store/store5/data/acp21rjf/data/emonet-voice-bench \
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

## VoiceMOS 2022

VoiceMOS 2022 / BVCC is supported as a MOS-style naturalness and quality
benchmark. The default data root is:

```bash
/store/store5/data/acp21rjf/data/voicemos-2022
```

Download, extract, and prepare manifests:

```bash
python scripts/prepare_voicemos.py all
```

The public Zenodo archive does not redistribute Blizzard Challenge audio. Those
rows stay in the manifests with `audio_path: null`, and request generation skips
them by default. On this server, the public archive currently gives these
usable-audio counts:

- `main_train`: 2,254 / 4,974 rows
- `main_dev`: 387 / 1,066 rows
- `main_test`: 741 / 1,066 rows
- OOD splits: 0 usable WAVs until the bundled Blizzard download/preprocess
  scripts are run

Build a small main-dev smoke request file:

```bash
python scripts/build_voicemos_requests.py \
  --split main_dev \
  --output runs/voicemos2022_main_dev_smoke_requests.jsonl \
  --limit 10
```

Build all available main-dev requests:

```bash
python scripts/build_voicemos_requests.py \
  --split main_dev \
  --output runs/voicemos2022_main_dev_requests.jsonl
```

The request prompt asks for a single 1-5 MOS score. Existing MOSS/Kimi runners
can consume these request files because they use the same `audio_path`, `prompt`,
and `raw_score_scale` fields as the EmoNet requests.

Summarize VoiceMOS predictions:

```bash
python scripts/summarize_voicemos_predictions.py \
  --predictions runs/moss_voicemos2022_main_dev_raw.jsonl \
  --manifest /store/store5/data/acp21rjf/data/voicemos-2022/manifests/main_dev.jsonl
```

The summarizer reports utterance-level and system-level Pearson, Spearman, MAE,
MSE, and RMSE on the 1-5 MOS scale.

The completed MOSS 4B 1-10 quality-prompt results for the available main-track
audio are recorded in `result/voicemos2022.md`. Current EmoNet-Voice Bench
results are recorded in `result/emonet.md`.

## SOMOS

SOMOS is supported as a MOS-style naturalness benchmark for neural TTS samples.
The default data root is:

```bash
/store/store5/data/acp21rjf/data/somos
```

Download, extract, and prepare manifests:

```bash
python scripts/prepare_somos.py all
```

The Zenodo release is a single outer ZIP containing metadata plus a nested
`audios.zip`. For a metadata-only setup, run:

```bash
python scripts/prepare_somos.py all --metadata-only
```

The preparation step writes both `clean_*` and `full_*` manifests. The default
request split is `clean_test`, which uses the filtered listener-quality scores.

Build a small clean-test smoke request file with the same 1-10 quality prompt
used for VoiceMOS:

```bash
python scripts/build_somos_requests.py \
  --split clean_test \
  --output runs/somos_clean_test_smoke_requests.jsonl \
  --limit 10
```

Build all clean-test requests:

```bash
python scripts/build_somos_requests.py \
  --split clean_test \
  --output runs/somos_clean_test_quality_1_10_requests.jsonl
```

Summarize SOMOS predictions:

```bash
python scripts/summarize_somos_predictions.py \
  --predictions runs/moss4b_somos_clean_test_quality_1_10_raw.jsonl \
  --manifest /store/store5/data/acp21rjf/data/somos/manifests/clean_test.jsonl
```

The planned SOMOS setup and pending results are recorded in `result/somos.md`.

## In-Context ASR

The in-context-asr probe is supported as a small transcription benchmark using
the data from `robflynnyh/in-context-asr`. Clone the data repo alongside this
repo:

```bash
git clone https://github.com/robflynnyh/in-context-asr ../in-context-asr
```

Build MOSS transcription requests:

```bash
python scripts/build_in_context_asr_requests.py \
  --data-root ../in-context-asr/data \
  --output runs/in_context_asr_moss4b_transcription_requests.jsonl
```

The same prompt is used for every request:

```text
Transcribe the speech in this audio. Return only the transcript.
```

An alternative prompt asks the model to preserve noisy and repeated segments:

```bash
python scripts/build_in_context_asr_requests.py \
  --data-root ../in-context-asr/data \
  --prompt-mode transcription_all_segments \
  --output runs/in_context_asr_moss4b_transcription_all_segments_requests.jsonl
```

Prompt:

```text
Transcribe all speech in this audio from start to finish, including noisy, unclear, interrupted, repeated, or corrected segments. Return only the transcript.
```

For a text-only one-shot prompt, build:

```bash
python scripts/build_in_context_asr_requests.py \
  --data-root ../in-context-asr/data \
  --prompt-mode transcription_text_fewshot \
  --output runs/in_context_asr_moss4b_transcription_text_fewshot_requests.jsonl
```

For an audio-paired one-shot prompt, first create the example WAV:

```bash
mkdir -p scratch/in_context_asr
espeak-ng -w scratch/in_context_asr/fewshot_example.wav \
  "The meeting starts at nine in the morning. Sorry, the line cut out. Could you repeat that? Sure thing. The meeting starts at nine in the morning."
```

Then build requests with the example audio paired to the example transcript:

```bash
python scripts/build_in_context_asr_requests.py \
  --data-root ../in-context-asr/data \
  --prompt-mode transcription_fewshot \
  --fewshot-audio-path scratch/in_context_asr/fewshot_example.wav \
  --output runs/in_context_asr_moss4b_transcription_fewshot_requests.jsonl
```

Run MOSS 4B Instruct:

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
.venv-moss/bin/python -u scripts/run_moss_in_context_asr_requests.py \
  --model-path /store/store5/data/acp21rjf/models/MOSS-Audio-4B-Instruct \
  --model-name OpenMOSS-Team/MOSS-Audio-4B-Instruct \
  --requests runs/in_context_asr_moss4b_transcription_requests.jsonl \
  --output runs/moss4b_in_context_asr_transcription_raw.jsonl \
  --overwrite \
  --max-new-tokens 256 \
  --device-map cuda:0
```

Summarize predictions:

```bash
.venv-moss/bin/python scripts/summarize_in_context_asr_predictions.py \
  --predictions runs/moss4b_in_context_asr_transcription_raw.jsonl
```

Completed MOSS 4B results are recorded in `result/in-context-asr.md`.

## TED-LIUM Real Vs Synthetic

This repo supports preparing a controlled real-vs-synthetic benchmark from the
TED-LIUM 3 legacy dev/test splits. For each target utterance, the real sample is
the original TED-LIUM segment and the synthetic sample is generated by
MOSS-TTS-Realtime from the same transcript. The TTS speaker prompt is a
different utterance from the same TED speaker.

Source TED-LIUM root on this server:

```bash
/store/store4/data/TEDLIUM_release-3
```

Generated dataset root:

```bash
/store/store5/data/acp21rjf/data/tedlium-moss-real-vs-synthetic
```

Build a tiny dev/test smoke setup:

```bash
python scripts/prepare_tedlium_real_vs_synthetic.py \
  --splits dev test \
  --max-items-per-split 2 \
  --overwrite-audio
```

Build the full dev/test manifests and extracted real/prompt clips:

```bash
python scripts/prepare_tedlium_real_vs_synthetic.py \
  --splits dev test \
  --overwrite-audio
```

The prep script writes:

- `manifests/dev.jsonl` and `manifests/test.jsonl`: one row per target
  utterance, with real/prompt/synthetic paths.
- `manifests/moss_texts_dev.jsonl` and `manifests/moss_texts_test.jsonl`:
  MOSS-TTS batch-rollout JSONL using per-row `prompt_wav`.
- `manifests/pairs_dev.jsonl` and `manifests/pairs_test.jsonl`: binary
  real/synthetic classifier rows.

Generate synthetic audio with the sibling MOSS realtime demo:

```bash
RUN_WITH_GPU_SCHEDULER=0 CUDA_VISIBLE_DEVICES=0 \
/exp/exp4/acp21rjf/moss-realtime-demo/scripts/run_batch_rollout.sh \
  --benchmark \
  --batch-size 32 \
  --max-audio-steps 768 \
  --texts-file /store/store5/data/acp21rjf/data/tedlium-moss-real-vs-synthetic/manifests/moss_texts_dev.jsonl \
  --out-dir /store/store5/data/acp21rjf/data/tedlium-moss-real-vs-synthetic/synthetic/moss-tts-realtime/dev \
  --seed 1234
```

Repeat with `moss_texts_test.jsonl` and the `test` output directory for the
test split. Current smoke-generation status is recorded in
`result/tedlium-real-vs-synthetic.md`.

Build blind quality-judge requests for both real and synthetic audio:

```bash
python scripts/build_tedlium_real_vs_synthetic_requests.py \
  --split dev \
  --mode quality_1_10 \
  --output runs/tedlium_rvs_dev_quality_1_10_requests.jsonl
```

The same builder supports a transcript-aware quality prompt:

```bash
python scripts/build_tedlium_real_vs_synthetic_requests.py \
  --split dev \
  --mode quality_1_10_with_transcript \
  --output runs/tedlium_rvs_dev_quality_1_10_with_transcript_requests.jsonl
```

To use the exact VoiceMOS 1-10 no-rubric prompt, use:

```bash
python scripts/build_tedlium_real_vs_synthetic_requests.py \
  --split dev \
  --mode quality_1_10_voicemos_exact \
  --output runs/tedlium_rvs_dev_quality_1_10_voicemos_exact_requests.jsonl
```

To ask directly for a real-vs-synthetic score, use:

```bash
python scripts/build_tedlium_real_vs_synthetic_requests.py \
  --split dev \
  --mode real_vs_synthetic_0_10 \
  --output runs/tedlium_rvs_dev_real_vs_synthetic_0_10_requests.jsonl
```

To include the expected transcript in the real-vs-synthetic prompt, use:

```bash
python scripts/build_tedlium_real_vs_synthetic_requests.py \
  --split dev \
  --mode real_vs_synthetic_0_10_with_transcript \
  --output runs/tedlium_rvs_dev_real_vs_synthetic_0_10_with_transcript_requests.jsonl
```

Each request includes metadata labels for analysis, but the model prompt never
tells the model whether the audio is real or synthetic.

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
HF_HOME=/store/store5/data/acp21rjf/hf-cache \
HUGGINGFACE_HUB_CACHE=/store/store5/data/acp21rjf/hf-cache/hub \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/run_kimi_emonet_requests.py \
  --model-path /store/store5/data/acp21rjf/models/Kimi-Audio-7B-Instruct \
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
HF_HOME=/store/store5/data/acp21rjf/hf-cache \
HUGGINGFACE_HUB_CACHE=/store/store5/data/acp21rjf/hf-cache/hub \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_moss_emonet_requests.py \
  --model-path /store/store5/data/acp21rjf/models/MOSS-Audio-4B-Instruct \
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
