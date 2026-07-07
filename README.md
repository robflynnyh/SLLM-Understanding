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

