# TED-LIUM Real Vs Synthetic Setup

This note records the current TED-LIUM real-vs-synthetic benchmark setup.

## Dataset

- Source: `/store/store4/data/TEDLIUM_release-3`
- Source splits: `legacy/dev`, `legacy/test`
- Generated root: `/store/store5/data/acp21rjf/data/tedlium-moss-real-vs-synthetic`
- Real audio: extracted TED-LIUM target utterance clips
- Synthetic audio: MOSS-TTS-Realtime generated from the target transcript
- Speaker conditioning: a different utterance from the same TED speaker

## Prepared Data

| Split | Target rows | Pair rows | Missing pair audio | Synthetic generation |
| --- | ---: | ---: | ---: | --- |
| dev | 503 | 1006 | 0 | complete |
| test | 1153 | 2306 | 0 | complete |

Rows without a different same-speaker prompt utterance are skipped and recorded
in `manifests/skipped_{split}.jsonl`: 4 dev rows and 2 test rows.

Generated data size at completion: 1.4G.

## Generation Settings

Full generation used MOSS-TTS-Realtime with per-row speaker-conditioning WAVs,
`max-audio-steps=768`, `seed=1234`, and the MOSS-TTS-Realtime recommended
sampled decoding settings:

```text
sample=True
temperature=0.8
top_p=0.6
top_k=30
repetition_penalty=1.1
repetition_window=50
```

`max-audio-steps=768` was chosen as the approximate one-minute cap after the
initial smoke showed `128` steps produced about 10.3s of audio. The smoke
batches stopped by EOS rather than by the cap: dev produced 26.72s total audio
and test produced 28.72s total audio for two files each.

The successful full run was completed through resumed shards:

- dev: `batch-size=32`, `codec-decode-batch-size=4`
- test: initial/resume runs used `batch-size=32`, `codec-decode-batch-size=4`;
  the final 152-row shard used `batch-size=16`, `codec-decode-batch-size=1`
  after a codec decode OOM on a long batch.

The final test shard was launched through the store5 cooperative GPU scheduler
with `GPU_POOL=all` via `moss-realtime-demo/scripts/run_batch_rollout.sh`.

## Smoke Hashes

| Split | File | SHA256 |
| --- | --- | --- |
| dev | `0000_dev__AlGore_2009__Al_Gore__13.040_23.460.wav` | `952bfcc11289b7dbd62cbe4e0acd203bafb7b225e32e6486ae03b8d55fba41de` |
| dev | `0001_dev__AlGore_2009__Al_Gore__23.460_40.050.wav` | `3e7134578fae7b3a6f300ac88770bf21943434e584897d88812a4557ae87c770` |
| test | `0000_test__AimeeMullins_2009P__AimeeMullins__17.820_28.810.wav` | `64621a48685816fc4165fc8913a8587d97bfecf015e733cfd3752842964d0f2c` |
| test | `0001_test__AimeeMullins_2009P__AimeeMullins__28.810_40.266.wav` | `5af76f6bee359c2e988944453814f5db264d11a8203458381e1acd80e9c186b8` |

## Artifacts

- `manifests/dev.jsonl`
- `manifests/test.jsonl`
- `manifests/moss_texts_dev.jsonl`
- `manifests/moss_texts_test.jsonl`
- `manifests/pairs_dev.jsonl`
- `manifests/pairs_test.jsonl`
- `manifests/skipped_dev.jsonl`
- `manifests/skipped_test.jsonl`
- `manifests/resume/*.jsonl`
- `synthetic/moss-tts-realtime/dev/manifest.json`
- `synthetic/moss-tts-realtime/test/manifest.json`
