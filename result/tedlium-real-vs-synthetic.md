# TED-LIUM Real Vs Synthetic Setup

This note records the current TED-LIUM real-vs-synthetic benchmark setup.

## Dataset

- Source: `/store/store4/data/TEDLIUM_release-3`
- Source splits: `legacy/dev`, `legacy/test`
- Generated root: `/store/store5/acp21rjf/data/tedlium-moss-real-vs-synthetic`
- Real audio: extracted TED-LIUM target utterance clips
- Synthetic audio: MOSS-TTS-Realtime generated from the target transcript
- Speaker conditioning: a different utterance from the same TED speaker

## Current Smoke

A two-row-per-split smoke set has been prepared and generated.

| Split | Target rows | Pair rows | Missing pair audio | Synthetic generation |
| --- | ---: | ---: | ---: | --- |
| dev | 2 | 4 | 0 | complete |
| test | 2 | 4 | 0 | complete |

Smoke generation commands used `batch-size=2`, `max-audio-steps=128`, and
`seed=1234`.

## Artifacts

- `manifests/dev.jsonl`
- `manifests/test.jsonl`
- `manifests/moss_texts_dev.jsonl`
- `manifests/moss_texts_test.jsonl`
- `manifests/pairs_dev.jsonl`
- `manifests/pairs_test.jsonl`
- `synthetic/moss-tts-realtime/dev/manifest.json`
- `synthetic/moss-tts-realtime/test/manifest.json`
