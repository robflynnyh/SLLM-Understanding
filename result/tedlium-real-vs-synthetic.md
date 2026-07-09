# TED-LIUM Real Vs Synthetic Setup

This note records the current TED-LIUM real-vs-synthetic benchmark setup.

## Dataset

- Source: `/store/store4/data/TEDLIUM_release-3`
- Source splits: `legacy/dev`, `legacy/test`
- Generated root: `/store/store5/data/acp21rjf/data/tedlium-moss-real-vs-synthetic`
- Real audio: extracted TED-LIUM target utterance clips
- Synthetic audio: MOSS-TTS-Realtime generated from the target transcript
- Speaker conditioning: a different utterance from the same TED speaker

## Current Smoke

A two-row-per-split smoke set has been prepared and generated.

| Split | Target rows | Pair rows | Missing pair audio | Synthetic generation |
| --- | ---: | ---: | ---: | --- |
| dev | 2 | 4 | 0 | complete |
| test | 2 | 4 | 0 | complete |

Smoke generation commands used `batch-size=2`, `max-audio-steps=128`,
`seed=1234`, and the MOSS-TTS-Realtime recommended sampled decoding settings:

```text
sample=True
temperature=0.8
top_p=0.6
top_k=30
repetition_penalty=1.1
repetition_window=50
```

Smoke output hashes:

| Split | File | SHA256 |
| --- | --- | --- |
| dev | `0000_dev__AlGore_2009__Al_Gore__13.040_23.460.wav` | `911e19b6cedf79086ccaa82b5f0692cf464315d2b97e62fb5b5fb748dc716f4d` |
| dev | `0001_dev__AlGore_2009__Al_Gore__23.460_40.050.wav` | `a5b0d0c3ded1ddfc65213ee02ae372af658c9443a54e597f5cf2b1d6b8b36859` |
| test | `0000_test__AimeeMullins_2009P__AimeeMullins__17.820_28.810.wav` | `7057b0037f9845d032a26f104643dd56d76ad075f6af235a393920b7204d4eb0` |
| test | `0001_test__AimeeMullins_2009P__AimeeMullins__28.810_40.266.wav` | `a5d16a89181ac5a0fbf4cc7a76c21a990b1d7480492dce4fa7bf66098c634041` |

## Artifacts

- `manifests/dev.jsonl`
- `manifests/test.jsonl`
- `manifests/moss_texts_dev.jsonl`
- `manifests/moss_texts_test.jsonl`
- `manifests/pairs_dev.jsonl`
- `manifests/pairs_test.jsonl`
- `synthetic/moss-tts-realtime/dev/manifest.json`
- `synthetic/moss-tts-realtime/test/manifest.json`
