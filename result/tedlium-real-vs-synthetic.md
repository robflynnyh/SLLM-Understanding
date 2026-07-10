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

The prepared dataset is backed up in plain Git, without Git LFS:

- Repository: `https://github.com/robflynnyh/tedlium-real-vs-synthetic`
- Commit: `92da3cb8a3404baa3a1826d8e0d36342baa72ee4`

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

## Judge Evaluation Results

All judge runs used deterministic decoding. Each request scores one audio clip;
both real and synthetic rows are included in every split. Pair delta is
`synthetic_score - real_score`.

For quality prompts, negative pair delta means the model rates synthetic speech
as lower quality than real speech. For real-vs-synthetic detector prompts,
positive pair delta means the model rates synthetic speech as more synthetic
because the scale is `0=real`, `10=synthetic`.

### MOSS-Audio 4B Instruct

Model: `OpenMOSS-Team/MOSS-Audio-4B-Instruct`

| Split | Prompt mode | Parsed | Real mean | Synthetic mean | Pair delta mean | Synthetic lower | Equal | Synthetic higher |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | `quality_1_10` | 1006 / 1006 | 6.738 | 7.143 | 0.406 | 49 | 264 | 190 |
| test | `quality_1_10` | 2306 / 2306 | 6.484 | 6.774 | 0.289 | 157 | 586 | 410 |
| dev | `quality_1_10_with_transcript` | 1006 / 1006 | 5.506 | 5.562 | 0.056 | 4 | 483 | 16 |
| test | `quality_1_10_with_transcript` | 2306 / 2306 | 5.551 | 5.559 | 0.008 | 54 | 1064 | 35 |
| dev | `real_vs_synthetic_0_10` | 1006 / 1006 | 4.970 | 4.980 | 0.010 | 7 | 488 | 8 |
| test | `real_vs_synthetic_0_10` | 2306 / 2306 | 4.983 | 5.221 | 0.239 | 13 | 1070 | 70 |
| dev | `real_vs_synthetic_0_10_with_transcript` | 1006 / 1006 | 4.980 | 4.938 | -0.042 | 6 | 496 | 1 |
| test | `real_vs_synthetic_0_10_with_transcript` | 2306 / 2306 | 4.965 | 4.952 | -0.013 | 7 | 1139 | 7 |

The MOSS direct detector prompts are mostly tied around 5 and do not provide a
useful real/synthetic separation on this dataset.

### Kimi-Audio 7B Instruct

Model: `moonshotai/Kimi-Audio-7B-Instruct`

| Split | Prompt mode | Parsed | Real mean | Synthetic mean | Pair delta mean | Synthetic lower | Equal | Synthetic higher |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | `quality_1_10` | 992 / 1006 | 8.786 | 8.102 | -0.679 | 280 | 198 | 11 |
| test | `quality_1_10` | 2259 / 2306 | 8.817 | 7.907 | -0.915 | 658 | 399 | 49 |
| dev | `quality_1_10_voicemos_exact` | 1002 / 1006 | 8.715 | 7.970 | -0.745 | 303 | 186 | 10 |
| test | `quality_1_10_voicemos_exact` | 2263 / 2306 | 8.746 | 7.717 | -1.042 | 716 | 366 | 28 |
| dev | `quality_1_10_with_transcript` | 1005 / 1006 | 8.597 | 7.685 | -0.911 | 322 | 168 | 12 |
| test | `quality_1_10_with_transcript` | 2279 / 2306 | 8.621 | 7.525 | -1.097 | 698 | 385 | 43 |
| dev | `real_vs_synthetic_0_10` | 784 / 1006 | 4.314 | 6.055 | 1.586 | 55 | 65 | 182 |
| test | `real_vs_synthetic_0_10` | 1819 / 2306 | 4.768 | 5.741 | 0.802 | 210 | 177 | 346 |
| dev | `real_vs_synthetic_0_10_with_transcript` | 999 / 1006 | 4.636 | 5.930 | 1.305 | 84 | 135 | 277 |
| test | `real_vs_synthetic_0_10_with_transcript` | 2266 / 2306 | 4.822 | 5.568 | 0.750 | 259 | 341 | 515 |

The strongest Kimi quality prompt is `quality_1_10_with_transcript` by pair
delta on both dev and test. The direct detector prompts also separate the
classes, but `real_vs_synthetic_0_10` has many parse failures; adding the
transcript greatly improves parse rate while reducing the separation magnitude.

### Pairwise A/B Real-Vs-Synthetic

Each pairwise request passes both recordings for the same utterance and asks the
model to choose which recording is synthetic. Each utterance is evaluated in both
directions:

- `real_a_synthetic_b`: audio A is real, audio B is synthetic.
- `synthetic_a_real_b`: audio A is synthetic, audio B is real.

`Pair score` is the mean correctness across the two directions for each
utterance, then averaged over complete pairs. A value of `1.0` means both
directions are correct, `0.5` means exactly one direction is correct, and `0.0`
means both directions are wrong.

| Model | Split | Prompt mode | Parsed | Direction accuracy | Pair score | Both correct | One correct | Both wrong | Choice A | Choice B |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MOSS-Audio 4B Instruct | dev | `pairwise_real_vs_synthetic` | 1006 / 1006 | 0.494 | 0.494 | 34 | 429 | 40 | 0.907 | 0.093 |
| MOSS-Audio 4B Instruct | test | `pairwise_real_vs_synthetic` | 2306 / 2306 | 0.537 | 0.537 | 164 | 910 | 79 | 0.882 | 0.118 |
| MOSS-Audio 4B Instruct | dev | `pairwise_real_vs_synthetic_with_transcript` | 1006 / 1006 | 0.496 | 0.496 | 1 | 497 | 5 | 0.994 | 0.006 |
| MOSS-Audio 4B Instruct | test | `pairwise_real_vs_synthetic_with_transcript` | 2306 / 2306 | 0.501 | 0.501 | 9 | 1137 | 7 | 0.993 | 0.007 |
| Kimi-Audio 7B Instruct | dev | `pairwise_real_vs_synthetic` | 1006 / 1006 | 0.800 | 0.800 | 311 | 183 | 9 | 0.402 | 0.598 |
| Kimi-Audio 7B Instruct | test | `pairwise_real_vs_synthetic` | 2306 / 2306 | 0.785 | 0.785 | 678 | 455 | 20 | 0.461 | 0.539 |
| Kimi-Audio 7B Instruct | dev | `pairwise_real_vs_synthetic_with_transcript` | 1006 / 1006 | 0.786 | 0.786 | 301 | 189 | 13 | 0.374 | 0.626 |
| Kimi-Audio 7B Instruct | test | `pairwise_real_vs_synthetic_with_transcript` | 2306 / 2306 | 0.764 | 0.764 | 635 | 491 | 27 | 0.394 | 0.606 |

MOSS is approximately chance after both-direction averaging and has a strong
choice-A bias, especially when the transcript is included. Kimi is substantially
better in this pairwise setup; unlike the single-audio quality-score setup, the
transcript-conditioned pairwise prompt is slightly worse than the no-transcript
pairwise prompt on both dev and test.

#### Question-Balanced Pairwise Early Stop

The `pairwise_real_vs_synthetic_question_balanced` mode crosses both audio
orders with both question targets, giving four requests per utterance pair:
ask which recording is synthetic, and ask which recording is real. For the
combined score, `ask-real` responses are converted through correctness rather
than by averaging raw `A`/`B` choices.

The Kimi run was stopped early on 2026-07-10 after dev completed and partial
test showed the same pattern. Reason: it was clearly worse than the best
non-transcript pairwise setup above. The completed dev score was `0.659`,
compared with `0.800` for `pairwise_real_vs_synthetic`; the degradation came
from the real-question half.

| Model | Split | Prompt mode | Parsed | Pair score | Ask synthetic accuracy | Ask real accuracy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Kimi-Audio 7B Instruct | dev | `pairwise_real_vs_synthetic_question_balanced` | 2012 / 2012 | 0.659 | 0.800 | 0.518 | complete dev |
| Kimi-Audio 7B Instruct | test | `pairwise_real_vs_synthetic_question_balanced` | 765 / 4612 | 0.671 | 0.872 | 0.469 | partial test at stop |

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

Ignored run artifacts used for the judge results:

- `runs/moss4b_tedlium_rvs_dev_quality_1_10_raw.jsonl`
- `runs/moss4b_tedlium_rvs_test_quality_1_10_raw.jsonl`
- `runs/moss4b_tedlium_rvs_dev_quality_1_10_with_transcript_raw.jsonl`
- `runs/moss4b_tedlium_rvs_test_quality_1_10_with_transcript_raw.jsonl`
- `runs/moss4b_tedlium_rvs_dev_real_vs_synthetic_0_10_raw.jsonl`
- `runs/moss4b_tedlium_rvs_test_real_vs_synthetic_0_10_raw.jsonl`
- `runs/moss4b_tedlium_rvs_dev_real_vs_synthetic_0_10_with_transcript_raw.jsonl`
- `runs/moss4b_tedlium_rvs_test_real_vs_synthetic_0_10_with_transcript_raw.jsonl`
- `runs/kimi_tedlium_rvs_dev_all_prompts_raw.jsonl`
- `runs/kimi_tedlium_rvs_test_all_prompts_raw.jsonl`
- `runs/tedlium_rvs_pairwise_all_requests.jsonl`
- `runs/moss4b_tedlium_rvs_pairwise_all_raw.jsonl`
- `runs/kimi_tedlium_rvs_pairwise_all_raw.jsonl`
- `runs/tedlium_rvs_pairwise_question_balanced_all_requests.jsonl`
- `runs/kimi_tedlium_rvs_pairwise_question_balanced_all_raw.jsonl`
