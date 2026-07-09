# VoiceMOS 2022 MOSS 4B Results

This note records the completed VoiceMOS 2022 main-track evaluation for
`OpenMOSS-Team/MOSS-Audio-4B-Instruct` using the direct 1-10 quality prompt.

The run used the public-archive audio currently available in
`/store/store5/data/acp21rjf/data/voicemos-2022`. Blizzard Challenge rows whose audio
is not redistributed by the public archive were skipped at request-generation
time, so these are audio-available main split results rather than full
challenge-set results.

## Setup

- Model: `OpenMOSS-Team/MOSS-Audio-4B-Instruct`
- Local model path: `/store/store5/data/acp21rjf/models/MOSS-Audio-4B-Instruct`
- Dataset: VoiceMOS 2022 / BVCC, main track
- Prompt mode: `quality_1_10_no_rubric`
- Raw model scale: 1-10
- Metric scale: 1-5 MOS, using the linear mapping
  `1 + ((score_1_10 - 1) / 9) * 4`
- Metrics: Pearson, Spearman, MSE, and RMSE at utterance and system level

Prompt:

```text
You are a trained assessor of speech quality. We are assessing the quality and naturalness of synthesized speech. Rate the quality and naturalness of this speech on a scale of 1-10. Do not give a rubric. Return only a single number from 1 to 10. Decimals are allowed.
```

## Split Coverage

| Split | Manifest | Parsed predictions | Utterances | Systems |
| --- | --- | ---: | ---: | ---: |
| main_train | `/store/store5/data/acp21rjf/data/voicemos-2022/manifests/main_train.jsonl` | 2254 / 2254 | 2254 | 83 |
| main_dev | `/store/store5/data/acp21rjf/data/voicemos-2022/manifests/main_dev.jsonl` | 387 / 387 | 387 | 86 |
| main_test | `/store/store5/data/acp21rjf/data/voicemos-2022/manifests/main_test.jsonl` | 741 / 741 | 741 | 89 |

## Results

| Split | Level | Pearson | Spearman | MSE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| main_train | utterance | 0.557 | 0.521 | 0.845 | 0.919 |
| main_train | system | 0.819 | 0.765 | 0.576 | 0.759 |
| main_dev | utterance | 0.482 | 0.429 | 1.212 | 1.101 |
| main_dev | system | 0.658 | 0.630 | 1.000 | 1.000 |
| main_test | utterance | 0.595 | 0.552 | 0.798 | 0.893 |
| main_test | system | 0.755 | 0.681 | 0.579 | 0.761 |

## Artifacts

The raw run outputs are intentionally left under gitignored `runs/` paths:

- `runs/moss4b_voicemos2022_main_train_quality_1_10_raw.jsonl`
- `runs/moss4b_voicemos2022_main_train_quality_1_10_summary.txt`
- `runs/moss4b_voicemos2022_main_dev_quality_1_10_raw.jsonl`
- `runs/moss4b_voicemos2022_main_dev_quality_1_10_summary.txt`
- `runs/moss4b_voicemos2022_main_test_quality_1_10_raw.jsonl`
- `runs/moss4b_voicemos2022_main_test_quality_1_10_summary.txt`
