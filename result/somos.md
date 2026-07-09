# SOMOS Results

This note records the completed SOMOS clean-test evaluation for
`OpenMOSS-Team/MOSS-Audio-4B-Instruct` using the direct 1-10 quality prompt.

## Setup

- Dataset: SOMOS, Zenodo record `7378801`
- Default data root: `/store/store5/data/acp21rjf/data/somos`
- Primary split: `clean_test`
- Model: `OpenMOSS-Team/MOSS-Audio-4B-Instruct`
- Local model path: `/store/store5/data/acp21rjf/models/MOSS-Audio-4B-Instruct`
- Prompt mode: `quality_1_10_no_rubric`
- Raw model scale: 1-10
- Metric scale: 1-5 MOS, using the linear mapping
  `1 + ((score_1_10 - 1) / 9) * 4`
- Metrics: Pearson, Spearman, MAE, MSE, and RMSE at utterance and system level

Prompt:

```text
You are a trained assessor of speech quality. We are assessing the quality and naturalness of synthesized speech. Rate the quality and naturalness of this speech on a scale of 1-10. Do not give a rubric. Return only a single number from 1 to 10. Decimals are allowed.
```

## Split Coverage

| Split | Manifest | Parsed predictions | Utterances | Systems |
| --- | --- | ---: | ---: | ---: |
| clean_test | `/store/store5/data/acp21rjf/data/somos/manifests/clean_test.jsonl` | 3000 / 3000 | 3000 | 201 |

## Results

| Split | Level | Pearson | Spearman | MAE | MSE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| clean_test | utterance | 0.235 | 0.231 | 0.499 | 0.398 | 0.631 |
| clean_test | system | 0.519 | 0.473 | 0.290 | 0.133 | 0.364 |

Raw 1-10 score distribution: `4.5=5`, `5=241`, `5.5=540`, `6=164`,
`6.5=321`, `7=1476`, `7.5=230`, `8.5=23`.

## Artifacts

The raw run outputs are intentionally left under gitignored `runs/` paths:

- `runs/somos_clean_test_quality_1_10_requests.jsonl`
- `runs/moss4b_somos_clean_test_quality_1_10_raw.jsonl`
- `runs/moss4b_somos_clean_test_quality_1_10_summary.txt`
