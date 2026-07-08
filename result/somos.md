# SOMOS Results

SOMOS is the next MOS-style naturalness benchmark planned for this repo. It has
not been evaluated yet.

## Planned Setup

- Dataset: SOMOS, Zenodo record `7378801`
- Default data root: `/store/store5/acp21rjf/data/somos`
- Primary split: `clean_test`
- Prompt mode: `quality_1_10_no_rubric`
- Raw model scale: 1-10
- Metric scale: 1-5 MOS, using the linear mapping
  `1 + ((score_1_10 - 1) / 9) * 4`
- Metrics: Pearson, Spearman, MSE, and RMSE at utterance and system level

## Results

Pending.
