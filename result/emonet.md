# EmoNet-Voice Bench Results

This note records the current EmoNet-Voice Bench target-emotion evaluation
results.

Metrics are reported against the human mean on the paper 0-10 scale. For runs
prompted on the human 0-2 rubric, model scores are multiplied by 5 before MAE,
MSE, and RMSE are computed. Pearson and Spearman are unaffected by that positive
linear rescaling.

## Setup

- Dataset: EmoNet-Voice Bench
- Manifest: `/store/store5/data/acp21rjf/data/emonet-voice-bench/manifests/train.jsonl`
- Task: score only each row's target emotion
- Primary metrics: accuracy versus majority human score, Pearson, Spearman, MAE,
  MSE, and RMSE

## Main Results

The primary run is the full 12,600-row train-manifest MOSS 4B target-only
evaluation from ROB-336.

| Model | Prompt mode | Parsed | Accuracy vs majority | Accuracy vs rounded mean | Pearson | Spearman | MAE | MSE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MOSS 4B Instruct | target paper 0-10 | 12600 / 12600 | 0.443 | 0.426 | 0.404 | 0.401 | 3.356 | 18.090 | 4.253 |

Score distribution: `0=2208`, `5=2652`, `8=3`, `10=7737`.

Human majority distribution: `0=5957`, `1=2757`, `2=3886`.

## Supplementary 100-Sample Results

These earlier runs used 100 target-emotion samples from the train manifest and
are retained for prompt/model comparison only.

| Model | Prompt mode | Parsed | Accuracy vs majority | Pearson | Spearman | MSE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MOSS 4B Instruct | target paper 0-10 | 100 / 100 | 0.440 | 0.473 | 0.435 | 15.599 | 3.950 |
| MOSS 4B Instruct | target human 0-2 rubric | 100 / 100 | 0.500 | 0.472 | 0.476 | 16.932 | 4.115 |
| MOSS 8B | target paper 0-10 | 100 / 100 | 0.430 | 0.392 | 0.371 | 19.682 | 4.436 |
| Kimi-Audio 7B Instruct | target human 0-2 rubric | 100 / 100 | 0.330 | 0.270 | 0.241 | 18.682 | 4.322 |
| Kimi-Audio 7B Instruct | contrastive 0-2 target | 99 / 100 | 0.283 | 0.157 | 0.185 | 20.653 | 4.545 |
| Kimi-Audio 7B Instruct | contrastive 0-2 swapped target | 99 / 100 | 0.303 | 0.312 | 0.240 | 18.506 | 4.302 |

## Current Takeaways

- The full 12,600-row MOSS 4B target paper 0-10 run has lower correlations and
  higher RMSE than its 100-sample subset estimate, so the full run should be
  treated as the main benchmark number.
- The strongest 100-sample Spearman is MOSS 4B with the target human
  0-2 rubric.
- The strongest 100-sample RMSE is MOSS 4B with the direct paper 0-10
  target-emotion prompt.
- The contrastive Kimi setup was worse than the non-contrastive human-rubric
  setup on these 100 samples.

## Artifacts

The raw run outputs are intentionally left under gitignored `runs/` paths:

- `/exp/exp4/acp21rjf/symphony-workspaces-SLLM-understanding/ROB-336/runs/moss_train_full_target_paper_0_10_raw.jsonl`
- `/exp/exp4/acp21rjf/symphony-workspaces-SLLM-understanding/ROB-336/runs/emonet_train_full_target_paper_0_10_requests.jsonl`
- `runs/moss_train100_target_paper_0_10_raw.jsonl`
- `runs/moss_train100_target_human_rubric_raw.jsonl`
- `runs/moss8b_train100_target_paper_0_10_raw.jsonl`
- `runs/kimi_train100_target_human_rubric_raw.jsonl`
- `runs/kimi_train100_target_contrastive_rubric_raw.jsonl`
- `runs/kimi_train100_target_contrastive_rubric_swapped_raw.jsonl`
