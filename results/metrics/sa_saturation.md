# Sampling-agreement saturation check (test split; answers = raw token-probability answers)

| Statistic | qwen3.5-4b | qwen3.5-27b |
|---|---|---|
| Questions | 2909 | 2909 |
| Accuracy of the answer (%) | 59.1 | 75.5 |
| Questions with all 10 samples agreeing (%) | 39.2 | 68.6 |
| Accuracy when all 10 agree (%) | 88.9 | 90.0 |
| Accuracy when fewer than 10 agree (%) | 39.9 | 43.9 |
| Share of wrong answers where all 10 agree (%) | 10.6 | 28.1 |
| Distinct sampling-agreement values | 11 | 11 |
| AUROC, sampling agreement (all) | 0.823 | 0.790 |
| AUROC, token probability (all) | 0.764 | 0.832 |
| AUROC, token probability within the all-agree group | 0.670 | 0.787 |
| AUROC, sampling agreement within the not-all-agree group | 0.722 | 0.664 |

## Distribution of sampling-agreement confidence

| Agreement | qwen3.5-4b (% of questions) | qwen3.5-27b (% of questions) |
|---|---|---|
| 1.0 | 39.2 | 68.6 |
| 0.9 | 8.9 | 6.4 |
| 0.8 | 7.1 | 4.0 |
| 0.7 | 5.4 | 3.2 |
| 0.6 | 5.7 | 3.2 |
| 0.5 | 5.6 | 1.4 |
| 0.4 | 4.7 | 2.2 |
| 0.3 | 4.8 | 2.2 |
| 0.2 | 5.1 | 2.3 |
| 0.1 | 6.0 | 2.4 |
| 0.0 | 7.6 | 4.2 |
