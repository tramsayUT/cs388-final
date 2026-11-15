# Calibration metrics: qwen3.5-27b (test split)

- Test questions: 2909; per-method n reflects parse failures dropped

- Temperatures (fit on calibration split): tp_raw T=1.735, tp_corr T=1.454

- Platt (vb): a=0.651, b=-1.114; Platt (sa): a=0.316, b=-0.226

- Brackets are bootstrap 95% CIs (1000 resamples)

## Overall

| Method | n | Accuracy % | Mean conf | ECE % (15 bins) | ECE % (10 mass) | Brier | AUROC | Sel. AUC | Acc@50% | Acc@80% |
|---|---|---|---|---|---|---|---|---|---|---|
| tp_raw | 2909 | 75.5 [74.0, 77.0] | 0.849 | 9.4 [8.2, 10.8] | 9.4 | 0.148 [0.140, 0.158] | 0.832 [0.815, 0.848] | 0.916 | 94.8 | 84.4 |
| tp_raw_ts | 2909 | 75.5 [74.0, 77.0] | 0.745 | 2.7 [2.1, 4.3] | 2.7 | 0.136 [0.129, 0.143] | 0.833 [0.818, 0.848] | 0.917 | 94.5 | 84.4 |
| tp_corr | 2909 | 76.3 [74.7, 77.9] | 0.825 | 6.3 [5.3, 7.9] | 6.2 | 0.139 [0.131, 0.149] | 0.834 [0.818, 0.850] | 0.922 | 94.7 | 84.5 |
| tp_corr_ts | 2909 | 76.3 [74.8, 77.7] | 0.757 | 1.2 [1.2, 3.1] | 1.4 | 0.134 [0.128, 0.141] | 0.834 [0.819, 0.848] | 0.922 | 94.7 | 84.6 |
| vb | 2909 | 75.3 [73.8, 76.9] | 0.953 | 20.1 [18.5, 21.6] | 20.0 | 0.221 [0.207, 0.235] | 0.614 [0.601, 0.628] | 0.832 | 80.4 | 77.1 |
| vb_platt | 2909 | 75.3 [73.7, 76.8] | 0.732 | 3.0 [1.5, 4.4] | 2.1 | 0.177 [0.170, 0.183] | 0.614 [0.600, 0.628] | 0.832 | 80.4 | 77.1 |
| sa | 2909 | 75.5 [73.9, 77.0] | 0.846 | 12.6 [11.4, 14.0] | 12.4 | 0.152 [0.141, 0.164] | 0.790 [0.770, 0.809] | 0.879 | 90.0 | 85.4 |
| sa_platt | 2909 | 75.5 [73.8, 77.0] | 0.735 | 2.6 [2.0, 4.2] | 2.4 | 0.134 [0.127, 0.142] | 0.790 [0.771, 0.808] | 0.879 | 90.0 | 85.4 |

## ECE % by training stage

| group | sa | sa_platt | tp_corr | tp_corr_ts | tp_raw | tp_raw_ts | vb | vb_platt |
|---|---|---|---|---|---|---|---|---|
| clinical | 12.7 | 4.3 | 5.8 | 3.3 | 9.4 | 3.6 | 18.2 | 5.8 |
| dental | 17.0 | 3.9 | 7.7 | 3.0 | 11.4 | 3.6 | 28.1 | 5.7 |
| non-dental | 10.7 | 4.5 | 5.7 | 1.5 | 8.6 | 2.6 | 16.7 | 5.6 |
| para-clinical | 10.9 | 4.9 | 7.7 | 4.4 | 9.4 | 3.8 | 16.2 | 6.2 |
| pre-clinical | 7.9 | 7.8 | 6.8 | 3.2 | 9.4 | 4.6 | 14.3 | 5.0 |

## AUROC by training stage

| group | sa | sa_platt | tp_corr | tp_corr_ts | tp_raw | tp_raw_ts | vb | vb_platt |
|---|---|---|---|---|---|---|---|---|
| clinical | 0.795 | 0.795 | 0.811 | 0.81 | 0.821 | 0.82 | 0.585 | 0.585 |
| dental | 0.742 | 0.742 | 0.799 | 0.801 | 0.795 | 0.796 | 0.567 | 0.567 |
| non-dental | 0.806 | 0.806 | 0.841 | 0.841 | 0.84 | 0.842 | 0.627 | 0.627 |
| para-clinical | 0.796 | 0.796 | 0.88 | 0.881 | 0.859 | 0.861 | 0.636 | 0.636 |
| pre-clinical | 0.85 | 0.85 | 0.841 | 0.842 | 0.849 | 0.854 | 0.703 | 0.703 |

## Letter bias (token probability)

| Answer | RStd (pts) | Chosen A/B/C/D % | Chi-square vs gold |
|---|---|---|---|
| tp_raw | 2.8 | 30.7 / 24.7 / 23.3 / 21.2 | 5.5 (p=0.14) |
| tp_corr | 1.6 | 30.6 / 26.7 / 22.9 / 19.9 | 1.3 (p=0.73) |

## ECE % sensitivity to binning

| Method | 10 bins | 15 bins | 20 bins | 10 equal-mass | RMS (equal-mass) |
|---|---|---|---|---|---|
| tp_raw | 9.4 | 9.4 | 9.4 | 9.4 | 11.3 |
| tp_raw_ts | 2.4 | 2.7 | 2.7 | 2.7 | 2.9 |
| tp_corr | 6.3 | 6.3 | 6.4 | 6.2 | 7.5 |
| tp_corr_ts | 1.5 | 1.2 | 1.7 | 1.4 | 2.1 |
| vb | 20.1 | 20.1 | 20.1 | 20.0 | 21.4 |
| vb_platt | 2.8 | 3.0 | 3.0 | 2.1 | 2.8 |
| sa | 12.6 | 12.6 | 12.6 | 12.4 | 13.2 |
| sa_platt | 2.3 | 2.6 | 2.6 | 2.4 | 2.8 |

## Paired bootstrap tests (A - B; ECE and accuracy in % points)

| Metric | Comparison | Difference | 95% CI | p |
|---|---|---|---|---|
| ece | tp_raw - tp_raw_ts | +6.7 | [+4.5, +8.1] | <0.001 |
| ece | tp_raw - tp_corr | +3.1 | [+2.0, +4.0] | <0.001 |
| ece | tp_raw_ts - tp_corr_ts | +1.4 | [-0.3, +2.3] | 0.15 |
| ece | vb - sa | +7.5 | [+6.2, +8.8] | <0.001 |
| ece | tp_raw_ts - vb_platt | -0.3 | [-1.5, +1.8] | 0.976 |
| ece | tp_raw_ts - sa_platt | +0.1 | [-1.3, +1.5] | 0.95 |
| auroc | sa - vb | +0.176 | [+0.153, +0.201] | <0.001 |
| auroc | sa - tp_raw | -0.042 | [-0.058, -0.025] | <0.001 |
| auroc | sa - tp_corr | -0.043 | [-0.064, -0.024] | <0.001 |
| auroc | tp_raw - vb | +0.218 | [+0.200, +0.236] | <0.001 |
| accuracy | tp_corr - tp_raw | +0.8 | [-0.1, +1.8] | 0.092 |
| accuracy | vb - tp_raw | -0.2 | [-1.0, +0.7] | 0.656 |

## Token probability vs. sampling agreement (calibrated; split at each median)

| Token prob. | Sampling agr. | n | % of questions | Accuracy % |
|---|---|---|---|---|
| high | high | 1412 | 48.5 | 95.8 |
| high | low | 43 | 1.5 | 53.5 |
| low | high | 583 | 20.0 | 76.0 |
| low | low | 871 | 29.9 | 43.4 |

Verbalized answer matches the raw token-probability answer on 92.8% of questions.
