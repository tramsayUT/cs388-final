# Disagreement groups: qwen3.5-4b (test split, n=2909)

Rank cutoffs used: sa_high_tp_low: high rank >= 0.75, low rank <= 0.25; tp_high_sa_low: high rank >= 0.67, low rank <= 0.33

| Group | Questions | % of test | Raw answer correct % |
|---|---|---|---|
| sa_high_tp_low | 50 | 1.7 | 84.0 |
| tp_high_sa_low | 26 | 0.9 | 23.1 |
| bias_flip | 610 | 21.0 | 27.2 |

Letter-bias flips: fixed (raw wrong, corrected right) 265; broke (raw right, corrected wrong) 166; both wrong 179.
