# Run summary: qwen3.5-27b_calibration

- Questions: 1000

| Method | Accuracy | Mean confidence | Parse failures |
|---|---|---|---|
| Token probability (raw) | 73.0% (n=1000) | 0.841 | 0 of 1000 questions with any rotation not ok |
| Token probability (letter-bias corrected) | 73.1% (n=1000) | 0.819 | — |
| Verbalized | 72.8% (n=1000) | 0.952 | 0 of 1000 ({'ok': 1000}) |
| Sampling agreement | (scored on raw token-probability answer) | 0.836 | 194 of 10000 samples unparsed |

## Reasoning samples

- Mean generated tokens per sample: 119
- Samples cut off at the token limit: 177 of 10000
- Lowest letter mass across rotations (token probability): 0.882

## Most common verbalized confidences

- 1.00: 156
- 0.99: 2
- 0.98: 27
- 0.95: 770
- 0.92: 1
- 0.90: 8
- 0.85: 31
- 0.65: 1
- 0.45: 1
- 0.25: 1

## Timing (this session)

- token_prob: 0.10 s/question
- verbalized: 0.06 s/question
- reasoning: 1.19 s/question
- Total: 1.35 s/question -> projected 1.47 h for all 3937 questions on this GPU (excludes model load)
