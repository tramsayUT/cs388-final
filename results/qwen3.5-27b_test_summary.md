# Run summary: qwen3.5-27b_test

- Questions: 2909

| Method | Accuracy | Mean confidence | Parse failures |
|---|---|---|---|
| Token probability (raw) | 75.5% (n=2909) | 0.849 | 1 of 2909 questions with any rotation not ok |
| Token probability (letter-bias corrected) | 76.3% (n=2909) | 0.825 | — |
| Verbalized | 75.3% (n=2909) | 0.953 | 0 of 2909 ({'ok': 2909}) |
| Sampling agreement | (scored on raw token-probability answer) | 0.846 | 521 of 29090 samples unparsed |

## Reasoning samples

- Mean generated tokens per sample: 118
- Samples cut off at the token limit: 476 of 29090
- Lowest letter mass across rotations (token probability): 0.433

## Most common verbalized confidences

- 1.00: 498
- 0.99: 4
- 0.98: 44
- 0.95: 2261
- 0.92: 2
- 0.90: 16
- 0.85: 71
- 0.75: 1
- 0.65: 3
- 0.25: 3

## Timing (this session)

- token_prob: 0.10 s/question
- verbalized: 0.06 s/question
- reasoning: 1.19 s/question
- Total: 1.34 s/question -> projected 1.47 h for all 3937 questions on this GPU (excludes model load)
