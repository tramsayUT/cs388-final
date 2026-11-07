# Run summary: qwen3.5-4b_calibration_n50

- Questions: 50

| Method | Accuracy | Mean confidence | Parse failures |
|---|---|---|---|
| Token probability (raw) | 62.0% (n=50) | 0.727 | 0 of 50 questions with any rotation not ok |
| Token probability (letter-bias corrected) | 66.0% (n=50) | 0.668 | — |
| Verbalized | 62.0% (n=50) | 0.973 | 0 of 50 ({'ok': 50}) |
| Sampling agreement | (scored on raw token-probability answer) | 0.688 | 3 of 500 samples unparsed |

## Reasoning samples

- Mean generated tokens per sample: 92
- Samples cut off at the token limit: 3 of 500
- Lowest letter mass across rotations (token probability): 0.995

## Most common verbalized confidences

- 1.00: 25
- 0.95: 23
- 0.90: 2

## Timing (this session)

- token_prob: 0.04 s/question
- verbalized: 0.03 s/question
- reasoning: 0.54 s/question
- Total: 0.61 s/question -> projected 0.67 h for all 3937 questions on this GPU (excludes model load)
