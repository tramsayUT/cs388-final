# Run summary: qwen3.5-4b_calibration

- Questions: 1000

| Method | Accuracy | Mean confidence | Parse failures |
|---|---|---|---|
| Token probability (raw) | 56.1% (n=1000) | 0.725 | 0 of 1000 questions with any rotation not ok |
| Token probability (letter-bias corrected) | 60.8% (n=1000) | 0.676 | — |
| Verbalized | 59.9% (n=999) | 0.972 | 1 of 1000 ({'ok': 999, 'no_match': 1}) |
| Sampling agreement | (scored on raw token-probability answer) | 0.684 | 86 of 10000 samples unparsed |

## Reasoning samples

- Mean generated tokens per sample: 94
- Samples cut off at the token limit: 55 of 10000
- Lowest letter mass across rotations (token probability): 0.860

## Most common verbalized confidences

- 1.00: 469
- 0.95: 516
- 0.90: 13
- 0.00: 1

## Timing (this session)

- token_prob: 0.04 s/question
- verbalized: 0.02 s/question
- reasoning: 0.49 s/question
- Total: 0.55 s/question -> projected 0.60 h for all 3937 questions on this GPU (excludes model load)
