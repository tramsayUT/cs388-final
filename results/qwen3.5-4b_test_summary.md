# Run summary: qwen3.5-4b_test

- Questions: 2909

| Method | Accuracy | Mean confidence | Parse failures |
|---|---|---|---|
| Token probability (raw) | 59.1% (n=2909) | 0.718 | 0 of 2909 questions with any rotation not ok |
| Token probability (letter-bias corrected) | 62.5% (n=2909) | 0.669 | — |
| Verbalized | 61.5% (n=2903) | 0.973 | 6 of 2909 ({'ok': 2903, 'no_match': 6}) |
| Sampling agreement | (scored on raw token-probability answer) | 0.678 | 202 of 29090 samples unparsed |

## Reasoning samples

- Mean generated tokens per sample: 93
- Samples cut off at the token limit: 139 of 29090
- Lowest letter mass across rotations (token probability): 0.963

## Most common verbalized confidences

- 1.00: 1357
- 0.98: 1
- 0.95: 1519
- 0.90: 26

## Timing (this session)

- token_prob: 0.04 s/question
- verbalized: 0.02 s/question
- reasoning: 0.48 s/question
- Total: 0.55 s/question -> projected 0.60 h for all 3937 questions on this GPU (excludes model load)
