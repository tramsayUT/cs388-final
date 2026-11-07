"""Smoke test for the CS388 calibration project.

Loads Qwen3.5-4B in vLLM and checks two single-pass confidence readouts on
one hand-written multiple-choice question:
  1. Token-probability confidence: probabilities of the A/B/C/D tokens at
     the first answer position, renormalized over the four letters.
  2. Verbalized confidence: the model's stated 0-100 confidence (Xiong et al.
     2024 vanilla multiple-choice prompt).
Results are written to results/smoke_test.json.
"""

import json
import logging
import math
import re
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID: str = "Qwen/Qwen3.5-4B"
LETTERS: tuple[str, ...] = ("A", "B", "C", "D")
OUTPUT_PATH: Path = Path("results/smoke_test.json")

QUESTION: str = (
    "Deficiency of which vitamin causes scurvy?\n"
    "A. Vitamin A\n"
    "B. Vitamin B12\n"
    "C. Vitamin C\n"
    "D. Vitamin D"
)
GOLD_ANSWER: str = "C"

LETTER_PROMPT: str = (
    "The following is a multiple choice question about medicine. "
    "Answer with only the letter of the correct option.\n\n" + QUESTION
)

VERBAL_PROMPT: str = (
    "Read the question, provide your answer and your confidence in this answer. "
    "Note: The confidence indicates how likely you think your answer is true.\n"
    "Use the following format to answer:\n"
    "Answer and Confidence (0-100): [ONLY the option letter; not a complete "
    "sentence], [Your confidence level, please only include the numerical number "
    "in the range of 0-100]%\n"
    "Only the answer and confidence, don't give me the explanation.\n"
    "Question: " + QUESTION + "\n"
    "Now, please answer this question and provide your confidence level."
)

VERBAL_PATTERN = re.compile(r"\b([ABCD])\b\s*,\s*(\d{1,3})\s*%?")


def build_prompt(llm: LLM, user_text: str) -> str:
    """Render a single-turn chat prompt with thinking mode disabled.

    Args:
        llm: Loaded vLLM engine (provides the model's tokenizer).
        user_text: Content of the user turn.

    Returns:
        The fully formatted prompt string, ending at the assistant turn.
    """
    tokenizer = llm.get_tokenizer()
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def letter_probabilities(llm: LLM, prompt: str) -> dict:
    """Read A-D probabilities from the first generated token.

    Args:
        llm: Loaded vLLM engine.
        prompt: Formatted prompt ending at the assistant turn.

    Returns:
        Dict with raw letter probabilities, total probability mass on the
        four letters, renormalized probabilities, and the top-10 tokens.

    Raises:
        ValueError: If no A-D token appears among the top-20 candidates.
    """
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    output = llm.generate([prompt], params)[0].outputs[0]
    candidates = output.logprobs[0]

    raw = {letter: 0.0 for letter in LETTERS}
    for entry in candidates.values():
        token = (entry.decoded_token or "").strip()
        if token in raw:
            raw[token] += math.exp(entry.logprob)

    letter_mass = sum(raw.values())
    if letter_mass == 0.0:
        raise ValueError("No A-D tokens among the top-20 first-token candidates.")

    ranked = sorted(candidates.values(), key=lambda entry: entry.rank)
    return {
        "raw": raw,
        "letter_mass": letter_mass,
        "normalized": {letter: prob / letter_mass for letter, prob in raw.items()},
        "top_tokens": [
            (entry.decoded_token, round(math.exp(entry.logprob), 4)) for entry in ranked[:10]
        ],
    }


def verbalized_confidence(llm: LLM, prompt: str) -> dict:
    """Generate a verbalized answer and parse the letter and confidence.

    Args:
        llm: Loaded vLLM engine.
        prompt: Formatted verbalized-confidence prompt.

    Returns:
        Dict with the raw generated text and the parsed letter and confidence
        (None when parsing fails).
    """
    params = SamplingParams(max_tokens=40, temperature=0.0)
    text = llm.generate([prompt], params)[0].outputs[0].text
    match = VERBAL_PATTERN.search(text)
    return {
        "text": text,
        "parsed_letter": match.group(1) if match else None,
        "parsed_confidence": int(match.group(2)) if match else None,
    }


def main() -> int:
    """Run both smoke tests and write results to disk.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    try:
        llm = LLM(model=MODEL_ID, max_model_len=4096, gpu_memory_utilization=0.85)
    except (RuntimeError, ValueError, OSError) as err:
        logger.error("Model failed to load: %s", err)
        return 1

    letter_prompt = build_prompt(llm, LETTER_PROMPT)
    verbal_prompt = build_prompt(llm, VERBAL_PROMPT)
    logger.info("Prompt tail (check for leftover thinking tags):\n%s", letter_prompt[-200:])

    try:
        token_result = letter_probabilities(llm, letter_prompt)
    except ValueError as err:
        logger.error("Token-probability test failed: %s", err)
        return 1
    verbal_result = verbalized_confidence(llm, verbal_prompt)

    results = {
        "model": MODEL_ID,
        "gold_answer": GOLD_ANSWER,
        "prompt_tail": letter_prompt[-200:],
        "token_probability": token_result,
        "verbalized": verbal_result,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", OUTPUT_PATH)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())