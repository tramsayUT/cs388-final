"""Output parsers for the CS388 calibration project.

Turns raw model outputs into answers and confidences for each method:

  Token-probability: A-D probabilities from the first-token top-k logprobs,
      renormalized over the four letters, mapped back to original option order
      for each rotation, then combined into raw (rotation 0) and
      letter-bias-corrected (mean over 4 rotations; Zheng et al. 2024) readouts.
  Verbalized: answer letter and 0-100 confidence from "C, 95%"-style text.
  Sampling agreement: final "Answer: X" from each reasoning sample, and the
      fraction of samples agreeing with an anchor answer (Xiong et al. 2024).

Parse failures are never dropped silently: every result carries a status
string so failure rates can be counted and reported.

Standard library only; self-test without a GPU:  python scripts/parsers.py
"""

import math
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

LETTERS: tuple[str, ...] = ("A", "B", "C", "D")
NUM_OPTIONS: int = len(LETTERS)

# Status values shared by all parsers.
OK: str = "ok"
NO_LETTER_TOKENS: str = "no_letter_tokens"
LOW_LETTER_MASS: str = "low_letter_mass"
NO_MATCH: str = "no_match"
CONFIDENCE_OUT_OF_RANGE: str = "confidence_out_of_range"
TRUNCATED_NO_ANSWER: str = "truncated_no_answer"

LOW_MASS_THRESHOLD: float = 0.5

VERBAL_PATTERN = re.compile(
    r"(?<![A-Za-z])\(?([ABCD])\)?(?![A-Za-z])\s*[,:;\-]?\s*(\d{1,3}(?:\.\d+)?)\s*%?"
)
ANSWER_PATTERN = re.compile(r"answer\s*(?:is)?\s*[:\-]?\s*\**\(?([ABCD])\)?\**(?![A-Za-z])", re.IGNORECASE)


@dataclass
class TokenProbResult:
    """Letter probabilities from one rotation, in ORIGINAL option order.

    Attributes:
        probs: Renormalized probability per original option index (sums to 1),
            or None when parsing failed.
        letter_mass: Total raw probability on the A-D tokens before renormalizing.
        status: OK, NO_LETTER_TOKENS, or LOW_LETTER_MASS (probs still filled).
    """

    probs: Optional[list[float]]
    letter_mass: float
    status: str


@dataclass
class VerbalResult:
    """Parsed verbalized answer.

    Attributes:
        answer_index: Original option index (0-3), or None on failure.
        confidence: Stated confidence scaled to 0-1, or None on failure.
        status: OK, NO_MATCH, or CONFIDENCE_OUT_OF_RANGE.
    """

    answer_index: Optional[int]
    confidence: Optional[float]
    status: str


@dataclass
class AgreementResult:
    """Sampling-agreement confidence for one question.

    Attributes:
        confidence: Fraction of samples whose answer matches the anchor.
        sample_answers: Parsed answer index per sample (None = unparseable).
        num_unparsed: Samples with no parseable answer (counted as disagreeing).
        statuses: Status per sample.
    """

    confidence: float
    sample_answers: list[Optional[int]]
    num_unparsed: int
    statuses: list[str] = field(default_factory=list)


def letter_probs_from_candidates(
    candidates: Iterable[tuple[str, float]],
    position_to_original: Sequence[int] = (0, 1, 2, 3),
) -> TokenProbResult:
    """Extract A-D probabilities from first-token candidates for one rotation.

    Tokens are matched after stripping whitespace, so "C" and " C" both count
    toward letter C. Probabilities are renormalized over the four letters and
    reordered from displayed letters back to original option indices.

    Args:
        candidates: (decoded_token, logprob) pairs, e.g. vLLM top-20 logprobs.
        position_to_original: Map from displayed letter index to original
            option index for this rotation (identity for rotation 0).

    Returns:
        TokenProbResult in original option order.
    """
    by_letter = [0.0] * NUM_OPTIONS
    for token, logprob in candidates:
        cleaned = (token or "").strip()
        if cleaned in LETTERS:
            by_letter[LETTERS.index(cleaned)] += math.exp(logprob)

    letter_mass = sum(by_letter)
    if letter_mass == 0.0:
        return TokenProbResult(probs=None, letter_mass=0.0, status=NO_LETTER_TOKENS)

    original = [0.0] * NUM_OPTIONS
    for position, prob in enumerate(by_letter):
        original[position_to_original[position]] = prob / letter_mass
    status = OK if letter_mass >= LOW_MASS_THRESHOLD else LOW_LETTER_MASS
    return TokenProbResult(probs=original, letter_mass=letter_mass, status=status)


def combine_rotations(results: Sequence[TokenProbResult]) -> dict:
    """Build raw and letter-bias-corrected readouts from 4 rotations.

    Raw = rotation 0. Corrected = mean of each original option's probability
    across all successfully parsed rotations (cyclic permutation averaging,
    Zheng et al. 2024).

    Args:
        results: One TokenProbResult per rotation; index 0 must be rotation 0.

    Returns:
        Dict with raw/corrected answer index, confidence, and probability
        vectors (None where unavailable) and the number of rotations used.
    """
    output = {
        "raw_answer": None, "raw_confidence": None, "raw_probs": None,
        "corrected_answer": None, "corrected_confidence": None, "corrected_probs": None,
        "rotations_used": 0,
    }
    if results and results[0].probs is not None:
        raw = results[0].probs
        output.update(raw_probs=raw, raw_answer=max(range(NUM_OPTIONS), key=raw.__getitem__),
                      raw_confidence=max(raw))

    usable = [result.probs for result in results if result.probs is not None]
    if usable:
        corrected = [sum(probs[index] for probs in usable) / len(usable) for index in range(NUM_OPTIONS)]
        output.update(corrected_probs=corrected, rotations_used=len(usable),
                      corrected_answer=max(range(NUM_OPTIONS), key=corrected.__getitem__),
                      corrected_confidence=max(corrected))
    return output


def parse_verbalized(text: str) -> VerbalResult:
    """Parse "C, 95%"-style output into an answer and a 0-1 confidence.

    Accepts optional parentheses, a missing percent sign, and the format's
    "Answer and Confidence (0-100):" prefix. Uses the first letter-number
    pair that follows the prefix; a confidence above 100 is a failure, not
    clipped.

    Args:
        text: Generated text.

    Returns:
        VerbalResult.
    """
    body = re.split(r"\(0-100\)\s*:", text, maxsplit=1)[-1]
    match = VERBAL_PATTERN.search(body)
    if not match:
        return VerbalResult(answer_index=None, confidence=None, status=NO_MATCH)
    value = float(match.group(2))
    if not 0 <= value <= 100:
        return VerbalResult(answer_index=LETTERS.index(match.group(1)), confidence=None,
                            status=CONFIDENCE_OUT_OF_RANGE)
    return VerbalResult(answer_index=LETTERS.index(match.group(1)), confidence=value / 100, status=OK)


def parse_reasoning_answer(text: str, finish_reason: Optional[str] = None) -> tuple[Optional[int], str]:
    """Parse the final "Answer: X" from one reasoning sample.

    Uses the LAST match, since the model may mention letters while reasoning.

    Args:
        text: Generated reasoning text.
        finish_reason: vLLM finish reason; "length" marks a truncated sample.

    Returns:
        Tuple of (answer index or None, status).
    """
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        return LETTERS.index(matches[-1].upper()), OK
    return None, TRUNCATED_NO_ANSWER if finish_reason == "length" else NO_MATCH


def sampling_agreement(
    samples: Sequence[tuple[str, Optional[str]]],
    anchor_index: int,
) -> AgreementResult:
    """Compute agreement between reasoning samples and an anchor answer.

    Args:
        samples: (text, finish_reason) per sample.
        anchor_index: Original option index of the anchor (token-probability answer).

    Returns:
        AgreementResult; unparseable samples count as disagreeing.

    Raises:
        ValueError: If there are no samples.
    """
    if not samples:
        raise ValueError("No samples to score")
    parsed = [parse_reasoning_answer(text, finish) for text, finish in samples]
    answers = [answer for answer, _ in parsed]
    agree = sum(1 for answer in answers if answer == anchor_index)
    return AgreementResult(
        confidence=agree / len(samples),
        sample_answers=answers,
        num_unparsed=sum(1 for answer in answers if answer is None),
        statuses=[status for _, status in parsed],
    )


def self_test() -> list[str]:
    """Check every parser on clean and messy sample outputs.

    Returns:
        List of failed checks (empty when all pass).
    """
    problems: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            problems.append(name)

    # Token probabilities: smoke-test-like candidates, leading-space duplicate.
    cands = [("C", math.log(0.90)), (" C", math.log(0.05)), ("D", math.log(0.03)), ("<think>", math.log(0.02))]
    result = letter_probs_from_candidates(cands)
    check("token: status ok", result.status == OK)
    check("token: C highest", result.probs is not None and result.probs.index(max(result.probs)) == 2)
    check("token: sums to 1", result.probs is not None and abs(sum(result.probs) - 1) < 1e-9)
    check("token: leading-space merged", abs(result.letter_mass - 0.98) < 1e-9)
    check("token: no letters", letter_probs_from_candidates([("The", 0.0)]).status == NO_LETTER_TOKENS)
    check("token: low mass flagged",
          letter_probs_from_candidates([("A", math.log(0.2)), ("The", math.log(0.8))]).status == LOW_LETTER_MASS)

    # Rotation 1: display order is original (1,2,3,0); model picks displayed B = original C.
    rotated = letter_probs_from_candidates([("B", math.log(0.9)), ("A", math.log(0.1))], (1, 2, 3, 0))
    check("rotation: mapped back to original C", rotated.probs is not None and rotated.probs[2] == 0.9)
    check("rotation: displayed A -> original B", rotated.probs is not None and abs(rotated.probs[1] - 0.1) < 1e-9)

    # Letter-biased model: always 0.7 on displayed A, 0.1 elsewhere.
    biased = [letter_probs_from_candidates([(l, math.log(0.7 if l == "A" else 0.1)) for l in LETTERS],
                                           tuple((pos + shift) % 4 for pos in range(4))) for shift in range(4)]
    combined = combine_rotations(biased)
    check("combine: raw follows the bias", combined["raw_answer"] == 0 and abs(combined["raw_confidence"] - 0.7) < 1e-9)
    check("combine: corrected is uniform", all(abs(p - 0.25) < 1e-9 for p in combined["corrected_probs"]))
    check("combine: 4 rotations used", combined["rotations_used"] == 4)
    partial = combine_rotations([TokenProbResult(None, 0.0, NO_LETTER_TOKENS)] + biased[1:])
    check("combine: missing rotation 0 -> no raw", partial["raw_answer"] is None and partial["rotations_used"] == 3)

    # Verbalized.
    cases = {
        "Answer and Confidence (0-100): C, 100%": (2, 1.0, OK),
        "C, 95": (2, 0.95, OK),
        "(B), 80%": (1, 0.8, OK),
        "A: 72.5%": (0, 0.725, OK),
        "D, 150%": (3, None, CONFIDENCE_OUT_OF_RANGE),
        "I am not sure about this question.": (None, None, NO_MATCH),
        "Answer and Confidence (0-100): B, 85% (because A is wrong, 90)": (1, 0.85, OK),
    }
    for text, (answer, confidence, status) in cases.items():
        parsed = parse_verbalized(text)
        check(f"verbal: {text!r}", (parsed.answer_index, parsed.confidence, parsed.status) == (answer, confidence, status))

    # Reasoning and agreement.
    samples = [
        ("Scurvy is vitamin C deficiency, not A or B.\nAnswer: C", "stop"),
        ("Collagen synthesis needs ascorbate.\nAnswer: (C)", "stop"),
        ("I think it is B.\nanswer: b", "stop"),
        ("The answer is **C**", "stop"),
        ("Scurvy relates to collagen and vitamin", "length"),
        ("No idea.", "stop"),
    ]
    agreement = sampling_agreement(samples, anchor_index=2)
    check("reasoning: answers parsed", agreement.sample_answers == [2, 2, 1, 2, None, None])
    check("reasoning: agreement 3/6", abs(agreement.confidence - 0.5) < 1e-9)
    check("reasoning: 2 unparsed", agreement.num_unparsed == 2)
    check("reasoning: truncation flagged", agreement.statuses[4] == TRUNCATED_NO_ANSWER)
    check("reasoning: plain miss flagged", agreement.statuses[5] == NO_MATCH)
    check("reasoning: last answer wins", parse_reasoning_answer("Answer: A? No.\nAnswer: D")[0] == 3)
    return problems


if __name__ == "__main__":
    failures = self_test()
    print("Self-test:", "PASSED" if not failures else f"FAILED ({len(failures)})")
    for failure in failures:
        print(" -", failure)
    sys.exit(0 if not failures else 1)
