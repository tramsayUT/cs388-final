"""Prompt templates and option-rotation logic for the CS388 calibration project.

Three confidence methods share one question format (Kadavath et al. 2022;
Zheng et al. 2024): lettered options, zero-shot, thinking mode off.

  Token-probability confidence: answer with only the letter; read A-D token
      probabilities at the first generated token. Run under 4 cyclic option
      rotations; rotation 0 is the raw readout, the average over rotations is
      the letter-bias-corrected readout (Zheng et al. 2024).
  Verbalized confidence: Xiong et al. (2024) vanilla multiple-choice prompt;
      the model states its answer and a 0-100 confidence.
  Sampling-agreement confidence: brief reasoning (<= 3 sentences) then
      "Answer: X"; 10 samples at temperature 0.7 (Xiong et al. 2024).

This module uses only the Python standard library, so it can be imported and
self-tested without a GPU:  python scripts/prompts.py
"""

import sys
from dataclasses import dataclass
from typing import Any, Mapping

LETTERS: tuple[str, ...] = ("A", "B", "C", "D")
NUM_OPTIONS: int = len(LETTERS)
NUM_ROTATIONS: int = NUM_OPTIONS
OPTION_COLUMNS: tuple[str, ...] = ("opa", "opb", "opc", "opd")

INTRO: str = "The following is a multiple choice question from a medical or dental entrance exam."

# Generation settings, passed later as vllm.SamplingParams(**settings).
TOKEN_PROB_SETTINGS: dict[str, Any] = {"max_tokens": 1, "temperature": 0.0, "logprobs": 20}
VERBAL_SETTINGS: dict[str, Any] = {"max_tokens": 40, "temperature": 0.0}
REASONING_SETTINGS: dict[str, Any] = {
    "n": 10,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 1.0,
    "seed": 388,
}


@dataclass(frozen=True)
class Question:
    """One multiple-choice question.

    Attributes:
        qid: MedMCQA question ID.
        text: Question stem.
        options: The four option texts in original order (A, B, C, D).
        answer_index: Index (0-3) of the correct option in original order.
    """

    qid: str
    text: str
    options: tuple[str, ...]
    answer_index: int


def question_from_row(row: Mapping[str, Any]) -> Question:
    """Build a Question from a MedMCQA record (dict or pandas row).

    Args:
        row: Record with keys id, question, opa-opd, cop.

    Returns:
        Question instance.

    Raises:
        ValueError: If the correct-option index is not 0-3.
    """
    answer_index = int(row["cop"])
    if answer_index not in range(NUM_OPTIONS):
        raise ValueError(f"Question {row['id']}: invalid cop {answer_index}")
    options = tuple(str(row[column]).strip() for column in OPTION_COLUMNS)
    return Question(qid=str(row["id"]), text=str(row["question"]).strip(), options=options, answer_index=answer_index)


def rotate_options(options: tuple[str, ...], shift: int) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Cyclically rotate option contents while keeping letters fixed.

    Position i (letter LETTERS[i]) shows original option (i + shift) % 4.
    Across shifts 0-3, every option appears once at every letter.

    Args:
        options: Option texts in original order.
        shift: Rotation amount, 0 to NUM_ROTATIONS - 1.

    Returns:
        Tuple of (rotated option texts, position_to_original index map).

    Raises:
        ValueError: If shift is out of range or there are not 4 options.
    """
    if len(options) != NUM_OPTIONS:
        raise ValueError(f"Expected {NUM_OPTIONS} options, got {len(options)}")
    if shift not in range(NUM_ROTATIONS):
        raise ValueError(f"Shift must be 0-{NUM_ROTATIONS - 1}, got {shift}")
    position_to_original = tuple((position + shift) % NUM_OPTIONS for position in range(NUM_OPTIONS))
    rotated = tuple(options[index] for index in position_to_original)
    return rotated, position_to_original


def format_question_block(text: str, options: tuple[str, ...]) -> str:
    """Render the question stem and lettered options.

    Args:
        text: Question stem.
        options: Option texts in display order.

    Returns:
        Formatted question block.
    """
    option_lines = "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, options))
    return f"Question: {text}\nOptions:\n{option_lines}"


def token_prob_prompt(question: Question, shift: int = 0) -> tuple[str, tuple[int, ...]]:
    """Build the token-probability prompt for one rotation.

    Args:
        question: The question.
        shift: Option rotation (0 = original order).

    Returns:
        Tuple of (user-turn text, position_to_original index map).
    """
    rotated, position_to_original = rotate_options(question.options, shift)
    text = (
        f"{INTRO}\n\n{format_question_block(question.text, rotated)}\n\n"
        "Answer with only the letter of the correct option (A, B, C, or D)."
    )
    return text, position_to_original


def verbalized_prompt(question: Question) -> str:
    """Build the verbalized-confidence prompt (Xiong et al. 2024, vanilla MC).

    Args:
        question: The question (original option order).

    Returns:
        User-turn text.
    """
    return (
        "Read the question, provide your answer and your confidence in this answer. "
        "Note: The confidence indicates how likely you think your answer is true.\n"
        "Use the following format to answer:\n"
        "Answer and Confidence (0-100): [ONLY the option letter; not a complete sentence], "
        "[Your confidence level, please only include the numerical number in the range of 0-100]%\n"
        "Only the answer and confidence, don't give me the explanation.\n\n"
        f"{INTRO}\n\n{format_question_block(question.text, question.options)}\n\n"
        "Now, please answer this question and provide your confidence level."
    )


def reasoning_prompt(question: Question) -> str:
    """Build the brief-reasoning prompt used for sampling agreement.

    Args:
        question: The question (original option order).

    Returns:
        User-turn text.
    """
    return (
        f"{INTRO}\n\n{format_question_block(question.text, question.options)}\n\n"
        "Think through the question briefly, in no more than 3 sentences. "
        "Then give your final answer on its own line in exactly this format:\n"
        "Answer: X\n"
        "where X is A, B, C, or D."
    )


def build_chat_prompt(tokenizer: Any, user_text: str) -> str:
    """Wrap user text in the model's chat template with thinking mode off.

    Args:
        tokenizer: Hugging Face tokenizer (e.g., from vllm.LLM.get_tokenizer()).
        user_text: Content of the user turn.

    Returns:
        Full prompt string ending at the assistant turn.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def self_test() -> list[str]:
    """Check rotation invariants and prompt construction on a sample question.

    Returns:
        List of problems found (empty when all checks pass).
    """
    problems = []
    sample = question_from_row({
        "id": "sample-1",
        "question": "Deficiency of which vitamin causes scurvy?",
        "opa": "Vitamin A",
        "opb": "Vitamin B12",
        "opc": "Vitamin C",
        "opd": "Vitamin D",
        "cop": 2,
    })

    maps = [rotate_options(sample.options, shift)[1] for shift in range(NUM_ROTATIONS)]
    if maps[0] != tuple(range(NUM_OPTIONS)):
        problems.append("Rotation 0 is not the identity")
    for position in range(NUM_OPTIONS):
        if sorted(mapping[position] for mapping in maps) != list(range(NUM_OPTIONS)):
            problems.append(f"Letter {LETTERS[position]} does not see every option exactly once")

    for shift in range(NUM_ROTATIONS):
        text, mapping = token_prob_prompt(sample, shift)
        gold_letter = LETTERS[mapping.index(sample.answer_index)]
        if f"{gold_letter}. Vitamin C" not in text:
            problems.append(f"Shift {shift}: correct option not at expected letter {gold_letter}")

    for name, text in [("verbalized", verbalized_prompt(sample)), ("reasoning", reasoning_prompt(sample))]:
        if "A. Vitamin A" not in text or "D. Vitamin D" not in text:
            problems.append(f"{name} prompt is missing options in original order")

    print("=== Token-probability prompt, rotation 1 ===")
    print(token_prob_prompt(sample, 1)[0])
    print("\n=== Verbalized prompt ===")
    print(verbalized_prompt(sample))
    print("\n=== Reasoning prompt ===")
    print(reasoning_prompt(sample))
    return problems


if __name__ == "__main__":
    found = self_test()
    print("\nSelf-test:", "PASSED" if not found else "FAILED")
    for problem in found:
        print(" -", problem)
    sys.exit(0 if not found else 1)
