"""Run all three confidence methods on a split of MedMCQA questions.

Used for both the pilot (e.g. --limit 50) and the full runs. For each question:
  1. Token probability: 4 option rotations, 1 token each, top-20 logprobs
     -> raw (rotation 0) and letter-bias-corrected (mean over rotations).
  2. Verbalized: Xiong et al. (2024) vanilla prompt, temperature 0.
  3. Sampling agreement: 10 brief-reasoning samples at temperature 0.7,
     scored against the raw token-probability answer (the anchor).

Questions are processed in chunks and appended to disk after each chunk, so
an interrupted run resumes where it stopped (rerun the same command).

Outputs (TAG = <model short name>_<split>[_n<limit>]):
  results/<TAG>_questions.csv   one row per question: answers, confidences,
                                statuses (small; commit to GitHub)
  results/<TAG>_raw.jsonl       generated texts for auditing (larger)
  results/<TAG>_summary.md      parse-failure rates, accuracy, timing

Example:
  uv run python scripts/run_inference.py --model Qwen/Qwen3.5-4B --split calibration --limit 50
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from datasets import load_dataset

from parsers import (
    LETTERS,
    OK,
    combine_rotations,
    letter_probs_from_candidates,
    parse_verbalized,
    sampling_agreement,
)
from prompts import (
    NUM_ROTATIONS,
    REASONING_SETTINGS,
    TOKEN_PROB_SETTINGS,
    VERBAL_SETTINGS,
    Question,
    build_chat_prompt,
    question_from_row,
    reasoning_prompt,
    token_prob_prompt,
    verbalized_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_ID: str = "openlifescienceai/medmcqa"
DATASET_SPLIT: str = "validation"
SPLIT_DIR: Path = Path("data/splits")
RESULTS_DIR: Path = Path("results")
CHUNK_SIZE: int = 200
FULL_RUN_QUESTIONS: int = 3937

CSV_COLUMNS: list[str] = [
    "id", "subject_name", "stage", "gold",
    "tp_raw_answer", "tp_raw_conf", "tp_corr_answer", "tp_corr_conf",
    "tp_rotations_used", "tp_min_letter_mass", "tp_status",
    "vb_answer", "vb_conf", "vb_status",
    "sa_conf", "sa_unparsed", "sa_truncated", "sa_mean_tokens",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model ID, e.g. Qwen/Qwen3.5-4B")
    parser.add_argument("--split", choices=["calibration", "test"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N questions (pilot)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def load_split_questions(split: str, limit: Optional[int]) -> tuple[list[Question], pd.DataFrame]:
    """Load the questions for one split, in split-file order.

    Args:
        split: "calibration" or "test".
        limit: Optional cap on the number of questions.

    Returns:
        Tuple of (Question list, split label DataFrame aligned to it).

    Raises:
        FileNotFoundError: If the split file is missing (run make_splits.py).
        ValueError: If split IDs are missing from the dataset.
    """
    split_path = SPLIT_DIR / f"{split}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} not found; run scripts/make_splits.py first")
    labels = pd.read_csv(split_path)
    if limit is not None:
        labels = labels.head(limit)

    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT).to_pandas().set_index("id")
    missing = set(labels["id"]) - set(dataset.index)
    if missing:
        raise ValueError(f"{len(missing)} split IDs not found in the dataset")
    rows = dataset.loc[labels["id"]].reset_index()
    questions = [question_from_row(row) for _, row in rows.iterrows()]
    return questions, labels.reset_index(drop=True)


def completed_ids(csv_path: Path) -> set[str]:
    """Return question IDs already written by an earlier, interrupted run.

    Args:
        csv_path: Per-question results CSV.

    Returns:
        Set of completed question IDs (empty if the file does not exist).
    """
    if not csv_path.exists():
        return set()
    return set(pd.read_csv(csv_path, usecols=["id"])["id"].astype(str))


def candidates_from_output(completion: Any) -> list[tuple[str, float]]:
    """Convert a vLLM completion's first-token logprobs to (token, logprob) pairs.

    Args:
        completion: vLLM CompletionOutput with logprobs requested.

    Returns:
        List of (decoded_token, logprob); empty if no logprobs were returned.
    """
    if not completion.logprobs:
        return []
    return [(entry.decoded_token or "", entry.logprob) for entry in completion.logprobs[0].values()]


def run_chunk(llm: Any, sampling: dict[str, Any], questions: list[Question]) -> tuple[list[dict], list[dict], dict]:
    """Run all three methods on one chunk of questions.

    Args:
        llm: Loaded vLLM engine.
        sampling: Map of method name -> vllm.SamplingParams.
        questions: Questions in this chunk.

    Returns:
        Tuple of (per-question result rows, raw-text records, seconds per method).
    """
    tokenizer = llm.get_tokenizer()
    timings: dict[str, float] = {}

    # 1. Token probability: 4 rotations per question.
    tp_prompts, tp_maps = [], []
    for question in questions:
        for shift in range(NUM_ROTATIONS):
            text, mapping = token_prob_prompt(question, shift)
            tp_prompts.append(build_chat_prompt(tokenizer, text))
            tp_maps.append(mapping)
    start = time.perf_counter()
    tp_outputs = llm.generate(tp_prompts, sampling["token_prob"], use_tqdm=False)
    timings["token_prob"] = time.perf_counter() - start

    # 2. Verbalized.
    vb_prompts = [build_chat_prompt(tokenizer, verbalized_prompt(question)) for question in questions]
    start = time.perf_counter()
    vb_outputs = llm.generate(vb_prompts, sampling["verbalized"], use_tqdm=False)
    timings["verbalized"] = time.perf_counter() - start

    # 3. Sampling agreement: 10 reasoning samples per question.
    sa_prompts = [build_chat_prompt(tokenizer, reasoning_prompt(question)) for question in questions]
    start = time.perf_counter()
    sa_outputs = llm.generate(sa_prompts, sampling["reasoning"], use_tqdm=False)
    timings["reasoning"] = time.perf_counter() - start

    rows, raw_records = [], []
    for index, question in enumerate(questions):
        rotation_results = [
            letter_probs_from_candidates(candidates_from_output(tp_outputs[index * NUM_ROTATIONS + shift].outputs[0]),
                                         tp_maps[index * NUM_ROTATIONS + shift])
            for shift in range(NUM_ROTATIONS)
        ]
        combined = combine_rotations(rotation_results)
        tp_statuses = [result.status for result in rotation_results]

        vb_text = vb_outputs[index].outputs[0].text
        verbal = parse_verbalized(vb_text)

        samples = [(completion.text, completion.finish_reason) for completion in sa_outputs[index].outputs]
        token_counts = [len(completion.token_ids) for completion in sa_outputs[index].outputs]
        anchor = combined["raw_answer"]
        agreement = sampling_agreement(samples, anchor) if anchor is not None else None

        rows.append({
            "id": question.qid,
            "gold": question.answer_index,
            "tp_raw_answer": anchor,
            "tp_raw_conf": combined["raw_confidence"],
            "tp_corr_answer": combined["corrected_answer"],
            "tp_corr_conf": combined["corrected_confidence"],
            "tp_rotations_used": combined["rotations_used"],
            "tp_min_letter_mass": min(result.letter_mass for result in rotation_results),
            "tp_status": OK if all(status == OK for status in tp_statuses) else "|".join(tp_statuses),
            "vb_answer": verbal.answer_index,
            "vb_conf": verbal.confidence,
            "vb_status": verbal.status,
            "sa_conf": agreement.confidence if agreement else None,
            "sa_unparsed": agreement.num_unparsed if agreement else None,
            "sa_truncated": sum(1 for _, finish in samples if finish == "length"),
            "sa_mean_tokens": sum(token_counts) / len(token_counts),
        })
        raw_records.append({
            "id": question.qid,
            "tp_probs_by_rotation": [result.probs for result in rotation_results],
            "verbalized_text": vb_text,
            "reasoning_samples": [text for text, _ in samples],
            "reasoning_answers": agreement.sample_answers if agreement else None,
        })
    return rows, raw_records, timings


def append_results(csv_path: Path, raw_path: Path, rows: list[dict], raw_records: list[dict],
                   labels: pd.DataFrame) -> None:
    """Append one chunk of results to the CSV and JSONL outputs.

    Args:
        csv_path: Per-question CSV path.
        raw_path: Raw-text JSONL path.
        rows: Result rows for the chunk.
        raw_records: Raw-text records for the chunk.
        labels: Split labels (for subject and stage).
    """
    label_map = labels.set_index("id")[["subject_name", "stage"]].to_dict("index")
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({**row, **label_map[row["id"]]})
    with raw_path.open("a") as handle:
        for record in raw_records:
            handle.write(json.dumps(record) + "\n")


def summarize(csv_path: Path, seconds: dict[str, float], questions_timed: int) -> str:
    """Build the run summary: parse failures, accuracy, confidence, timing.

    Args:
        csv_path: Per-question CSV path.
        seconds: Total seconds per method for questions processed this session.
        questions_timed: Number of questions processed this session.

    Returns:
        Markdown summary.
    """
    df = pd.read_csv(csv_path)
    n = len(df)

    def accuracy(column: str) -> str:
        valid = df[column].notna()
        if not valid.any():
            return "n/a"
        return f"{100 * (df.loc[valid, column] == df.loc[valid, 'gold']).mean():.1f}% (n={int(valid.sum())})"

    def mean(column: str) -> str:
        return f"{df[column].mean():.3f}" if df[column].notna().any() else "n/a"

    vb_counts = df["vb_status"].value_counts().to_dict()
    lines = [
        "| Method | Accuracy | Mean confidence | Parse failures |",
        "|---|---|---|---|",
        f"| Token probability (raw) | {accuracy('tp_raw_answer')} | {mean('tp_raw_conf')} | "
        f"{int((df['tp_status'] != OK).sum())} of {n} questions with any rotation not ok |",
        f"| Token probability (letter-bias corrected) | {accuracy('tp_corr_answer')} | {mean('tp_corr_conf')} | — |",
        f"| Verbalized | {accuracy('vb_answer')} | {mean('vb_conf')} | {n - vb_counts.get(OK, 0)} of {n} ({vb_counts}) |",
        f"| Sampling agreement | (scored on raw token-probability answer) | {mean('sa_conf')} | "
        f"{int(df['sa_unparsed'].fillna(0).sum())} of {10 * n} samples unparsed |",
    ]

    timing = []
    if questions_timed:
        per_question = {method: total / questions_timed for method, total in seconds.items()}
        total_per_q = sum(per_question.values())
        timing = ["## Timing (this session)\n\n" + "\n".join([
            *[f"- {method}: {value:.2f} s/question" for method, value in per_question.items()],
            f"- Total: {total_per_q:.2f} s/question -> projected {total_per_q * FULL_RUN_QUESTIONS / 3600:.2f} h "
            f"for all {FULL_RUN_QUESTIONS} questions on this GPU (excludes model load)",
        ])]

    verbal_hist = df["vb_conf"].dropna().round(2).value_counts().sort_index(ascending=False).head(10)
    return "\n\n".join([
        f"# Run summary: {csv_path.stem.removesuffix('_questions')}",
        f"- Questions: {n}",
        "\n".join(lines),
        "## Reasoning samples\n\n"
        f"- Mean generated tokens per sample: {df['sa_mean_tokens'].mean():.0f}\n"
        f"- Samples cut off at the token limit: {int(df['sa_truncated'].sum())} of {10 * n}\n"
        f"- Lowest letter mass across rotations (token probability): {df['tp_min_letter_mass'].min():.3f}",
        "## Most common verbalized confidences\n\n"
        + "\n".join(f"- {value:.2f}: {count}" for value, count in verbal_hist.items()),
        *timing,
    ])


def main() -> int:
    """Run inference on the requested split and write results.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    args = parse_args()
    tag = f"{args.model.split('/')[-1].lower()}_{args.split}" + (f"_n{args.limit}" if args.limit else "")
    csv_path = RESULTS_DIR / f"{tag}_questions.csv"
    raw_path = RESULTS_DIR / f"{tag}_raw.jsonl"
    summary_path = RESULTS_DIR / f"{tag}_summary.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        questions, labels = load_split_questions(args.split, args.limit)
    except (FileNotFoundError, ValueError, OSError) as err:
        logger.error("%s", err)
        return 1

    done = completed_ids(csv_path)
    remaining = [question for question in questions if question.qid not in done]
    logger.info("%d questions in split, %d already done, %d to run", len(questions), len(done), len(remaining))

    seconds = {"token_prob": 0.0, "verbalized": 0.0, "reasoning": 0.0}
    if remaining:
        from vllm import LLM, SamplingParams  # imported here so --help works without a GPU

        try:
            llm = LLM(model=args.model, max_model_len=args.max_model_len,
                      gpu_memory_utilization=args.gpu_memory_utilization, seed=388)
        except (RuntimeError, ValueError, OSError) as err:
            logger.error("Model failed to load: %s", err)
            return 1
        sampling = {
            "token_prob": SamplingParams(**TOKEN_PROB_SETTINGS),
            "verbalized": SamplingParams(**VERBAL_SETTINGS),
            "reasoning": SamplingParams(**REASONING_SETTINGS),
        }
        for start in range(0, len(remaining), CHUNK_SIZE):
            chunk = remaining[start:start + CHUNK_SIZE]
            rows, raw_records, timings = run_chunk(llm, sampling, chunk)
            append_results(csv_path, raw_path, rows, raw_records, labels)
            for method, value in timings.items():
                seconds[method] += value
            logger.info("Saved %d / %d remaining questions (chunk took %.0f s)",
                        start + len(chunk), len(remaining), sum(timings.values()))

    summary = summarize(csv_path, seconds, len(remaining))
    summary_path.write_text(summary + "\n")
    print(summary)
    logger.info("Wrote %s, %s, %s", csv_path, raw_path, summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
