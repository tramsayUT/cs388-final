"""Extract confidently wrong answers for manual error categorization.

Two samples per model, drawn from the TEST split with a fixed seed:
  1. confident_wrong: wrong answers whose calibrated, letter-bias-corrected
     token-probability confidence (tp_corr_ts) is >= MIN_CONFIDENCE.
  2. verbal100_wrong: wrong answers where the model stated 100% confidence.

Each row has the question, options, gold and model answers, all three
confidence signals, the verbalized output, and one reasoning sample, plus
empty "category" and "notes" columns to fill in by hand.

Suggested categories (edit as patterns emerge):
  key_error        gold answer looks wrong or the question is ambiguous
  knowledge_gap    model lacks the fact; answer is plausible but wrong
  near_miss        model picks a closely related distractor
  multi_statement  "multi" question combining numbered statements
  artifact         digitization damage (missing letters, garbled text)
  other            anything else (explain in notes)

Outputs:
  results/error_analysis/<tag>_confident_wrong.csv
  results/error_analysis/<tag>_verbal100_wrong.csv

Example:
  uv run python scripts/error_analysis.py --tag qwen3.5-4b
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_ID: str = "openlifescienceai/medmcqa"
DATASET_SPLIT: str = "validation"
RESULTS_DIR: Path = Path("results")
METRICS_DIR: Path = RESULTS_DIR / "metrics"
OUT_DIR: Path = RESULTS_DIR / "error_analysis"
LETTERS: tuple[str, ...] = ("A", "B", "C", "D")
OPTION_COLUMNS: tuple[str, ...] = ("opa", "opb", "opc", "opd")
MIN_CONFIDENCE: float = 0.8
N_CONFIDENT_WRONG: int = 40
N_VERBAL100_WRONG: int = 20
SEED: int = 388
PROB_FLOOR: float = 1e-8


def load_inputs(tag: str) -> tuple[pd.DataFrame, dict, float]:
    """Load test results, raw outputs, and the fitted temperature.

    Args:
        tag: Model tag, e.g. "qwen3.5-4b".

    Returns:
        Tuple of (test results DataFrame, raw records by ID, tp_corr temperature).

    Raises:
        FileNotFoundError: If an input file is missing.
    """
    csv_path = RESULTS_DIR / f"{tag}_test_questions.csv"
    raw_path = RESULTS_DIR / f"{tag}_test_raw.jsonl"
    params_path = METRICS_DIR / f"{tag}_params.json"
    for path in (csv_path, raw_path, params_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run run_inference.py and compute_metrics.py first")
    raw = {}
    with raw_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            raw[record["id"]] = record
    temperature = json.loads(params_path.read_text())["tp_corr_temperature"]
    return pd.read_csv(csv_path), raw, temperature


def calibrated_corrected(raw_record: dict, temperature: float) -> tuple[int, float] | None:
    """Recompute the temperature-scaled, letter-bias-corrected answer and confidence.

    Args:
        raw_record: Raw output record with tp_probs_by_rotation.
        temperature: Fitted temperature for tp_corr.

    Returns:
        Tuple of (answer index, confidence), or None if no rotation parsed.
    """
    rotations = [np.array(p, dtype=float) for p in raw_record["tp_probs_by_rotation"] if p is not None]
    if not rotations:
        return None
    log_probs = np.log(np.clip(np.mean(rotations, axis=0), PROB_FLOOR, 1.0)) / temperature
    scaled = np.exp(log_probs - log_probs.max())
    scaled /= scaled.sum()
    return int(scaled.argmax()), float(scaled.max())


def sample_rows(frame: pd.DataFrame, n: int, sort_col: str) -> pd.DataFrame:
    """Draw a seeded random sample, sorted by confidence for easier reading.

    Args:
        frame: Candidate rows.
        n: Sample size (all rows if fewer).
        sort_col: Column to sort the sample by, descending.

    Returns:
        Sampled DataFrame.
    """
    sample = frame.sample(n=min(n, len(frame)), random_state=SEED) if len(frame) else frame
    return sample.sort_values(sort_col, ascending=False)


def build_table(sample: pd.DataFrame, questions: pd.DataFrame, raw: dict, answer_col: str) -> pd.DataFrame:
    """Assemble the review table with question text and empty category columns.

    Args:
        sample: Sampled result rows.
        questions: MedMCQA rows indexed by ID.
        raw: Raw records by ID.
        answer_col: Column holding the model answer being reviewed.

    Returns:
        Review DataFrame.
    """
    rows = []
    for _, row in sample.iterrows():
        q = questions.loc[row["id"]]
        record = raw[row["id"]]
        model_answer = int(row[answer_col])
        samples = record.get("reasoning_samples") or []
        reasoning = next(
            (text for text, ans in zip(samples, record.get("reasoning_answers") or []) if ans == model_answer),
            samples[0] if samples else "",
        )
        rows.append({
            "id": row["id"], "subject": row["subject_name"], "stage": row["stage"],
            "choice_type": q.get("choice_type", ""), "question": q["question"],
            **{f"option_{LETTERS[i]}": q[col] for i, col in enumerate(OPTION_COLUMNS)},
            "gold": LETTERS[int(row["gold"])], "model_answer": LETTERS[model_answer],
            "tp_corr_ts_conf": round(row.get("tp_corr_ts_conf", np.nan), 3),
            "sa_conf": row["sa_conf"], "vb_conf": row["vb_conf"],
            "verbalized_text": record.get("verbalized_text", ""), "reasoning_sample": reasoning,
            "explanation_in_dataset": q.get("exp", ""),
            "category": "", "notes": "",
        })
    return pd.DataFrame(rows)


def main() -> int:
    """Write the two review tables for one model.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="Model tag, e.g. qwen3.5-4b")
    args = parser.parse_args()

    try:
        test, raw, temperature = load_inputs(args.tag)
        questions = load_dataset(DATASET_ID, split=DATASET_SPLIT).to_pandas().set_index("id")
    except (FileNotFoundError, ConnectionError, OSError, ValueError) as err:
        logger.error("%s", err)
        return 1

    recomputed = test["id"].map(lambda qid: calibrated_corrected(raw[qid], temperature))
    test["tp_corr_ts_answer"] = recomputed.map(lambda value: value[0] if value else np.nan)
    test["tp_corr_ts_conf"] = recomputed.map(lambda value: value[1] if value else np.nan)

    confident_wrong = test[(test["tp_corr_ts_answer"] != test["gold"]) & (test["tp_corr_ts_conf"] >= MIN_CONFIDENCE)]
    verbal_wrong = test[(test["vb_conf"] >= 1.0) & (test["vb_answer"] != test["gold"]) & test["vb_answer"].notna()]
    logger.info("Candidates: %d confident-wrong (conf >= %.2f), %d stated-100%%-wrong",
                len(confident_wrong), MIN_CONFIDENCE, len(verbal_wrong))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "confident_wrong": build_table(sample_rows(confident_wrong, N_CONFIDENT_WRONG, "tp_corr_ts_conf"),
                                       questions, raw, "tp_corr_ts_answer"),
        "verbal100_wrong": build_table(sample_rows(verbal_wrong, N_VERBAL100_WRONG, "vb_conf"),
                                       questions, raw, "vb_answer"),
    }
    for name, table in outputs.items():
        path = OUT_DIR / f"{args.tag}_{name}.csv"
        table.to_csv(path, index=False)
        logger.info("Wrote %d rows to %s", len(table), path)

    print(f"Confident-wrong candidates: {len(confident_wrong)} of {len(test)} test questions "
          f"({100 * len(confident_wrong) / len(test):.1f}%); sampled {len(outputs['confident_wrong'])}")
    print(f"Stated-100%-and-wrong candidates: {len(verbal_wrong)}; sampled {len(outputs['verbal100_wrong'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
