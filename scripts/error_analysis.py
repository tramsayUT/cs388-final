"""Extract examples where confidence methods disagree, for manual categorization.

Draws three groups of test-split questions (fixed seed), 10 each by default:
  sa_high_tp_low   sampling agreement ranks the answer highly, but calibrated
                   token probability ranks it low (top vs. bottom quartile;
                   widened to terciles, then halves, if a group is too small)
  tp_high_sa_low   the reverse
  bias_flip        letter-bias correction changed the token-probability answer

Both confidence signals score the same raw token-probability answer, so the
first two groups show which signal is right when they disagree. Ranks are
percentile ranks within the test split (ties averaged), so no fitted scaling
is needed to compare the two signals.

Each row has the question, options, gold answer, raw and corrected answers,
all confidences, the verbalized output, up to two reasoning samples (one
agreeing with the raw answer, one disagreeing), MedMCQA's explanation, and
empty "category" and "notes" columns to fill in by hand.

Suggested categories (edit as patterns emerge):
  key_error        gold answer looks wrong or the question is ambiguous
  knowledge_gap    model lacks the fact; answer is plausible but wrong
  near_miss        model picks a closely related distractor
  multi_statement  "multi" question combining numbered statements
  artifact         digitization damage (missing letters, garbled text)
  other            anything else (explain in notes)

Outputs:
  results/error_analysis/<tag>_disagreements.csv
  results/error_analysis/<tag>_disagreements_summary.md  (counts for each group)

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
# Rank cutoffs (high, low), tried in order until each disagreement group has enough examples.
RANK_CUTOFFS: tuple[tuple[float, float], ...] = ((0.75, 0.25), (2 / 3, 1 / 3), (0.5, 0.5))
DEFAULT_PER_GROUP: int = 10
SEED: int = 388
PROB_FLOOR: float = 1e-8


def load_inputs(tag: str) -> tuple[pd.DataFrame, dict, float]:
    """Load test results, raw outputs, and the fitted tp_raw temperature.

    Args:
        tag: Model tag, e.g. "qwen3.5-4b".

    Returns:
        Tuple of (test results DataFrame, raw records by ID, tp_raw temperature).

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
    temperature = json.loads(params_path.read_text())["tp_raw_temperature"]
    return pd.read_csv(csv_path), raw, temperature


def scaled_raw_confidence(raw_record: dict, temperature: float) -> float:
    """Temperature-scaled confidence of the rotation-0 token-probability answer.

    Args:
        raw_record: Raw output record with tp_probs_by_rotation.
        temperature: Fitted temperature for tp_raw.

    Returns:
        Max scaled probability, or NaN if rotation 0 did not parse.
    """
    first = raw_record["tp_probs_by_rotation"][0]
    if first is None:
        return float("nan")
    log_probs = np.log(np.clip(np.array(first, dtype=float), PROB_FLOOR, 1.0)) / temperature
    scaled = np.exp(log_probs - log_probs.max())
    return float(scaled.max() / scaled.sum())


def reasoning_pair(record: dict, answer: int) -> tuple[str, str]:
    """Pick one reasoning sample agreeing with an answer and one disagreeing.

    Args:
        record: Raw output record.
        answer: Option index to compare against.

    Returns:
        Tuple of (agreeing sample, disagreeing sample); empty strings if none.
    """
    samples = record.get("reasoning_samples") or []
    answers = record.get("reasoning_answers") or [None] * len(samples)
    agree = next((text for text, ans in zip(samples, answers) if ans == answer), "")
    disagree = next((text for text, ans in zip(samples, answers) if ans != answer), "")
    return agree, disagree


def describe_groups(test: pd.DataFrame, masks: dict[str, pd.Series]) -> str:
    """Summarize each group's size and outcome counts.

    Args:
        test: Test results with derived columns.
        masks: Group name -> boolean mask over test.

    Returns:
        Markdown summary.
    """
    lines = ["| Group | Questions | % of test | Raw answer correct % |", "|---|---|---|---|"]
    for name, mask in masks.items():
        sub = test[mask]
        acc = 100 * (sub["tp_raw_answer"] == sub["gold"]).mean() if len(sub) else float("nan")
        lines.append(f"| {name} | {len(sub)} | {100 * len(sub) / len(test):.1f} | {acc:.1f} |")
    flips = test[masks["bias_flip"]]
    raw_ok = flips["tp_raw_answer"] == flips["gold"]
    corr_ok = flips["tp_corr_answer"] == flips["gold"]
    lines += [
        "",
        "Letter-bias flips: "
        f"fixed (raw wrong, corrected right) {int((~raw_ok & corr_ok).sum())}; "
        f"broke (raw right, corrected wrong) {int((raw_ok & ~corr_ok).sum())}; "
        f"both wrong {int((~raw_ok & ~corr_ok).sum())}.",
    ]
    return "\n".join(lines)


def build_rows(group: str, sample: pd.DataFrame, questions: pd.DataFrame, raw: dict) -> list[dict]:
    """Assemble review rows for one group.

    Args:
        group: Group name.
        sample: Sampled result rows.
        questions: MedMCQA rows indexed by ID.
        raw: Raw records by ID.

    Returns:
        List of row dicts.
    """
    rows = []
    for _, row in sample.iterrows():
        q = questions.loc[row["id"]]
        record = raw[row["id"]]
        raw_answer = int(row["tp_raw_answer"])
        agree, disagree = reasoning_pair(record, raw_answer)
        rows.append({
            "group": group, "id": row["id"], "subject": row["subject_name"], "stage": row["stage"],
            "choice_type": q.get("choice_type", ""), "question": q["question"],
            **{f"option_{LETTERS[i]}": q[col] for i, col in enumerate(OPTION_COLUMNS)},
            "gold": LETTERS[int(row["gold"])],
            "tp_raw_answer": LETTERS[raw_answer],
            "tp_corr_answer": LETTERS[int(row["tp_corr_answer"])] if pd.notna(row["tp_corr_answer"]) else "",
            "tp_raw_ts_conf": round(row["tp_raw_ts_conf"], 3),
            "sa_conf": row["sa_conf"], "tp_rank": round(row["tp_rank"], 2), "sa_rank": round(row["sa_rank"], 2),
            "vb_answer": LETTERS[int(row["vb_answer"])] if pd.notna(row["vb_answer"]) else "",
            "vb_conf": row["vb_conf"], "verbalized_text": record.get("verbalized_text", ""),
            "reasoning_agreeing": agree, "reasoning_disagreeing": disagree,
            "explanation_in_dataset": q.get("exp", ""),
            "category": "", "notes": "",
        })
    return rows


def main() -> int:
    """Write the disagreement review table and group summary for one model.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="Model tag, e.g. qwen3.5-4b")
    parser.add_argument("--per-group", type=int, default=DEFAULT_PER_GROUP, help="Examples per group")
    args = parser.parse_args()

    try:
        test, raw, temperature = load_inputs(args.tag)
        questions = load_dataset(DATASET_ID, split=DATASET_SPLIT).to_pandas().set_index("id")
    except (FileNotFoundError, ConnectionError, OSError, ValueError) as err:
        logger.error("%s", err)
        return 1

    test["tp_raw_ts_conf"] = test["id"].map(lambda qid: scaled_raw_confidence(raw[qid], temperature))
    test = test.dropna(subset=["tp_raw_ts_conf", "sa_conf", "tp_raw_answer"]).reset_index(drop=True)
    test["tp_rank"] = test["tp_raw_ts_conf"].rank(pct=True)
    test["sa_rank"] = test["sa_conf"].rank(pct=True)

    masks: dict[str, pd.Series] = {}
    cutoffs_used: dict[str, tuple[float, float]] = {}
    for name, (high_col, low_col) in (("sa_high_tp_low", ("sa_rank", "tp_rank")),
                                      ("tp_high_sa_low", ("tp_rank", "sa_rank"))):
        for high, low in RANK_CUTOFFS:
            mask = (test[high_col] >= high) & (test[low_col] <= low)
            if mask.sum() >= args.per_group or (high, low) == RANK_CUTOFFS[-1]:
                masks[name], cutoffs_used[name] = mask, (high, low)
                break
    masks["bias_flip"] = test["tp_corr_answer"].notna() & (test["tp_corr_answer"] != test["tp_raw_answer"])
    rows = []
    for name, mask in masks.items():
        candidates = test[mask]
        sample = candidates.sample(n=min(args.per_group, len(candidates)), random_state=SEED)
        rows += build_rows(name, sample, questions, raw)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUT_DIR / f"{args.tag}_disagreements.csv"
    summary_path = OUT_DIR / f"{args.tag}_disagreements_summary.md"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    cutoff_text = "; ".join(f"{name}: high rank >= {high:.2f}, low rank <= {low:.2f}"
                            for name, (high, low) in cutoffs_used.items())
    summary = (f"# Disagreement groups: {args.tag} (test split, n={len(test)})\n\n"
               f"Rank cutoffs used: {cutoff_text}\n\n" + describe_groups(test, masks))
    summary_path.write_text(summary + "\n")
    print(summary)
    logger.info("Wrote %d review rows to %s", len(rows), table_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
