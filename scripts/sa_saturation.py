import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR: Path = Path("results")
OUTPUT_PATH: Path = RESULTS_DIR / "metrics" / "sa_saturation.md"


def safe_auroc(labels: pd.Series, scores: pd.Series) -> float:
    """AUROC, or NaN when only one class is present.

    Args:
        labels: 0/1 correctness.
        scores: Confidence scores.

    Returns:
        AUROC or NaN.
    """
    if labels.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def analyze(tag: str) -> tuple[dict, pd.Series]:
    """Compute saturation statistics for one model.

    Args:
        tag: Model tag, e.g. "qwen3.5-27b".

    Returns:
        Tuple of (summary statistics, share of questions at each confidence value).

    Raises:
        FileNotFoundError: If the test results file is missing.
    """
    path = RESULTS_DIR / f"{tag}_test_questions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path).dropna(subset=["sa_conf", "tp_raw_answer", "tp_raw_conf"])
    df["correct"] = (df["tp_raw_answer"] == df["gold"]).astype(int)
    full = df["sa_conf"] >= 1.0 - 1e-9
    wrong = df["correct"] == 0
    stats = {
        "n": len(df),
        "accuracy": 100 * df["correct"].mean(),
        "share_at_1": 100 * full.mean(),
        "acc_at_1": 100 * df.loc[full, "correct"].mean(),
        "acc_below_1": 100 * df.loc[~full, "correct"].mean(),
        "wrong_at_1_share": 100 * (full & wrong).sum() / wrong.sum(),
        "distinct_values": df["sa_conf"].round(2).nunique(),
        "auroc_sa": safe_auroc(df["correct"], df["sa_conf"]),
        "auroc_tp": safe_auroc(df["correct"], df["tp_raw_conf"]),
        "auroc_tp_within_1": safe_auroc(df.loc[full, "correct"], df.loc[full, "tp_raw_conf"]),
        "auroc_sa_below_1": safe_auroc(df.loc[~full, "correct"], df.loc[~full, "sa_conf"]),
    }
    distribution = (100 * df["sa_conf"].round(1).value_counts(normalize=True)).sort_index(ascending=False)
    return stats, distribution


def main() -> int:
    """Run the saturation check for each tag and write a summary.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="+", required=True, help="Model tags, e.g. qwen3.5-4b qwen3.5-27b")
    args = parser.parse_args()

    try:
        results = {tag: analyze(tag) for tag in args.tags}
    except FileNotFoundError as err:
        logger.error("%s", err)
        return 1

    rows = [
        ("Questions", "n", "{:.0f}"),
        ("Accuracy of the answer (%)", "accuracy", "{:.1f}"),
        ("Questions with all 10 samples agreeing (%)", "share_at_1", "{:.1f}"),
        ("Accuracy when all 10 agree (%)", "acc_at_1", "{:.1f}"),
        ("Accuracy when fewer than 10 agree (%)", "acc_below_1", "{:.1f}"),
        ("Share of wrong answers where all 10 agree (%)", "wrong_at_1_share", "{:.1f}"),
        ("Distinct sampling-agreement values", "distinct_values", "{:.0f}"),
        ("AUROC, sampling agreement (all)", "auroc_sa", "{:.3f}"),
        ("AUROC, token probability (all)", "auroc_tp", "{:.3f}"),
        ("AUROC, token probability within the all-agree group", "auroc_tp_within_1", "{:.3f}"),
        ("AUROC, sampling agreement within the not-all-agree group", "auroc_sa_below_1", "{:.3f}"),
    ]
    header = "| Statistic | " + " | ".join(args.tags) + " |\n|---|" + "---|" * len(args.tags)
    body = "\n".join("| " + label + " | " + " | ".join(fmt.format(results[t][0][key]) for t in args.tags) + " |"
                     for label, key, fmt in rows)

    values = sorted({v for t in args.tags for v in results[t][1].index}, reverse=True)
    dist_header = "| Agreement | " + " | ".join(f"{t} (% of questions)" for t in args.tags) + " |\n|---|" + "---|" * len(args.tags)
    dist_body = "\n".join(f"| {v:.1f} | " + " | ".join(f"{results[t][1].get(v, 0.0):.1f}" for t in args.tags) + " |"
                          for v in values)

    summary = "\n\n".join([
        "# Sampling-agreement saturation check (test split; answers = raw token-probability answers)",
        header + "\n" + body,
        "## Distribution of sampling-agreement confidence\n\n" + dist_header + "\n" + dist_body,
    ])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(summary + "\n")
    print(summary)
    logger.info("Wrote %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())