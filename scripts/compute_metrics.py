"""Compute calibration metrics for every confidence method and model.

Reads the per-question results written by run_inference.py:
  results/<tag>_calibration_questions.csv + results/<tag>_calibration_raw.jsonl
  results/<tag>_test_questions.csv        + results/<tag>_test_raw.jsonl

Confidence variants evaluated on the TEST split (fitting uses CALIBRATION only):
  tp_raw          token probability, rotation 0
  tp_raw_ts       tp_raw after temperature scaling (Guo et al. 2017)
  tp_corr         letter-bias corrected (mean over 4 rotations; Zheng et al. 2024)
  tp_corr_ts      tp_corr after temperature scaling
  vb              verbalized confidence (Xiong et al. 2024)
  vb_platt        vb after Platt scaling
  sa              sampling agreement with the tp_raw answer
  sa_platt        sa after Platt scaling

Metrics: accuracy, mean confidence, ECE (15 equal-width bins; primary),
ECE (10 equal-mass bins; Kadavath et al. 2022), RMS calibration error, Brier
score, AUROC, selective-prediction AUC (area under the accuracy-coverage curve;
Tian et al. 2023), accuracy at 50% and 80% coverage, with bootstrap 95% CIs.
Also: per-stage and per-subject breakdowns, reliability-diagram bins,
accuracy-coverage curves, and letter-bias statistics (recall standard
deviation across gold letters; chi-square on chosen letters).

Outputs go to results/metrics/<tag>_*.csv and results/metrics/<tag>_summary.md.

Example:
  uv run python scripts/compute_metrics.py --tag qwen3.5-4b
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chisquare
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR: Path = Path("results")
METRICS_DIR: Path = RESULTS_DIR / "metrics"
LETTERS: tuple[str, ...] = ("A", "B", "C", "D")
NUM_BINS_WIDTH: int = 15
NUM_BINS_MASS: int = 10
BOOTSTRAP_SAMPLES: int = 1000
BOOTSTRAP_SEED: int = 388
PROB_FLOOR: float = 1e-8
CONF_CLIP: float = 1e-3
MIN_SUBJECT_TEST_N: int = 150
COVERAGE_TARGETS: tuple[float, ...] = (0.5, 0.8)
METHODS: tuple[str, ...] = ("tp_raw", "tp_raw_ts", "tp_corr", "tp_corr_ts", "vb", "vb_platt", "sa", "sa_platt")
CI_METRICS: tuple[str, ...] = ("accuracy", "ece", "auroc", "brier")


# ---------------------------------------------------------------- loading

def load_split(tag: str, split: str) -> pd.DataFrame:
    """Load per-question results and attach probability vectors from the raw file.

    Args:
        tag: Model tag, e.g. "qwen3.5-4b".
        split: "calibration" or "test".

    Returns:
        DataFrame with one row per question plus "probs_raw" (rotation 0) and
        "probs_corr" (mean over usable rotations) columns, each a length-4 array
        or None.

    Raises:
        FileNotFoundError: If either results file is missing.
        ValueError: If the two files do not cover the same question IDs.
    """
    csv_path = RESULTS_DIR / f"{tag}_{split}_questions.csv"
    raw_path = RESULTS_DIR / f"{tag}_{split}_raw.jsonl"
    for path in (csv_path, raw_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run run_inference.py first")

    df = pd.read_csv(csv_path)
    probs_raw, probs_corr = {}, {}
    with raw_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            rotations = [np.array(p, dtype=float) for p in record["tp_probs_by_rotation"] if p is not None]
            first = record["tp_probs_by_rotation"][0]
            probs_raw[record["id"]] = np.array(first, dtype=float) if first is not None else None
            probs_corr[record["id"]] = np.mean(rotations, axis=0) if rotations else None

    if set(df["id"]) != set(probs_raw):
        raise ValueError(f"{split}: CSV and raw JSONL cover different question IDs")
    df["probs_raw"] = df["id"].map(probs_raw)
    df["probs_corr"] = df["id"].map(probs_corr)
    return df


# ---------------------------------------------------------------- calibration fits

def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale probability vectors: p_i^(1/T), renormalized.

    Equivalent to dividing logits by T, since log p_i = z_i - log Z.

    Args:
        probs: Array of shape (n, 4).
        temperature: T > 0.

    Returns:
        Scaled probabilities, shape (n, 4).
    """
    log_probs = np.log(np.clip(probs, PROB_FLOOR, 1.0)) / temperature
    log_probs -= log_probs.max(axis=1, keepdims=True)
    scaled = np.exp(log_probs)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, gold: np.ndarray) -> float:
    """Fit T by minimizing negative log-likelihood of the gold option.

    Args:
        probs: Calibration-split probabilities, shape (n, 4).
        gold: Gold option indices, shape (n,).

    Returns:
        Fitted temperature.
    """
    def nll(log_t: float) -> float:
        scaled = apply_temperature(probs, math.exp(log_t))
        return float(-np.mean(np.log(np.clip(scaled[np.arange(len(gold)), gold], PROB_FLOOR, 1.0))))

    result = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return math.exp(result.x)


def logit(values: np.ndarray) -> np.ndarray:
    """Logit of clipped confidences.

    Args:
        values: Confidences in [0, 1].

    Returns:
        log(p / (1 - p)) with p clipped to [CONF_CLIP, 1 - CONF_CLIP].
    """
    clipped = np.clip(values, CONF_CLIP, 1 - CONF_CLIP)
    return np.log(clipped / (1 - clipped))


def fit_platt(confidence: np.ndarray, correct: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling: correct ~ sigmoid(a * logit(conf) + b).

    Args:
        confidence: Calibration-split confidences.
        correct: Calibration-split correctness (0/1).

    Returns:
        Fitted (effectively unregularized) logistic regression.
    """
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(logit(confidence).reshape(-1, 1), correct)
    return model


# ---------------------------------------------------------------- method table

def build_methods(cal: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Build (confidence, correct) tables on the test split for every method.

    Args:
        cal: Calibration-split results.
        test: Test-split results.

    Returns:
        Tuple of (method name -> DataFrame with id, stage, subject_name,
        confidence, correct, answer, gold) and fitted parameters.
    """
    params: dict = {}
    base = test[["id", "stage", "subject_name", "gold"]]

    def table(frame: pd.DataFrame, confidence, answer) -> pd.DataFrame:
        out = frame[["id", "stage", "subject_name", "gold"]].copy()
        out["confidence"] = np.asarray(confidence, dtype=float)
        out["answer"] = np.asarray(answer, dtype=float)
        out["correct"] = (out["answer"] == out["gold"]).astype(int)
        return out.dropna(subset=["confidence", "answer"]).reset_index(drop=True)

    tables: dict[str, pd.DataFrame] = {}
    for variant, column in (("tp_raw", "probs_raw"), ("tp_corr", "probs_corr")):
        cal_ok, test_ok = cal[cal[column].notna()], test[test[column].notna()]
        cal_probs, test_probs = np.stack(cal_ok[column].to_list()), np.stack(test_ok[column].to_list())
        tables[variant] = table(test_ok, test_probs.max(axis=1), test_probs.argmax(axis=1))
        temperature = fit_temperature(cal_probs, cal_ok["gold"].to_numpy())
        params[f"{variant}_temperature"] = temperature
        scaled = apply_temperature(test_probs, temperature)
        tables[f"{variant}_ts"] = table(test_ok, scaled.max(axis=1), scaled.argmax(axis=1))

    tables["vb"] = table(test, test["vb_conf"], test["vb_answer"])
    tables["sa"] = table(test, test["sa_conf"], test["tp_raw_answer"])
    for variant, conf_col, answer_col in (("vb", "vb_conf", "vb_answer"), ("sa", "sa_conf", "tp_raw_answer")):
        cal_ok = cal.dropna(subset=[conf_col, answer_col])
        model = fit_platt(cal_ok[conf_col].to_numpy(), (cal_ok[answer_col] == cal_ok["gold"]).astype(int).to_numpy())
        params[f"{variant}_platt"] = {"a": float(model.coef_[0][0]), "b": float(model.intercept_[0])}
        src = tables[variant]
        calibrated = model.predict_proba(logit(src["confidence"].to_numpy()).reshape(-1, 1))[:, 1]
        out = src.copy()
        out["confidence"] = calibrated
        tables[f"{variant}_platt"] = out

    params["test_n_by_method"] = {name: len(frame) for name, frame in tables.items()}
    params["test_n_total"] = len(base)
    return tables, params


# ---------------------------------------------------------------- metrics

def ece_equal_width(confidence: np.ndarray, correct: np.ndarray, num_bins: int = NUM_BINS_WIDTH) -> float:
    """Expected calibration error with equal-width bins (Guo et al. 2017).

    Args:
        confidence: Confidences in [0, 1].
        correct: 0/1 correctness.
        num_bins: Number of bins over [0, 1].

    Returns:
        Weighted mean |accuracy - confidence| across bins.
    """
    bins = np.minimum((confidence * num_bins).astype(int), num_bins - 1)
    counts = np.bincount(bins, minlength=num_bins)
    conf_sums = np.bincount(bins, weights=confidence, minlength=num_bins)
    correct_sums = np.bincount(bins, weights=correct, minlength=num_bins)
    return float(np.abs(correct_sums - conf_sums).sum() / len(confidence))


def equal_mass_bins(confidence: np.ndarray, num_bins: int = NUM_BINS_MASS) -> list[np.ndarray]:
    """Split indices into bins of (nearly) equal size, sorted by confidence.

    Args:
        confidence: Confidences.
        num_bins: Number of bins.

    Returns:
        List of index arrays.
    """
    order = np.argsort(confidence, kind="stable")
    return [chunk for chunk in np.array_split(order, num_bins) if len(chunk)]


def ece_equal_mass(confidence: np.ndarray, correct: np.ndarray) -> float:
    """ECE with equal-mass bins (Kadavath et al. 2022).

    Args:
        confidence: Confidences.
        correct: 0/1 correctness.

    Returns:
        Weighted mean |accuracy - confidence| across equal-mass bins.
    """
    return sum(len(i) * abs(correct[i].mean() - confidence[i].mean())
               for i in equal_mass_bins(confidence)) / len(confidence)


def rms_calibration_error(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Root-mean-square calibration error over equal-mass bins.

    Args:
        confidence: Confidences.
        correct: 0/1 correctness.

    Returns:
        sqrt(weighted mean of squared bin gaps).
    """
    return math.sqrt(sum(len(i) * (correct[i].mean() - confidence[i].mean()) ** 2
                         for i in equal_mass_bins(confidence)) / len(confidence))


def expected_accuracy_curve(confidence: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Accuracy of the top-k most confident answers, for k = 1..n.

    Ties in confidence are handled by expectation over random tie-breaking:
    every item in a tie group contributes the group's mean correctness.

    Args:
        confidence: Confidences.
        correct: 0/1 correctness.

    Returns:
        Array of length n: accuracy when keeping the k most confident answers.
    """
    frame = pd.DataFrame({"conf": confidence, "correct": correct})
    frame["expected"] = frame.groupby("conf")["correct"].transform("mean")
    expected = frame.sort_values("conf", ascending=False, kind="stable")["expected"].to_numpy()
    return np.cumsum(expected) / np.arange(1, len(expected) + 1)


def safe_auroc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """AUROC for separating correct from incorrect answers (NaN if one class).

    Args:
        confidence: Confidences.
        correct: 0/1 correctness.

    Returns:
        AUROC or NaN.
    """
    if correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, confidence))


def point_metrics(confidence: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """Compute every metric for one set of predictions.

    Args:
        confidence: Confidences.
        correct: 0/1 correctness.

    Returns:
        Dict of metric name -> value.
    """
    curve = expected_accuracy_curve(confidence, correct)
    n = len(confidence)
    metrics = {
        "n": n,
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "ece": ece_equal_width(confidence, correct),
        "ece_mass10": ece_equal_mass(confidence, correct),
        "rms_ce": rms_calibration_error(confidence, correct),
        "brier": float(np.mean((confidence - correct) ** 2)),
        "auroc": safe_auroc(confidence, correct),
        "selective_auc": float(curve.mean()),
    }
    for target in COVERAGE_TARGETS:
        metrics[f"acc_at_{int(target * 100)}"] = float(curve[max(1, math.ceil(target * n)) - 1])
    return metrics


def bootstrap_ci(confidence: np.ndarray, correct: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Percentile bootstrap 95% CIs for the headline metrics.

    Args:
        confidence: Confidences.
        correct: 0/1 correctness.
        rng: Seeded random generator.

    Returns:
        Dict with "<metric>_lo" and "<metric>_hi" for each CI metric.
    """
    n = len(confidence)
    draws = {metric: [] for metric in CI_METRICS}
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, n)
        conf_b, corr_b = confidence[idx], correct[idx]
        draws["accuracy"].append(corr_b.mean())
        draws["ece"].append(ece_equal_width(conf_b, corr_b))
        draws["auroc"].append(safe_auroc(conf_b, corr_b))
        draws["brier"].append(np.mean((conf_b - corr_b) ** 2))
    out = {}
    for metric, values in draws.items():
        values = np.array(values, dtype=float)
        out[f"{metric}_lo"], out[f"{metric}_hi"] = np.nanpercentile(values, [2.5, 97.5])
    return out


def metrics_table(tables: dict[str, pd.DataFrame], group_col: str | None, with_ci: bool,
                  min_n: int = 1) -> pd.DataFrame:
    """Compute metrics per method, optionally per group.

    Args:
        tables: Method name -> prediction table.
        group_col: Column to group by ("stage", "subject_name"), or None.
        with_ci: Whether to add bootstrap CIs.
        min_n: Skip groups with fewer test questions than this.

    Returns:
        Long-format metrics DataFrame.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for method in METHODS:
        frame = tables[method]
        groups = [("all", frame)] if group_col is None else list(frame.groupby(group_col))
        if group_col == "stage":
            groups.append(("non-dental", frame[frame["stage"] != "dental"]))
        for name, sub in groups:
            if len(sub) < min_n:
                continue
            conf, corr = sub["confidence"].to_numpy(), sub["correct"].to_numpy()
            row = {"method": method, "group": name, **point_metrics(conf, corr)}
            if with_ci:
                row.update(bootstrap_ci(conf, corr, rng))
            rows.append(row)
    return pd.DataFrame(rows)


def reliability_bins(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-bin counts, accuracy, and confidence for reliability diagrams.

    Args:
        tables: Method name -> prediction table.

    Returns:
        DataFrame with one row per (method, bin).
    """
    rows = []
    for method in METHODS:
        frame = tables[method]
        conf, corr = frame["confidence"].to_numpy(), frame["correct"].to_numpy()
        bins = np.minimum((conf * NUM_BINS_WIDTH).astype(int), NUM_BINS_WIDTH - 1)
        for b in range(NUM_BINS_WIDTH):
            mask = bins == b
            rows.append({
                "method": method, "bin": b,
                "bin_lo": b / NUM_BINS_WIDTH, "bin_hi": (b + 1) / NUM_BINS_WIDTH,
                "count": int(mask.sum()),
                "accuracy": float(corr[mask].mean()) if mask.any() else np.nan,
                "mean_confidence": float(conf[mask].mean()) if mask.any() else np.nan,
            })
    return pd.DataFrame(rows)


def coverage_curves(tables: dict[str, pd.DataFrame], points: int = 100) -> pd.DataFrame:
    """Accuracy-coverage curves sampled at evenly spaced coverage levels.

    Args:
        tables: Method name -> prediction table.
        points: Number of coverage levels.

    Returns:
        DataFrame with method, coverage, accuracy.
    """
    rows = []
    for method in METHODS:
        frame = tables[method]
        curve = expected_accuracy_curve(frame["confidence"].to_numpy(), frame["correct"].to_numpy())
        n = len(curve)
        for coverage in np.linspace(1 / points, 1.0, points):
            rows.append({"method": method, "coverage": coverage,
                         "accuracy": float(curve[max(1, math.ceil(coverage * n)) - 1])})
    return pd.DataFrame(rows)


def letter_bias(test: pd.DataFrame) -> pd.DataFrame:
    """Letter-bias statistics for raw and corrected token-probability answers.

    Recall standard deviation (RStd; Zheng et al. 2024) is the standard
    deviation of per-gold-letter accuracy, in percentage points. The
    chi-square test compares chosen-letter counts with gold-letter counts.

    Args:
        test: Test-split results.

    Returns:
        DataFrame with one row per answer type.
    """
    rows = []
    gold = test["gold"].astype(int)
    gold_counts = np.bincount(gold, minlength=4)
    for name, column in (("tp_raw", "tp_raw_answer"), ("tp_corr", "tp_corr_answer")):
        valid = test[column].notna()
        answers = test.loc[valid, column].astype(int)
        recalls = [100 * (answers[gold[valid] == k] == k).mean() for k in range(4)]
        chosen = np.bincount(answers, minlength=4)
        expected = gold_counts / gold_counts.sum() * chosen.sum()
        stat, p_value = chisquare(chosen, expected)
        rows.append({
            "answer": name, "rstd_pts": float(np.std(recalls)),
            **{f"recall_{LETTERS[k]}": recalls[k] for k in range(4)},
            **{f"chosen_{LETTERS[k]}_pct": 100 * chosen[k] / chosen.sum() for k in range(4)},
            "chi2": float(stat), "chi2_p": float(p_value),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- reporting

def fmt_ci(row: pd.Series, metric: str, scale: float = 1.0, digits: int = 1) -> str:
    """Format a metric with its CI, e.g. '12.3 [10.1, 14.2]'.

    Args:
        row: Metrics row.
        metric: Metric name.
        scale: Multiplier (100 for percentages).
        digits: Decimal places.

    Returns:
        Formatted string.
    """
    value = row[metric] * scale
    if f"{metric}_lo" not in row or pd.isna(row[f"{metric}_lo"]):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{row[f'{metric}_lo'] * scale:.{digits}f}, {row[f'{metric}_hi'] * scale:.{digits}f}]"


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a DataFrame (with index) as a Markdown table without extra packages.

    Args:
        frame: DataFrame to render.

    Returns:
        Markdown table string.
    """
    header = "| " + " | ".join([str(frame.index.name or "")] + [str(c) for c in frame.columns]) + " |"
    divider = "|" + "---|" * (len(frame.columns) + 1)
    body = ["| " + " | ".join([str(idx)] + ["" if pd.isna(v) else str(v) for v in row]) + " |"
            for idx, row in zip(frame.index, frame.to_numpy())]
    return "\n".join([header, divider, *body])


def write_summary(tag: str, main: pd.DataFrame, stage: pd.DataFrame, bias: pd.DataFrame, params: dict) -> str:
    """Build the Markdown summary.

    Args:
        tag: Model tag.
        main: Overall metrics.
        stage: Per-stage metrics.
        bias: Letter-bias statistics.
        params: Fitted calibration parameters.

    Returns:
        Markdown string.
    """
    head = ("| Method | n | Accuracy % | Mean conf | ECE % (15 bins) | ECE % (10 mass) | Brier | AUROC | "
            "Sel. AUC | Acc@50% | Acc@80% |\n|---|---|---|---|---|---|---|---|---|---|---|")
    main_rows = [
        f"| {r.method} | {r.n} | {fmt_ci(r, 'accuracy', 100)} | {r.mean_confidence:.3f} | {fmt_ci(r, 'ece', 100)} | "
        f"{100 * r.ece_mass10:.1f} | {fmt_ci(r, 'brier', 1, 3)} | {fmt_ci(r, 'auroc', 1, 3)} | "
        f"{r.selective_auc:.3f} | {100 * r.acc_at_50:.1f} | {100 * r.acc_at_80:.1f} |"
        for _, r in main.iterrows()
    ]
    stage_pivot = stage.pivot(index="group", columns="method", values="ece").mul(100).round(1)
    auroc_pivot = stage.pivot(index="group", columns="method", values="auroc").round(3)
    bias_rows = [
        f"| {r.answer} | {r.rstd_pts:.1f} | " + " / ".join(f"{r[f'chosen_{l}_pct']:.1f}" for l in LETTERS)
        + f" | {r.chi2:.1f} (p={r.chi2_p:.2g}) |"
        for _, r in bias.iterrows()
    ]
    return "\n\n".join([
        f"# Calibration metrics: {tag} (test split)",
        f"- Test questions: {params['test_n_total']}; per-method n reflects parse failures dropped",
        f"- Temperatures (fit on calibration split): tp_raw T={params['tp_raw_temperature']:.3f}, "
        f"tp_corr T={params['tp_corr_temperature']:.3f}",
        f"- Platt (vb): a={params['vb_platt']['a']:.3f}, b={params['vb_platt']['b']:.3f}; "
        f"Platt (sa): a={params['sa_platt']['a']:.3f}, b={params['sa_platt']['b']:.3f}",
        "- Brackets are bootstrap 95% CIs (1000 resamples)",
        "## Overall\n\n" + head + "\n" + "\n".join(main_rows),
        "## ECE % by training stage\n\n" + frame_to_markdown(stage_pivot),
        "## AUROC by training stage\n\n" + frame_to_markdown(auroc_pivot),
        "## Letter bias (token probability)\n\n| Answer | RStd (pts) | Chosen A/B/C/D % | Chi-square vs gold |\n"
        "|---|---|---|---|\n" + "\n".join(bias_rows),
    ])


def main() -> int:
    """Compute and save all metrics for the requested model tag.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="Model tag used in results file names, e.g. qwen3.5-4b")
    args = parser.parse_args()

    try:
        cal, test = load_split(args.tag, "calibration"), load_split(args.tag, "test")
    except (FileNotFoundError, ValueError) as err:
        logger.error("%s", err)
        return 1

    tables, params = build_methods(cal, test)
    main_metrics = metrics_table(tables, None, with_ci=True)
    stage_metrics = metrics_table(tables, "stage", with_ci=True)
    subject_metrics = metrics_table(tables, "subject_name", with_ci=False, min_n=MIN_SUBJECT_TEST_N)
    bias = letter_bias(test)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = METRICS_DIR / args.tag
    main_metrics.to_csv(f"{prefix}_main.csv", index=False)
    stage_metrics.to_csv(f"{prefix}_by_stage.csv", index=False)
    subject_metrics.to_csv(f"{prefix}_by_subject.csv", index=False)
    reliability_bins(tables).to_csv(f"{prefix}_reliability.csv", index=False)
    coverage_curves(tables).to_csv(f"{prefix}_coverage.csv", index=False)
    bias.to_csv(f"{prefix}_letter_bias.csv", index=False)
    Path(f"{prefix}_params.json").write_text(json.dumps(params, indent=2))

    summary = write_summary(args.tag, main_metrics, stage_metrics, bias, params)
    Path(f"{prefix}_summary.md").write_text(summary + "\n")
    print(summary)
    logger.info("Wrote metrics to %s", METRICS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
