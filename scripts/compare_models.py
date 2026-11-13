"""Compare two models (e.g. Qwen3.5-4B vs Qwen3.5-27B) on every confidence method.

For each confidence variant, both models are scored on the same test
questions and compared with a paired bootstrap (resampling questions), so
the test accounts for shared question difficulty. Tests hypothesis H7
(the larger model is better calibrated and discriminates better).

Requires compute_metrics.py to have been run for both tags (it reuses that
script's functions and its *_main.csv outputs).

Outputs:
  results/metrics/compare_<small>_vs_<large>.csv   paired differences (large - small)
  results/metrics/compare_<small>_vs_<large>.md    summary table
  figures/compare_<small>_vs_<large>.pdf / .png     accuracy, ECE, AUROC by method

Example:
  uv run python scripts/compare_models.py --small qwen3.5-4b --large qwen3.5-27b
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compute_metrics import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, METHODS, build_methods, load_split, metric_value  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)

METRICS_DIR: Path = Path("results/metrics")
FIGURES_DIR: Path = Path("figures")
COMPARE_METRICS: tuple[str, ...] = ("accuracy", "ece", "auroc", "brier")
SMALL_COLOR, LARGE_COLOR = "#2a78d6", "#eb6834"
INK_MUTED, GRID = "#52514e", "#e4e3df"


def paired_model_tests(small_tables: dict[str, pd.DataFrame], large_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Paired bootstrap of (large - small) for each method and metric.

    Args:
        small_tables: Method -> prediction table for the smaller model.
        large_tables: Method -> prediction table for the larger model.

    Returns:
        DataFrame with method, metric, both values, difference, 95% CI, p.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for method in METHODS:
        merged = small_tables[method][["id", "confidence", "correct"]].merge(
            large_tables[method][["id", "confidence", "correct"]], on="id", suffixes=("_s", "_l"))
        conf_s, corr_s = merged["confidence_s"].to_numpy(), merged["correct_s"].to_numpy()
        conf_l, corr_l = merged["confidence_l"].to_numpy(), merged["correct_l"].to_numpy()
        n = len(merged)
        draws = {metric: np.empty(BOOTSTRAP_SAMPLES) for metric in COMPARE_METRICS}
        for i in range(BOOTSTRAP_SAMPLES):
            idx = rng.integers(0, n, n)
            for metric in COMPARE_METRICS:
                draws[metric][i] = (metric_value(metric, conf_l[idx], corr_l[idx])
                                    - metric_value(metric, conf_s[idx], corr_s[idx]))
        for metric in COMPARE_METRICS:
            small_value = metric_value(metric, conf_s, corr_s)
            large_value = metric_value(metric, conf_l, corr_l)
            diffs = draws[metric]
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            p_value = min(1.0, 2 * min(np.nanmean(diffs <= 0), np.nanmean(diffs >= 0)))
            rows.append({"method": method, "metric": metric, "n": n, "small": small_value, "large": large_value,
                         "diff_large_minus_small": large_value - small_value, "ci_lo": lo, "ci_hi": hi,
                         "p_value": p_value})
    return pd.DataFrame(rows)


def summary_markdown(tests: pd.DataFrame, small: str, large: str) -> str:
    """Render the comparison as Markdown.

    Args:
        tests: Output of paired_model_tests.
        small: Smaller model tag.
        large: Larger model tag.

    Returns:
        Markdown string.
    """
    lines = [f"# {large} vs {small} (test split, paired bootstrap, {BOOTSTRAP_SAMPLES} resamples)",
             "Difference = large - small. ECE, accuracy, and Brier: lower ECE/Brier is better; "
             "higher accuracy/AUROC is better. ECE and accuracy in % points.",
             "| Method | Metric | " + small + " | " + large + " | Difference | 95% CI | p |",
             "|---|---|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        scale, digits = (100, 1) if r.metric in ("ece", "accuracy") else (1, 3)
        p_text = "<0.001" if r.p_value == 0 else f"{r.p_value:.3g}"
        lines.append(f"| {r.method} | {r.metric} | {scale * r.small:.{digits}f} | {scale * r.large:.{digits}f} | "
                     f"{scale * r.diff_large_minus_small:+.{digits}f} | "
                     f"[{scale * r.ci_lo:+.{digits}f}, {scale * r.ci_hi:+.{digits}f}] | {p_text} |")
    return "\n".join(lines[:2]) + "\n\n" + "\n".join(lines[2:])


def comparison_figure(small: str, large: str) -> None:
    """Dot plot of accuracy, ECE, and AUROC per method for both models, with 95% CIs.

    Args:
        small: Smaller model tag.
        large: Larger model tag.
    """
    main_small = pd.read_csv(METRICS_DIR / f"{small}_main.csv").set_index("method").reindex(list(METHODS))
    main_large = pd.read_csv(METRICS_DIR / f"{large}_main.csv").set_index("method").reindex(list(METHODS))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    y = np.arange(len(METHODS))
    for ax, metric, label, scale in ((axes[0], "accuracy", "Accuracy %", 100),
                                     (axes[1], "ece", "ECE % (lower = better)", 100),
                                     (axes[2], "auroc", "AUROC (higher = better)", 1)):
        for frame, tag, color, offset in ((main_small, small, SMALL_COLOR, -0.15), (main_large, large, LARGE_COLOR, 0.15)):
            value = scale * frame[metric]
            err = np.vstack([value - scale * frame[f"{metric}_lo"], scale * frame[f"{metric}_hi"] - value])
            ax.errorbar(value, y + offset, xerr=err, fmt="o", color=color, markersize=6, capsize=0,
                        elinewidth=1.5, label=tag)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(METHODS)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"{large} vs {small} on the test split (95% bootstrap CIs)", fontsize=10.5)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURES_DIR / f"compare_{small}_vs_{large}"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s.pdf / .png", stem)


def main() -> int:
    """Run the paired model comparison and write outputs.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--small", required=True, help="Smaller model tag, e.g. qwen3.5-4b")
    parser.add_argument("--large", required=True, help="Larger model tag, e.g. qwen3.5-27b")
    args = parser.parse_args()

    try:
        small_tables, _ = build_methods(load_split(args.small, "calibration"), load_split(args.small, "test"))
        large_tables, _ = build_methods(load_split(args.large, "calibration"), load_split(args.large, "test"))
        for tag in (args.small, args.large):
            if not (METRICS_DIR / f"{tag}_main.csv").exists():
                raise FileNotFoundError(f"Run compute_metrics.py --tag {tag} first")
    except (FileNotFoundError, ValueError) as err:
        logger.error("%s", err)
        return 1

    tests = paired_model_tests(small_tables, large_tables)
    stem = METRICS_DIR / f"compare_{args.small}_vs_{args.large}"
    tests.to_csv(f"{stem}.csv", index=False)
    summary = summary_markdown(tests, args.small, args.large)
    Path(f"{stem}.md").write_text(summary + "\n")
    comparison_figure(args.small, args.large)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
