"""Make report figures from the metrics written by compute_metrics.py.

Reads results/metrics/<tag>_*.csv and results/<tag>_test_questions.csv and
writes PDF (for LaTeX) and PNG (for viewing) files to figures/:

  <tag>_reliability   reliability diagrams with per-bin counts, before and
                      after calibration, one column per confidence method
  <tag>_coverage      accuracy vs. coverage when answering only the most
                      confident questions
  <tag>_stage         AUROC and ECE by training stage with 95% CIs
  <tag>_letter_bias   share of answers at each letter: gold key vs. raw vs.
                      letter-bias-corrected token-probability answers

Example:
  uv run python scripts/make_figures.py --tag qwen3.5-4b
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)

METRICS_DIR: Path = Path("results/metrics")
RESULTS_DIR: Path = Path("results")
FIGURES_DIR: Path = Path("figures")
LETTERS: tuple[str, ...] = ("A", "B", "C", "D")
STAGES: tuple[str, ...] = ("pre-clinical", "para-clinical", "clinical", "dental")

# Fixed categorical order (validated default palette, slots 1-4); one color per method family.
COLORS: dict[str, str] = {"tp_raw": "#2a78d6", "tp_corr": "#eb6834", "vb": "#1baf7a", "sa": "#eda100"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS: dict[str, str] = {
    "tp_raw": "Token prob. (raw)",
    "tp_corr": "Token prob. (bias-corrected)",
    "vb": "Verbalized",
    "sa": "Sampling agreement",
}
CALIBRATED: dict[str, str] = {"tp_raw": "tp_raw_ts", "tp_corr": "tp_corr_ts", "vb": "vb_platt", "sa": "sa_platt"}
CALIBRATION_NAME: dict[str, str] = {"tp_raw": "temp. scaled", "tp_corr": "temp. scaled",
                                     "vb": "Platt scaled", "sa": "Platt scaled"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def load_inputs(tag: str) -> dict[str, pd.DataFrame]:
    """Load every metrics file needed for the figures.

    Args:
        tag: Model tag, e.g. "qwen3.5-4b".

    Returns:
        Dict of DataFrames keyed by name.

    Raises:
        FileNotFoundError: If any input is missing (run compute_metrics.py first).
    """
    paths = {
        "main": METRICS_DIR / f"{tag}_main.csv",
        "stage": METRICS_DIR / f"{tag}_by_stage.csv",
        "reliability": METRICS_DIR / f"{tag}_reliability.csv",
        "coverage": METRICS_DIR / f"{tag}_coverage.csv",
        "bias": METRICS_DIR / f"{tag}_letter_bias.csv",
        "test": RESULTS_DIR / f"{tag}_test_questions.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    return {name: pd.read_csv(path) for name, path in paths.items()}


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure as PDF and PNG and close it.

    Args:
        fig: Figure to save.
        name: File stem in FIGURES_DIR.
    """
    for suffix, dpi in ((".pdf", None), (".png", 200)):
        fig.savefig(FIGURES_DIR / f"{name}{suffix}", dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s.pdf / .png", FIGURES_DIR / name)


def reliability_figure(data: dict[str, pd.DataFrame], tag: str) -> None:
    """Reliability diagrams (accuracy vs. confidence per bin) with count strips.

    Top block: uncalibrated confidence. Bottom block: after temperature or
    Platt scaling (fit on the calibration split). Bars sit at bin centers;
    the dashed diagonal is perfect calibration.

    Args:
        data: Loaded inputs.
        tag: Model tag (file name).
    """
    bins, main = data["reliability"], data["main"].set_index("method")
    fig = plt.figure(figsize=(10, 7.2))
    grid = fig.add_gridspec(5, 4, height_ratios=[3, 1, 0.9, 3, 1], hspace=0.15, wspace=0.25)
    rows = ((0, 1), (3, 4))

    for block, use_calibrated in enumerate((False, True)):
        for col, family in enumerate(COLORS):
            method = CALIBRATED[family] if use_calibrated else family
            sub = bins[bins["method"] == method]
            ax = fig.add_subplot(grid[rows[block][0], col])
            ax_count = fig.add_subplot(grid[rows[block][1], col], sharex=ax)
            width = (sub["bin_hi"] - sub["bin_lo"]).iloc[0]
            centers = (sub["bin_lo"] + sub["bin_hi"]) / 2
            ax.bar(centers, sub["accuracy"].fillna(0), width=width * 0.85, color=COLORS[family], zorder=2)
            ax.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1, zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(labelbottom=False)
            ece = 100 * main.loc[method, "ece"]
            ax.text(0.04, 0.93, f"ECE {ece:.1f}%", transform=ax.transAxes, va="top", fontsize=8.5)
            title = LABELS[family] if not use_calibrated else f"{LABELS[family]}\n({CALIBRATION_NAME[family]})"
            ax.set_title(title, fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel("Accuracy")
                ax_count.set_ylabel("Count")
            ax_count.bar(centers, sub["count"], width=width * 0.85, color=INK_MUTED, zorder=2)
            ax_count.set_yscale("symlog", linthresh=10)
            ax_count.set_ylim(0, max(10, sub["count"].max()) * 2)
            ax_count.grid(False)
            if block == 1:
                ax_count.set_xlabel("Confidence")
            else:
                ax_count.tick_params(labelbottom=False)
    fig.suptitle(f"Reliability diagrams on the test split ({tag})", fontsize=10.5, y=0.995)
    save(fig, f"{tag}_reliability")


def coverage_figure(data: dict[str, pd.DataFrame], tag: str) -> None:
    """Accuracy when answering only the most confident fraction of questions.

    Args:
        data: Loaded inputs.
        tag: Model tag (file name).
    """
    curves = data["coverage"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for family, color in COLORS.items():
        sub = curves[curves["method"] == family]
        ax.plot(100 * sub["coverage"], 100 * sub["accuracy"], color=color, linewidth=2, label=LABELS[family])
    accept_all = 100 * curves[curves["method"] == "tp_raw"]["accuracy"].iloc[-1]
    ax.axhline(accept_all, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(2, accept_all - 2.2, f"Answer everything, raw token prob. ({accept_all:.1f}%)", color=INK_MUTED, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=accept_all - 6)
    ax.set_xlabel("Coverage: % of questions answered (most confident first)")
    ax.set_ylabel("Accuracy on answered questions (%)")
    ax.set_title(f"Selective answering ({tag}, test split)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, f"{tag}_coverage")


def stage_figure(data: dict[str, pd.DataFrame], tag: str) -> None:
    """AUROC and ECE by training stage, with bootstrap 95% CIs.

    Two separate panels (no shared or dual axis).

    Args:
        data: Loaded inputs.
        tag: Model tag (file name).
    """
    stage = data["stage"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    offsets = np.linspace(-0.27, 0.27, len(COLORS))
    for ax, metric, label, scale in ((axes[0], "auroc", "AUROC (higher = better ranking)", 1),
                                     (axes[1], "ece", "ECE % (lower = better calibrated)", 100)):
        for offset, (family, color) in zip(offsets, COLORS.items()):
            sub = stage[stage["method"] == family].set_index("group").reindex(list(STAGES))
            x = np.arange(len(STAGES)) + offset
            y = scale * sub[metric]
            err = np.vstack([y - scale * sub[f"{metric}_lo"], scale * sub[f"{metric}_hi"] - y])
            ax.errorbar(x, y, yerr=err, fmt="o", color=color, markersize=6, capsize=0, elinewidth=1.5,
                        label=LABELS[family])
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(STAGES)
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    axes[0].axhline(0.5, color=INK_MUTED, linestyle="--", linewidth=1)
    axes[0].text(-0.45, 0.51, "chance", color=INK_MUTED, fontsize=8)
    axes[1].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Discrimination and calibration by training stage ({tag}, test split, uncalibrated)",
                 fontsize=10.5)
    save(fig, f"{tag}_stage")


def letter_bias_figure(data: dict[str, pd.DataFrame], tag: str) -> None:
    """Share of answers at each letter: gold key vs. raw vs. corrected answers.

    Args:
        data: Loaded inputs.
        tag: Model tag (file name).
    """
    bias = data["bias"].set_index("answer")
    gold = data["test"]["gold"].astype(int).value_counts(normalize=True).reindex(range(4), fill_value=0) * 100
    series = [
        ("Correct answer (gold key)", gold.to_numpy(), INK_MUTED),
        ("Raw token-prob. answer", bias.loc["tp_raw", [f"chosen_{l}_pct" for l in LETTERS]].to_numpy(float),
         COLORS["tp_raw"]),
        ("Letter-bias-corrected answer", bias.loc["tp_corr", [f"chosen_{l}_pct" for l in LETTERS]].to_numpy(float),
         COLORS["tp_corr"]),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    width = 0.26
    for index, (label, values, color) in enumerate(series):
        ax.bar(np.arange(4) + (index - 1) * (width + 0.01), values, width=width, color=color, label=label, zorder=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(LETTERS)
    ax.set_xlabel("Answer letter")
    ax.set_ylabel("% of test questions")
    ax.grid(axis="x", visible=False)
    p_raw, p_corr = bias.loc["tp_raw", "chi2_p"], bias.loc["tp_corr", "chi2_p"]
    ax.set_title(f"Letter bias ({tag}): chi-square vs. gold p={p_raw:.1g} raw, p={p_corr:.2f} corrected",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, f"{tag}_letter_bias")


def main() -> int:
    """Make all figures for one model tag.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="Model tag, e.g. qwen3.5-4b")
    args = parser.parse_args()
    try:
        data = load_inputs(args.tag)
    except FileNotFoundError as err:
        logger.error("%s", err)
        return 1
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    reliability_figure(data, args.tag)
    coverage_figure(data, args.tag)
    stage_figure(data, args.tag)
    letter_bias_figure(data, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
