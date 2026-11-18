"""Tally hand-labeled error-analysis categories from the disagreements spreadsheet."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("tally_error_analysis")

GROUPS: list[str] = ["sa_high_tp_low", "tp_high_sa_low", "bias_flip"]
GROUP_LABELS: dict[str, str] = {
    "sa_high_tp_low": "SA high / TP low",
    "tp_high_sa_low": "TP high / SA low",
    "bias_flip": "Bias flip",
}
CATEGORIES: list[str] = [
    "correct",
    "key_error",
    "knowledge_gap",
    "near_miss",
    "multi_statement",
    "artifact",
    "other",
]
CATEGORY_LABELS: dict[str, str] = {
    "correct": "Correct (per key)",
    "key_error": "Key error / ambiguous",
    "knowledge_gap": "Knowledge gap",
    "near_miss": "Near miss",
    "multi_statement": "Multi-statement",
    "artifact": "Text artifact",
    "other": "Other",
}
# Note tags: label -> lowercase substrings searched for in the notes column.
NOTE_TAGS: dict[str, list[str]] = {
    "Negation question": ["negation"],
    "Garbled dataset text": ["garbled"],
    "Letter bias": ["letter bias"],
    "Flawed reasoning, right answer": ["flawed reasoning"],
    "Key questionable": ["key_error?", "ambiguous key", "key questionable"],
    "Disagreeing sample was right": ["disagreeing sample gets"],
    "Two plausible options": ["two plausible"],
}
FLIP_OUTCOMES: list[str] = ["fixed", "broke", "both wrong"]
REQUIRED_COLUMNS: list[str] = [
    "group", "id", "gold", "tp_raw_answer", "tp_corr_answer",
    "category", "notes",
]
EXPECTED_PER_GROUP = 10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with ``tag``, ``input`` and ``output_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="qwen3.5-4b",
                        help="Model tag used in file names.")
    parser.add_argument("--input", type=Path, default=None,
                        help="Labeled .xlsx (default: results/error_analysis/"
                             "<tag>_disagreements_labeled.xlsx).")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/error_analysis"),
                        help="Directory for tally outputs.")
    return parser.parse_args()


def load_labels(path: Path) -> pd.DataFrame:
    """Load and validate the labeled workbook.

    Args:
        path: Path to the labeled ``.xlsx`` file (first sheet is used).

    Returns:
        DataFrame with string columns, category lowercased and stripped.

    Raises:
        FileNotFoundError: If the workbook does not exist.
        ValueError: If columns, row counts, groups or labels are invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Labeled workbook not found: {path}")
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Expected an .xlsx file, got: {path.name}")

    frame = pd.read_excel(path, sheet_name=0, dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame = frame[frame["id"].str.strip() != ""].copy()
    for column in ("group", "gold", "tp_raw_answer", "tp_corr_answer"):
        frame[column] = frame[column].str.strip()
    frame["category"] = frame["category"].str.strip().str.lower()
    frame["notes"] = frame["notes"].str.strip()

    problems: list[str] = []
    counts = frame["group"].value_counts()
    for group in GROUPS:
        if counts.get(group, 0) != EXPECTED_PER_GROUP:
            problems.append(
                f"group {group}: {counts.get(group, 0)} rows, "
                f"expected {EXPECTED_PER_GROUP}")
    unknown_groups = sorted(set(frame["group"]) - set(GROUPS))
    if unknown_groups:
        problems.append(f"unknown groups: {unknown_groups}")
    if frame["id"].duplicated().any():
        problems.append("duplicate ids present")
    blank = frame.index[frame["category"] == ""].tolist()
    if blank:
        problems.append(f"blank category in {len(blank)} row(s)")
    if problems:
        raise ValueError("Validation failed: " + "; ".join(problems))

    extra = sorted(set(frame["category"]) - set(CATEGORIES))
    if extra:
        LOGGER.warning("Categories outside the agreed list (kept): %s", extra)
    blank_notes = int((frame["notes"] == "").sum())
    if blank_notes:
        LOGGER.warning("%d row(s) have blank notes", blank_notes)
    LOGGER.info("Loaded %d labeled rows from %s", len(frame), path)
    return frame


def flip_outcome(row: pd.Series) -> str:
    """Classify a bias-flip row from its answer columns.

    Args:
        row: Row with ``gold``, ``tp_raw_answer`` and ``tp_corr_answer``.

    Returns:
        ``"fixed"``, ``"broke"``, ``"both wrong"`` or ``"no flip"``.
    """
    raw_ok = row["tp_raw_answer"] == row["gold"]
    corr_ok = row["tp_corr_answer"] == row["gold"]
    if row["tp_raw_answer"] == row["tp_corr_answer"]:
        return "no flip"
    if corr_ok and not raw_ok:
        return "fixed"
    if raw_ok and not corr_ok:
        return "broke"
    return "both wrong"


def note_tags(note: str) -> list[str]:
    """Return the note-tag labels whose keywords appear in a note.

    Args:
        note: Free-text note for one example.

    Returns:
        Tag labels in ``NOTE_TAGS`` order.
    """
    lowered = note.lower()
    return [label for label, keys in NOTE_TAGS.items()
            if any(key in lowered for key in keys)]


def build_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build category, tag, accuracy and flip tables.

    Args:
        frame: Validated labeled rows.

    Returns:
        Dict of named DataFrames (index = item, columns = groups + Total).
    """
    frame = frame.copy()
    frame["tags"] = frame["notes"].map(note_tags)

    categories = [c for c in CATEGORIES if c in set(frame["category"])]
    categories += sorted(set(frame["category"]) - set(CATEGORIES))
    cat_table = (pd.crosstab(frame["category"], frame["group"])
                 .reindex(index=categories, columns=GROUPS, fill_value=0))

    tag_rows = {label: [int(frame.loc[frame["group"] == g, "tags"]
                            .map(lambda t, lab=label: lab in t).sum())
                        for g in GROUPS]
                for label in NOTE_TAGS}
    tag_table = pd.DataFrame.from_dict(tag_rows, orient="index",
                                       columns=GROUPS)

    raw_correct = frame["tp_raw_answer"] == frame["gold"]
    acc_table = pd.DataFrame(
        [[int(raw_correct[frame["group"] == g].sum()) for g in GROUPS]],
        index=["Raw answer correct"], columns=GROUPS)

    flips = frame[frame["group"] == "bias_flip"].copy()
    flips["outcome"] = flips.apply(flip_outcome, axis=1)
    no_flip = int((flips["outcome"] == "no flip").sum())
    if no_flip:
        LOGGER.warning("%d bias_flip row(s) have raw == corrected answer",
                       no_flip)
    flip_table = (pd.crosstab(flips["outcome"], flips["tp_raw_answer"])
                  .reindex(index=FLIP_OUTCOMES, fill_value=0))
    flip_table = flip_table.reindex(columns=sorted(flip_table.columns),
                                    fill_value=0)

    for table in (cat_table, tag_table, acc_table, flip_table):
        table["Total"] = table.sum(axis=1)

    per_row = frame[["group", "id", "gold", "tp_raw_answer",
                     "tp_corr_answer", "category"]].copy()
    per_row["tags"] = frame["tags"].map(lambda t: "; ".join(t))
    per_row["flip"] = frame.apply(
        lambda r: flip_outcome(r) if r["group"] == "bias_flip" else "",
        axis=1)
    return {"categories": cat_table, "tags": tag_table,
            "accuracy": acc_table, "flips": flip_table, "per_row": per_row}


def frame_to_markdown(table: pd.DataFrame, index_name: str) -> str:
    """Render a DataFrame as a GitHub markdown table without tabulate.

    Args:
        table: Table to render; the index becomes the first column.
        index_name: Header for the index column.

    Returns:
        Markdown table string.
    """
    headers = [index_name] + [str(c) for c in table.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for idx, row in table.iterrows():
        cells = [str(idx)] + [str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters in a short label.

    Args:
        text: Plain text.

    Returns:
        LaTeX-safe text.
    """
    for char, repl in (("\\", r"\textbackslash{}"), ("&", r"\&"),
                       ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(char, repl)
    return text


def to_latex(tables: dict[str, pd.DataFrame], tag: str) -> str:
    """Build a booktabs LaTeX table of categories and note tags.

    Args:
        tables: Output of :func:`build_tables`.
        tag: Model tag for the caption and label.

    Returns:
        LaTeX source for a ``table`` environment (needs booktabs).
    """
    cols = GROUPS + ["Total"]
    header = " & ".join(["Category"] + [GROUP_LABELS.get(c, c)
                                        for c in cols])

    def rows(table: pd.DataFrame, labels: dict[str, str]) -> list[str]:
        """Format table rows as LaTeX lines."""
        return [" & ".join([latex_escape(labels.get(i, str(i)))]
                           + [str(int(v)) for v in table.loc[i, cols]])
                + r" \\" for i in table.index]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Hand-labeled disagreement examples for "
        + latex_escape(tag) + r" (10 per group). Note tags can overlap.}",
        r"\label{tab:error-analysis}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *rows(tables["categories"], CATEGORY_LABELS),
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Note tags}} \\",
        *rows(tables["tags"], {}),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def write_outputs(tables: dict[str, pd.DataFrame], tag: str,
                  output_dir: Path) -> list[Path]:
    """Write CSV, markdown and LaTeX outputs.

    Args:
        tables: Output of :func:`build_tables`.
        tag: Model tag used in file names.
        output_dir: Destination directory (created if missing).

    Returns:
        Paths of the files written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    long_parts = []
    for section in ("categories", "tags", "accuracy", "flips"):
        part = (tables[section].rename_axis("item").reset_index()
                .melt(id_vars="item", var_name="group", value_name="count"))
        part.insert(0, "section", section)
        long_parts.append(part)
    csv_path = output_dir / f"{tag}_tally.csv"
    pd.concat(long_parts).to_csv(csv_path, index=False)

    md_path = output_dir / f"{tag}_tally.md"
    md = [
        f"# Error-analysis tally: {tag}", "",
        "## Categories by group", "",
        frame_to_markdown(tables["categories"], "category"), "",
        "## Note tags by group (can overlap)", "",
        frame_to_markdown(tables["tags"], "tag"), "",
        "## Raw token-probability answer correct (out of 10)", "",
        frame_to_markdown(tables["accuracy"], "measure"), "",
        "## Bias flips: outcome by raw letter", "",
        frame_to_markdown(tables["flips"], "outcome"), "",
        "## Per-row tags (check for missed keywords)", "",
        frame_to_markdown(tables["per_row"].set_index("id"), "id"), "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")

    tex_path = output_dir / f"{tag}_tally.tex"
    tex_path.write_text(to_latex(tables, tag), encoding="utf-8")
    return [csv_path, md_path, tex_path]


def main() -> int:
    """Run the tally end to end.

    Returns:
        Process exit code (0 on success, 1 on validation or I/O error).
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")
    args = parse_args()
    input_path = args.input or (
        Path("results/error_analysis")
        / f"{args.tag}_disagreements_labeled.xlsx")
    try:
        frame = load_labels(input_path)
        tables = build_tables(frame)
        paths = write_outputs(tables, args.tag, args.output_dir)
    except (FileNotFoundError, ValueError, PermissionError) as exc:
        LOGGER.error("%s", exc)
        return 1

    print(frame_to_markdown(tables["categories"], "category"))
    print()
    print(frame_to_markdown(tables["tags"], "tag"))
    print()
    print(frame_to_markdown(tables["accuracy"], "measure"))
    print()
    print(frame_to_markdown(tables["flips"], "outcome"))
    for path in paths:
        LOGGER.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())