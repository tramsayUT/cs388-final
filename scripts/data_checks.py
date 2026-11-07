import logging
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_ID: str = "openlifescienceai/medmcqa"
SPLIT: str = "validation"
EXPECTED_ROWS: int = 4183
OPTION_COLUMNS: list[str] = ["opa", "opb", "opc", "opd"]
REQUIRED_COLUMNS: list[str] = ["id", "question", *OPTION_COLUMNS, "cop", "choice_type", "subject_name"]
LETTERS: list[str] = ["A", "B", "C", "D"]

SUMMARY_PATH: Path = Path("results/data_quality_summary.md")
EXCLUDED_PATH: Path = Path("data/splits/excluded_cross_reference_ids.csv")
REVIEW_PATH: Path = Path("data/raw/flagged_review.csv")

CROSS_REFERENCE_PATTERNS: list[re.Pattern] = [
    # "All of the above", "None of the following", "Both of these", ...
    re.compile(r"\b(all|none|both|neither|any)\s+of\s+(the\s+)?(above|following|these)\b", re.IGNORECASE),
    # Options that are combinations of other option letters: "a and b", "(a) & (c)", "b, c"
    re.compile(r"^\s*(both\s+|only\s+)?\(?[a-d]\)?(\s*[,&]\s*|\s+(and|or)\s+)\(?[a-d]\)?(\s|$|\)|\.)", re.IGNORECASE),
    # Explicit references: "option a"
    re.compile(r"\boption\s+[a-d]\b", re.IGNORECASE),
]


def load_validation() -> pd.DataFrame:
    """Download the MedMCQA validation split and return it as a DataFrame.

    Returns:
        DataFrame with one row per question.

    Raises:
        RuntimeError: If the dataset cannot be downloaded or loaded.
    """
    try:
        dataset = load_dataset(DATASET_ID, split=SPLIT)
    except (ConnectionError, OSError, ValueError) as err:
        raise RuntimeError(f"Could not load {DATASET_ID} [{SPLIT}]: {err}") from err
    return dataset.to_pandas()


def missing_values(df: pd.DataFrame) -> pd.Series:
    """Count null or whitespace-only values in each required column.

    Args:
        df: Questions DataFrame.

    Returns:
        Series of missing-value counts indexed by column name.
    """
    counts = {}
    for column in REQUIRED_COLUMNS:
        series = df[column]
        blank = series.isna() | series.astype(str).str.strip().eq("")
        counts[column] = int(blank.sum())
    return pd.Series(counts)


def duplicate_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count duplicate IDs, duplicate question texts, and duplicate full items.

    Args:
        df: Questions DataFrame.

    Returns:
        Dict mapping each duplicate check to the number of extra copies.
    """
    question_key = df["question"].astype(str).str.strip().str.lower()
    full_key = question_key + "||" + df[OPTION_COLUMNS].astype(str).agg("||".join, axis=1).str.lower()
    return {
        "duplicate ids": int(df["id"].duplicated().sum()),
        "duplicate question text": int(question_key.duplicated().sum()),
        "duplicate question + options": int(full_key.duplicated().sum()),
    }


def flag_cross_reference(df: pd.DataFrame) -> pd.Series:
    """Flag questions where any option refers to other options.

    Args:
        df: Questions DataFrame.

    Returns:
        Boolean Series, True where at least one option matches a pattern.
    """
    def option_refers(text: object) -> bool:
        value = str(text)
        return any(pattern.search(value) for pattern in CROSS_REFERENCE_PATTERNS)

    return df[OPTION_COLUMNS].map(option_refers).any(axis=1)


def letter_distribution(cop: pd.Series) -> pd.Series:
    """Compute the share of correct answers at each letter.

    Args:
        cop: Correct-option indices (0-3).

    Returns:
        Series of percentages indexed by letter A-D.
    """
    counts = cop.map(dict(enumerate(LETTERS))).value_counts().reindex(LETTERS, fill_value=0)
    return (100 * counts / counts.sum()).round(1)


def series_to_markdown(series: pd.Series, key_header: str, value_header: str) -> str:
    """Render a Series as a two-column Markdown table.

    Args:
        series: Values to render.
        key_header: Header for the index column.
        value_header: Header for the value column.

    Returns:
        Markdown table string.
    """
    lines = [f"| {key_header} | {value_header} |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in series.items()]
    return "\n".join(lines)


def main() -> int:
    """Run all checks, print a summary, and write output files.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    try:
        df = load_validation()
    except RuntimeError as err:
        logger.error("%s", err)
        return 1

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        logger.error("Missing expected columns: %s", missing_columns)
        return 1
    if len(df) != EXPECTED_ROWS:
        logger.warning("Expected %d rows, found %d", EXPECTED_ROWS, len(df))

    invalid_cop = int((~df["cop"].isin([0, 1, 2, 3])).sum())
    missing = missing_values(df)
    duplicates = duplicate_counts(df)
    choice_types = df["choice_type"].value_counts()

    flagged = flag_cross_reference(df)
    kept = df[~flagged]
    subjects = kept["subject_name"].value_counts()

    EXCLUDED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.loc[flagged, ["id"]].to_csv(EXCLUDED_PATH, index=False)
    df.loc[flagged, ["id", "question", *OPTION_COLUMNS, "cop", "subject_name"]].to_csv(REVIEW_PATH, index=False)

    summary = "\n\n".join([
        "# MedMCQA validation split: data quality summary",
        f"- Source: `{DATASET_ID}` [{SPLIT}], license Apache-2.0 per dataset card",
        f"- Rows: {len(df)} (expected {EXPECTED_ROWS})",
        f"- Invalid correct-option values: {invalid_cop}",
        f"- Flagged cross-reference questions (excluded): {int(flagged.sum())}",
        f"- Remaining questions: {len(kept)}",
        "## Missing or empty values\n\n" + series_to_markdown(missing, "Column", "Count"),
        "## Duplicates\n\n" + series_to_markdown(pd.Series(duplicates), "Check", "Count"),
        "## Choice type\n\n" + series_to_markdown(choice_types, "Type", "Count"),
        "## Answer-key distribution (% correct at each letter)\n\n"
        + series_to_markdown(letter_distribution(df["cop"]), "Letter", "All questions (%)")
        + "\n\n"
        + series_to_markdown(letter_distribution(kept["cop"]), "Letter", "After exclusions (%)"),
        "## Subject counts after exclusions\n\n" + series_to_markdown(subjects, "Subject", "Questions"),
    ])
    SUMMARY_PATH.write_text(summary + "\n")
    print(summary)

    examples = df.loc[flagged, ["question", *OPTION_COLUMNS]].head(15)
    print("\n## First 15 flagged questions (manual spot check)\n")
    for _, row in examples.iterrows():
        options = " | ".join(str(row[column])[:40] for column in OPTION_COLUMNS)
        print(f"- {str(row['question'])[:80]}\n    {options}")

    logger.info("Wrote %s, %s, %s", SUMMARY_PATH, EXCLUDED_PATH, REVIEW_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())