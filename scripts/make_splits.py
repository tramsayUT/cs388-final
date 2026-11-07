"""Create the stratified calibration/test split for the MedMCQA validation set.

Steps:
  1. Load the MedMCQA validation split from Hugging Face.
  2. Drop the 244 cross-referencing questions (data/splits/excluded_cross_reference_ids.csv)
     and the 2 questions with subject "Unknown".
  3. Label each question with its training stage (pre-clinical, para-clinical,
     clinical, dental) and its correct answer letter.
  4. Split into ~1,000 calibration / ~2,937 test questions, stratified by
     individual subject, with a fixed random seed.
  5. Validate the split and write a summary.

Outputs (IDs and labels only; no question text):
  data/splits/calibration.csv
  data/splits/test.csv
  results/split_summary.md
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_ID: str = "openlifescienceai/medmcqa"
SPLIT: str = "validation"
EXCLUDED_PATH: Path = Path("data/splits/excluded_cross_reference_ids.csv")
CALIBRATION_PATH: Path = Path("data/splits/calibration.csv")
TEST_PATH: Path = Path("data/splits/test.csv")
SUMMARY_PATH: Path = Path("results/split_summary.md")

EXPECTED_POOL_SIZE: int = 3937
CALIBRATION_SIZE: int = 1000
RANDOM_SEED: int = 388
DROP_SUBJECTS: set[str] = {"Unknown"}
LETTERS: list[str] = ["A", "B", "C", "D"]
OUTPUT_COLUMNS: list[str] = ["id", "subject_name", "stage", "choice_type", "answer"]
MAX_SHARE_GAP_PCT: float = 2.0

STAGE_BY_SUBJECT: dict[str, str] = {
    "Anatomy": "pre-clinical",
    "Physiology": "pre-clinical",
    "Biochemistry": "pre-clinical",
    "Pathology": "para-clinical",
    "Pharmacology": "para-clinical",
    "Microbiology": "para-clinical",
    "Forensic Medicine": "para-clinical",
    "Social & Preventive Medicine": "para-clinical",
    "Medicine": "clinical",
    "Surgery": "clinical",
    "Pediatrics": "clinical",
    "Gynaecology & Obstetrics": "clinical",
    "Radiology": "clinical",
    "Ophthalmology": "clinical",
    "ENT": "clinical",
    "Anaesthesia": "clinical",
    "Orthopaedics": "clinical",
    "Skin": "clinical",
    "Psychiatry": "clinical",
    "Dental": "dental",
}


def load_questions() -> pd.DataFrame:
    """Download the MedMCQA validation split as a DataFrame.

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


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove cross-referencing questions and questions with dropped subjects.

    Args:
        df: Full validation-split DataFrame.

    Returns:
        Filtered DataFrame (the question pool for the project).

    Raises:
        FileNotFoundError: If the exclusion list is missing.
    """
    if not EXCLUDED_PATH.exists():
        raise FileNotFoundError(f"Exclusion list not found: {EXCLUDED_PATH} (run data_checks.py first)")
    excluded_ids = set(pd.read_csv(EXCLUDED_PATH)["id"])
    pool = df[~df["id"].isin(excluded_ids) & ~df["subject_name"].isin(DROP_SUBJECTS)].copy()
    logger.info(
        "Pool: %d rows (removed %d cross-referencing, %d dropped-subject)",
        len(pool),
        int(df["id"].isin(excluded_ids).sum()),
        int(df["subject_name"].isin(DROP_SUBJECTS).sum()),
    )
    return pool


def add_labels(pool: pd.DataFrame) -> pd.DataFrame:
    """Add the training-stage label and the correct answer letter.

    Args:
        pool: Filtered question DataFrame.

    Returns:
        DataFrame with added "stage" and "answer" columns.

    Raises:
        ValueError: If any subject has no stage mapping.
    """
    unmapped = sorted(set(pool["subject_name"]) - set(STAGE_BY_SUBJECT))
    if unmapped:
        raise ValueError(f"Subjects with no stage mapping: {unmapped}")
    labeled = pool.copy()
    labeled["stage"] = labeled["subject_name"].map(STAGE_BY_SUBJECT)
    labeled["answer"] = labeled["cop"].map(dict(enumerate(LETTERS)))
    return labeled


def make_split(labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into calibration and test sets, stratified by subject.

    Args:
        labeled: Labeled question DataFrame.

    Returns:
        Tuple of (calibration, test) DataFrames.
    """
    calibration, test = train_test_split(
        labeled,
        train_size=CALIBRATION_SIZE,
        stratify=labeled["subject_name"],
        random_state=RANDOM_SEED,
    )
    return calibration, test


def share_table(calibration: pd.DataFrame, test: pd.DataFrame, column: str) -> tuple[str, float]:
    """Build a Markdown table comparing a column's distribution across splits.

    Args:
        calibration: Calibration split.
        test: Test split.
        column: Column to compare.

    Returns:
        Tuple of (Markdown table, largest percentage-point gap between splits).
    """
    cal_counts = calibration[column].value_counts()
    test_counts = test[column].value_counts()
    keys = sorted(set(cal_counts.index) | set(test_counts.index), key=lambda k: -test_counts.get(k, 0))
    lines = [
        f"| {column} | Calibration n | Calibration % | Test n | Test % |",
        "|---|---|---|---|---|",
    ]
    max_gap = 0.0
    for key in keys:
        cal_n, test_n = int(cal_counts.get(key, 0)), int(test_counts.get(key, 0))
        cal_pct, test_pct = 100 * cal_n / len(calibration), 100 * test_n / len(test)
        max_gap = max(max_gap, abs(cal_pct - test_pct))
        lines.append(f"| {key} | {cal_n} | {cal_pct:.1f} | {test_n} | {test_pct:.1f} |")
    return "\n".join(lines), max_gap


def validate_split(labeled: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Check the split for overlap, size, and coverage problems.

    Args:
        labeled: Full labeled pool.
        calibration: Calibration split.
        test: Test split.

    Returns:
        List of problem descriptions (empty when the split is valid).
    """
    problems = []
    overlap = set(calibration["id"]) & set(test["id"])
    if overlap:
        problems.append(f"{len(overlap)} IDs appear in both splits")
    if len(calibration) + len(test) != len(labeled):
        problems.append("Split sizes do not add up to the pool size")
    if len(labeled) != EXPECTED_POOL_SIZE:
        problems.append(f"Pool size {len(labeled)} differs from expected {EXPECTED_POOL_SIZE}")
    missing_in_cal = set(labeled["subject_name"]) - set(calibration["subject_name"])
    if missing_in_cal:
        problems.append(f"Subjects missing from calibration split: {sorted(missing_in_cal)}")
    return problems


def main() -> int:
    """Build, validate, and save the calibration/test split.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    try:
        labeled = add_labels(apply_exclusions(load_questions()))
    except (RuntimeError, FileNotFoundError, ValueError) as err:
        logger.error("%s", err)
        return 1

    calibration, test = make_split(labeled)
    problems = validate_split(labeled, calibration, test)
    if problems:
        for problem in problems:
            logger.error("Validation failed: %s", problem)
        return 1

    subject_table, subject_gap = share_table(calibration, test, "subject_name")
    stage_table, stage_gap = share_table(calibration, test, "stage")
    answer_table, answer_gap = share_table(calibration, test, "answer")
    choice_table, _ = share_table(calibration, test, "choice_type")
    if max(subject_gap, stage_gap) > MAX_SHARE_GAP_PCT:
        logger.warning("Subject/stage shares differ by more than %.1f points between splits", MAX_SHARE_GAP_PCT)

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    calibration[OUTPUT_COLUMNS].sort_values("id").to_csv(CALIBRATION_PATH, index=False)
    test[OUTPUT_COLUMNS].sort_values("id").to_csv(TEST_PATH, index=False)

    summary = "\n\n".join([
        "# Calibration/test split summary",
        f"- Pool: {len(labeled)} questions; seed {RANDOM_SEED}; stratified by subject",
        f"- Calibration: {len(calibration)}; Test: {len(test)}",
        f"- Largest share gap between splits: subject {subject_gap:.2f} pts, stage {stage_gap:.2f} pts, "
        f"answer letter {answer_gap:.2f} pts",
        "## Training stage\n\n" + stage_table,
        "## Subject\n\n" + subject_table,
        "## Correct answer letter\n\n" + answer_table,
        "## Choice type\n\n" + choice_table,
    ])
    SUMMARY_PATH.write_text(summary + "\n")
    print(summary)
    logger.info("Wrote %s, %s, %s", CALIBRATION_PATH, TEST_PATH, SUMMARY_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
