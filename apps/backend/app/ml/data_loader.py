"""Data loader and validation module for CareerFit ML.

Loads, validates, and splits labelled resume–job matching datasets from documented CSV files,
ensuring data integrity, class balance, and absence of data leakage before model training or evaluation.
"""

import argparse
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import sys
from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger("app.ml.data_loader")

REQUIRED_COLUMNS = {"resume_text", "job_description", "label"}
VALID_LABELS = {0, 1}


@dataclass
class ValidationReport:
    """Detailed structural and semantic validation report for a resume–job dataset."""

    is_valid: bool
    total_rows: int
    valid_rows: int
    positive_rows: int
    negative_rows: int
    empty_text_rows: int
    invalid_label_rows: int
    duplicate_count: int
    conflicting_count: int
    class_balance_ratio: float  # minority class percentage (0.0 to 0.5)
    category_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetStats:
    """Statistical summary of loaded dataset."""

    total_samples: int
    positive_samples: int  # label == 1
    negative_samples: int  # label == 0
    categories: list[str]
    is_balanced: bool

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "positive_samples": self.positive_samples,
            "negative_samples": self.negative_samples,
            "categories": self.categories,
            "is_balanced": self.is_balanced,
        }


@dataclass
class DatasetSplit:
    """Container for train/test dataset splits."""

    X_train: list[str]
    X_test: list[str]
    y_train: list[int]
    y_test: list[int]
    stats: DatasetStats


def validate_dataset(
    df: pd.DataFrame,
    min_samples: int = 2,
    check_both_classes: bool = True,
) -> ValidationReport:
    """Perform comprehensive integrity validation on a resume–job DataFrame.

    Checks:
    - Required columns exist ('resume_text', 'job_description', 'label')
    - Text fields are non-empty after trimming whitespace
    - Labels belong strictly to {0, 1}
    - Both classes are present if check_both_classes is True
    - Exact duplicate pairs are identified
    - Conflicting pairs (identical texts with differing labels) are identified
    - Class balance ratio and category distributions are computed

    Args:
        df: Input DataFrame to validate.
        min_samples: Minimum valid samples required.
        check_both_classes: Whether to enforce presence of both class 0 and class 1.

    Returns:
        ValidationReport with validation flags, statistics, errors, and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    total_rows = len(df)
    if total_rows == 0:
        return ValidationReport(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            positive_rows=0,
            negative_rows=0,
            empty_text_rows=0,
            invalid_label_rows=0,
            duplicate_count=0,
            conflicting_count=0,
            class_balance_ratio=0.0,
            errors=["Dataset is completely empty (0 rows)."],
        )

    # 1. Required columns check
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(
            f"Dataset is missing required columns: {sorted(missing_cols)}. "
            f"Required columns: {sorted(REQUIRED_COLUMNS)}"
        )
        return ValidationReport(
            is_valid=False,
            total_rows=total_rows,
            valid_rows=0,
            positive_rows=0,
            negative_rows=0,
            empty_text_rows=0,
            invalid_label_rows=0,
            duplicate_count=0,
            conflicting_count=0,
            class_balance_ratio=0.0,
            errors=errors,
        )

    # 2. Text cleanliness and emptiness check
    resume_cleaned = df["resume_text"].astype(str).str.strip()
    job_cleaned = df["job_description"].astype(str).str.strip()
    empty_mask = (resume_cleaned == "") | (job_cleaned == "")
    empty_count = int(empty_mask.sum())
    if empty_count > 0:
        errors.append(f"Found {empty_count} rows with empty or whitespace-only resume_text or job_description.")

    # 3. Label validity check
    invalid_label_count = 0
    valid_labels_mask = pd.Series(True, index=df.index)
    for idx, val in df["label"].items():
        try:
            int_val = int(val)
            if int_val not in VALID_LABELS:
                invalid_label_count += 1
                valid_labels_mask.loc[idx] = False
        except (ValueError, TypeError):
            invalid_label_count += 1
            valid_labels_mask.loc[idx] = False

    if invalid_label_count > 0:
        errors.append(f"Found {invalid_label_count} rows with invalid labels. Supported labels must be 0 or 1.")

    # 4. Valid subset for deep inspection
    valid_mask = (~empty_mask) & valid_labels_mask
    valid_df = df[valid_mask].copy()
    valid_rows = len(valid_df)

    if valid_rows < min_samples:
        errors.append(
            f"Dataset contains only {valid_rows} valid non-empty samples, "
            f"which is below the required minimum of {min_samples}."
        )

    # 5. Duplicate and conflicting pair detection
    duplicate_count = 0
    conflicting_count = 0
    if valid_rows > 0:
        # Exact duplicates
        dup_mask = valid_df.duplicated(subset=["resume_text", "job_description"], keep=False)
        duplicate_count = int(valid_df.duplicated(subset=["resume_text", "job_description", "label"]).sum())
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} exact duplicate (resume, job, label) records.")

        # Conflicting labels (same resume and job with different labels)
        if dup_mask.any():
            grouped = valid_df[dup_mask].groupby(["resume_text", "job_description"])["label"].nunique()
            conflicts = grouped[grouped > 1]
            conflicting_count = len(conflicts)
            if conflicting_count > 0:
                errors.append(
                    f"Found {conflicting_count} conflicting examples where the same "
                    f"resume and job description are assigned contradictory labels (0 and 1)."
                )

    # 6. Class distribution and balance
    pos_count = 0
    neg_count = 0
    balance_ratio = 0.0
    if valid_rows > 0:
        valid_labels = valid_df["label"].astype(int)
        pos_count = int((valid_labels == 1).sum())
        neg_count = int((valid_labels == 0).sum())
        minority_count = min(pos_count, neg_count)
        balance_ratio = minority_count / valid_rows

        if check_both_classes:
            if pos_count == 0 or neg_count == 0:
                errors.append(
                    f"Dataset must contain both positive (1) and negative (0) examples. "
                    f"Found Positives={pos_count}, Negatives={neg_count}."
                )
            elif balance_ratio < 0.20:
                warnings.append(
                    f"Significant class imbalance detected: minority class represents only "
                    f"{balance_ratio * 100:.1f}% of data (Positives={pos_count}, Negatives={neg_count})."
                )

    # 7. Category counts
    cat_counts: dict[str, int] = {}
    if "category" in valid_df.columns:
        cat_counts = valid_df["category"].dropna().value_counts().to_dict()

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        total_rows=total_rows,
        valid_rows=valid_rows,
        positive_rows=pos_count,
        negative_rows=neg_count,
        empty_text_rows=empty_count,
        invalid_label_rows=invalid_label_count,
        duplicate_count=duplicate_count,
        conflicting_count=conflicting_count,
        class_balance_ratio=round(balance_ratio, 4),
        category_counts=cat_counts,
        errors=errors,
        warnings=warnings,
    )


def load_labeled_data(
    file_path: Union[str, Path],
    min_samples: int = 2,
    strict_validation: bool = False,
) -> tuple[pd.DataFrame, DatasetStats]:
    """Load and validate a CSV dataset of labelled resume–job pairs.

    Expected CSV columns:
    - resume_text: String text of candidate resume (required, non-empty)
    - job_description: String text of job posting (required, non-empty)
    - label: Binary integer 0 (not a match) or 1 (match / qualified) (required)
    - category: Optional domain/role category

    Args:
        file_path: Path to the CSV file.
        min_samples: Minimum valid rows required for the dataset.
        strict_validation: If True, raises ValueError on duplicate or conflicting records.

    Returns:
        Tuple of (validated DataFrame, DatasetStats)

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are missing, data is empty, or labels are invalid.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file at {path}: {exc}") from exc

    report = validate_dataset(df, min_samples=min_samples)
    if not report.is_valid:
        error_summary = " | ".join(report.errors)
        raise ValueError(f"Dataset validation failed for '{path.name}': {error_summary}")

    if strict_validation and report.conflicting_count > 0:
        raise ValueError(f"Strict validation failed: found {report.conflicting_count} conflicting rows.")

    # Clean and filter non-empty text rows
    df["resume_text"] = df["resume_text"].astype(str).str.strip()
    df["job_description"] = df["job_description"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    valid_mask = (df["resume_text"] != "") & (df["job_description"] != "") & (df["label"].isin(VALID_LABELS))
    df_clean = df[valid_mask].copy()

    categories = sorted(df_clean["category"].dropna().unique().tolist()) if "category" in df_clean.columns else []

    stats = DatasetStats(
        total_samples=report.valid_rows,
        positive_samples=report.positive_rows,
        negative_samples=report.negative_rows,
        categories=categories,
        is_balanced=report.class_balance_ratio >= 0.20,
    )

    return df_clean, stats


def prepare_pair_texts(df: pd.DataFrame) -> list[str]:
    """Format resume and job description pairs into unified text inputs for NLP vectorization.

    Format: 'RESUME: <clean_resume> \n\n JOB: <clean_job>'

    Args:
        df: DataFrame containing 'resume_text' and 'job_description'.

    Returns:
        List of combined text strings.
    """
    return [
        f"RESUME: {row['resume_text']}\n\nJOB: {row['job_description']}"
        for _, row in df.iterrows()
    ]


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> DatasetSplit:
    """Split dataset into train and test sets for model training and evaluation.

    Args:
        df: Validated DataFrame of labelled examples.
        test_size: Fraction of samples reserved for test set (0.0 to 1.0).
        random_state: Random seed for reproducibility.
        stratify: Whether to preserve class proportions in splits.

    Returns:
        DatasetSplit container with combined pair strings and labels.
    """
    X = prepare_pair_texts(df)
    y = df["label"].astype(int).tolist()

    strat = y if (stratify and len(set(y)) > 1 and min(y.count(0), y.count(1)) >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    pos_count = sum(1 for label in y if label == 1)
    neg_count = sum(1 for label in y if label == 0)
    categories = sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else []

    stats = DatasetStats(
        total_samples=len(y),
        positive_samples=pos_count,
        negative_samples=neg_count,
        categories=categories,
        is_balanced=(min(pos_count, neg_count) / len(y)) >= 0.20 if len(y) > 0 else False,
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        stats=stats,
    )


def save_train_test_split(
    df: pd.DataFrame,
    train_output_path: Union[str, Path],
    test_output_path: Union[str, Path],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test subsets and save them directly as CSV files.

    Guarantees no data leakage between train and test datasets.
    Preserves all original columns (including category).

    Args:
        df: Input DataFrame to split.
        train_output_path: Destination path for train.csv.
        test_output_path: Destination path for test.csv.
        test_size: Fraction of data for the test set (default 0.2 for 80/20 split).
        random_state: Fixed random seed for deterministic reproducibility.
        stratify: Whether to preserve class proportions.

    Returns:
        Tuple of (train_df, test_df)
    """
    y = df["label"].astype(int).tolist()
    strat = y if (stratify and len(set(y)) > 1 and min(y.count(0), y.count(1)) >= 2) else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    train_path = Path(train_output_path)
    test_path = Path(test_output_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    logger.info(
        "Saved %d training rows to %s and %d test rows to %s",
        len(train_df),
        train_path,
        len(test_df),
        test_path,
    )
    return train_df, test_df


def main() -> None:
    """CLI entry point for dataset validation and splitting."""
    parser = argparse.ArgumentParser(description="CareerFit ML — Dataset Validator & Splitter")
    parser.add_argument("--validate", type=str, help="Path to CSV dataset to validate.")
    parser.add_argument("--split", type=str, help="Path to full CSV dataset to split into train/test.")
    parser.add_argument("--train-out", type=str, default="data/ml_dataset/train.csv", help="Destination path for train.csv.")
    parser.add_argument("--test-out", type=str, default="data/ml_dataset/test.csv", help="Destination path for test.csv.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (default 0.2 = 80/20 split).")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed (default 42).")
    args = parser.parse_args()

    if args.validate:
        print(f"\n=== CareerFit ML: Validating Dataset: {args.validate} ===")
        try:
            df = pd.read_csv(args.validate)
            report = validate_dataset(df)
            print(f"Status:             {'VALID' if report.is_valid else 'INVALID'}")
            print(f"Total Rows:         {report.total_rows}")
            print(f"Valid Rows:         {report.valid_rows}")
            print(f"Positive Matches:   {report.positive_rows} ({report.positive_rows / max(report.valid_rows, 1) * 100:.1f}%)")
            print(f"Negative Matches:   {report.negative_rows} ({report.negative_rows / max(report.valid_rows, 1) * 100:.1f}%)")
            print(f"Balance Ratio:      {report.class_balance_ratio:.4f}")
            print(f"Duplicates:         {report.duplicate_count}")
            print(f"Conflicts:          {report.conflicting_count}")
            if report.category_counts:
                print("\nCategories:")
                for cat, count in report.category_counts.items():
                    print(f"  • {cat}: {count} rows")
            if report.warnings:
                print("\nWarnings:")
                for w in report.warnings:
                    print(f"  [WARN] {w}")
            if report.errors:
                print("\nErrors:")
                for e in report.errors:
                    print(f"  [ERROR] {e}")
                sys.exit(1)
            print("\nDataset successfully validated with 0 errors!")
        except Exception as exc:
            print(f"[FATAL ERROR] Could not validate dataset: {exc}")
            sys.exit(1)

    elif args.split:
        print(f"\n=== CareerFit ML: Splitting Dataset: {args.split} ===")
        try:
            df, stats = load_labeled_data(args.split)
            train_df, test_df = save_train_test_split(
                df,
                train_output_path=args.train_out,
                test_output_path=args.test_out,
                test_size=args.test_size,
                random_state=args.seed,
            )
            print(f"Original samples: {len(df)}")
            print(f"Train samples:    {len(train_df)} (Pos: {(train_df['label'] == 1).sum()}, Neg: {(train_df['label'] == 0).sum()}) -> {args.train_out}")
            print(f"Test samples:     {len(test_df)} (Pos: {(test_df['label'] == 1).sum()}, Neg: {(test_df['label'] == 0).sum()}) -> {args.test_out}")
            print("Stratified train/test split successfully created with zero data leakage!")
        except Exception as exc:
            print(f"[FATAL ERROR] Could not split dataset: {exc}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
