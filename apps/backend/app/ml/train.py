"""Supervised model training module for CareerFit ML.

Trains a supervised Logistic Regression classifier using TF-IDF features
on labelled resume–job matching pairs and persists the fitted pipeline using joblib.
Supports k-fold cross-validation, metadata recording, and zero data leakage.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from app.ml.data_loader import load_labeled_data, prepare_pair_texts
from app.ml.feature_extraction import TECH_TOKEN_PATTERN
from app.ml.preprocessor import clean_text


logger = logging.getLogger("app.ml.train")

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ml_models"
DEFAULT_MODEL_FILE = DEFAULT_MODEL_DIR / "careerfit_logistic_regression.joblib"
DEFAULT_META_FILE = DEFAULT_MODEL_DIR / "training_metadata.json"

MIN_REQUIRED_SAMPLES = 4


class InsufficientDataError(ValueError):
    """Raised when training data does not meet minimum quality or sample size thresholds."""


def build_pipeline(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    c_regularization: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Construct an end-to-end scikit-learn Pipeline for resume-job classification.

    Args:
        max_features: Maximum TF-IDF vocabulary features.
        ngram_range: N-gram lower and upper range.
        c_regularization: Inverse regularization strength for Logistic Regression.
        random_state: Random state seed.

    Returns:
        Unfitted scikit-learn Pipeline instance.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
        token_pattern=TECH_TOKEN_PATTERN,
    )
    clf = LogisticRegression(
        C=c_regularization,
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )
    return Pipeline([("tfidf", tfidf), ("classifier", clf)])


def train_classifier(
    data: Union[str, Path, pd.DataFrame],
    output_model_path: Optional[Union[str, Path]] = None,
    max_features: int = 5000,
    c_regularization: float = 1.0,
    random_state: int = 42,
    cv_folds: int = 5,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train a supervised Logistic Regression classifier on labelled resume–job pairs.

    Args:
        data: Path to CSV dataset or an already-loaded pandas DataFrame.
        output_model_path: Destination path for saving the fitted joblib artifact.
        max_features: Maximum TF-IDF feature vocabulary size.
        c_regularization: Regularization parameter for Logistic Regression.
        random_state: Random seed for reproducibility.
        cv_folds: Number of stratified cross-validation folds on training set (0 to disable).

    Returns:
        Tuple of (fitted Pipeline, training metadata dict)

    Raises:
        InsufficientDataError: If dataset contains fewer than MIN_REQUIRED_SAMPLES
            or lacks representations of both positive (1) and negative (0) classes.
    """
    dataset_source = "DataFrame"
    if isinstance(data, (str, Path)):
        dataset_source = str(Path(data).name)
        df, stats = load_labeled_data(data, min_samples=MIN_REQUIRED_SAMPLES)
    else:
        df = data.copy()
        if len(df) < MIN_REQUIRED_SAMPLES:
            raise InsufficientDataError(
                f"Insufficient training samples: provided {len(df)} samples, but at least "
                f"{MIN_REQUIRED_SAMPLES} samples are required."
            )

    # Check class distribution
    unique_labels = set(df["label"].unique())
    if len(unique_labels) < 2:
        raise InsufficientDataError(
            f"Training failed: Dataset contains only one class ({unique_labels}). "
            f"Supervised binary classification requires both positive (1) and negative (0) examples. "
            f"Please add counter-examples to the dataset."
        )

    pos_samples = int((df["label"] == 1).sum())
    neg_samples = int((df["label"] == 0).sum())
    if pos_samples == 0 or neg_samples == 0:
        raise InsufficientDataError(
            f"Training failed: Both positive (found {pos_samples}) and negative "
            f"(found {neg_samples}) samples must be present."
        )

    X_texts = prepare_pair_texts(df)
    y_labels = df["label"].astype(int).tolist()

    pipeline = build_pipeline(
        max_features=max_features,
        c_regularization=c_regularization,
        random_state=random_state,
    )

    # Cross-validation within training split to assess variance without data leakage
    cv_results_summary: dict[str, Any] = {}
    min_class_count = min(pos_samples, neg_samples)
    actual_folds = min(cv_folds, min_class_count)

    if cv_folds >= 2 and actual_folds >= 2:
        logger.info("Running %d-fold stratified cross-validation on training data...", actual_folds)
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=random_state)
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }
        try:
            cv_scores = cross_validate(pipeline, X_texts, y_labels, cv=skf, scoring=scoring, n_jobs=1)
            cv_results_summary = {
                "folds": actual_folds,
                "accuracy_mean": round(float(np.mean(cv_scores["test_accuracy"])), 4),
                "accuracy_std": round(float(np.std(cv_scores["test_accuracy"])), 4),
                "precision_mean": round(float(np.mean(cv_scores["test_precision"])), 4),
                "precision_std": round(float(np.std(cv_scores["test_precision"])), 4),
                "recall_mean": round(float(np.mean(cv_scores["test_recall"])), 4),
                "recall_std": round(float(np.std(cv_scores["test_recall"])), 4),
                "f1_mean": round(float(np.mean(cv_scores["test_f1"])), 4),
                "f1_std": round(float(np.std(cv_scores["test_f1"])), 4),
                "roc_auc_mean": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
                "roc_auc_std": round(float(np.std(cv_scores["test_roc_auc"])), 4),
            }
        except Exception as exc:
            logger.warning("Cross-validation could not be completed: %s", exc)
            cv_results_summary = {"error": str(exc)}

    # Fit final model on all training data
    logger.info("Fitting TF-IDF and Logistic Regression on %d total training samples...", len(X_texts))
    pipeline.fit(X_texts, y_labels)

    # Gather feature count
    tfidf_step = pipeline.named_steps["tfidf"]
    vocab_size = len(tfidf_step.vocabulary_)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": dataset_source,
        "total_training_samples": len(X_texts),
        "positive_samples": pos_samples,
        "negative_samples": neg_samples,
        "vocabulary_size": vocab_size,
        "model_type": "LogisticRegression",
        "vectorizer_type": "TfidfVectorizer",
        "c_regularization": c_regularization,
        "max_features": max_features,
        "random_state": random_state,
        "cross_validation": cv_results_summary,
        "status": "trained_successfully",
    }

    # Save to disk if requested or default
    save_path = Path(output_model_path) if output_model_path else DEFAULT_MODEL_FILE
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info("Saved trained pipeline model to %s", save_path)

    # Save metadata JSON adjacent to model
    meta_path = save_path.parent / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return pipeline, metadata


def save_pipeline(pipeline: Pipeline, path: Union[str, Path]) -> None:
    """Save pipeline to disk using joblib."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, dest)


def load_pipeline(path: Union[str, Path]) -> Pipeline:
    """Load a previously saved model pipeline from disk."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Trained model not found at: {src.resolve()}")
    return joblib.load(src)


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="CareerFit ML — Train Resume-Job Classifier")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to CSV dataset containing 'resume_text', 'job_description', and 'label'.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(DEFAULT_MODEL_FILE),
        help="Path to save the fitted joblib model.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum vocabulary features for TF-IDF.",
    )
    parser.add_argument(
        "--c-regularization",
        type=float,
        default=1.0,
        help="Inverse regularization strength C (default 1.0).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds on training set (default 5, 0 to disable).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        pipeline, meta = train_classifier(
            data=args.data_path,
            output_model_path=args.output_model,
            max_features=args.max_features,
            c_regularization=args.c_regularization,
            random_state=args.random_state,
            cv_folds=args.cv_folds,
        )
        print("\n=== CareerFit ML: Training Succeeded ===")
        print(f"Model saved to:         {args.output_model}")
        print(f"Dataset source:         {meta['dataset_source']}")
        print(f"Training Samples:       {meta['total_training_samples']} (Pos: {meta['positive_samples']}, Neg: {meta['negative_samples']})")
        print(f"TF-IDF Vocabulary size: {meta['vocabulary_size']}")
        print(f"Regularization C:       {meta['c_regularization']}")
        print(f"Random State:           {meta['random_state']}")

        cv = meta.get("cross_validation", {})
        if cv and "accuracy_mean" in cv:
            print("\n--- 5-Fold Cross-Validation on Training Data ---")
            print(f"CV Accuracy:            {cv['accuracy_mean']:.4f} (+/- {cv['accuracy_std']:.4f})")
            print(f"CV Precision:           {cv['precision_mean']:.4f} (+/- {cv['precision_std']:.4f})")
            print(f"CV Recall:              {cv['recall_mean']:.4f} (+/- {cv['recall_std']:.4f})")
            print(f"CV F1-Score:            {cv['f1_mean']:.4f} (+/- {cv['f1_std']:.4f})")
            print(f"CV ROC-AUC:             {cv['roc_auc_mean']:.4f} (+/- {cv['roc_auc_std']:.4f})")
    except InsufficientDataError as e:
        print(f"\n[DATASET ERROR] Training aborted: {e}")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
