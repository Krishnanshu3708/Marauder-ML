"""Unit tests for CareerFit ML dataset loading, validation, training, and evaluation."""

import json
from pathlib import Path
import pandas as pd
import pytest

from app.ml.data_loader import (
    load_labeled_data,
    get_train_test_split,
    save_train_test_split,
    validate_dataset,
    ValidationReport,
)
from app.ml.evaluate import (
    evaluate_classifier,
    export_evaluation_report,
    EvaluationReport,
)
from app.ml.train import (
    InsufficientDataError,
    load_pipeline,
    save_pipeline,
    train_classifier,
)


@pytest.fixture
def sample_dataset_df():
    """Create a minimal valid DataFrame with both positive and negative classes."""
    data = [
        {"resume_text": "Python FastAPI backend engineer", "job_description": "Seeking Python backend developer", "label": 1, "category": "Backend"},
        {"resume_text": "Experienced Python and Docker developer", "job_description": "Hiring Python backend engineer", "label": 1, "category": "Backend"},
        {"resume_text": "React TypeScript UI developer", "job_description": "Seeking React frontend engineer", "label": 1, "category": "Frontend"},
        {"resume_text": "Corporate accountant balance sheet audit", "job_description": "Seeking Python backend developer", "label": 0, "category": "Accounting"},
        {"resume_text": "High school chemistry educator", "job_description": "Hiring Python backend engineer", "label": 0, "category": "Education"},
        {"resume_text": "Executive chef pastry kitchen lead", "job_description": "Seeking React frontend engineer", "label": 0, "category": "Culinary"},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_path(sample_dataset_df, tmp_path):
    """Save sample DataFrame to temporary CSV."""
    csv_file = tmp_path / "temp_train.csv"
    sample_dataset_df.to_csv(csv_file, index=False)
    return csv_file


class TestDataLoader:
    """Tests for load_labeled_data, validate_dataset, and get_train_test_split."""

    def test_load_valid_csv(self, sample_csv_path):
        df, stats = load_labeled_data(sample_csv_path)
        assert len(df) == 6
        assert stats.total_samples == 6
        assert stats.positive_samples == 3
        assert stats.negative_samples == 3
        assert stats.is_balanced is True

    def test_missing_file_raises_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_labeled_data("non_existent_file_path_12345.csv")

    def test_missing_columns_raises_error(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_df = pd.DataFrame({"resume_text": ["text"], "label": [1]})
        bad_df.to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_labeled_data(bad_csv)

    def test_invalid_label_raises_error(self, tmp_path):
        bad_csv = tmp_path / "invalid_label.csv"
        bad_df = pd.DataFrame({
            "resume_text": ["r1", "r2"],
            "job_description": ["j1", "j2"],
            "label": [1, 99],  # 99 is invalid
        })
        bad_df.to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="validation failed"):
            load_labeled_data(bad_csv)

    def test_get_train_test_split(self, sample_dataset_df):
        split = get_train_test_split(sample_dataset_df, test_size=0.33, random_state=42)
        assert len(split.X_train) + len(split.X_test) == len(sample_dataset_df)
        assert len(split.y_train) + len(split.y_test) == len(sample_dataset_df)

    def test_validate_dataset_detects_empty_text(self):
        bad_df = pd.DataFrame({
            "resume_text": ["  ", "valid resume"],
            "job_description": ["valid job", ""],
            "label": [1, 0],
        })
        report = validate_dataset(bad_df)
        assert report.is_valid is False
        assert report.empty_text_rows == 2
        assert any("empty or whitespace-only" in err for err in report.errors)

    def test_validate_dataset_detects_conflicting_labels(self):
        conflict_df = pd.DataFrame({
            "resume_text": ["Identical Resume Text", "Identical Resume Text"],
            "job_description": ["Identical Job Description", "Identical Job Description"],
            "label": [1, 0],  # Contradictory labels for identical text pair
        })
        report = validate_dataset(conflict_df)
        assert report.is_valid is False
        assert report.conflicting_count == 1
        assert any("conflicting examples" in err for err in report.errors)

    def test_validate_dataset_identifies_duplicates(self, sample_dataset_df):
        dup_df = pd.concat([sample_dataset_df, sample_dataset_df.iloc[[0]]], ignore_index=True)
        report = validate_dataset(dup_df)
        assert report.duplicate_count == 1
        assert any("exact duplicate" in warn for warn in report.warnings)

    def test_save_train_test_split_ensures_zero_leakage_and_stratification(self, sample_dataset_df, tmp_path):
        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"

        train_df, test_df = save_train_test_split(
            sample_dataset_df,
            train_output_path=train_path,
            test_output_path=test_path,
            test_size=0.33,
            random_state=42,
        )

        assert len(train_df) == 4
        assert len(test_df) == 2
        assert train_path.exists()
        assert test_path.exists()

        # Check no overlap between train and test
        train_pairs = set(zip(train_df["resume_text"], train_df["job_description"]))
        test_pairs = set(zip(test_df["resume_text"], test_df["job_description"]))
        assert len(train_pairs.intersection(test_pairs)) == 0


class TestTraining:
    """Tests for model training and persistence."""

    def test_train_and_predict(self, sample_dataset_df, tmp_path):
        model_path = tmp_path / "model.joblib"
        pipeline, meta = train_classifier(
            data=sample_dataset_df,
            output_model_path=model_path,
            cv_folds=2,
        )
        assert meta["status"] == "trained_successfully"
        assert meta["total_training_samples"] == 6
        assert model_path.exists()

        # Check prediction works
        test_pair = ["RESUME: Python FastAPI engineer\n\nJOB: Looking for Python developer"]
        pred = pipeline.predict(test_pair)
        assert pred[0] in (0, 1)

        # Check predict_proba works
        probs = pipeline.predict_proba(test_pair)
        assert probs.shape == (1, 2)
        assert abs(sum(probs[0]) - 1.0) < 1e-5

        # Check CV stats recorded
        assert "cross_validation" in meta
        assert "accuracy_mean" in meta["cross_validation"]

    def test_save_and_load_pipeline(self, sample_dataset_df, tmp_path):
        model_path = tmp_path / "persisted_model.joblib"
        pipeline, _ = train_classifier(sample_dataset_df, output_model_path=model_path, cv_folds=0)

        loaded = load_pipeline(model_path)
        test_pair = ["RESUME: React engineer\n\nJOB: Looking for React engineer"]
        assert pipeline.predict(test_pair) == loaded.predict(test_pair)

    def test_single_class_training_aborts_with_error(self):
        single_class_df = pd.DataFrame({
            "resume_text": ["Python dev", "React dev", "FastAPI dev", "AWS dev"],
            "job_description": ["Python job", "React job", "FastAPI job", "AWS job"],
            "label": [1, 1, 1, 1],  # Only class 1
        })
        with pytest.raises(InsufficientDataError, match="Dataset contains only one class"):
            train_classifier(single_class_df)

    def test_insufficient_samples_aborts_with_error(self):
        tiny_df = pd.DataFrame({
            "resume_text": ["Python dev"],
            "job_description": ["Python job"],
            "label": [1],
        })
        with pytest.raises(InsufficientDataError, match="Insufficient training samples"):
            train_classifier(tiny_df)


class TestEvaluation:
    """Tests for model evaluation metrics, baseline comparison, and report export."""

    def test_evaluate_with_two_classes_computes_roc_auc_and_baseline(self, sample_dataset_df, tmp_path):
        model_path = tmp_path / "eval_model.joblib"
        pipeline, _ = train_classifier(sample_dataset_df, output_model_path=model_path, cv_folds=0)

        test_df = pd.DataFrame([
            {"resume_text": "Python and FastAPI engineer", "job_description": "Python backend role", "label": 1},
            {"resume_text": "Corporate accountant", "job_description": "Python backend role", "label": 0},
        ])

        report = evaluate_classifier(model=pipeline, test_data=test_df, train_data=sample_dataset_df)
        assert isinstance(report, EvaluationReport)
        assert report.total_test_samples == 2
        assert 0.0 <= report.accuracy <= 1.0
        assert report.confusion_matrix is not None
        assert len(report.confusion_matrix) == 2  # 2x2
        assert "true_negatives" in report.confusion_matrix_dict
        assert "true_positives" in report.confusion_matrix_dict

        # ROC-AUC supported
        assert report.roc_auc is not None
        assert 0.0 <= report.roc_auc <= 1.0
        assert "Calculated successfully" in report.roc_auc_status

        # Baseline comparison
        assert report.baseline_metrics is not None
        assert "accuracy" in report.baseline_metrics
        assert report.lift_over_baseline is not None
        assert "accuracy_lift" in report.lift_over_baseline

    def test_evaluate_with_single_class_sets_roc_auc_none(self, sample_dataset_df, tmp_path):
        model_path = tmp_path / "eval_model.joblib"
        pipeline, _ = train_classifier(sample_dataset_df, output_model_path=model_path, cv_folds=0)

        # Test set contains ONLY class 1
        test_df = pd.DataFrame([
            {"resume_text": "Python engineer", "job_description": "Python role", "label": 1},
            {"resume_text": "React developer", "job_description": "React role", "label": 1},
        ])

        report = evaluate_classifier(model=pipeline, test_data=test_df, include_baseline=False)
        assert report.roc_auc is None
        assert "ROC-AUC is undefined because the test set contains only one class" in report.roc_auc_status

    def test_export_evaluation_report_json_and_markdown(self, sample_dataset_df, tmp_path):
        model_path = tmp_path / "eval_model.joblib"
        pipeline, _ = train_classifier(sample_dataset_df, output_model_path=model_path, cv_folds=0)

        test_df = pd.DataFrame([
            {"resume_text": "Python and FastAPI engineer", "job_description": "Python backend role", "label": 1},
            {"resume_text": "Corporate accountant", "job_description": "Python backend role", "label": 0},
        ])

        report = evaluate_classifier(model=pipeline, test_data=test_df, train_data=sample_dataset_df)
        out_dir = tmp_path / "eval_output"
        json_path, md_path = export_evaluation_report(report, out_dir, dataset_name="unit_test.csv")

        assert json_path.exists()
        assert md_path.exists()

        # Check JSON contents
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["accuracy"] == report.accuracy
            assert "confusion_matrix_dict" in data

        # Check Markdown contents
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "# CareerFit Logistic Regression — Evaluation Report" in content
            assert "Quantitative Performance Summary" in content
            assert "Confusion Matrix" in content
