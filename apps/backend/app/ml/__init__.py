"""CareerFit ML: Machine Learning Package for Resume Matching and Skill Gap Analysis.

Provides preprocessing, feature extraction, text similarity calculation,
skill gap analysis, supervised classifier training, and performance evaluation.
"""

from typing import Any

__all__ = [
    "clean_text",
    "normalize_text",
    "tokenize_terms",
    "TfidfFeatureExtractor",
    "calculate_similarity",
    "MatchResult",
    "SKILL_TAXONOMY",
    "SKILL_ALIASES",
    "ALL_CANONICAL_SKILLS",
    "normalize_skill",
    "extract_skills",
    "analyze_skill_gap",
    "SkillGapResult",
    "load_labeled_data",
    "validate_dataset",
    "ValidationReport",
    "DatasetSplit",
    "DatasetStats",
    "save_train_test_split",
    "build_pipeline",
    "train_classifier",
    "save_pipeline",
    "load_pipeline",
    "InsufficientDataError",
    "evaluate_classifier",
    "EvaluationReport",
    "export_evaluation_report",
]


def __getattr__(name: str) -> Any:
    """Lazy-load submodules to avoid eager circular imports when executing via -m."""
    if name in ("clean_text", "normalize_text", "tokenize_terms"):
        from app.ml import preprocessor
        return getattr(preprocessor, name)
    if name == "TfidfFeatureExtractor":
        from app.ml import feature_extraction
        return getattr(feature_extraction, name)
    if name in ("calculate_similarity", "MatchResult"):
        from app.ml import matcher
        return getattr(matcher, name)
    if name in (
        "SKILL_TAXONOMY",
        "SKILL_ALIASES",
        "ALL_CANONICAL_SKILLS",
        "normalize_skill",
        "extract_skills",
        "analyze_skill_gap",
        "SkillGapResult",
    ):
        from app.ml import skill_gap
        return getattr(skill_gap, name)
    if name in (
        "load_labeled_data",
        "validate_dataset",
        "ValidationReport",
        "DatasetSplit",
        "DatasetStats",
        "save_train_test_split",
    ):
        from app.ml import data_loader
        return getattr(data_loader, name)
    if name in (
        "build_pipeline",
        "train_classifier",
        "save_pipeline",
        "load_pipeline",
        "InsufficientDataError",
    ):
        from app.ml import train
        return getattr(train, name)
    if name in ("evaluate_classifier", "EvaluationReport", "export_evaluation_report"):
        from app.ml import evaluate
        return getattr(evaluate, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
