"""Pydantic schemas for Marauder ML API endpoints."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class MatchAnalysisRequest(BaseModel):
    """Request payload for Marauder ML resume and job analysis."""

    resume_text: str = Field(
        ...,
        description="Raw or plain text content of candidate resume.",
        min_length=1,
    )
    job_description: str = Field(
        ...,
        description="Raw or plain text content of target job description.",
        min_length=1,
    )
    model_path: Optional[str] = Field(
        None,
        description="Optional custom path to a trained joblib model artifact.",
    )


class ModelPredictionDetail(BaseModel):
    """Details from the supervised machine learning model prediction."""

    predicted_label: int = Field(..., description="Binary prediction: 1 (Match) or 0 (Mismatch).")
    is_match: bool = Field(..., description="True if model classifies candidate as a qualified match.")
    match_probability: float = Field(..., description="Estimated probability (0.0 to 1.0) of positive class match.")
    class_probabilities: dict[str, float] = Field(
        ...,
        description="Probability distribution across classes (e.g. {'0': 0.15, '1': 0.85}).",
    )
    model_type: str = Field("LogisticRegression", description="Classification algorithm used.")


class MatchAnalysisResponse(BaseModel):
    """Comprehensive analysis report from Marauder ML."""

    similarity_score: float = Field(
        ...,
        description="TF-IDF Cosine Similarity percentage score (0.0 to 100.0).",
    )
    raw_cosine: float = Field(
        ...,
        description="Unscaled Cosine Similarity metric (0.0 to 1.0).",
    )
    match_category: str = Field(
        ...,
        description="Interpretable fit tier: 'High', 'Moderate', 'Low', 'Minimal', or 'None'.",
    )
    top_shared_terms: list[str] = Field(
        default_factory=list,
        description="Top informative vocabulary terms overlapping between resume and job.",
    )
    matched_skills: list[str] = Field(
        default_factory=list,
        description="Technical skills found in both candidate resume and job requirements.",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="Technical skills required by job that are absent from the resume.",
    )
    additional_skills: list[str] = Field(
        default_factory=list,
        description="Technical skills candidate possesses that were not explicitly listed in job.",
    )
    skill_match_percentage: float = Field(
        ...,
        description="Percentage of detected job skills satisfied by candidate (0.0 to 100.0).",
    )
    readiness_level: str = Field(
        ...,
        description="Skill readiness status: 'High Fit', 'Moderate Fit', 'Skill Gap Identified', or 'No Skills Specified'.",
    )
    prediction: Optional[ModelPredictionDetail] = Field(
        None,
        description="Supervised ML prediction details if a trained model is available.",
    )
    model_status: str = Field(
        ...,
        description="Status of the supervised model: 'model_loaded', 'model_unavailable', or 'model_error'.",
    )
    notes: Optional[str] = Field(
        None,
        description="Operational context or diagnostic note (e.g., instructions on model training).",
    )
