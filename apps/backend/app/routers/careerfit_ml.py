"""FastAPI router for Marauder ML — Resume Matching and Skill Gap Analysis.

Provides dedicated endpoints for TF-IDF Cosine Similarity, skill gap extraction,
and supervised Logistic Regression match classification.
Runs completely locally with zero external LLM calls and zero personal data logging.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
from fastapi import APIRouter, HTTPException, status
from sklearn.pipeline import Pipeline

from app.ml.data_loader import prepare_pair_texts
from app.ml.matcher import calculate_similarity
from app.ml.preprocessor import clean_text
from app.ml.skill_gap import analyze_skill_gap
from app.ml.train import DEFAULT_MODEL_FILE, load_pipeline
from app.schemas.careerfit_ml import (
    MatchAnalysisRequest,
    MatchAnalysisResponse,
    ModelPredictionDetail,
)

logger = logging.getLogger("app.routers.careerfit_ml")
router = APIRouter(prefix="/ml", tags=["Marauder ML"])

# In-memory cache for the loaded model pipeline: (model_path_str, last_modified_time, pipeline)
_MODEL_CACHE: dict[str, Tuple[float, Pipeline]] = {}


def get_or_load_model(model_path: Optional[str] = None) -> Tuple[Optional[Pipeline], str, Optional[str]]:
    """Safely retrieve or lazily load the trained classifier pipeline from disk.

    Avoids retraining during requests by maintaining an in-memory cached instance
    that reloads only if the underlying file changes.

    Args:
        model_path: Optional explicit model path. Defaults to DEFAULT_MODEL_FILE.

    Returns:
        Tuple of (loaded_pipeline_or_None, status_code, diagnostic_notes)
    """
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_FILE

    if not target_path.is_file():
        return (
            None,
            "model_unavailable",
            f"Supervised model artifact not found at '{target_path.name}'. "
            f"Run offline training via CLI: 'python -m app.ml.train --data-path <dataset.csv>'",
        )

    try:
        current_mtime = target_path.stat().st_mtime
        cache_key = str(target_path.resolve())

        # Check if already in cache and not modified
        if cache_key in _MODEL_CACHE:
            cached_mtime, cached_pipeline = _MODEL_CACHE[cache_key]
            if cached_mtime == current_mtime:
                return cached_pipeline, "model_loaded", None

        # Load from disk
        logger.info("Loading CareerFit ML model artifact from disk: %s", target_path)
        pipeline = load_pipeline(target_path)
        _MODEL_CACHE[cache_key] = (current_mtime, pipeline)
        return pipeline, "model_loaded", None
    except Exception as exc:
        logger.warning("Failed to load CareerFit ML model artifact: %s", exc)
        return (
            None,
            "model_error",
            f"Error loading model artifact: {exc}. The model file may be corrupted.",
        )


@router.post(
    "/match-analysis",
    response_model=MatchAnalysisResponse,
    summary="Analyze resume-job match and skill gaps using CareerFit ML",
)
async def analyze_resume_job_match(request: MatchAnalysisRequest) -> MatchAnalysisResponse:
    """Analyze resume-job text similarity, skill gaps, and ML classification fit.

    Validates inputs, calculates TF-IDF cosine similarity, identifies matched and
    missing skills, and executes the supervised classification pipeline if a fitted
    model artifact is available.
    """
    # 1. Input Validation
    resume_cleaned = clean_text(request.resume_text)
    if not resume_cleaned:
        raise HTTPException(
            status_code=422,
            detail="Resume text cannot be empty, whitespace-only, or contain only punctuation.",
        )

    jd_cleaned = clean_text(request.job_description)
    if not jd_cleaned:
        raise HTTPException(
            status_code=422,
            detail="Job description cannot be empty, whitespace-only, or contain only punctuation.",
        )

    # Privacy confirmation: Log request metadata only, never raw resume text
    logger.info(
        "CareerFit ML analysis requested (resume chars: %d, job chars: %d)",
        len(request.resume_text),
        len(request.job_description),
    )

    # 2. Text Similarity (TF-IDF + Cosine Similarity)
    similarity_result = calculate_similarity(
        resume_text=resume_cleaned,
        job_description=jd_cleaned,
    )

    # 3. Skill Gap Analysis (Canonical taxonomy + alias normalization)
    skill_result = analyze_skill_gap(
        resume_input=resume_cleaned,
        job_input=jd_cleaned,
    )

    # 4. Supervised Model Classification
    pipeline, model_status, model_notes = get_or_load_model(request.model_path)
    prediction_detail: Optional[ModelPredictionDetail] = None

    if pipeline is not None:
        try:
            pair_input = [f"RESUME: {resume_cleaned}\n\nJOB: {jd_cleaned}"]
            pred_label = int(pipeline.predict(pair_input)[0])
            is_match = pred_label == 1

            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(pair_input)[0]
                # Classes in scikit-learn LogisticRegression
                classes = [str(c) for c in pipeline.classes_]
                class_probs = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)}
                match_prob = class_probs.get("1", round(float(probs[-1]), 4))
            else:
                match_prob = 1.0 if is_match else 0.0
                class_probs = {"1": match_prob, "0": round(1.0 - match_prob, 4)}

            prediction_detail = ModelPredictionDetail(
                predicted_label=pred_label,
                is_match=is_match,
                match_probability=round(float(match_prob), 4),
                class_probabilities=class_probs,
                model_type="LogisticRegression",
            )
        except Exception as exc:
            logger.warning("Prediction inference failed: %s", exc)
            model_status = "model_error"
            model_notes = f"Inference failed on loaded model: {exc}"

    return MatchAnalysisResponse(
        similarity_score=similarity_result.similarity_score,
        raw_cosine=similarity_result.raw_cosine,
        match_category=similarity_result.match_level,
        top_shared_terms=similarity_result.shared_terms,
        matched_skills=skill_result.matched_skills,
        missing_skills=skill_result.missing_skills,
        additional_skills=skill_result.additional_skills,
        skill_match_percentage=skill_result.match_percentage,
        readiness_level=skill_result.readiness_level,
        prediction=prediction_detail,
        model_status=model_status,
        notes=model_notes,
    )


@router.get("/status", summary="Get CareerFit ML model status")
async def get_model_status() -> dict:
    """Return status of the trained CareerFit ML model artifact."""
    target_path = DEFAULT_MODEL_FILE
    meta_path = DEFAULT_MODEL_FILE.parent / "training_metadata.json"

    is_available = target_path.is_file()
    metadata = None
    if meta_path.is_file():
        import json
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    return {
        "model_available": is_available,
        "model_path": str(target_path.resolve()),
        "model_type": "LogisticRegression (TF-IDF)",
        "metadata": metadata,
    }
