"""Resume matching module using TF-IDF and Cosine Similarity for CareerFit ML.

Calculates quantitative, interpretable similarity scores between resume text
and job descriptions without calling external LLM APIs.
"""

from dataclasses import asdict, dataclass
from typing import Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.feature_extraction import TECH_TOKEN_PATTERN
from app.ml.preprocessor import clean_text


@dataclass
class MatchResult:
    """Interpretable resume-to-job matching result."""

    similarity_score: float       # Percentage score: 0.0 to 100.0
    raw_cosine: float             # Exact cosine similarity: 0.0 to 1.0
    shared_terms: list[str]       # Top terms occurring in both documents
    shared_term_count: int        # Number of overlapping vocabulary terms
    is_strong_match: bool         # Boolean flag (score >= 60.0)
    match_level: str              # Categorical fit: High, Moderate, Low, or None
    message: str                  # Diagnostic message or summary

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a standard dictionary."""
        return asdict(self)


def calculate_similarity(
    resume_text: Optional[str],
    job_description: Optional[str],
    vectorizer: Optional[TfidfVectorizer] = None,
    top_n_terms: int = 10,
) -> MatchResult:
    """Calculate cosine similarity and shared features between resume and job description.

    Args:
        resume_text: Raw or preprocessed text of the resume.
        job_description: Raw or preprocessed text of the job description.
        vectorizer: Optional pre-fitted TfidfVectorizer. If None, a vectorizer
            is dynamically fit on the pair.
        top_n_terms: Maximum number of top shared keywords to return.

    Returns:
        MatchResult with similarity score (0-100), raw cosine (0-1),
        shared terms, and match categorization.
    """
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(job_description)

    # Handle empty input cases cleanly
    if not clean_resume and not clean_jd:
        return MatchResult(
            similarity_score=0.0,
            raw_cosine=0.0,
            shared_terms=[],
            shared_term_count=0,
            is_strong_match=False,
            match_level="None",
            message="Both resume text and job description are empty.",
        )
    if not clean_resume:
        return MatchResult(
            similarity_score=0.0,
            raw_cosine=0.0,
            shared_terms=[],
            shared_term_count=0,
            is_strong_match=False,
            match_level="None",
            message="Resume text is empty.",
        )
    if not clean_jd:
        return MatchResult(
            similarity_score=0.0,
            raw_cosine=0.0,
            shared_terms=[],
            shared_term_count=0,
            is_strong_match=False,
            match_level="None",
            message="Job description text is empty.",
        )

    # Use existing or dynamic vectorizer
    if vectorizer is None:
        local_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            token_pattern=TECH_TOKEN_PATTERN,
        )
        tfidf_matrix = local_vec.fit_transform([clean_resume, clean_jd])
        resume_vec = tfidf_matrix[0]
        jd_vec = tfidf_matrix[1]
        feature_names = local_vec.get_feature_names_out()
    else:
        resume_vec = vectorizer.transform([clean_resume])
        jd_vec = vectorizer.transform([clean_jd])
        feature_names = vectorizer.get_feature_names_out()

    # Calculate Cosine Similarity: dot(u, v) / (||u|| * ||v||)
    cosine_val = float(cosine_similarity(resume_vec, jd_vec)[0][0])
    # Guard against float rounding quirks
    cosine_clamped = max(0.0, min(1.0, cosine_val))
    similarity_score = round(cosine_clamped * 100.0, 2)

    # Extract shared terms based on element-wise product of TF-IDF weights
    # (terms that carry high combined weight in both documents)
    resume_dense = resume_vec.toarray()[0]
    jd_dense = jd_vec.toarray()[0]
    overlap_weights = resume_dense * jd_dense

    shared_indices = np.where(overlap_weights > 0)[0]
    shared_term_count = int(len(shared_indices))

    # Sort shared indices by combined TF-IDF weight descending
    if shared_term_count > 0:
        top_indices = shared_indices[np.argsort(overlap_weights[shared_indices])[::-1]]
        top_terms = [str(feature_names[idx]) for idx in top_indices[:top_n_terms]]
    else:
        top_terms = []

    # Assign match category
    if similarity_score >= 70.0:
        match_level = "High"
        is_strong = True
    elif similarity_score >= 45.0:
        match_level = "Moderate"
        is_strong = False
    elif similarity_score >= 20.0:
        match_level = "Low"
        is_strong = False
    else:
        match_level = "Minimal"
        is_strong = False

    return MatchResult(
        similarity_score=similarity_score,
        raw_cosine=round(cosine_clamped, 4),
        shared_terms=top_terms,
        shared_term_count=shared_term_count,
        is_strong_match=is_strong,
        match_level=match_level,
        message=f"{match_level} match ({similarity_score}%) with {shared_term_count} overlapping terms.",
    )
