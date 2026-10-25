"""Unit tests for CareerFit ML Cosine Similarity matcher."""

import pytest
from app.ml.matcher import calculate_similarity, MatchResult


class TestMatcher:
    """Tests for calculate_similarity and MatchResult."""

    def test_identical_texts_near_100(self):
        text = "Senior Python developer experienced in FastAPI, Docker, and PostgreSQL databases."
        result = calculate_similarity(text, text)
        assert isinstance(result, MatchResult)
        assert result.similarity_score >= 99.0
        assert result.raw_cosine >= 0.99
        assert result.is_strong_match is True
        assert result.match_level == "High"
        assert len(result.shared_terms) > 0

    def test_orthogonal_texts_low_similarity(self):
        resume = "Senior Certified Public Accountant specializing in corporate tax filings and GAAP ledger auditing."
        job = "Seeking Senior DevOps Engineer with deep expertise in Kubernetes, Terraform, and AWS cloud networking."
        result = calculate_similarity(resume, job)
        assert result.similarity_score < 20.0
        assert result.match_level in ("Low", "Minimal")
        assert result.is_strong_match is False

    def test_partial_match(self):
        resume = (
            "Python backend developer with experience building scalable REST APIs, "
            "PostgreSQL databases, Docker containers, and Git workflows."
        )
        job = (
            "We are hiring a Python backend developer to build REST APIs using "
            "PostgreSQL databases and Docker containers."
        )
        result = calculate_similarity(resume, job)
        assert result.similarity_score >= 30.0
        assert "python" in result.shared_terms
        assert "postgresql" in result.shared_terms

    def test_empty_resume_returns_zero(self):
        result = calculate_similarity("", "Seeking Python developer with AWS experience.")
        assert result.similarity_score == 0.0
        assert result.raw_cosine == 0.0
        assert result.shared_terms == []
        assert result.match_level == "None"
        assert "Resume text is empty" in result.message

    def test_empty_job_description_returns_zero(self):
        result = calculate_similarity("Experienced React developer", "")
        assert result.similarity_score == 0.0
        assert result.raw_cosine == 0.0
        assert result.shared_terms == []
        assert result.match_level == "None"
        assert "Job description text is empty" in result.message

    def test_both_empty_returns_zero(self):
        result = calculate_similarity(None, None)
        assert result.similarity_score == 0.0
        assert result.raw_cosine == 0.0
        assert result.shared_terms == []
        assert result.match_level == "None"

    def test_to_dict_serialization(self):
        result = calculate_similarity("Python developer", "Python developer")
        data = result.to_dict()
        assert isinstance(data, dict)
        assert "similarity_score" in data
        assert "raw_cosine" in data
        assert "shared_terms" in data
        assert "match_level" in data
