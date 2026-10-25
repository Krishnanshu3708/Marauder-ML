"""Unit tests for CareerFit ML skill gap analysis and alias normalization."""

import pytest
from app.ml.skill_gap import (
    analyze_skill_gap,
    extract_skills,
    normalize_skill,
    SkillGapResult,
)


class TestSkillNormalization:
    """Tests for skill alias normalization."""

    def test_alias_normalization(self):
        assert normalize_skill("js") == "javascript"
        assert normalize_skill("py") == "python"
        assert normalize_skill("python3") == "python"
        assert normalize_skill("ts") == "typescript"
        assert normalize_skill("k8s") == "kubernetes"
        assert normalize_skill("postgres") == "postgresql"
        assert normalize_skill("psql") == "postgresql"
        assert normalize_skill("react.js") == "react"
        assert normalize_skill("reactjs") == "react"
        assert normalize_skill("node.js") == "nodejs"
        assert normalize_skill("aws") == "aws"
        assert normalize_skill("amazon web services") == "aws"
        assert normalize_skill("gcp") == "gcp"
        assert normalize_skill("google cloud") == "gcp"
        assert normalize_skill("cpp") == "c++"
        assert normalize_skill("csharp") == "c#"
        assert normalize_skill("sklearn") == "scikit-learn"
        assert normalize_skill("ml") == "machine learning"

    def test_empty_and_none(self):
        assert normalize_skill("") == ""
        assert normalize_skill(None) == ""
        assert normalize_skill("   ") == ""


class TestSkillExtraction:
    """Tests for skill extraction from text."""

    def test_extract_skills_from_prose(self):
        text = (
            "Experienced in building microservices using Python and FastAPI. "
            "Proficient in Docker, K8s, PostgreSQL, and React.js on AWS."
        )
        skills = extract_skills(text)
        assert "python" in skills
        assert "fastapi" in skills
        assert "docker" in skills
        assert "kubernetes" in skills  # k8s normalized
        assert "postgresql" in skills
        assert "react" in skills       # react.js normalized
        assert "aws" in skills

    def test_whole_word_boundaries_prevent_false_positives(self):
        # 'go' should not match 'good' or 'ongoing'
        text = "Doing good work on an ongoing project."
        skills = extract_skills(text)
        assert "go" not in skills

    def test_empty_text_returns_empty_set(self):
        assert extract_skills("") == set()
        assert extract_skills(None) == set()


class TestSkillGapAnalysis:
    """Tests for analyze_skill_gap function."""

    def test_full_skill_match(self):
        resume = "Expertise in Python, FastAPI, Docker, and PostgreSQL."
        job = "Seeking Python developer with FastAPI, PostgreSQL, and Docker."
        result = analyze_skill_gap(resume, job)

        assert isinstance(result, SkillGapResult)
        assert set(result.matched_skills) == {"python", "fastapi", "docker", "postgresql"}
        assert result.missing_skills == []
        assert result.match_percentage == 100.0
        assert result.readiness_level == "High Fit"

    def test_skill_gap_identified(self):
        resume = "Frontend engineer with React, TypeScript, HTML, and CSS."
        job = "Looking for a full stack engineer proficient in React, TypeScript, Python, and PostgreSQL."
        result = analyze_skill_gap(resume, job)

        assert "react" in result.matched_skills
        assert "typescript" in result.matched_skills
        assert "python" in result.missing_skills
        assert "postgresql" in result.missing_skills
        assert "html" in result.additional_skills
        assert result.matched_count == 2
        assert result.missing_count == 2
        assert result.total_job_skills == 4
        assert result.match_percentage == 50.0
        assert result.readiness_level == "Moderate Fit"

    def test_list_inputs_supported(self):
        resume_skills = ["py", "Docker", "postgres", "Redis"]
        job_skills = ["Python", "Kubernetes", "PostgreSQL", "AWS"]
        result = analyze_skill_gap(resume_skills, job_skills)

        assert set(result.matched_skills) == {"python", "postgresql"}
        assert set(result.missing_skills) == {"kubernetes", "aws"}
        assert "docker" in result.additional_skills

    def test_empty_job_skills(self):
        result = analyze_skill_gap("Python developer", "No specific tech skills required here.")
        assert result.matched_skills == []
        assert result.missing_skills == []
        assert result.total_job_skills == 0
        assert result.match_percentage == 0.0
        assert result.readiness_level == "No Skills Specified"

    def test_to_dict_serialization(self):
        result = analyze_skill_gap("Python developer", "Python developer")
        data = result.to_dict()
        assert isinstance(data, dict)
        assert "matched_skills" in data
        assert "missing_skills" in data
        assert "match_percentage" in data
