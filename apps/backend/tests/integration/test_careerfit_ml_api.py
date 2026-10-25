"""Integration tests for CareerFit ML API endpoints (/api/v1/ml/...)."""

import logging
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    """Async HTTP test client bound directly to the FastAPI app via ASGI transport."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_successful_match_analysis(client: AsyncClient):
    """Test standard successful analysis with valid inputs."""
    payload = {
        "resume_text": "Experienced Python developer with expertise in FastAPI, PostgreSQL, and Docker.",
        "job_description": "We are seeking a Python engineer skilled in FastAPI, Docker, and Kubernetes.",
    }
    response = await client.post("/api/v1/ml/match-analysis", json=payload)
    assert response.status_code == 200

    data = response.json()
    # Check similarity fields
    assert "similarity_score" in data
    assert 0.0 <= data["similarity_score"] <= 100.0
    assert "raw_cosine" in data
    assert 0.0 <= data["raw_cosine"] <= 1.0
    assert data["match_category"] in ("High", "Moderate", "Low", "Minimal")

    # Check skill gap fields
    assert "matched_skills" in data
    assert "fastapi" in data["matched_skills"]
    assert "python" in data["matched_skills"]
    assert "docker" in data["matched_skills"]
    assert "kubernetes" in data["missing_skills"]
    assert "postgresql" in data["additional_skills"]
    assert 0.0 <= data["skill_match_percentage"] <= 100.0
    assert "readiness_level" in data

    # Check model status
    assert "model_status" in data
    assert data["model_status"] in ("model_loaded", "model_unavailable")
    if data["model_status"] == "model_loaded":
        assert data["prediction"] is not None
        assert data["prediction"]["predicted_label"] in (0, 1)
        assert 0.0 <= data["prediction"]["match_probability"] <= 1.0


@pytest.mark.asyncio
async def test_empty_resume_returns_422(client: AsyncClient):
    """Test validation rejection for empty or whitespace resume text."""
    payload = {
        "resume_text": "    ",
        "job_description": "Python backend role",
    }
    response = await client.post("/api/v1/ml/match-analysis", json=payload)
    assert response.status_code == 422
    assert "Resume text cannot be empty" in response.json()["detail"]


@pytest.mark.asyncio
async def test_empty_job_description_returns_422(client: AsyncClient):
    """Test validation rejection for empty or whitespace job description."""
    payload = {
        "resume_text": "Python developer",
        "job_description": "   ",
    }
    response = await client.post("/api/v1/ml/match-analysis", json=payload)
    assert response.status_code == 422
    assert "Job description cannot be empty" in response.json()["detail"]


@pytest.mark.asyncio
async def test_model_unavailable_graceful_fallback(client: AsyncClient):
    """Test graceful fallback when specified model artifact does not exist."""
    payload = {
        "resume_text": "Senior React and TypeScript developer",
        "job_description": "Frontend engineer needed with React expertise",
        "model_path": "non_existent_dir/missing_model.joblib",
    }
    response = await client.post("/api/v1/ml/match-analysis", json=payload)
    assert response.status_code == 200

    data = response.json()
    # Text similarity and skill gaps still work accurately
    assert data["similarity_score"] > 0.0
    assert "react" in data["matched_skills"]
    # Model is reported as unavailable without crashing
    assert data["model_status"] == "model_unavailable"
    assert data["prediction"] is None
    assert "not found" in data["notes"].lower()


@pytest.mark.asyncio
async def test_model_status_endpoint(client: AsyncClient):
    """Test GET /api/v1/ml/status endpoint."""
    response = await client.get("/api/v1/ml/status")
    assert response.status_code == 200
    data = response.json()
    assert "model_available" in data
    assert "model_path" in data
    assert "model_type" in data


@pytest.mark.asyncio
async def test_privacy_no_personal_data_logged(client: AsyncClient, caplog):
    """Confirm that raw resume text is not written into application logs."""
    sensitive_name = "Jonathan-SuperSecretCandidate-98765"
    payload = {
        "resume_text": f"Name: {sensitive_name}. Python FastAPI developer with 5 years experience.",
        "job_description": "Seeking Python FastAPI developer.",
    }
    with caplog.at_level(logging.INFO):
        response = await client.post("/api/v1/ml/match-analysis", json=payload)
        assert response.status_code == 200

    # Ensure sensitive resume content is absent from all logged records
    for record in caplog.records:
        assert sensitive_name not in record.message
