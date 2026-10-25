"""Unit tests for CareerFit ML TF-IDF feature extraction."""

import pytest
from app.ml.feature_extraction import TfidfFeatureExtractor


class TestTfidfFeatureExtractor:
    """Tests for TfidfFeatureExtractor."""

    def test_fit_and_transform(self):
        corpus = [
            "Senior Python developer with experience in FastAPI and PostgreSQL.",
            "Frontend React developer with skills in TypeScript and CSS.",
            "DevOps engineer skilled in Docker, Kubernetes, and AWS cloud.",
        ]
        extractor = TfidfFeatureExtractor(max_features=100)
        assert not extractor.is_fitted

        extractor.fit(corpus)
        assert extractor.is_fitted
        assert extractor.vocabulary_size > 0

        feature_names = extractor.get_feature_names()
        assert "python" in feature_names
        assert "react" in feature_names

        matrix = extractor.transform(["Python and FastAPI engineer"])
        assert matrix.shape[0] == 1
        assert matrix.shape[1] == len(feature_names)
        assert matrix.nnz > 0

    def test_fit_transform_shortcut(self):
        corpus = [
            "Data scientist working with PyTorch and Pandas.",
            "Backend engineer building microservices with Go and Docker.",
        ]
        extractor = TfidfFeatureExtractor()
        matrix = extractor.fit_transform(corpus)
        assert matrix.shape[0] == 2
        assert extractor.is_fitted

    def test_transform_before_fit_raises_error(self):
        extractor = TfidfFeatureExtractor()
        with pytest.raises(RuntimeError, match="must be fitted before"):
            extractor.transform(["Sample document"])

    def test_fit_empty_corpus_raises_error(self):
        extractor = TfidfFeatureExtractor()
        with pytest.raises(ValueError, match="Cannot fit TF-IDF vectorizer on an empty"):
            extractor.fit(["", "   ", None])

    def test_get_feature_names_unfitted_returns_empty(self):
        extractor = TfidfFeatureExtractor()
        assert extractor.get_feature_names() == []
        assert extractor.vocabulary_size == 0
