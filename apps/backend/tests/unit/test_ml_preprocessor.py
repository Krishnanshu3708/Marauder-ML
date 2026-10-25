"""Unit tests for CareerFit ML text preprocessing."""

import pytest
from app.ml.preprocessor import clean_text, normalize_text, tokenize_terms


class TestCleanText:
    """Tests for clean_text function."""

    def test_normal_text_cleaned(self):
        raw = "Software Engineer with Experience in Python and SQL."
        expected = "software engineer with experience in python and sql"
        assert clean_text(raw) == expected

    def test_case_normalization(self):
        assert clean_text("FASTAPI and DJANGO") == "fastapi and django"

    def test_whitespace_normalization(self):
        raw = "  Python   \t\n  Developer \n\n with   Docker   "
        assert clean_text(raw) == "python developer with docker"

    def test_empty_and_whitespace_inputs(self):
        assert clean_text("") == ""
        assert clean_text("   ") == ""
        assert clean_text("\n\t  \r") == ""

    def test_none_and_non_string_inputs(self):
        assert clean_text(None) == ""
        assert clean_text(12345) == ""
        assert clean_text(["python"]) == ""

    def test_url_removal(self):
        raw = "Check out my portfolio at https://github.com/example/repo and www.example.com"
        cleaned = clean_text(raw)
        assert "http" not in cleaned
        assert "github.com" not in cleaned
        assert "www.example.com" not in cleaned
        assert "portfolio at and" in cleaned

    def test_email_removal_privacy(self):
        raw = "Contact candidate at john.doe@email.com or hr@company.org for inquiries."
        cleaned = clean_text(raw)
        assert "john.doe@email.com" not in cleaned
        assert "hr@company.org" not in cleaned
        assert "contact candidate at or for inquiries" in cleaned

    def test_tech_terms_preserved(self):
        raw = "Proficient in C++, C#, .NET core, and Node.js backend development."
        cleaned = clean_text(raw)
        assert "c++" in cleaned
        assert "c#" in cleaned
        assert ".net" in cleaned
        assert "node.js" in cleaned

    def test_markdown_bullet_points_cleaned(self):
        raw = "• Developed REST API\n* Managed Kubernetes clusters\n- Wrote unit tests"
        cleaned = clean_text(raw)
        assert cleaned == "developed rest api managed kubernetes clusters wrote unit tests"

    def test_normalize_text_alias(self):
        assert normalize_text("Python Developer") == "python developer"


class TestTokenizeTerms:
    """Tests for tokenize_terms function."""

    def test_tokenize_standard(self):
        tokens = tokenize_terms("Python, Docker & AWS!")
        assert tokens == ["python", "docker", "aws"]

    def test_tokenize_empty(self):
        assert tokenize_terms("") == []
        assert tokenize_terms(None) == []
