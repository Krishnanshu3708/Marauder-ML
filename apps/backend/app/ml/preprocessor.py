"""Text preprocessing module for CareerFit ML pipeline.

Cleans and normalizes resume and job description text for NLP feature extraction,
handling whitespace, case normalization, special characters, and edge cases.
"""

import re
from typing import Optional


# Pre-compiled regex patterns for efficiency
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_BULLET_PATTERN = re.compile(r"^[\s•*–—\-·•]+\s*", re.MULTILINE)

# Characters to preserve for programming languages and technical terms
# (e.g., c++, c#, .net, node.js)
_TECH_TERM_PROTECTED = {
    "c++": "cplusplus",
    "c#": "csharp",
    ".net": "dotnet",
    "node.js": "nodejs",
    "vue.js": "vuejs",
    "next.js": "nextjs",
    "react.js": "reactjs",
}


def clean_text(text: Optional[str]) -> str:
    """Clean and normalize raw text from resumes or job descriptions.

    Performs the following cleaning operations:
    1. Returns empty string for None, non-string, or whitespace-only inputs.
    2. Protects well-known technical terms (e.g. 'c++', 'c#', '.net') from punctuation stripping.
    3. Removes URLs, web links, and email addresses (privacy protection).
    4. Removes non-alphanumeric noise while preserving standard punctuation.
    5. Normalizes all whitespace (tabs, newlines) to single spaces.
    6. Converts to lowercase and strips leading/trailing spaces.

    Args:
        text: Raw input string or None.

    Returns:
        Cleaned, lowercased, and normalized text string.
    """
    if not isinstance(text, str):
        return ""

    stripped = text.strip()
    if not stripped:
        return ""

    # Normalize unicode characters and remove URLs & emails (protects personal info)
    cleaned = _URL_PATTERN.sub(" ", stripped)
    cleaned = _EMAIL_PATTERN.sub(" ", cleaned)

    # Remove markdown bullets and list markers at line starts
    cleaned = _BULLET_PATTERN.sub(" ", cleaned)

    # Protect specific technical symbols before stripping punctuation
    cleaned_lower = cleaned.lower()
    for term, placeholder in _TECH_TERM_PROTECTED.items():
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        cleaned_lower = re.sub(pattern, placeholder, cleaned_lower)

    # Replace punctuation and special characters with spaces, keeping alphanumeric
    # and spaces
    cleaned_chars = re.sub(r"[^\w\s]", " ", cleaned_lower)

    # Restore protected tech terms
    for term, placeholder in _TECH_TERM_PROTECTED.items():
        cleaned_chars = re.sub(rf"\b{placeholder}\b", term, cleaned_chars)

    # Collapse multiple whitespaces and newlines into a single space
    normalized = _WHITESPACE_PATTERN.sub(" ", cleaned_chars).strip()
    return normalized


def normalize_text(text: Optional[str]) -> str:
    """Convenience alias for clean_text."""
    return clean_text(text)


def tokenize_terms(text: Optional[str]) -> list[str]:
    """Tokenize cleaned text into a list of word tokens.

    Args:
        text: Input string.

    Returns:
        List of non-empty token strings.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()
