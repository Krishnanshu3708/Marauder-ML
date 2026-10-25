"""Feature extraction module using TF-IDF for CareerFit ML.

Provides scikit-learn based TF-IDF vectorization for transforming resume and
job description texts into numerical sparse feature representations.
"""

from typing import Any, Optional, Sequence
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app.ml.preprocessor import clean_text


# Default token pattern to capture standard words plus tech terms like c++, c#, .net
TECH_TOKEN_PATTERN = r"(?u)\b[\w\.]+\b|c\+\+|c\#|\.net"


class TfidfFeatureExtractor:
    """TF-IDF vectorizer wrapper tailored for resume and job description texts."""

    def __init__(
        self,
        max_features: Optional[int] = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        stop_words: Optional[str] = "english",
        sublinear_tf: bool = True,
        token_pattern: str = TECH_TOKEN_PATTERN,
    ) -> None:
        """Initialize the TF-IDF feature extractor.

        Args:
            max_features: Maximum vocabulary size (None for unlimited).
            ngram_range: Lower and upper boundary of n-values for n-grams.
            stop_words: Language for stop words ('english' or None).
            sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).
            token_pattern: Regular expression defining what constitutes a token.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.sublinear_tf = sublinear_tf
        self.token_pattern = token_pattern

        self.vectorizer = TfidfVectorizer(
            preprocessor=clean_text,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            sublinear_tf=self.sublinear_tf,
            token_pattern=self.token_pattern,
        )
        self.is_fitted: bool = False

    def fit(self, documents: Sequence[str]) -> "TfidfFeatureExtractor":
        """Fit the TF-IDF vectorizer on a collection of documents.

        Args:
            documents: Sequence of document text strings.

        Returns:
            self
        """
        cleaned_docs = [clean_text(d) for d in documents]
        non_empty = [d for d in cleaned_docs if d]
        if not non_empty:
            raise ValueError("Cannot fit TF-IDF vectorizer on an empty or whitespace-only corpus.")

        self.vectorizer.fit(cleaned_docs)
        self.is_fitted = True
        return self

    def transform(self, documents: Sequence[str]) -> scipy.sparse.csr_matrix:
        """Transform documents into TF-IDF sparse matrix.

        Args:
            documents: Sequence of document text strings.

        Returns:
            Sparse matrix of TF-IDF feature representations.
        """
        if not self.is_fitted:
            raise RuntimeError("TfidfFeatureExtractor must be fitted before calling transform().")
        cleaned_docs = [clean_text(d) for d in documents]
        return self.vectorizer.transform(cleaned_docs)

    def fit_transform(self, documents: Sequence[str]) -> scipy.sparse.csr_matrix:
        """Fit vectorizer and transform documents in one step.

        Args:
            documents: Sequence of document text strings.

        Returns:
            Sparse matrix of TF-IDF feature representations.
        """
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self) -> list[str]:
        """Return list of feature names in the learned vocabulary.

        Returns:
            List of vocabulary terms.
        """
        if not self.is_fitted:
            return []
        return list(self.vectorizer.get_feature_names_out())

    @property
    def vocabulary_size(self) -> int:
        """Return the size of the vocabulary."""
        if not self.is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)
