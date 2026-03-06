
import numpy as np

from robustvdb.core.embeddings import EmbeddingModel, DEFAULT_MODEL_NAME
from robustvdb.core.index import VectorIndex
from robustvdb.explainability.scorer import (
    compute_keyword_overlap,
    compute_matched_terms,
    compute_confidence,
)
from robustvdb.metrics.hardquery import hard_query_check


class RobustVDB:
    """Main search orchestration layer for the RobustVDB project."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, baseline_distances: np.ndarray | None = None):
        self.embedder = EmbeddingModel(model_name)
        self.index: VectorIndex | None = None
        self.baseline_distances = baseline_distances

    def add(self, texts: list[str]) -> None:
        """Embed and store documents in the FAISS index."""
        if not texts:
            raise ValueError("texts must be a non-empty list.")

        embeddings = self.embedder.embed_documents(texts)

        # Initialise the index on first call, using the embedding dimension
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = VectorIndex(dimension)

        self.index.add_documents(texts, embeddings)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search the index and return the top-k results.

        Returns a list of dicts with 'text', 'vector_score',
        'keyword_overlap', 'matched_terms', 'confidence', and 'robustness_flag'.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if self.index is None or len(self.index) == 0:
            raise ValueError("No documents in the index. Call add() first.")

        query_embedding = self.embedder.embed_query(query)
        scores, indices = self.index.search(query_embedding, k)

        # Hard-query detection is query-level, so compute once for all results
        if self.baseline_distances is not None:
            robustness_flag = hard_query_check(scores[0], self.baseline_distances)
        else:
            robustness_flag = "stable"

        results = []
        for score, idx in zip(scores[0], indices[0]):
            # FAISS may return -1 for unfilled slots when k > index size
            if idx == -1:
                continue

            doc_text = self.index.get_document(int(idx))
            vs = float(score)
            ko = compute_keyword_overlap(query, doc_text)
            mt = compute_matched_terms(query, doc_text)
            conf = compute_confidence(vs, ko)

            results.append({
                "text": doc_text,
                "vector_score": vs,
                "keyword_overlap": ko,
                "matched_terms": mt,
                "confidence": conf,
                "robustness_flag": robustness_flag,
            })

        return results
