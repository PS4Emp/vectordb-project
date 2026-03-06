
import numpy as np
import faiss


class VectorIndex:
    """FAISS IndexFlatIP wrapper that stores document texts alongside their embeddings."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[str] = []

    def add_documents(self, texts: list[str], embeddings: np.ndarray) -> None:
        """Add documents and their precomputed embeddings to the index."""
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(texts):
            raise ValueError(
                f"number of texts ({len(texts)}) must match number of embedding rows ({embeddings.shape[0]})"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"embedding dimension ({embeddings.shape[1]}) must match index dimension ({self.dimension})"
            )
        # Copy so we don't normalise the caller's array in place
        emb = np.array(embeddings, dtype=np.float32, copy=True)
        faiss.normalize_L2(emb)
        self.index.add(emb)
        self.documents.extend(texts)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the k nearest neighbours.

        Returns:
            (scores, indices) — both of shape (1, k).
        """
        if query_embedding.ndim != 2:
            raise ValueError(f"query_embedding must be 2D, got shape {query_embedding.shape}")
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"query dimension ({query_embedding.shape[1]}) must match index dimension ({self.dimension})"
            )
        # Copy so we don't normalise the caller's array in place
        qe = np.array(query_embedding, dtype=np.float32, copy=True)
        faiss.normalize_L2(qe)
        scores, indices = self.index.search(qe, k)
        return scores, indices

    def get_document(self, doc_id: int) -> str:
        """Return the original text for a given document index."""
        return self.documents[doc_id]

    def __len__(self) -> int:
        return self.index.ntotal
