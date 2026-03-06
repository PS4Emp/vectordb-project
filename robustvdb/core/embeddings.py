
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level cache: only load each model once across all instances
_model_cache: dict[str, SentenceTransformer] = {}


class EmbeddingModel:
    """Simple wrapper around SentenceTransformer for encoding texts into FAISS-ready vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        if model_name not in _model_cache:
            _model_cache[model_name] = SentenceTransformer(model_name)
        self.model = _model_cache[model_name]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Encode a list of document strings. Returns float32 array of shape (n, dim)."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a single query string. Returns float32 array of shape (1, dim)."""
        embedding = self.model.encode([text], show_progress_bar=False)
        return np.array(embedding, dtype=np.float32)
