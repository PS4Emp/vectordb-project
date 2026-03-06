
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from robustvdb.core.search import RobustVDB
from robustvdb.core.embeddings import EmbeddingModel


# --- FastAPI app ---
app = FastAPI(title="RobustVDB API", version="0.1.0")


# --- Demo corpus (same as robustvdb/main.py) ---
DEMO_DOCUMENTS = [
    "Neural networks for signal processing applications",
    "Introduction to deep learning and backpropagation",
    "Natural language processing with transformers",
    "Computer vision using convolutional neural networks",
    "Reinforcement learning for game playing agents",
]


def _compute_baseline(documents: list[str]) -> np.ndarray:
    """Compute corpus-based calibration baseline for hard-query detection."""
    embedder = EmbeddingModel()
    doc_embeddings = embedder.embed_documents(documents)

    # L2-normalize so dot product = cosine similarity
    norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    doc_normed = doc_embeddings / norms

    # Cosine similarity matrix
    sim_matrix = doc_normed @ doc_normed.T
    np.fill_diagonal(sim_matrix, 0.0)

    # For each document, average distance to its top-k neighbours
    cal_k = min(3, len(documents) - 1)
    avg_dists = []
    for i in range(len(documents)):
        top_k_sims = np.sort(sim_matrix[i])[::-1][:cal_k]
        avg_dists.append(float(np.mean(1.0 - top_k_sims)))

    return np.array(avg_dists, dtype=np.float32)


# --- Initialise the database at startup ---
baseline_distances = _compute_baseline(DEMO_DOCUMENTS)
db = RobustVDB(baseline_distances=baseline_distances)
db.add(DEMO_DOCUMENTS)


# --- Request / Response models ---
class SearchRequest(BaseModel):
    query: str
    k: int = 5

    @field_validator("query")
    @classmethod
    def query_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must be a non-empty string.")
        return v

    @field_validator("k")
    @classmethod
    def k_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("k must be a positive integer.")
        return v


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    results = db.search(req.query, req.k)
    return {
        "query": req.query,
        "k": req.k,
        "results": results,
    }
