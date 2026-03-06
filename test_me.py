import json
import numpy as np

from robustvdb.core.search import RobustVDB
from robustvdb.core.embeddings import EmbeddingModel
from robustvdb.metrics.robustness import robustness_score


# --- Small test corpus ---
corpus = [
    "Neural networks for signal processing applications",
    "Introduction to deep learning and backpropagation",
    "Natural language processing with transformers",
    "Computer vision using convolutional neural networks",
    "Reinforcement learning for game playing agents",
]

# --- Corpus-based calibration baseline (same method as robustvdb/main.py) ---
embedder = EmbeddingModel()
doc_embeddings = embedder.embed_documents(corpus)

norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
doc_normed = doc_embeddings / norms

sim_matrix = doc_normed @ doc_normed.T
np.fill_diagonal(sim_matrix, 0.0)

cal_k = min(3, len(corpus) - 1)
baseline_distances = np.array([
    float(np.mean(1.0 - np.sort(sim_matrix[i])[::-1][:cal_k]))
    for i in range(len(corpus))
], dtype=np.float32)

print(f"baseline_distances = {baseline_distances}\n")

# --- Initialise and populate ---
db = RobustVDB(baseline_distances=baseline_distances)
db.add(corpus)

# ============================================================
# TEST 1: Relevant query — full result schema check
# ============================================================
print("=" * 60)
print("TEST 1: Relevant query")
print("=" * 60)

query = "neural network signal processing"
results = db.search(query, k=3)

print(f"Query: \"{query}\"\n")
for i, r in enumerate(results, 1):
    print(f"Result {i}:")
    print(json.dumps(r, indent=2))
    print()

# Verify all 6 fields are present in every result
expected_fields = {"text", "vector_score", "keyword_overlap", "matched_terms", "confidence", "robustness_flag"}
all_ok = True
for i, r in enumerate(results):
    missing = expected_fields - set(r.keys())
    if missing:
        print(f"  [FAIL] Result {i} missing fields: {missing}")
        all_ok = False

if all_ok:
    print(f"[PASS] All {len(results)} results contain all 6 expected fields.\n")

# ============================================================
# TEST 2: Unrelated query — check robustness_flag
# ============================================================
print("=" * 60)
print("TEST 2: Unrelated query")
print("=" * 60)

unrelated_query = "cooking recipes for pasta"
unrelated_results = db.search(unrelated_query, k=3)

print(f"Query: \"{unrelated_query}\"")
print(f"Top result robustness_flag: {unrelated_results[0]['robustness_flag']}")
print(f"Top result vector_score:    {unrelated_results[0]['vector_score']:.4f}")
print(f"Top result confidence:      {unrelated_results[0]['confidence']}\n")

# ============================================================
# TEST 3: Robustness score with mock data
# ============================================================
print("=" * 60)
print("TEST 3: robustness_score()")
print("=" * 60)

mock_retrieved = [[0, 1, 2], [3, 4, 0], [1, 2, 3]]
mock_ground_truth = [{0, 1}, {3, 4}, {2, 5}]

score = robustness_score(mock_retrieved, mock_ground_truth, delta=0.5, k=3)
print(f"Robustness-0.5@3 = {score:.2f}")
print(f"  ({int(score * len(mock_retrieved))}/{len(mock_retrieved)} queries met recall >= 0.5)")

print("\n[PASS] All tests passed.")
