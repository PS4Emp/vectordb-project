import json
import numpy as np

from robustvdb.core.search import RobustVDB
from robustvdb.core.embeddings import EmbeddingModel
from robustvdb.metrics.robustness import robustness_score


# --- Demo corpus ---
documents = [
    "Neural networks for signal processing applications",
    "Introduction to deep learning and backpropagation",
    "Natural language processing with transformers",
    "Computer vision using convolutional neural networks",
    "Reinforcement learning for game playing agents",
]

# --- Compute calibration baseline from the corpus ---
# 1. Embed all documents
embedder = EmbeddingModel()
doc_embeddings = embedder.embed_documents(documents)

# 2. L2-normalize so dot product = cosine similarity
norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
doc_normed = doc_embeddings / norms

# 3. Cosine similarity matrix via dot product
sim_matrix = doc_normed @ doc_normed.T

# 4. Zero out the diagonal (ignore self-similarity)
np.fill_diagonal(sim_matrix, 0.0)

# 5. For each document, take top-k neighbor similarities and convert to distances
cal_k = min(3, len(documents) - 1)
baseline_distances_list = []
for i in range(len(documents)):
    row = sim_matrix[i]
    top_k_sims = np.sort(row)[::-1][:cal_k]       # highest similarities first
    top_k_dists = 1.0 - top_k_sims                 # convert to distance-like values
    baseline_distances_list.append(float(np.mean(top_k_dists)))

baseline_distances = np.array(baseline_distances_list, dtype=np.float32)

print("=== Calibration Baseline ===")
print(f"baseline_distances = {baseline_distances}\n")

# --- Initialise and populate the database ---
db = RobustVDB(baseline_distances=baseline_distances)
db.add(documents)

# --- Run a search query ---
query = "neural network signal processing"
results = db.search(query, k=3)

# Raw results for schema verification
print("=== Raw Results Object ===")
print(results)
print()

print("=== Search Results ===")
print(f"Query: \"{query}\"\n")
for i, r in enumerate(results, 1):
    print(f"Result {i}:")
    print(json.dumps(r, indent=2))
    print()

# --- Robustness score demo ---
# Mock data: 3 queries, each with retrieved doc IDs and ground-truth relevant IDs
mock_retrieved = [[0, 1, 2], [3, 4, 0], [1, 2, 3]]
mock_ground_truth = [{0, 1}, {3, 4}, {2, 5}]

score = robustness_score(mock_retrieved, mock_ground_truth, delta=0.5, k=3)

print("=== Robustness Score ===")
print(f"Robustness-0.5@3 = {score:.2f}")
