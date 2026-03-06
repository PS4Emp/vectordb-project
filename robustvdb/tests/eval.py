
import numpy as np

from robustvdb.core.search import RobustVDB
from robustvdb.core.embeddings import EmbeddingModel
from robustvdb.metrics.robustness import robustness_score, recall_at_k


# --- Evaluation corpus ---
corpus = [
    "Neural networks for signal processing applications",           # 0
    "Introduction to deep learning and backpropagation",            # 1
    "Natural language processing with transformers",                # 2
    "Computer vision using convolutional neural networks",          # 3
    "Reinforcement learning for game playing agents",               # 4
    "Recurrent neural networks for time series forecasting",        # 5
    "Generative adversarial networks for image synthesis",          # 6
    "Support vector machines for text classification",              # 7
]

# --- Evaluation queries with ground-truth relevant document IDs ---
eval_set = [
    {"query": "neural network signal processing",     "relevant": {0, 5}},
    {"query": "deep learning backpropagation",        "relevant": {1}},
    {"query": "language models and transformers",      "relevant": {2}},
    {"query": "image recognition with neural nets",    "relevant": {3, 6}},
    {"query": "reinforcement learning game AI",        "relevant": {4}},
    {"query": "text classification methods",           "relevant": {7}},
]

K = 3
DELTA = 0.5

# --- Compute corpus-based calibration baseline ---
embedder = EmbeddingModel()
doc_embeddings = embedder.embed_documents(corpus)

norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
doc_normed = doc_embeddings / norms

sim_matrix = doc_normed @ doc_normed.T
np.fill_diagonal(sim_matrix, 0.0)

cal_k = min(K, len(corpus) - 1)
baseline_distances = np.array([
    float(np.mean(1.0 - np.sort(sim_matrix[i])[::-1][:cal_k]))
    for i in range(len(corpus))
], dtype=np.float32)

print(f"Calibration baseline_distances = {baseline_distances}\n")

# --- Initialise and populate RobustVDB ---
db = RobustVDB(baseline_distances=baseline_distances)
db.add(corpus)

# --- Run evaluation ---
all_retrieved_ids = []
all_ground_truth = []

print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

for entry in eval_set:
    query = entry["query"]
    relevant = entry["relevant"]

    results = db.search(query, k=K)

    # Map returned texts back to corpus IDs
    retrieved_ids = [corpus.index(r["text"]) for r in results]

    # Compute recall@k for this query
    r_at_k = recall_at_k(retrieved_ids, relevant, K)

    # Hard-query flag from the first result (query-level, same for all results)
    flag = results[0]["robustness_flag"] if results else "n/a"

    print(f"\nQuery: \"{query}\"")
    print(f"  Retrieved IDs:  {retrieved_ids}")
    print(f"  Ground Truth:   {sorted(relevant)}")
    print(f"  Recall@{K}:      {r_at_k:.2f}")
    print(f"  Robustness Flag: {flag}")

    all_retrieved_ids.append(retrieved_ids)
    all_ground_truth.append(relevant)

# --- Overall robustness score ---
overall = robustness_score(all_retrieved_ids, all_ground_truth, delta=DELTA, k=K)

print("\n" + "=" * 60)
print(f"OVERALL  Robustness-{DELTA}@{K} = {overall:.2f}")
print(f"  ({int(overall * len(eval_set))}/{len(eval_set)} queries met recall >= {DELTA})")
print("=" * 60)
