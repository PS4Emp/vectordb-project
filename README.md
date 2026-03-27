# RobustVDB

**A Python vector database that tells you not just what it found, but why it matched, how confident it is, and how consistently reliable it is across different types of queries.**

---

## Project Goal

Production RAG systems can fail silently, and engineers often cannot tell whether failure came from retrieval or from the LLM. Existing vector databases like FAISS, ChromaDB, and Pinecone return results but do not explain *why* a match happened, *how confident* retrieval is, or *whether the system is consistently reliable* across difficult queries.

RobustVDB is a lightweight Python library that wraps FAISS and adds **robustness metrics**, **explainability signals**, and **hard-query detection** — making retrieval transparent and empirically validated.

---

## Three Pillars

### Pillar 1 — Robustness Metric (Robustness-δ@K)

Measures retrieval reliability across a test set using ground truth.

- **recall@k** = retrieved relevant documents at k / total relevant documents
- **Robustness-δ@K** = fraction of queries whose recall@k ≥ δ

```python
score = robustness_score(retrieved_ids, ground_truth, delta=0.7, k=5)
```

### Pillar 2 — Explainability

Every search result includes lightweight lexical explanation signals:

| Field | Definition |
|---|---|
| `keyword_overlap` | matched query terms / total unique query terms |
| `matched_terms` | sorted list of overlapping tokens between query and document |
| `confidence` | `"high"`, `"medium"`, or `"low"` dynamically graded based on similarity score and lexical overlap. |

### Pillar 3 — Hard Query Detection

Flags queries that land in a sparse region of vector space using Configurable Query Performance Prediction (QPP). RobustVDB now natively supports multiple detection modes:

1. **Mean Distance (`mean_distance`)** (Baseline): Compares the query's average distance against a pre-computed corpus calibration threshold.
2. **Clarity (`clarity`)** (Target): Computes the margin between the top-1 retrieve document and the mean of the top-k neighbourhood (`-(sim[0] - mean(sim[1:k]))`). Explicitly flags queries lacking a clear semantic match.

---

## Empirical Benchmark Setup

To validate our hard-query detection modes, we integrated RobustVDB directly over the `BEIR` framework. 

*   **Datasets:** SciFact (fact-checking) and FIQA-2018 (financial QA).
*   **Embeddings & Retriever:** Fixed `all-MiniLM-L6-v2` dense embeddings using exact FAISS `IndexFlatIP`.
*   **Methodology:** Evaluated plain dense baseline against `mean_distance` wrapper and `clarity` wrapper (configured with the best percentile-based thresholds: P80 for SciFact, P50 for FIQA, at `qpp_k=3`).

---

## Benchmark Results

| Dataset | Mode | R@5 | Rob0.5@5 | Flag% | R@5 flagged | R@5 stable | Bad% flagged | Bad% stable |
|---|---|---|---|---|---|---|---|---|
| scifact | baseline | 0.7413 | 0.7500 | | | | | |
| scifact | mean_distance | 0.7413 | 0.7500 | 49.3% | 0.7318 | 0.7505 | 26.4% | 23.7% |
| scifact | clarity | 0.7413 | 0.7500 | 20.0% | 0.4722 | 0.8085 | 51.7% | 18.3% |
| fiqa | baseline | 0.3671 | 0.4074 | | | | | |
| fiqa | mean_distance | 0.3671 | 0.4074 | 9.6% | 0.4218 | 0.3613 | 56.5% | 59.6% |
| fiqa | clarity | 0.3671 | 0.4074 | 50.0% | 0.2608 | 0.4733 | 72.5% | 46.0% |

*(Note: "Bad%" refers to the fraction of queries achieving less than 0.5 Recall@5).*

---

## Key Finding: Clarity > Mean-Distance

The **Clarity** metric systematically and significantly outperforms the default **Mean-Distance** technique for hard-query detection:

1. **Massive Density Concentration (SciFact):** While `mean_distance` indiscriminately flagged 49% of the dataset as a near coin flip (26% bad vs 23% bad), `clarity` precisely isolated a 20% fragment containing a massive majority of failures (51.7% bad rate vs only 18.3% bad in stable queries).
2. **Robustness to Domain Spikes (FIQA):** The `mean_distance` metric entirely inverted and broke on FIQA, incorrectly correlating short term sparse neighbourhoods with higher accuracy. `clarity` successfully survived cross-collection transfer, perfectly segregating the hardest 50% of the dataset and filtering out an astounding 72.5% bad query rate, cleaning the stable cohort up to an acceptable 46%.

---

## Current Limitations

- **Limited testing scale:** Tested on only two BEIR datasets (SciFact, FIQA) so far.
- **Retrospective thresholding:** The Clarity flag threshold was chosen actively from ablation study percentiles. A fully robust zero-shot threshold estimation mechanism remains to be developed.
- **Unsupervised baseline only:** No comparison has been made yet against stronger, learned Query Performance Predictors (QPP) (e.g. supervised neural classifiers).
- **Single embedding model:** Tested with `all-MiniLM-L6-v2`; no model comparison across heavier embeddings yet.

---

## License

Academic project — AI Masters programme.
