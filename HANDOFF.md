# RobustVDB — Full Project Handoff Document

## Project Overview

**RobustVDB** is an academic Python library (AI Masters project) that wraps FAISS and adds robustness metrics, explainability signals, and hard-query detection.

**One-line pitch:** "A Python vector database that tells you not just what it found, but why it matched, how confident it is, and how consistently reliable it is across different types of queries."

**Core problem:** Production RAG systems can fail, and engineers often cannot tell whether failure came from retrieval or from the LLM. Existing tools return results but do not explain why a match happened, how confident retrieval is, or whether the system is consistently reliable across difficult queries.

---

## Environment

- **OS:** Windows
- **Project root:** `C:\vectordb-project`
- **Python:** 3.12 (virtual environment at `.venv`)
- **Run commands** must use `& .\.venv\Scripts\python.exe` or activate the venv first
- **IDE:** Antigravity
- **All packages already installed:** faiss-cpu, sentence-transformers, FastAPI, uvicorn, numpy, scikit-learn, pytest

---

## Three Pillars (ALL COMPLETE)

### Pillar 1 — Robustness Metric (Robustness-δ@K)
- `recall@k = |retrieved[:k] ∩ relevant| / |relevant|`
- `Robustness-δ@K = fraction of queries where recall@k >= delta`
- Requires ground truth: `robustness_score(retrieved_ids, ground_truth, delta=0.7, k=5)`
- Implemented in `robustvdb/metrics/robustness.py`

### Pillar 2 — Explainability
Each search result includes:
- `keyword_overlap` = matched_query_terms / total_unique_query_terms
- `matched_terms` = sorted list of overlapping tokens
- `confidence` = "high" (score≥0.80 AND overlap≥0.50), "medium" (≥0.60/≥0.25), "low" (otherwise)
- Token matching: lowercase, remove punctuation, split whitespace, deduplicate via sets
- Implemented in `robustvdb/explainability/scorer.py`, integrated into `core/search.py`

### Pillar 3 — Hard Query Detection
- Converts similarity scores to distances: `distance = 1.0 - similarity`
- Threshold = `mean(baseline_distances) + 1 * std(baseline_distances)`
- Returns `"hard_query_warning"` or `"stable"`
- Baseline is a fixed calibration array computed from the corpus (NOT dynamic)
- Implemented in `robustvdb/metrics/hardquery.py`, integrated into `core/search.py`

---

## Complete File Structure

```
C:\vectordb-project\
├── README.md                              # Project README
├── test_me.py                             # Standalone verification script
├── main.py                                # (root-level, not used — see robustvdb/main.py)
├── robustvdb/
│   ├── __init__.py                        # Empty package marker
│   ├── main.py                            # Demo entry point
│   ├── core/
│   │   ├── __init__.py                    # Empty
│   │   ├── embeddings.py                  # SentenceTransformer wrapper (singleton cached)
│   │   ├── index.py                       # FAISS IndexFlatIP wrapper
│   │   └── search.py                      # RobustVDB main class
│   ├── explainability/
│   │   ├── __init__.py                    # Empty
│   │   └── scorer.py                      # tokenize, keyword_overlap, matched_terms, confidence
│   ├── metrics/
│   │   ├── __init__.py                    # Empty
│   │   ├── robustness.py                  # recall_at_k, robustness_score
│   │   └── hardquery.py                   # hard_query_check (similarity → distance conversion)
│   ├── api/
│   │   ├── __init__.py                    # Empty
│   │   └── main.py                        # FastAPI: GET /health, POST /search
│   └── tests/
│       ├── __init__.py                    # Empty
│       └── eval.py                        # Local evaluation with ground truth
└── .venv/                                 # Virtual environment (do not modify)
```

---

## File-by-File Summary

### `robustvdb/core/embeddings.py`
- Class `EmbeddingModel` wrapping `SentenceTransformer`
- Default model: `all-MiniLM-L6-v2`
- **Singleton pattern**: module-level `_model_cache` dict prevents loading model twice
- `embed_documents(texts) → np.ndarray` shape (n, dim), float32
- `embed_query(text) → np.ndarray` shape (1, dim), float32

### `robustvdb/core/index.py`
- Class `VectorIndex` wrapping `faiss.IndexFlatIP`
- Stores document texts in `self.documents: list[str]`
- `add_documents(texts, embeddings)` — validates shape, copies before L2 normalization
- `search(query_embedding, k)` — validates shape, copies before normalization, returns (scores, indices)
- `get_document(doc_id)`, `__len__()`

### `robustvdb/core/search.py`
- Class `RobustVDB` — main orchestration layer
- `__init__(model_name, baseline_distances)` — creates EmbeddingModel, stores optional baseline
- `add(texts)` — validates non-empty, embeds, lazily creates VectorIndex
- `search(query, k)` — returns list of dicts with all 6 fields
- Imports and uses: `compute_keyword_overlap`, `compute_matched_terms`, `compute_confidence`, `hard_query_check`
- Hard-query flag computed once per query (query-level), applied to all results

### `robustvdb/explainability/scorer.py`
- `tokenize(text) → set[str]` — lowercase, strip punctuation, split, deduplicate
- `compute_matched_terms(query, document) → list[str]` — sorted intersection
- `compute_keyword_overlap(query, document) → float` — returns 0.0 if no query tokens
- `compute_confidence(vector_score, keyword_overlap) → str` — fixed rules

### `robustvdb/metrics/hardquery.py`
- `_validate_1d_nonempty(arr, name)` — shape/empty check
- `compute_threshold(baseline_distances) → float` — mean + std
- `classify_query(avg_distance, threshold) → str`
- `hard_query_check(neighbor_scores, baseline_distances) → str` — converts similarity to distance first

### `robustvdb/metrics/robustness.py`
- `recall_at_k(retrieved, relevant, k) → float` — returns 0.0 if no relevant docs
- `robustness_score(retrieved_ids, ground_truth, delta, k) → float` — validates lengths, delta, k

### `robustvdb/api/main.py`
- FastAPI app with `GET /health` and `POST /search`
- Computes corpus-based baseline at startup
- Pydantic `SearchRequest` validates query (non-empty) and k (positive)
- Same 5-doc demo corpus as `main.py`

### `robustvdb/main.py`
- Demo entry point: embeds corpus, computes calibration baseline, runs search, prints results
- Also demos `robustness_score()` with mock data

### `robustvdb/tests/eval.py`
- 8-doc corpus with 6 queries and ground truth
- Per-query: prints retrieved IDs, ground truth, recall@k, robustness flag
- Overall: prints Robustness-δ@K

### `test_me.py` (project root)
- Three tests: schema check, unrelated query hard-query detection, robustness_score

---

## Target Search Result Schema

Every `db.search()` result returns:

```json
{
  "text": "Neural networks for signal processing applications",
  "vector_score": 0.861343502998352,
  "keyword_overlap": 0.75,
  "matched_terms": ["neural", "processing", "signal"],
  "confidence": "high",
  "robustness_flag": "stable"
}
```

---

## Verified Test Results

### test_me.py output (all passing):
```
TEST 1: Relevant query "neural network signal processing"
  Result 1: vector_score=0.8613, keyword_overlap=0.75, confidence=high, robustness_flag=stable
  Result 2: vector_score=0.4781, keyword_overlap=0.0, confidence=low, robustness_flag=stable
  Result 3: vector_score=0.3565, keyword_overlap=0.25, confidence=low, robustness_flag=stable
  [PASS] All 3 results contain all 6 expected fields.

TEST 2: Unrelated query "cooking recipes for pasta"
  robustness_flag: hard_query_warning
  vector_score: 0.2199, confidence: low

TEST 3: robustness_score()
  Robustness-0.5@3 = 1.00 (3/3 queries met recall >= 0.5)
  [PASS] All tests passed.
```

### tests/eval.py output:
```
Query: "neural network signal processing"    → Recall@3: 1.00, Flag: stable
Query: "deep learning backpropagation"       → Recall@3: 1.00, Flag: stable
Query: "language models and transformers"    → Recall@3: 1.00, Flag: stable
Query: "image recognition with neural nets"  → Recall@3: 0.50, Flag: stable
Query: "reinforcement learning game AI"      → Recall@3: 1.00, Flag: stable
Query: "text classification methods"         → Recall@3: 1.00, Flag: stable

OVERALL  Robustness-0.5@3 = 1.00 (6/6 queries met recall >= 0.5)
```

---

## How to Run

```bash
# Demo
python -m robustvdb.main

# API
uvicorn robustvdb.api.main:app --reload

# Evaluation
python -m robustvdb.tests.eval

# Quick verification
python test_me.py
```

---

## Key Design Decisions Made

1. **FAISS IndexFlatIP + L2 normalization** = effective cosine similarity search
2. **Copy-before-normalize** — caller arrays never mutated in place
3. **Lazy index initialization** — dimension inferred from first `add()` call
4. **Singleton model cache** — `_model_cache` dict in embeddings.py prevents double model loading
5. **Similarity → distance conversion** in hard-query detection: `distance = 1.0 - similarity`
6. **Corpus-based calibration baseline** — not synthetic, computed from actual indexed document embeddings
7. **Query-level robustness flag** — same flag for all results from one query (detection is about the query, not individual results)

---

## What Has NOT Been Built (by design)

- No distributed systems / cloud infrastructure
- No authentication / caching / async queues
- No persistence (in-memory only)
- No BEIR integration yet
- No pytest test framework yet (script-based evaluation only)
- No plotting / visualization
- No features outside the three pillars

---

## Current Limitations

- Small demo corpus only — not yet tested on large-scale benchmarks
- Single embedding model (all-MiniLM-L6-v2)
- Lexical explainability only (lightweight keyword overlap, not token-level attribution)
- Fixed hard-query threshold (mean + 1*std, not domain-tuned)
- No persistent storage
- Calibration baseline computation is duplicated across main.py, api/main.py, and tests/eval.py

---

## Workstyle Rules for Continued Development

- Write only the file asked for
- Do not modify other files unless asked
- Do not invent extra architecture
- Do not add features outside the three pillars
- Keep code simple, readable, and academically defensible
- Use type hints where reasonable
- Keep dependencies minimal
