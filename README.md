# RobustVDB

**A Python vector database that tells you not just what it found, but why it matched, how confident it is, and how consistently reliable it is across different types of queries.**

---

## Project Goal

Production RAG systems can fail silently, and engineers often cannot tell whether failure came from retrieval or from the LLM. Existing vector databases like FAISS, ChromaDB, and Pinecone return results but do not explain *why* a match happened, *how confident* retrieval is, or *whether the system is consistently reliable* across difficult queries.

RobustVDB is a lightweight Python library that wraps FAISS and adds **robustness metrics**, **explainability signals**, and **hard-query detection** — making retrieval transparent and academically defensible.

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
| `confidence` | `"high"` if vector_score >= 0.80 AND overlap >= 0.50; `"medium"` if >= 0.60 / >= 0.25; `"low"` otherwise |

### Pillar 3 — Hard Query Detection

Flags queries that land in a sparse region of vector space.

- Converts similarity scores to distances: `distance = 1.0 - similarity`
- Computes average distance from query to its top-k neighbours
- Compares against a fixed calibration threshold: `mean + 1 * std`
- Returns `"hard_query_warning"` or `"stable"`

---

## Example Result

```json
{
  "text": "Neural networks for signal processing applications",
  "vector_score": 0.8613,
  "keyword_overlap": 0.75,
  "matched_terms": ["neural", "processing", "signal"],
  "confidence": "high",
  "robustness_flag": "stable"
}
```

---

## Folder Structure

```
robustvdb/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── embeddings.py      # SentenceTransformer wrapper
│   ├── index.py            # FAISS IndexFlatIP wrapper
│   └── search.py           # Main RobustVDB orchestration class
├── explainability/
│   ├── __init__.py
│   └── scorer.py           # Tokenizer, keyword overlap, confidence
├── metrics/
│   ├── __init__.py
│   ├── robustness.py       # recall@k and Robustness-delta@K
│   └── hardquery.py        # Hard-query detection
├── api/
│   ├── __init__.py
│   └── main.py             # FastAPI endpoints
├── tests/
│   ├── __init__.py
│   └── eval.py             # Local evaluation script
└── main.py                 # Demo entry point
```

---

## How to Run

### Prerequisites

```
Python 3.10+
Virtual environment with: faiss-cpu, sentence-transformers, fastapi, uvicorn, numpy, scikit-learn
```

### Demo

```bash
python -m robustvdb.main
```

Runs a small demo corpus, prints calibration baseline, search results with all 6 fields, and a mock robustness score.

### API

```bash
uvicorn robustvdb.api.main:app --reload
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}` |
| POST | `/search` | Accepts `{"query": "...", "k": 5}`, returns enriched results |

### Evaluation Script

```bash
python -m robustvdb.tests.eval
```

Runs queries against an 8-document corpus with ground truth, prints per-query recall@k, robustness flags, and overall Robustness-delta@K.

### Quick Verification

```bash
python test_me.py
```

Runs three checks: schema verification, hard-query detection on an unrelated query, and robustness_score computation.

---

## Tech Stack

| Package | Purpose |
|---|---|
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Text embedding (all-MiniLM-L6-v2) |
| `FastAPI` | Minimal API layer |
| `numpy` | Numerical operations |
| `scikit-learn` | Available for future metrics |

---

## Current Limitations

- **Small demo corpus only** — not yet tested on large-scale benchmarks like BEIR
- **No persistence** — index lives in memory and is rebuilt on each run
- **Single embedding model** — uses all-MiniLM-L6-v2; no model comparison yet
- **Lexical explainability only** — keyword overlap is a lightweight proxy, not strict token-level attribution
- **Fixed threshold** — hard-query threshold uses mean + 1*std; not tuned for specific domains
- **No pytest suite** — evaluation is script-based, not integrated into a test framework yet

---

## License

Academic project — AI Masters programme.
