# RobustVDB — Hard-Query Calibration Sweep
#
# Experiment script that sweeps threshold rules and neighbor counts
# for hard-query detection on BEIR SciFact.
#
# The retrieval pipeline is held FIXED.  Only the query-difficulty
# estimation parameters are varied.
#
# Run:
#   cd C:\vectordb-project
#   & .\.venv\Scripts\python.exe -m robustvdb.tests.calibration_sweep

import os
import sys
from typing import Dict, List, Set

import faiss
import numpy as np

try:
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader
except ImportError:
    print("ERROR: 'beir' package is required.  pip install beir")
    sys.exit(1)

from robustvdb.core.embeddings import EmbeddingModel
from robustvdb.core.index import VectorIndex
from robustvdb.metrics.robustness import recall_at_k

# ── Configuration ────────────────────────────────────────────────

DATASET = "scifact"
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")
RETRIEVAL_K = 10   # fixed retrieval depth
EVAL_K = 5         # recall is measured at this k
BAD_RECALL_THRESHOLD = 0.5  # queries below this are "bad"

QPP_K_VALUES = [3, 5, 10]

THRESHOLD_RULES = {
    "mean+0.5σ": lambda bl: float(np.mean(bl) + 0.5 * np.std(bl)),
    "mean+1.0σ": lambda bl: float(np.mean(bl) + 1.0 * np.std(bl)),
    "mean+1.5σ": lambda bl: float(np.mean(bl) + 1.5 * np.std(bl)),
    "P75":       lambda bl: float(np.percentile(bl, 75)),
    "P85":       lambda bl: float(np.percentile(bl, 85)),
    "P90":       lambda bl: float(np.percentile(bl, 90)),
}


# ── Dataset loading (mirrors beir_eval.py logic) ────────────────

def corpus_entry_to_text(entry: dict) -> str:
    """Combine a BEIR corpus entry's title and text."""
    title = (entry.get("title") or "").strip()
    text = (entry.get("text") or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text or ""


def load_dataset(name: str):
    """
    Download (if needed) and load a BEIR dataset.

    Returns:
        doc_texts      – corpus documents in stable order
        queries        – dict  query_id → query_text
        ground_truth   – dict  query_id → set of relevant local doc IDs
        eval_query_ids – sorted list of query IDs that have ground truth
    """
    print(f"[sweep] Downloading / locating {name} ...")
    data_path = beir_util.download_and_unzip(
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip",
        DOWNLOAD_DIR,
    )
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    sorted_corpus_ids: List[str] = sorted(corpus.keys())
    beir_id_to_local: Dict[str, int] = {
        bid: idx for idx, bid in enumerate(sorted_corpus_ids)
    }
    doc_texts: List[str] = [
        corpus_entry_to_text(corpus[bid]) for bid in sorted_corpus_ids
    ]

    ground_truth: Dict[str, Set[int]] = {}
    for qid, judgments in qrels.items():
        relevant = {
            beir_id_to_local[did]
            for did, score in judgments.items()
            if score > 0 and did in beir_id_to_local
        }
        if relevant:
            ground_truth[qid] = relevant

    eval_query_ids = sorted(ground_truth.keys())
    return doc_texts, queries, ground_truth, eval_query_ids


# ── Retrieval (fixed, not part of the sweep) ─────────────────────

def run_retrieval(
    doc_texts: List[str],
    doc_embeddings: np.ndarray,
    queries: dict,
    eval_query_ids: List[str],
    embedder: EmbeddingModel,
) -> tuple:
    """
    Run dense retrieval for every evaluation query.

    Returns:
        query_scores  – dict  qid → 1-D array of similarity scores  (length RETRIEVAL_K)
        query_indices – dict  qid → 1-D array of corpus doc indices  (length RETRIEVAL_K)
    """
    dim = doc_embeddings.shape[1]
    index = VectorIndex(dim)
    index.add_documents(doc_texts, doc_embeddings)

    query_scores: Dict[str, np.ndarray] = {}
    query_indices: Dict[str, np.ndarray] = {}

    for qid in eval_query_ids:
        qe = embedder.embed_query(queries[qid])
        scores, indices = index.search(qe, k=RETRIEVAL_K)
        query_scores[qid] = scores[0]
        query_indices[qid] = indices[0]

    return query_scores, query_indices


# ── Per-query recall ─────────────────────────────────────────────

def compute_recalls(
    query_indices: Dict[str, np.ndarray],
    ground_truth: Dict[str, Set[int]],
    eval_query_ids: List[str],
    k: int,
) -> Dict[str, float]:
    """Compute recall@k for each query."""
    return {
        qid: recall_at_k(query_indices[qid].tolist(), ground_truth[qid], k)
        for qid in eval_query_ids
    }


# ── Calibration baseline ────────────────────────────────────────

def compute_calibration_baseline(doc_embeddings: np.ndarray, cal_k: int) -> np.ndarray:
    """
    Per-document average distance to its cal_k nearest corpus neighbours.
    Uses FAISS for scalability.  Same logic as beir_eval.py.
    """
    emb = doc_embeddings.astype(np.float32).copy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-12)

    n, dim = emb.shape
    cal_k_clamped = min(cal_k, n - 1)

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    sims, _ = index.search(emb, cal_k_clamped + 1)
    sims = sims[:, 1:]  # drop self-similarity

    dists = 1.0 - sims
    return dists.mean(axis=1).astype(np.float32)


# ── Sweep logic ──────────────────────────────────────────────────

def run_sweep(
    query_scores: Dict[str, np.ndarray],
    recalls: Dict[str, float],
    bad_qids: Set[str],
    eval_query_ids: List[str],
    doc_embeddings: np.ndarray,
) -> List[dict]:
    """
    Sweep all (qpp_k, threshold_rule) combinations.

    For each query, the average distance to its top-qpp_k neighbours
    is compared against a threshold derived from the corpus calibration
    baseline.  If the distance exceeds the threshold the query is
    flagged as hard.

    Returns a list of result dicts, one per (qpp_k, rule) combination.
    """
    total = len(eval_query_ids)
    num_bad = len(bad_qids)
    results: List[dict] = []

    for qpp_k in QPP_K_VALUES:
        print(f"  qpp_k={qpp_k}: computing calibration baseline ...")
        baseline = compute_calibration_baseline(doc_embeddings, cal_k=qpp_k)

        # Per-query average distance using top-qpp_k similarity scores
        query_avg_dists: Dict[str, float] = {}
        for qid in eval_query_ids:
            top_scores = query_scores[qid][:qpp_k]
            query_avg_dists[qid] = float(np.mean(1.0 - top_scores))

        # Apply each threshold rule
        for rule_name, rule_fn in THRESHOLD_RULES.items():
            threshold = rule_fn(baseline)

            flagged = [qid for qid in eval_query_ids if query_avg_dists[qid] > threshold]
            stable  = [qid for qid in eval_query_ids if query_avg_dists[qid] <= threshold]

            flag_rate = len(flagged) / total

            r5_flagged = float(np.mean([recalls[qid] for qid in flagged])) if flagged else float("nan")
            r5_stable  = float(np.mean([recalls[qid] for qid in stable]))  if stable  else float("nan")

            gap = (r5_stable - r5_flagged) if not (np.isnan(r5_stable) or np.isnan(r5_flagged)) else float("nan")

            # Bad-query counts per group
            bad_flagged = sum(1 for qid in flagged if qid in bad_qids)
            bad_stable  = sum(1 for qid in stable  if qid in bad_qids)

            # What % of each group is bad?
            bad_pct_flagged = bad_flagged / len(flagged) if flagged else float("nan")
            bad_pct_stable  = bad_stable  / len(stable)  if stable  else float("nan")

            # Sensitivity: of all bad queries, how many were flagged?
            sensitivity = bad_flagged / num_bad if num_bad > 0 else float("nan")

            results.append({
                "qpp_k":          qpp_k,
                "rule":           rule_name,
                "threshold":      threshold,
                "flag_rate":      flag_rate,
                "r5_flagged":     r5_flagged,
                "r5_stable":      r5_stable,
                "gap":            gap,
                "n_flagged":      len(flagged),
                "n_stable":       len(stable),
                "n_bad_flagged":  bad_flagged,
                "n_bad_stable":   bad_stable,
                "bad_pct_flagged": bad_pct_flagged,
                "bad_pct_stable":  bad_pct_stable,
                "sensitivity":    sensitivity,
            })

    return results


# ── Output ───────────────────────────────────────────────────────

def _fmt(v: float, pct: bool = False) -> str:
    """Format a float for the results table.  Returns 'n/a' for NaN."""
    if np.isnan(v):
        return "   n/a"
    if pct:
        return f"{v:6.1%}"
    return f"{v:7.4f}"


def print_report(
    results: List[dict],
    total_queries: int,
    num_bad: int,
    avg_recall: float,
):
    """Print the full sweep results table and highlight the best setting."""

    print()
    print("=" * 95)
    print(f"  {DATASET.upper()} — CALIBRATION SWEEP RESULTS")
    print(f"  Queries: {total_queries}  |  Bad (R@{EVAL_K}<{BAD_RECALL_THRESHOLD}): {num_bad}  |  Mean R@{EVAL_K}: {avg_recall:.4f}")
    print("=" * 95)

    header = (
        f"  {'k':>3} | {'Rule':<10} | {'Flag%':>6} | "
        f"{'R@5 Flag':>8} | {'R@5 Stbl':>8} | {'Gap':>7} | "
        f"{'Bad%F':>6} | {'Bad%S':>6} | {'Sens':>6}"
    )
    sep = (
        f"  {'─'*3}─┼─{'─'*10}─┼─{'─'*6}─┼─"
        f"{'─'*8}─┼─{'─'*8}─┼─{'─'*7}─┼─"
        f"{'─'*6}─┼─{'─'*6}─┼─{'─'*6}"
    )
    print(header)
    print(sep)

    prev_k = None
    for r in results:
        if prev_k is not None and r["qpp_k"] != prev_k:
            print(sep)
        prev_k = r["qpp_k"]

        print(
            f"  {r['qpp_k']:>3} | {r['rule']:<10} | "
            f"{_fmt(r['flag_rate'], pct=True)} | "
            f"{_fmt(r['r5_flagged']):>8} | "
            f"{_fmt(r['r5_stable']):>8} | "
            f"{_fmt(r['gap']):>7} | "
            f"{_fmt(r['bad_pct_flagged'], pct=True)} | "
            f"{_fmt(r['bad_pct_stable'], pct=True)} | "
            f"{_fmt(r['sensitivity'], pct=True)}"
        )

    print("=" * 95)

    # ── Select best setting ────────────────────────────────────────
    #
    # Ranking logic:
    #   1. Only consider settings where gap > 0 (flagged recall < stable recall)
    #   2. Maximise gap, but penalise extreme flag rates
    #   3. Score = gap * weight, where weight is:
    #        1.0  if flag rate is in [5%, 50%]  (comfortable range)
    #        0.5  if flag rate is outside        (still considered, but penalised)
    #
    candidates = [
        r for r in results
        if not np.isnan(r["gap"]) and r["gap"] > 0
    ]

    def _rank_score(r: dict) -> float:
        weight = 1.0 if 0.05 <= r["flag_rate"] <= 0.50 else 0.5
        return r["gap"] * weight

    if candidates:
        best = max(candidates, key=_rank_score)
        in_range = "yes" if 0.05 <= best["flag_rate"] <= 0.50 else "no (penalised)"

        print()
        print("-" * 60)
        print(f"  BEST SETTING")
        print("-" * 60)
        print(f"  qpp_k            : {best['qpp_k']}")
        print(f"  Threshold rule   : {best['rule']}")
        print(f"  Flag rate        : {best['flag_rate']:.1%}  (in 5–50%: {in_range})")
        print(f"  R@5 flagged      : {best['r5_flagged']:.4f}")
        print(f"  R@5 stable       : {best['r5_stable']:.4f}")
        print(f"  Gap              : {best['gap']:.4f}")
        print(f"  Bad% flagged     : {_fmt(best['bad_pct_flagged'], pct=True).strip()}")
        print(f"  Bad% stable      : {_fmt(best['bad_pct_stable'], pct=True).strip()}")
        print(f"  Sensitivity      : {_fmt(best['sensitivity'], pct=True).strip()}")
        print("-" * 60)
    else:
        print()
        print("  No setting produced a positive gap (flagged R@5 < stable R@5).")
        print("  The distance signal may not predict query difficulty here.")


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RobustVDB — Hard-Query Calibration Sweep")
    print(f"  Dataset  : {DATASET}")
    print(f"  Retrieval held fixed — only flag parameters vary")
    print("=" * 60)

    # 1. Load dataset
    doc_texts, queries, ground_truth, eval_query_ids = load_dataset(DATASET)

    # 2. Embed corpus
    embedder = EmbeddingModel()
    print(f"[sweep] Embedding {len(doc_texts):,} corpus documents ...")
    doc_embeddings = embedder.embed_documents(doc_texts)

    # 3. Run fixed retrieval
    print(f"[sweep] Running retrieval (k={RETRIEVAL_K}) for {len(eval_query_ids)} queries ...")
    query_scores, query_indices = run_retrieval(
        doc_texts, doc_embeddings, queries, eval_query_ids, embedder,
    )

    # 4. Compute per-query Recall@5
    recalls = compute_recalls(query_indices, ground_truth, eval_query_ids, EVAL_K)

    # 5. Identify bad queries
    bad_qids = {qid for qid, r in recalls.items() if r < BAD_RECALL_THRESHOLD}

    # ── Dataset summary ──
    avg_recall = np.mean(list(recalls.values()))
    print()
    print("-" * 60)
    print(f"  {DATASET.upper()} — Dataset Summary")
    print("-" * 60)
    print(f"  Corpus size        : {len(doc_texts):,}")
    print(f"  Queries evaluated  : {len(eval_query_ids)}")
    print(f"  Mean Recall@{EVAL_K}     : {avg_recall:.4f}")
    print(f"  Bad queries (R@{EVAL_K}<{BAD_RECALL_THRESHOLD}): {len(bad_qids)}/{len(eval_query_ids)}")
    print("-" * 60)

    # 6. Run calibration sweep
    print()
    print("[sweep] Running calibration sweep ...")
    results = run_sweep(
        query_scores, recalls, bad_qids, eval_query_ids, doc_embeddings,
    )

    # 7. Print results
    print_report(results, len(eval_query_ids), len(bad_qids), avg_recall)


if __name__ == "__main__":
    main()
