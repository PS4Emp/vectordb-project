# RobustVDB -- QPP Signal Ablation
#
# Compares lightweight query-difficulty signals on a fixed dense retriever.
# Evaluates on BEIR SciFact and FIQA without modifying the retrieval pipeline.
#
# Signals compared:
#   1. MeanDist     -- mean distance to top-k neighbours
#   2. MeanDist+Std -- mean distance + distance std (penalises spread)
#   3. Clarity      -- gap between top-1 score and mean of rest (low = hard)
#   4. LexDisagree  -- vector score minus keyword overlap (high = semantics-only)
#
# Run:
#   cd C:\vectordb-project
#   & .\.venv\Scripts\python.exe -u -m robustvdb.tests.qpp_ablation

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
from robustvdb.explainability.scorer import compute_keyword_overlap

# -- Configuration --------------------------------------------------------

DATASETS = ["scifact", "fiqa"]
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")
RETRIEVAL_K = 10
EVAL_K = 5
BAD_RECALL_THRESHOLD = 0.5
QPP_K = 3  # fixed from calibration sweep findings

# Percentile thresholds: queries with difficulty above P_x are flagged.
# P50 -> ~50% flagged, P90 -> ~10% flagged.
PERCENTILES = [50, 60, 70, 80, 90]

SIGNAL_NAMES = ["MeanDist", "MeanDist+Std", "Clarity", "LexDisagree"]


# -- Dataset loading (same as calibration_sweep.py) -----------------------

def corpus_entry_to_text(entry: dict) -> str:
    title = (entry.get("title") or "").strip()
    text = (entry.get("text") or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text or ""


def load_dataset(name: str):
    print(f"[qpp] Downloading / locating {name} ...")
    data_path = beir_util.download_and_unzip(
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip",
        DOWNLOAD_DIR,
    )
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    sorted_ids = sorted(corpus.keys())
    id_map = {bid: idx for idx, bid in enumerate(sorted_ids)}
    doc_texts = [corpus_entry_to_text(corpus[bid]) for bid in sorted_ids]

    ground_truth: Dict[str, Set[int]] = {}
    for qid, judgments in qrels.items():
        relevant = {
            id_map[did] for did, score in judgments.items()
            if score > 0 and did in id_map
        }
        if relevant:
            ground_truth[qid] = relevant

    eval_qids = sorted(ground_truth.keys())
    return doc_texts, queries, ground_truth, eval_qids


# -- Retrieval (fixed) ----------------------------------------------------

def run_retrieval(doc_texts, doc_embeddings, queries, eval_qids, embedder):
    dim = doc_embeddings.shape[1]
    index = VectorIndex(dim)
    index.add_documents(doc_texts, doc_embeddings)

    q_scores: Dict[str, np.ndarray] = {}
    q_indices: Dict[str, np.ndarray] = {}
    for qid in eval_qids:
        qe = embedder.embed_query(queries[qid])
        scores, indices = index.search(qe, k=RETRIEVAL_K)
        q_scores[qid] = scores[0]
        q_indices[qid] = indices[0]
    return q_scores, q_indices


# -- Signal computation ---------------------------------------------------

def compute_signals(
    eval_qids: List[str],
    queries: dict,
    query_scores: Dict[str, np.ndarray],
    query_indices: Dict[str, np.ndarray],
    doc_texts: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute all four difficulty signals for every query.

    Returns dict of signal_name -> {qid -> difficulty_score}.
    Higher score = harder query for all signals.
    """
    signals: Dict[str, Dict[str, float]] = {name: {} for name in SIGNAL_NAMES}

    for qid in eval_qids:
        sims = query_scores[qid][:QPP_K]
        idxs = query_indices[qid][:QPP_K]
        query_text = queries[qid]

        dists = 1.0 - sims

        # 1. MeanDist: average distance to top-k
        signals["MeanDist"][qid] = float(np.mean(dists))

        # 2. MeanDist+Std: penalise high spread in distances
        signals["MeanDist+Std"][qid] = float(np.mean(dists) + np.std(dists))

        # 3. Clarity: margin between top-1 and mean of rest
        #    Low margin = ambiguous retrieval = hard query
        #    Invert so higher = harder
        if len(sims) > 1:
            margin = float(sims[0] - np.mean(sims[1:]))
        else:
            margin = 0.0
        signals["Clarity"][qid] = -margin  # negate: high = hard

        # 4. LexDisagree: mean vector score minus mean keyword overlap
        #    High = vector says "match" but no lexical evidence
        overlaps = [
            compute_keyword_overlap(query_text, doc_texts[int(idx)])
            for idx in idxs if idx != -1
        ]
        avg_sim = float(np.mean(sims))
        avg_ovl = float(np.mean(overlaps)) if overlaps else 0.0
        signals["LexDisagree"][qid] = avg_sim - avg_ovl

    return signals


# -- Threshold sweep per signal -------------------------------------------

def sweep_signal(
    signal_values: Dict[str, float],
    recalls: Dict[str, float],
    bad_qids: Set[str],
    eval_qids: List[str],
) -> List[dict]:
    """Sweep percentile thresholds for one signal. Returns list of result dicts."""
    total = len(eval_qids)
    vals = np.array([signal_values[qid] for qid in eval_qids])
    results = []

    for pct in PERCENTILES:
        threshold = float(np.percentile(vals, pct))
        flagged = [qid for qid in eval_qids if signal_values[qid] > threshold]
        stable  = [qid for qid in eval_qids if signal_values[qid] <= threshold]

        flag_rate = len(flagged) / total
        r5_flag = float(np.mean([recalls[qid] for qid in flagged])) if flagged else float("nan")
        r5_stbl = float(np.mean([recalls[qid] for qid in stable]))  if stable  else float("nan")
        gap = (r5_stbl - r5_flag) if not (np.isnan(r5_stbl) or np.isnan(r5_flag)) else float("nan")

        bad_f = sum(1 for qid in flagged if qid in bad_qids)
        bad_s = sum(1 for qid in stable  if qid in bad_qids)
        bpct_f = bad_f / len(flagged) if flagged else float("nan")
        bpct_s = bad_s / len(stable)  if stable  else float("nan")

        results.append({
            "pct": pct, "flag_rate": flag_rate,
            "r5_flag": r5_flag, "r5_stbl": r5_stbl, "gap": gap,
            "bpct_f": bpct_f, "bpct_s": bpct_s,
        })
    return results


# -- Output ----------------------------------------------------------------

def _fmt(v, pct=False):
    if isinstance(v, float) and np.isnan(v):
        return "   n/a"
    return f"{v:6.1%}" if pct else f"{v:7.4f}"


def print_signal_table(signal_name, results):
    print(f"\n  Signal: {signal_name}")
    print(f"  {'Pct':>5} | {'Flag%':>6} | {'R@5 Flag':>8} | {'R@5 Stbl':>8} | {'Gap':>7} | {'Bad%F':>6} | {'Bad%S':>6}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}")
    for r in results:
        print(
            f"  P{r['pct']:<4}| "
            f"{_fmt(r['flag_rate'], pct=True)} | "
            f"{_fmt(r['r5_flag']):>8} | "
            f"{_fmt(r['r5_stbl']):>8} | "
            f"{_fmt(r['gap']):>7} | "
            f"{_fmt(r['bpct_f'], pct=True)} | "
            f"{_fmt(r['bpct_s'], pct=True)}"
        )


def print_best_per_signal(all_signal_results, dataset_name):
    """Print best setting per signal, then rank signals against each other."""
    print(f"\n  {'='*70}")
    print(f"  {dataset_name.upper()} -- BEST SETTING PER SIGNAL  (qpp_k={QPP_K})")
    print(f"  {'='*70}")

    bests = []
    for sig_name in SIGNAL_NAMES:
        results = all_signal_results[sig_name]
        valid = [r for r in results if not np.isnan(r["gap"]) and r["gap"] > 0
                 and 0.05 <= r["flag_rate"] <= 0.50]
        if valid:
            best = max(valid, key=lambda r: r["gap"])
        else:
            # fall back to best positive gap at any flag rate
            pos = [r for r in results if not np.isnan(r["gap"]) and r["gap"] > 0]
            best = max(pos, key=lambda r: r["gap"]) if pos else None

        if best:
            bests.append((sig_name, best))
            flag_ok = "yes" if 0.05 <= best["flag_rate"] <= 0.50 else "no"
            print(f"\n  {sig_name}")
            print(f"    Threshold  : P{best['pct']}")
            print(f"    Flag rate  : {best['flag_rate']:.1%}  (in 5-50%: {flag_ok})")
            print(f"    R@5 flag   : {best['r5_flag']:.4f}")
            print(f"    R@5 stable : {best['r5_stbl']:.4f}")
            print(f"    Gap        : {best['gap']:.4f}")
            print(f"    Bad% flag  : {_fmt(best['bpct_f'], pct=True).strip()}")
            print(f"    Bad% stable: {_fmt(best['bpct_s'], pct=True).strip()}")
        else:
            print(f"\n  {sig_name}")
            print(f"    No positive gap at any threshold.")

    # Rank signals by best gap
    if bests:
        bests.sort(key=lambda x: x[1]["gap"], reverse=True)
        print(f"\n  {'='*70}")
        print(f"  SIGNAL RANKING (by best gap)")
        print(f"  {'='*70}")
        for rank, (name, best) in enumerate(bests, 1):
            print(f"  {rank}. {name:<15} gap={best['gap']:.4f}  flag={best['flag_rate']:.0%}  bad%F={_fmt(best['bpct_f'], pct=True).strip()}")


# -- Main ------------------------------------------------------------------

def evaluate_dataset(dataset_name, embedder):
    print(f"\n{'='*60}")
    print(f"  QPP Ablation -- {dataset_name.upper()}")
    print(f"{'='*60}")

    doc_texts, queries, ground_truth, eval_qids = load_dataset(dataset_name)

    print(f"[qpp] Embedding {len(doc_texts):,} corpus documents ...")
    doc_embeddings = embedder.embed_documents(doc_texts)

    print(f"[qpp] Running retrieval (k={RETRIEVAL_K}) for {len(eval_qids)} queries ...")
    q_scores, q_indices = run_retrieval(
        doc_texts, doc_embeddings, queries, eval_qids, embedder,
    )

    recalls = {
        qid: recall_at_k(q_indices[qid].tolist(), ground_truth[qid], EVAL_K)
        for qid in eval_qids
    }
    bad_qids = {qid for qid, r in recalls.items() if r < BAD_RECALL_THRESHOLD}
    avg_recall = np.mean(list(recalls.values()))

    print(f"\n  Corpus: {len(doc_texts):,}  |  Queries: {len(eval_qids)}")
    print(f"  Mean R@{EVAL_K}: {avg_recall:.4f}  |  Bad (R@{EVAL_K}<{BAD_RECALL_THRESHOLD}): {len(bad_qids)}/{len(eval_qids)}")

    # Compute all signals
    print(f"\n[qpp] Computing difficulty signals (qpp_k={QPP_K}) ...")
    signals = compute_signals(eval_qids, queries, q_scores, q_indices, doc_texts)

    # Sweep each signal
    all_results: Dict[str, List[dict]] = {}
    for sig_name in SIGNAL_NAMES:
        results = sweep_signal(signals[sig_name], recalls, bad_qids, eval_qids)
        all_results[sig_name] = results
        print_signal_table(sig_name, results)

    # Best per signal + ranking
    print_best_per_signal(all_results, dataset_name)


def main():
    print("=" * 60)
    print("  RobustVDB -- QPP Signal Ablation")
    print(f"  Comparing 4 query-difficulty signals on BEIR")
    print(f"  Retriever fixed  |  qpp_k={QPP_K}")
    print("=" * 60)

    embedder = EmbeddingModel()

    for dset in DATASETS:
        evaluate_dataset(dset, embedder)

    print(f"\n{'='*60}")
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
