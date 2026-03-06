# RobustVDB — BEIR Evaluation
#
# Requirements:
#   pip install beir
# SciFact (~6 MB) and FiQA-2018 (~18 MB) are downloaded automatically on first run.
#
# Run (Windows, project venv):
#   cd C:\vectordb-project
#   & .\.venv\Scripts\python.exe -m robustvdb.tests.beir_eval

import os
import sys
import faiss
from typing import Dict, List, Set

import numpy as np

try:
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader
except ImportError:
    print("ERROR: 'beir' package is required.  Install with:  pip install beir")
    sys.exit(1)

from robustvdb.core.embeddings import EmbeddingModel
from robustvdb.core.index import VectorIndex
from robustvdb.core.search import RobustVDB

DATASETS = ["scifact", "fiqa"]
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")
K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)


def corpus_entry_to_text(entry: dict) -> str:
    """Combine a BEIR corpus entry's title and text into a single string."""
    title = (entry.get("title") or "").strip()
    text = (entry.get("text") or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text or ""


def compute_calibration_baseline(doc_embeddings: np.ndarray, cal_k: int) -> np.ndarray:
    """
    Compute corpus-based calibration distances using FAISS nearest neighbours
    instead of a full NxN similarity matrix, so it scales to larger BEIR datasets.
    """
    emb = doc_embeddings.astype(np.float32).copy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-12)

    n, dim = emb.shape
    cal_k_clamped = min(cal_k, n - 1)

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    sims, _ = index.search(emb, cal_k_clamped + 1)  # includes self at rank 0
    sims = sims[:, 1:]  # drop self-similarity

    dists = 1.0 - sims
    baseline_distances = dists.mean(axis=1).astype(np.float32)
    return baseline_distances



def evaluate_dataset(dataset_name: str) -> dict:
    """Run the evaluation pipeline for a single dataset and return the metrics."""
    # -- Load dataset via BEIR --
    print(f"\n[beir_eval] [{dataset_name}] Downloading / locating dataset ...")
    data_path = beir_util.download_and_unzip(
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
        DOWNLOAD_DIR,
    )
    
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    assert len(corpus) > 0, "Corpus is empty."
    assert len(queries) > 0, "Queries are empty."
    assert len(qrels) > 0, "Qrels are empty."

    print(f"[beir_eval] [{dataset_name}] Corpus : {len(corpus):,} docs")
    print(f"[beir_eval] [{dataset_name}] Queries: {len(queries):,}")
    print(f"[beir_eval] [{dataset_name}] Qrels  : {len(qrels):,}")

    # -- Stable BEIR-doc-ID → local-integer-ID mapping --
    sorted_corpus_ids: List[str] = sorted(corpus.keys())
    beir_id_to_local: Dict[str, int] = {
        beir_id: idx for idx, beir_id in enumerate(sorted_corpus_ids)
    }

    doc_texts: List[str] = [
        corpus_entry_to_text(corpus[beir_id]) for beir_id in sorted_corpus_ids
    ]

    # -- Ground truth: query_id → set of relevant local IDs (score > 0) --
    ground_truth: Dict[str, Set[int]] = {}
    for query_id, doc_judgments in qrels.items():
        relevant = {
            beir_id_to_local[doc_id]
            for doc_id, score in doc_judgments.items()
            if score > 0 and doc_id in beir_id_to_local
        }
        if relevant:
            ground_truth[query_id] = relevant

    eval_query_ids: List[str] = sorted(ground_truth.keys())
    print(f"[beir_eval] [{dataset_name}] Queries with ground-truth: {len(eval_query_ids):,}")

    # -- Embed corpus and build baseline FAISS index --
    print(f"[beir_eval] [{dataset_name}] Embedding {len(doc_texts):,} corpus documents ...")
    embedder = EmbeddingModel()
    doc_embeddings = embedder.embed_documents(doc_texts)

    dimension = doc_embeddings.shape[1]
    baseline_index = VectorIndex(dimension)
    baseline_index.add_documents(doc_texts, doc_embeddings)

    # -- Run baseline dense retrieval --
    baseline_results: Dict[str, Dict[int, List[int]]] = {}
    for query_id in eval_query_ids:
        query_text = queries[query_id]
        query_embedding = embedder.embed_query(query_text)
        scores, indices = baseline_index.search(query_embedding, k=MAX_K)
        retrieved_ids: List[int] = indices[0].tolist()
        baseline_results[query_id] = {k: retrieved_ids[:k] for k in K_VALUES}

    # -- Compute calibration baseline --
    print(f"[beir_eval] [{dataset_name}] Computing calibration baseline ...")
    baseline_distances = compute_calibration_baseline(doc_embeddings, cal_k=MAX_K)

    # -- Initialise RobustVDB and add corpus --
    db = RobustVDB(baseline_distances=baseline_distances)
    db.add(doc_texts)

    # Build duplicate-safe mapping from text to all local IDs that share that text
    text_to_local_ids: Dict[str, List[int]] = {}
    for idx, text in enumerate(doc_texts):
        if text not in text_to_local_ids:
            text_to_local_ids[text] = []
        text_to_local_ids[text].append(idx)

    # -- Run RobustVDB retrieval --
    robustdb_results: Dict[str, Dict[int, List[int]]] = {}
    robustdb_flags: Dict[str, str] = {}

    for query_id in eval_query_ids:
        query_text = queries[query_id]
        results = db.search(query_text, k=MAX_K)
        
        retrieved_ids = []
        used_ids = set()
        
        for r in results:
            text = r["text"]
            candidates = text_to_local_ids[text]
            
            # Find an ID that hasn't been used for this query yet
            chosen_id = None
            for cid in candidates:
                if cid not in used_ids:
                    chosen_id = cid
                    break
            
            # If all candidates for this exact text have been used, fall back to the first
            if chosen_id is None:
                chosen_id = candidates[0]
                
            retrieved_ids.append(chosen_id)
            used_ids.add(chosen_id)

        robustdb_results[query_id] = {k: retrieved_ids[:k] for k in K_VALUES}
        robustdb_flags[query_id] = results[0]["robustness_flag"] if results else "n/a"

    # -- Compute metrics --
    from robustvdb.metrics.robustness import recall_at_k, robustness_score
    print(f"[beir_eval] [{dataset_name}] Computing metrics...")

    gt_list = [ground_truth[qid] for qid in eval_query_ids]
    base_ret_5 = [baseline_results[qid][5] for qid in eval_query_ids]
    base_ret_10 = [baseline_results[qid][10] for qid in eval_query_ids]
    rvdb_ret_5 = [robustdb_results[qid][5] for qid in eval_query_ids]
    rvdb_ret_10 = [robustdb_results[qid][10] for qid in eval_query_ids]
    
    # Baseline
    base_r5 = np.mean([recall_at_k(ret, gt, 5) for ret, gt in zip(base_ret_5, gt_list)])
    base_r10 = np.mean([recall_at_k(ret, gt, 10) for ret, gt in zip(base_ret_10, gt_list)])
    base_rob_05_5 = robustness_score(base_ret_5, gt_list, delta=0.5, k=5)
    base_rob_07_5 = robustness_score(base_ret_5, gt_list, delta=0.7, k=5)

    # RobustVDB
    rvdb_r5 = np.mean([recall_at_k(ret, gt, 5) for ret, gt in zip(rvdb_ret_5, gt_list)])
    rvdb_r10 = np.mean([recall_at_k(ret, gt, 10) for ret, gt in zip(rvdb_ret_10, gt_list)])
    rvdb_rob_05_5 = robustness_score(rvdb_ret_5, gt_list, delta=0.5, k=5)
    rvdb_rob_07_5 = robustness_score(rvdb_ret_5, gt_list, delta=0.7, k=5)
    
    # Hard-Query Flag Analysis
    flags = [robustdb_flags[qid] for qid in eval_query_ids]
    flagged_mask = [f == "hard_query_warning" for f in flags]
    stable_mask = [f == "stable" for f in flags]

    num_flagged = sum(flagged_mask)
    num_stable = sum(stable_mask)
    flag_rate = num_flagged / len(eval_query_ids) if eval_query_ids else 0.0

    # Recall on subsets
    flagged_r5 = "n/a"
    flagged_bad_count = "n/a"
    flagged_bad_frac = "n/a"
    if num_flagged > 0:
        flagged_recalls = [
            recall_at_k(rvdb_ret_5[i], gt_list[i], 5) 
            for i in range(len(eval_query_ids)) if flagged_mask[i]
        ]
        flagged_r5 = np.mean(flagged_recalls)
        flagged_bad_count = sum(1 for r in flagged_recalls if r < 0.5)
        flagged_bad_frac = flagged_bad_count / num_flagged

    stable_r5 = "n/a"
    stable_bad_count = "n/a"
    stable_bad_frac = "n/a"
    if num_stable > 0:
        stable_recalls = [
            recall_at_k(rvdb_ret_5[i], gt_list[i], 5) 
            for i in range(len(eval_query_ids)) if stable_mask[i]
        ]
        stable_r5 = np.mean(stable_recalls)
        stable_bad_count = sum(1 for r in stable_recalls if r < 0.5)
        stable_bad_frac = stable_bad_count / num_stable

    # -- Per-dataset Summary --
    print("\n" + "-" * 60)
    print(f"[{dataset_name.upper()}] SUMMARY")
    print("-" * 60)
    print("Baseline (Plain Dense)")
    print(f"  Recall@5           : {base_r5:.4f}")
    print(f"  Recall@10          : {base_r10:.4f}")
    print(f"  Robustness-0.5@5   : {base_rob_05_5:.4f}")
    print(f"  Robustness-0.7@5   : {base_rob_07_5:.4f}")
    print("-" * 60)
    print("RobustVDB")
    print(f"  Recall@5           : {rvdb_r5:.4f}")
    print(f"  Recall@10          : {rvdb_r10:.4f}")
    print(f"  Robustness-0.5@5   : {rvdb_rob_05_5:.4f}")
    print(f"  Robustness-0.7@5   : {rvdb_rob_07_5:.4f}")
    print("-" * 60)
    print("Hard-Query Diagnostics")
    print(f"  Queries Evaluated  : {len(eval_query_ids)}")
    print(f"  Flag Rate          : {flag_rate:.1%} ({num_flagged} flagged)")
    
    f_r5_str = f"{flagged_r5:.4f}" if isinstance(flagged_r5, float) else flagged_r5
    s_r5_str = f"{stable_r5:.4f}" if isinstance(stable_r5, float) else stable_r5
    
    f_bad_str = f"{flagged_bad_count}/{num_flagged} ({flagged_bad_frac:.1%})" if isinstance(flagged_bad_frac, float) else "n/a"
    s_bad_str = f"{stable_bad_count}/{num_stable} ({stable_bad_frac:.1%})" if isinstance(stable_bad_frac, float) else "n/a"
    
    print(f"  Recall@5 (Flagged) : {f_r5_str}")
    print(f"  Recall@5 (Stable)  : {s_r5_str}")
    print(f"  Bad Qs (<0.5) Flagged : {f_bad_str}")
    print(f"  Bad Qs (<0.5) Stable  : {s_bad_str}")
    print("-" * 60)

    return {
        "dataset": dataset_name,
        "base_r5": base_r5,
        "base_rob_05_5": base_rob_05_5,
        "rvdb_r5": rvdb_r5,
        "rvdb_rob_05_5": rvdb_rob_05_5,
        "flag_rate": flag_rate,
        "flagged_r5": flagged_r5,
        "stable_r5": stable_r5,
    }


def main():
    all_metrics = []
    for dset in DATASETS:
        all_metrics.append(evaluate_dataset(dset))

    # -- Cross-dataset Summary Table --
    print("\n" + "=" * 115)
    print("ROBUSTVDB CROSS-DATASET EVALUATION SUMMARY")
    print("=" * 115)
    
    header = (
        f"{'Dataset':<12} | "
        f"{'Base R@5':<10} | "
        f"{'RVDB R@5':<10} | "
        f"{'Base Rob0.5':<13} | "
        f"{'RVDB Rob0.5':<13} | "
        f"{'Flag Rate':<10} | "
        f"{'R@5 Flagged':<12} | "
        f"{'R@5 Stable':<10}"
    )
    print(header)
    print("-" * 115)

    for m in all_metrics:
        f_r5 = f"{m['flagged_r5']:.4f}" if isinstance(m['flagged_r5'], float) else str(m['flagged_r5'])
        s_r5 = f"{m['stable_r5']:.4f}" if isinstance(m['stable_r5'], float) else str(m['stable_r5'])
        
        row = (
            f"{m['dataset']:<12} | "
            f"{m['base_r5']:<10.4f} | "
            f"{m['rvdb_r5']:<10.4f} | "
            f"{m['base_rob_05_5']:<13.4f} | "
            f"{m['rvdb_rob_05_5']:<13.4f} | "
            f"{m['flag_rate']:<10.1%} | "
            f"{f_r5:<12} | "
            f"{s_r5:<10}"
        )
        print(row)
    print("=" * 115)


if __name__ == "__main__":
    main()
