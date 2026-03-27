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

    # Build duplicate-safe mapping from text to all local IDs that share that text
    text_to_local_ids: Dict[str, List[int]] = {}
    for idx, text in enumerate(doc_texts):
        if text not in text_to_local_ids:
            text_to_local_ids[text] = []
        text_to_local_ids[text].append(idx)

    # -- Initialise RobustVDB instances --
    db_md = RobustVDB(baseline_distances=baseline_distances, qpp_mode="mean_distance")
    db_md.index = baseline_index
    
    clarity_thresh = {"scifact": -0.026044, "fiqa": -0.037147}[dataset_name]
    db_cl = RobustVDB(qpp_mode="clarity", qpp_threshold=clarity_thresh, qpp_k=3)
    db_cl.index = baseline_index

    # -- Run RobustVDB retrieval (mean_distance) --
    print(f"[beir_eval] [{dataset_name}] Evaluating RobustVDB (mean_distance) ...")
    rvdb_results: Dict[str, Dict[int, List[int]]] = {}
    md_flags: Dict[str, str] = {}
    for query_id in eval_query_ids:
        query_text = queries[query_id]
        results = db_md.search(query_text, k=MAX_K)
        retrieved_ids = []
        used_ids = set()
        for r in results:
            candidates = text_to_local_ids[r["text"]]
            chosen_id = next((cid for cid in candidates if cid not in used_ids), candidates[0])
            retrieved_ids.append(chosen_id)
            used_ids.add(chosen_id)
        rvdb_results[query_id] = {k: retrieved_ids[:k] for k in K_VALUES}
        md_flags[query_id] = results[0]["robustness_flag"] if results else "n/a"

    # -- Run RobustVDB retrieval (clarity) --
    print(f"[beir_eval] [{dataset_name}] Evaluating RobustVDB (clarity, thresh={clarity_thresh:.4f}, qpp_k=3) ...")
    cl_flags: Dict[str, str] = {}
    for query_id in eval_query_ids:
        results = db_cl.search(queries[query_id], k=MAX_K)
        cl_flags[query_id] = results[0]["robustness_flag"] if results else "n/a"

    # -- Compute metrics --
    from robustvdb.metrics.robustness import recall_at_k, robustness_score
    print(f"[beir_eval] [{dataset_name}] Computing metrics...")

    gt_list = [ground_truth[qid] for qid in eval_query_ids]
    base_ret_5 = [baseline_results[qid][5] for qid in eval_query_ids]
    base_ret_10 = [baseline_results[qid][10] for qid in eval_query_ids]
    rvdb_ret_5 = [rvdb_results[qid][5] for qid in eval_query_ids]
    rvdb_ret_10 = [rvdb_results[qid][10] for qid in eval_query_ids]
    
    base_r5 = np.mean([recall_at_k(ret, gt, 5) for ret, gt in zip(base_ret_5, gt_list)])
    base_r10 = np.mean([recall_at_k(ret, gt, 10) for ret, gt in zip(base_ret_10, gt_list)])
    base_rob = robustness_score(base_ret_5, gt_list, delta=0.5, k=5)
    
    rvdb_r5 = np.mean([recall_at_k(ret, gt, 5) for ret, gt in zip(rvdb_ret_5, gt_list)])
    rvdb_r10 = np.mean([recall_at_k(ret, gt, 10) for ret, gt in zip(rvdb_ret_10, gt_list)])
    rvdb_rob = robustness_score(rvdb_ret_5, gt_list, delta=0.5, k=5)
    
    def analyze_flags(flags_dict):
        flags = [flags_dict[qid] for qid in eval_query_ids]
        flag_mask = [f == "hard_query_warning" for f in flags]
        stbl_mask = [f == "stable" for f in flags]
        flag_rate = sum(flag_mask) / len(flags) if flags else 0.0
        
        f_r5, f_bad = "n/a", "n/a"
        if sum(flag_mask) > 0:
            recalls = [recall_at_k(rvdb_ret_5[i], gt_list[i], 5) for i in range(len(flags)) if flag_mask[i]]
            f_r5 = np.mean(recalls)
            f_bad = sum(1 for r in recalls if r < 0.5) / len(recalls)
            
        s_r5, s_bad = "n/a", "n/a"
        if sum(stbl_mask) > 0:
            recalls = [recall_at_k(rvdb_ret_5[i], gt_list[i], 5) for i in range(len(flags)) if stbl_mask[i]]
            s_r5 = np.mean(recalls)
            s_bad = sum(1 for r in recalls if r < 0.5) / len(recalls)
            
        return flag_rate, f_r5, s_r5, f_bad, s_bad

    md_stats = analyze_flags(md_flags)
    cl_stats = analyze_flags(cl_flags)

    return {
        "dataset": dataset_name,
        "base_r5": base_r5,
        "base_r10": base_r10,
        "base_rob": base_rob,
        "rvdb_r5": rvdb_r5,
        "rvdb_r10": rvdb_r10,
        "rvdb_rob": rvdb_rob,
        "md_stats": md_stats,
        "cl_stats": cl_stats,
    }

def main():
    all_metrics = []
    for dset in DATASETS:
        all_metrics.append(evaluate_dataset(dset))

    # -- Cross-dataset Summary Table --
    print("\n" + "=" * 115)
    print("ROBUSTVDB CROSS-DATASET EVALUATION SUMMARY")
    print("=" * 115)
    
    with open("C:\\vectordb-project\\beir_eval_results.txt", "w") as f:
        f.write("=" * 115 + "\n")
        f.write("ROBUSTVDB CROSS-DATASET EVALUATION SUMMARY\n")
        f.write("=" * 115 + "\n")
    
    header = (
        f"{'Dataset':<10} | {'Mode':<14} | "
        f"{'R@5':<6} | {'R@10':<6} | {'Rob0.5':<7} | "
        f"{'Flag%':<6} | {'R@5(F)':<8} | {'R@5(S)':<8} | "
        f"{'Bad%(F)':<8} | {'Bad%(S)':<8}"
    )
    print(header)
    print("-" * 115)
    with open("C:\\vectordb-project\\beir_eval_results.txt", "a") as f:
        f.write(header + "\n")
        f.write("-" * 115 + "\n")

    def _fmt(v, is_pct=False):
        if isinstance(v, str): return f"{v:>8}"
        return f"{v:>8.1%}" if is_pct else f"{v:>8.4f}"

    for m in all_metrics:
        row_base = (
            f"{m['dataset']:<10} | {'baseline':<14} | "
            f"{m['base_r5']:<6.4f} | {m['base_r10']:<6.4f} | {m['base_rob']:<7.4f} | "
            f"{'':>6} | {'':>8} | {'':>8} | {'':>8} | {'':>8}"
        )
        print(row_base)
        
        md_fr, md_fr5, md_sr5, md_fbad, md_sbad = m["md_stats"]
        row_md = (
            f"{'':<10} | {'mean_distance':<14} | "
            f"{m['rvdb_r5']:<6.4f} | {m['rvdb_r10']:<6.4f} | {m['rvdb_rob']:<7.4f} | "
            f"{md_fr:>6.1%} | {_fmt(md_fr5)} | {_fmt(md_sr5)} | {_fmt(md_fbad, True)} | {_fmt(md_sbad, True)}"
        )
        print(row_md)

        cl_fr, cl_fr5, cl_sr5, cl_fbad, cl_sbad = m["cl_stats"]
        row_cl = (
            f"{'':<10} | {'clarity':<14} | "
            f"{m['rvdb_r5']:<6.4f} | {m['rvdb_r10']:<6.4f} | {m['rvdb_rob']:<7.4f} | "
            f"{cl_fr:>6.1%} | {_fmt(cl_fr5)} | {_fmt(cl_sr5)} | {_fmt(cl_fbad, True)} | {_fmt(cl_sbad, True)}"
        )
        print(row_cl)
        print("-" * 115)
        
        # Save to file to ensure it's not lost to console buffer
        with open("C:\\vectordb-project\\beir_eval_results.txt", "a") as f:
            f.write(row_base + "\n")
            f.write(row_md + "\n")
            f.write(row_cl + "\n")
            f.write("-" * 115 + "\n")
            
if __name__ == "__main__":
    main()
