import sys
import numpy as np
sys.path.append(r"C:\vectordb-project")
from robustvdb.core.search import RobustVDB

def main():
    docs = [
        "Python is a high-level programming language.",
        "FAISS is an index library for vector similarity search.",
        "Machine learning models often use vector embeddings."
    ]
    query = "What is Python?"
    required_keys = {
        "text", "vector_score", "keyword_overlap", 
        "matched_terms", "confidence", "robustness_flag"
    }

    try:
        # Default mean_distance mode
        print("\n--- 1. Default mean_distance mode ---")
        db1 = RobustVDB(baseline_distances=np.array([0.2, 0.3]))
        db1.add(docs)
        res1 = db1.search(query, k=1)[0]
        print("Raw result:", res1)
        missing1 = required_keys - set(res1.keys())
        extra1 = set(res1.keys()) - required_keys
        if missing1: print(f"Missing keys: {missing1}")
        if extra1: print(f"Extra keys: {extra1}")
        
        # Clarity mode
        print("\n--- 2. Clarity mode with qpp_k=3 and explicit qpp_threshold ---")
        db2 = RobustVDB(qpp_mode="clarity", qpp_threshold=0.1, qpp_k=3)
        db2.add(docs)
        res2 = db2.search(query, k=1)[0]
        print("Raw result:", res2)
        missing2 = required_keys - set(res2.keys())
        extra2 = set(res2.keys()) - required_keys
        if missing2: print(f"Missing keys: {missing2}")
        if extra2: print(f"Extra keys: {extra2}")

    except Exception as e:
        print(f"ERROR OCCURRED: {e}")

if __name__ == "__main__":
    main()
