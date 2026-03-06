
def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    Compute recall@k for a single query.

    recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    If a query has zero relevant documents, returns 0.0.
    """
    if not relevant:
        return 0.0

    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def robustness_score(
    retrieved_ids: list[list],
    ground_truth: list[set],
    delta: float = 0.7,
    k: int = 5,
) -> float:
    """
    Compute Robustness-delta@K across multiple queries.

    Robustness-delta@K = fraction of queries whose recall@k >= delta.

    Args:
        retrieved_ids: List of retrieved document-id lists, one per query.
        ground_truth:  List of relevant document-id sets, one per query.
        delta:         Minimum recall threshold (between 0 and 1).
        k:             Number of top results to consider (positive int).

    Returns:
        Score between 0.0 and 1.0.
    """
    if len(retrieved_ids) != len(ground_truth):
        raise ValueError(
            f"retrieved_ids length ({len(retrieved_ids)}) must match "
            f"ground_truth length ({len(ground_truth)})."
        )
    if not (0.0 <= delta <= 1.0):
        raise ValueError(f"delta must be between 0 and 1, got {delta}.")
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}.")

    total = len(retrieved_ids)
    if total == 0:
        return 0.0

    passed = sum(
        1 for ret, rel in zip(retrieved_ids, ground_truth)
        if recall_at_k(ret, set(rel), k) >= delta
    )

    return passed / total
