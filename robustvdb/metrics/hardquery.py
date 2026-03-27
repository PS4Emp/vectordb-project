
import numpy as np


def _validate_1d_nonempty(arr: np.ndarray, name: str) -> None:
    """Validate that an array is 1D and non-empty."""
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")


def compute_threshold(baseline_distances: np.ndarray) -> float:
    """
    Compute the hard-query threshold from a fixed calibration baseline.

    threshold = mean(baseline_distances) + 1 * std(baseline_distances)
    """
    _validate_1d_nonempty(baseline_distances, "baseline_distances")
    return float(np.mean(baseline_distances) + np.std(baseline_distances))


def classify_query(avg_distance: float, threshold: float) -> str:
    """
    Classify a query as hard or stable.

    If the average distance exceeds the threshold,
    the query is in a sparse region of vector space.
    """
    if avg_distance > threshold:
        return "hard_query_warning"
    return "stable"


def hard_query_check(
    neighbor_scores: np.ndarray,
    baseline_distances: np.ndarray | None = None,
    qpp_mode: str = "mean_distance",
    qpp_threshold: float | None = None,
    qpp_k: int | None = None,
) -> str:
    """
    End-to-end hard-query detection for a single query.

    Args:
        neighbor_scores:    1D array of similarity scores from the query
                            to its top-k nearest neighbours.
        baseline_distances: 1D array of pre-computed average distances from
                            a calibration set (fixed, not updated dynamically).
                            Only used if qpp_mode == "mean_distance".
        qpp_mode:           "mean_distance" (default) or "clarity".
        qpp_threshold:      Explicit threshold (required for "clarity").
        qpp_k:              How many neighbor scores to consider. Defaults to all.

    Returns:
        "hard_query_warning" or "stable".
    """
    _validate_1d_nonempty(neighbor_scores, "neighbor_scores")

    scores_to_use = neighbor_scores[:qpp_k] if qpp_k is not None else neighbor_scores

    if qpp_mode == "clarity":
        if qpp_threshold is None:
            raise ValueError("qpp_mode 'clarity' requires an explicit qpp_threshold")
        if len(scores_to_use) > 1:
            margin = float(scores_to_use[0] - np.mean(scores_to_use[1:]))
        else:
            margin = 0.0
        qpp_score = -margin
        threshold = qpp_threshold

    elif qpp_mode == "mean_distance":
        distances = 1.0 - scores_to_use
        qpp_score = float(np.mean(distances))
        if qpp_threshold is not None:
            threshold = qpp_threshold
        elif baseline_distances is not None:
            threshold = compute_threshold(baseline_distances)
        else:
            return "stable"
    else:
        raise ValueError(f"Unknown qpp_mode: {qpp_mode}")

    return classify_query(qpp_score, threshold)
