
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
    baseline_distances: np.ndarray,
) -> str:
    """
    End-to-end hard-query detection for a single query.

    Since our index uses IndexFlatIP with normalised embeddings, search
    returns cosine similarity scores (higher = more similar).  We convert
    them to distance-like values so the threshold logic stays intuitive:

        distance = 1.0 - similarity

    Args:
        neighbor_scores:    1D array of similarity scores from the query
                            to its top-k nearest neighbours.
        baseline_distances: 1D array of pre-computed average distances from
                            a calibration set (fixed, not updated dynamically).

    Returns:
        "hard_query_warning" or "stable".
    """
    _validate_1d_nonempty(neighbor_scores, "neighbor_scores")

    # Convert similarity scores to distance-like values
    distances = 1.0 - neighbor_scores
    avg_distance = float(np.mean(distances))

    threshold = compute_threshold(baseline_distances)
    return classify_query(avg_distance, threshold)
