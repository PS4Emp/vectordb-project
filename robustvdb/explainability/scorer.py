
import string


def tokenize(text: str) -> set[str]:
    """
    Normalize text into a set of unique tokens.

    Steps: lowercase -> remove punctuation -> split on whitespace -> deduplicate.
    """
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    return set(cleaned.split())


def compute_matched_terms(query: str, document: str) -> list[str]:
    """Return a sorted list of tokens that appear in both query and document."""
    query_tokens = tokenize(query)
    doc_tokens = tokenize(document)
    return sorted(query_tokens & doc_tokens)


def compute_keyword_overlap(query: str, document: str) -> float:
    """
    Compute keyword overlap between query and document.

    keyword_overlap = matched_query_terms / total_unique_query_terms
    Returns 0.0 if the query has zero valid tokens.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    doc_tokens = tokenize(document)
    matched = query_tokens & doc_tokens
    return len(matched) / len(query_tokens)


def compute_confidence(vector_score: float, keyword_overlap: float) -> str:
    """
    Determine confidence level from vector_score and keyword_overlap.

    Rules (locked):
      - high:   vector_score >= 0.80 AND keyword_overlap >= 0.50
      - medium: vector_score >= 0.60 AND keyword_overlap >= 0.25
      - low:    otherwise
    """
    if vector_score >= 0.80 and keyword_overlap >= 0.50:
        return "high"
    if vector_score >= 0.60 and keyword_overlap >= 0.25:
        return "medium"
    return "low"
