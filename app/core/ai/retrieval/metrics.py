from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any


def calculate_recall_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Collection[Any],
    k: int,
) -> float:
    """Calculate Recall@K: fraction of known relevant items found in top-k results.

    Args:
        retrieved_ids: Ordered sequence of retrieved identifiers.
        relevant_ids: Collection of ground-truth relevant identifiers.
        k: Cutoff rank (must be > 0).

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if relevant_ids is empty or k <= 0.
    """
    relevant_set = set(relevant_ids)
    if not relevant_set or k <= 0:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    return len(retrieved_set & relevant_set) / len(relevant_set)


def calculate_precision_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Collection[Any],
    k: int,
) -> float:
    """Calculate Precision@K: fraction of top-k retrieved items that are relevant.

    Args:
        retrieved_ids: Ordered sequence of retrieved identifiers.
        relevant_ids: Collection of ground-truth relevant identifiers.
        k: Cutoff rank (must be > 0).

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if k <= 0 or no items were retrieved.
    """
    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    relevant_set = set(relevant_ids)
    retrieved_set = set(top_k)
    return len(retrieved_set & relevant_set) / len(top_k)
