"""Precision@k and Recall@k evaluation metrics."""
from typing import List

class RetrievalMetrics:
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        top_k = retrieved_ids[:k]
        hits = len(set(top_k).intersection(set(relevant_ids)))
        return hits / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        top_k = retrieved_ids[:k]
        hits = len(set(top_k).intersection(set(relevant_ids)))
        return hits / len(relevant_ids) if relevant_ids else 0.0
