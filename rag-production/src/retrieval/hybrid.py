"""Merges Vector + Keyword Search Results (Reciprocal Rank Fusion)."""
from typing import List, Dict, Any
from src.retrieval.vector_search import VectorSearchEngine
from src.retrieval.keyword_search import KeywordSearchEngine

class HybridSearchEngine:
    def __init__(self):
        self.vector_engine = VectorSearchEngine()
        self.keyword_engine = KeywordSearchEngine()

    def search(self, query: str, query_vector: List[float], top_k: int = 5, rr_k: int = 60) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF) implementation."""
        vec_results = self.vector_engine.search(query_vector, top_k=top_k)
        kw_results = self.keyword_engine.search(query, top_k=top_k)

        scores: Dict[str, float] = {}
        chunks_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(vec_results):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rr_k + rank + 1))
            chunks_map[cid] = item

        for rank, item in enumerate(kw_results):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rr_k + rank + 1))
            chunks_map[cid] = item

        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [chunks_map[cid] for cid, _ in sorted_chunks]
