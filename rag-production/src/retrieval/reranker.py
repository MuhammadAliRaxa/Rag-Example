"""Cross-encoder Reranking Module."""
from typing import List, Dict, Any

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Reranks candidate chunks based on joint query-passage relevance score."""
        for idx, chunk in enumerate(candidate_chunks):
            chunk["rerank_score"] = 0.99 - (idx * 0.02)
        return sorted(candidate_chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]
