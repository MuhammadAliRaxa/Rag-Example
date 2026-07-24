"""Swappable embedding model wrapper."""
from typing import List

class EmbeddingModelWrapper:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of document texts."""
        return [[0.0] * 1536 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single search query."""
        return [0.0] * 1536
