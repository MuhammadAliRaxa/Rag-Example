"""Free local embedding model wrapper using sentence-transformers."""
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from src.config.logging_config import logger


class EmbeddingModelWrapper:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Free, local embedding model — no API key or cost.

        Good default models:
        - "all-MiniLM-L6-v2"      -> fast, 384-dim, great for most use cases
        - "all-mpnet-base-v2"     -> slower, 768-dim, higher quality
        - "BAAI/bge-small-en-v1.5"-> strong retrieval performance, 384-dim
        """
        self.model_name = model_name
        logger.info(f"Loading local embedding model: {model_name}")
        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of document texts (batched)."""
        if not texts:
            return []

        # Replace empty strings with a single space to avoid issues
        clean_texts = [t if t.strip() else " " for t in texts]

        embeddings = self._model.encode(
            clean_texts,
            batch_size=100,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # makes cosine similarity = dot product
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single search query."""
        results = self.embed_documents([text])
        return results[0] if results else []

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# Shared singleton instance for process-wide model reuse
shared_embedder = EmbeddingModelWrapper()