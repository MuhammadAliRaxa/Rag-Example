"""OpenAI Embedding model wrapper using text-embedding-3-small."""
import numpy as np
from typing import List, Optional
from openai import OpenAI
from src.config.settings import settings
from src.config.logging_config import logger


class EmbeddingModelWrapper:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or settings.OPENAI_API_KEY
        self._client: Optional[OpenAI] = None
        if self.api_key:
            self._client = OpenAI(api_key=self.api_key)

    def _get_client(self) -> OpenAI:
        if not self._client:
            key = self.api_key or settings.OPENAI_API_KEY
            if not key:
                raise ValueError("OPENAI_API_KEY is missing. Please set it in environment or .env file.")
            self._client = OpenAI(api_key=key)
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of document texts (batched)."""
        if not texts:
            return []
            
        client = self._get_client()
        batch_size = 100
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Replace empty strings with a single space to avoid API error
            clean_batch = [t if t.strip() else " " for t in batch]
            response = client.embeddings.create(
                input=clean_batch,
                model=self.model_name
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

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
