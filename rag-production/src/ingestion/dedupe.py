"""Exact hash and cosine similarity near-duplicate detection."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Set, Optional, List, Tuple

from src.config.settings import settings
from src.config.logging_config import logger
from src.ingestion.embedder import EmbeddingModelWrapper
from src.ingestion.vectorstore import VectorStoreManager

if TYPE_CHECKING:
    from src.ingestion.chunkers import Chunk



@dataclass
class DedupeResult:
    is_duplicate: bool
    reason: str  # "exact_hash", "cosine_similarity", or "unique"
    similar_chunk_id: Optional[str] = None
    similarity_score: float = 0.0


class Deduplicator:
    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        similarity_threshold: float = settings.SEMANTIC_SIMILARITY_THRESHOLD,
    ):
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.seen_exact_hashes: Set[str] = set()

    def compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash of normalized text."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def check_exact_duplicate(self, text: str) -> bool:
        """Check if identical text has already been seen in the current pipeline run."""
        text_hash = self.compute_hash(text)
        if text_hash in self.seen_exact_hashes:
            return True
        self.seen_exact_hashes.add(text_hash)
        return False

    def check_near_duplicate(
        self, chunk_text: str, chunk_embedding: List[float]
    ) -> DedupeResult:
        """Two-stage deduplication: Exact hash check followed by ChromaDB Cosine Similarity check."""
        # 1. Exact hash check
        if self.check_exact_duplicate(chunk_text):
            return DedupeResult(
                is_duplicate=True,
                reason="exact_hash",
                similarity_score=1.0
            )

        # 2. Near-duplicate cosine similarity check against existing vector store
        if self.vector_store and self.vector_store.get_collection_count() > 0:
            query_res = self.vector_store.query_similar_embeddings(chunk_embedding, top_k=1)
            
            distances = query_res.get("distances", [[]])[0]
            ids = query_res.get("ids", [[]])[0]

            if distances and ids:
                # Cosine distance = 1 - cosine_similarity for normalized embeddings in ChromaDB
                cosine_dist = distances[0]
                cosine_sim = 1.0 - cosine_dist

                if cosine_sim >= self.similarity_threshold:
                    matched_id = ids[0]
                    logger.info(
                        f"Flagged near-duplicate chunk (similarity={cosine_sim:.4f} >= {self.similarity_threshold}): {matched_id}"
                    )
                    return DedupeResult(
                        is_duplicate=True,
                        reason="cosine_similarity",
                        similar_chunk_id=matched_id,
                        similarity_score=cosine_sim
                    )

        return DedupeResult(is_duplicate=False, reason="unique", similarity_score=0.0)

    def check_near_duplicates_batch(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> List[DedupeResult]:
        """Batch version of near-duplicate check."""
        results: List[Optional[DedupeResult]] = [None] * len(chunks)
        to_query_indices: List[int] = []
        to_query_embeddings: List[List[float]] = []

        # 1. Exact hash check
        for i, chunk in enumerate(chunks):
            if self.check_exact_duplicate(chunk.text):
                results[i] = DedupeResult(
                    is_duplicate=True,
                    reason="exact_hash",
                    similarity_score=1.0
                )
            else:
                to_query_indices.append(i)
                to_query_embeddings.append(embeddings[i])

        # 2. Near-duplicate cosine similarity check in batch against existing vector store
        if to_query_embeddings and self.vector_store and self.vector_store.get_collection_count() > 0:
            query_results = self.vector_store.query_similar_embeddings_batch(to_query_embeddings, top_k=1)
            
            for idx, query_res in zip(to_query_indices, query_results):
                distances = query_res.get("distances", [])
                ids = query_res.get("ids", [])

                if distances and ids:
                    cosine_dist = distances[0]
                    cosine_sim = 1.0 - cosine_dist

                    if cosine_sim >= self.similarity_threshold:
                        matched_id = ids[0]
                        logger.info(
                            f"Flagged near-duplicate chunk (similarity={cosine_sim:.4f} >= {self.similarity_threshold}): {matched_id}"
                        )
                        results[idx] = DedupeResult(
                            is_duplicate=True,
                            reason="cosine_similarity",
                            similar_chunk_id=matched_id,
                            similarity_score=cosine_sim
                        )

        # Fill remaining None with unique DedupeResult
        for i in range(len(results)):
            if results[i] is None:
                results[i] = DedupeResult(is_duplicate=False, reason="unique", similarity_score=0.0)

        return results


# Backward compatibility alias
HashDeduplicator = Deduplicator
