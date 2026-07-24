import pytest
from src.ingestion.dedupe import Deduplicator, DedupeResult


def test_exact_hash_deduplication():
    deduper = Deduplicator(vector_store=None)
    text = "Unique content string for testing deduplication."

    res1 = deduper.check_near_duplicate(text, [0.1] * 1536)
    assert res1.is_duplicate is False
    assert res1.reason == "unique"

    res2 = deduper.check_near_duplicate(text, [0.1] * 1536)
    assert res2.is_duplicate is True
    assert res2.reason == "exact_hash"
    assert res2.similarity_score == 1.0


class DummyVectorStore:
    def __init__(self, count: int = 1, similarity: float = 0.96):
        self._count = count
        self.similarity = similarity

    def get_collection_count(self) -> int:
        return self._count

    def query_similar_embeddings(self, query_embedding, top_k=1):
        # Distance = 1 - similarity
        return {
            "distances": [[1.0 - self.similarity]],
            "ids": [["existing_chunk_999"]],
            "metadatas": [[]],
            "documents": [[]]
        }


def test_near_duplicate_cosine_similarity():
    dummy_store = DummyVectorStore(count=1, similarity=0.96)
    deduper = Deduplicator(vector_store=dummy_store, similarity_threshold=0.95)

    res = deduper.check_near_duplicate("Different sentence text", [0.1] * 1536)
    assert res.is_duplicate is True
    assert res.reason == "cosine_similarity"
    assert res.similar_chunk_id == "existing_chunk_999"
    assert res.similarity_score == pytest.approx(0.96, rel=1e-3)
