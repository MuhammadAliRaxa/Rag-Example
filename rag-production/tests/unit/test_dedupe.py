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

    def query_similar_embeddings_batch(self, query_embeddings, top_k=1):
        return [
            {
                "distances": [1.0 - self.similarity],
                "ids": ["existing_chunk_999"],
                "metadatas": [{}],
                "documents": [""]
            }
            for _ in query_embeddings
        ]


def test_near_duplicate_cosine_similarity():
    dummy_store = DummyVectorStore(count=1, similarity=0.96)
    deduper = Deduplicator(vector_store=dummy_store, similarity_threshold=0.95)

    res = deduper.check_near_duplicate("Different sentence text", [0.1] * 1536)
    assert res.is_duplicate is True
    assert res.reason == "cosine_similarity"
    assert res.similar_chunk_id == "existing_chunk_999"
    assert res.similarity_score == pytest.approx(0.96, rel=1e-3)


def test_near_duplicate_batch_deduplication():
    from src.ingestion.chunkers import Chunk

    # --- Part 1: Test exact-hash batch detection (no vector store) ---
    deduper_no_store = Deduplicator(vector_store=None, similarity_threshold=0.95)

    chunks = [
        Chunk(chunk_id="c1", text="Unique string 1", source="src", raw_hash="h1", chunk_index=0, section_heading="", chunk_strategy="test", char_count=15, page_number=1),
        Chunk(chunk_id="c2", text="Unique string 1", source="src", raw_hash="h1", chunk_index=1, section_heading="", chunk_strategy="test", char_count=15, page_number=1), # exact duplicate of c1
        Chunk(chunk_id="c3", text="Totally different text", source="src", raw_hash="h2", chunk_index=2, section_heading="", chunk_strategy="test", char_count=22, page_number=1),
    ]
    embeddings = [[0.1]*1536, [0.1]*1536, [0.2]*1536]

    res = deduper_no_store.check_near_duplicates_batch(chunks, embeddings)
    assert len(res) == 3

    # c1: first time seen -> unique
    assert res[0].is_duplicate is False
    assert res[0].reason == "unique"

    # c2: exact same text as c1 -> exact_hash duplicate
    assert res[1].is_duplicate is True
    assert res[1].reason == "exact_hash"

    # c3: different text, no vector store -> unique
    assert res[2].is_duplicate is False
    assert res[2].reason == "unique"

    # --- Part 2: Test cosine batch detection (with vector store returning high similarity) ---
    dummy_store = DummyVectorStore(count=1, similarity=0.96)
    deduper_with_store = Deduplicator(vector_store=dummy_store, similarity_threshold=0.95)

    chunks2 = [
        Chunk(chunk_id="d1", text="Near dup sentence A", source="src", raw_hash="h3", chunk_index=0, section_heading="", chunk_strategy="test", char_count=19, page_number=1),
    ]
    embeddings2 = [[0.5]*1536]

    res2 = deduper_with_store.check_near_duplicates_batch(chunks2, embeddings2)
    assert len(res2) == 1
    # DummyVectorStore always returns 0.96 >= 0.95 threshold -> near-duplicate
    assert res2[0].is_duplicate is True
    assert res2[0].reason == "cosine_similarity"
    assert res2[0].similar_chunk_id == "existing_chunk_999"
    assert res2[0].similarity_score == pytest.approx(0.96, rel=1e-3)

