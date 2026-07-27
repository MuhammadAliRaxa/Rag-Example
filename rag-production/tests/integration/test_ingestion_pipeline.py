import tempfile
from pathlib import Path
import pytest

from src.ingestion.pipeline import IngestionPipeline, IngestionResult
from src.ingestion.vectorstore import VectorStoreManager
from src.ingestion.embedder import EmbeddingModelWrapper


class MockEmbeddingModelWrapper(EmbeddingModelWrapper):
    """Mock embedder for unit/integration testing without live API keys."""
    def embed_documents(self, texts):
        # Generate deterministic mock 1536-dim embeddings based on text length / hash
        embeddings = []
        for text in texts:
            val = float(len(text) % 100) / 100.0
            vec = [val] * 1536
            embeddings.append(vec)
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def temp_environment():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        chroma_dir = tmp / "chroma"
        bm25_path = tmp / "bm25.pkl"
        raw_dir = tmp / "data" / "raw"
        processed_dir = tmp / "data" / "processed"

        mock_embedder = MockEmbeddingModelWrapper()
        vstore = VectorStoreManager(
            chroma_dir=str(chroma_dir),
            bm25_path=str(bm25_path),
            collection_name="test_rag_docs"
        )
        pipeline = IngestionPipeline(
            embedder=mock_embedder,
            vector_store=vstore
        )
        pipeline.loader.raw_dir = raw_dir
        pipeline.loader.processed_dir = processed_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        yield pipeline, tmp, vstore


def test_full_pipeline_ingestion_and_deduplication(temp_environment):
    pipeline, tmp_dir, vstore = temp_environment

    # Create sample document
    doc_path = tmp_dir / "sample_guide.md"
    doc_path.write_text(
        "# Production RAG System\n\n"
        "Retrieval-Augmented Generation enhances LLM answers with retrieved facts.\n\n"
        "## Ingestion Pipeline\n\n"
        "Document loaders process markdown, text, HTML, and PDF formats into clean normalized text.",
        encoding="utf-8"
    )

    # 1. First Ingestion Run
    res1: IngestionResult = pipeline.process_file(str(doc_path), strategy="recursive")

    assert res1.total_chunks_generated >= 1
    assert res1.inserted_chunks > 0
    assert res1.skipped_exact_duplicates == 0
    assert vstore.get_collection_count() == res1.inserted_chunks
    assert len(vstore.bm25_chunks) == res1.inserted_chunks
    assert vstore.bm25_index is not None

    # 2. Re-ingest exact same file (should trigger deduplication)
    res2: IngestionResult = pipeline.process_file(str(doc_path), strategy="recursive")
    assert res2.inserted_chunks == 0
    assert (res2.skipped_exact_duplicates + res2.skipped_near_duplicates) == res2.total_chunks_generated


def test_pipeline_different_strategies(temp_environment):
    pipeline, tmp_dir, vstore = temp_environment

    doc_path = tmp_dir / "tech_spec.txt"
    doc_path.write_text(
        "Architecture Overview: RAG pipelines consist of ingestion, retrieval, ranking, and generation stages.\n"
        "Ingestion processes raw data into vector stores and sparse keyword indexes.",
        encoding="utf-8"
    )

    res_fixed = pipeline.process_file(str(doc_path), strategy="fixed")
    assert res_fixed.chunking_strategy == "fixed"
    assert res_fixed.inserted_chunks > 0
    assert vstore.get_collection_count() == res_fixed.inserted_chunks

    # Verify metadata in ChromaDB
    collection_items = vstore.collection.get()
    for meta in collection_items["metadatas"]:
        assert meta["chunking_strategy"] == "fixed"
        assert "character_count" in meta
        assert "source" in meta
        assert "section_heading" in meta
