"""Demo / Smoke Test Script for running the full Ingestion and Chunking Pipeline."""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.ingestion.pipeline import IngestionPipeline, IngestionResult
from src.ingestion.embedder import EmbeddingModelWrapper


class MockEmbeddingModelWrapper(EmbeddingModelWrapper):
    """Mock embedder so the demo runs without requiring a live OpenAI API Key."""
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            val = float(len(text) % 100) / 100.0
            vec = [val] * 1536
            embeddings.append(vec)
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def main():
    print("=" * 60)
    print("🚀 Running RAG Ingestion & Chunking Pipeline Smoke Test")
    print("=" * 60)

    # Create sample document
    sample_file = root_dir / "data" / "raw" / "demo_doc.md"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text(
        "# Enterprise RAG Architecture\n\n"
        "Retrieval-Augmented Generation (RAG) merges LLM capabilities with internal vector search.\n\n"
        "## Ingestion and Chunking Strategy\n\n"
        "Document loaders normalize markdown, plain text, HTML, and PDF formats into structured clean plaintext.\n"
        "Configurable chunking strategies include fixed-size token splitting, recursive section header splitting, and semantic boundary detection.\n\n"
        "## Dual Vector & Keyword Indexing\n\n"
        "Chunks are indexed synchronously into ChromaDB vector store and BM25 sparse keyword index.\n"
        "Two-stage deduplication uses SHA-256 exact matching and cosine similarity thresholding (>0.95) to prevent redundancy.",
        encoding="utf-8"
    )

    mock_embedder = MockEmbeddingModelWrapper(api_key="demo-key")
    pipeline = IngestionPipeline(embedder=mock_embedder)

    print(f"\nProcessing file: {sample_file}")
    print("Running with strategy: 'recursive'...")
    res: IngestionResult = pipeline.process_file(str(sample_file), strategy="recursive")

    print("\n" + "-" * 40)
    print("📊 INGESTION PIPELINE RESULT")
    print("-" * 40)
    print(f"Source File             : {res.source_file}")
    print(f"Strategy Used           : {res.chunking_strategy}")
    print(f"Raw Document Pages      : {res.total_raw_docs}")
    print(f"Chunks Generated        : {res.total_chunks_generated}")
    print(f"Chunks Inserted         : {res.inserted_chunks}")
    print(f"Exact Dupes Skipped     : {res.skipped_exact_duplicates}")
    print(f"Near Dupes Skipped      : {res.skipped_near_duplicates}")
    print(f"ChromaDB Total Count    : {pipeline.vector_store.get_collection_count()}")
    print(f"BM25 Indexed Chunks     : {len(pipeline.vector_store.bm25_chunks)}")
    print("-" * 40)

    print("\n🔁 Testing Deduplication (Re-ingesting same file)...")
    res_dupe = pipeline.process_file(str(sample_file), strategy="recursive")
    print(f"Re-ingest Inserted Chunks: {res_dupe.inserted_chunks}")
    print(f"Re-ingest Skipped Dupes : {res_dupe.skipped_exact_duplicates + res_dupe.skipped_near_duplicates}")

    print("\n✅ Ingestion Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
