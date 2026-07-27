"""Demo / Smoke Test Script for running the full Ingestion and Chunking Pipeline."""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.ingestion.pipeline import IngestionPipeline, IngestionResult
from src.ingestion.embedder import EmbeddingModelWrapper

def main():
    print("=" * 60)
    print("🚀 Running RAG Ingestion & Chunking Pipeline Smoke Test")
    print("=" * 60)

    # Create sample document
    sample_file = root_dir / "data" / "raw" / "NIPS-2017-attention-is-all-you-need-Paper.pdf"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    print("sample_file", sample_file)

    embedder = EmbeddingModelWrapper(model_name="all-MiniLM-L6-v2", device="mps")
    pipeline = IngestionPipeline(embedder=embedder)

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
