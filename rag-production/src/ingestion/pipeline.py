"""Orchestrates document loading, chunking strategy, deduplication, embedding generation, and dual-index insertion."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from src.config.settings import settings
from src.config.logging_config import logger
from src.ingestion.loaders import DocumentLoader, RawDocument
from src.ingestion.chunkers import get_chunker, BaseChunker, Chunk
from src.ingestion.embedder import EmbeddingModelWrapper, shared_embedder
from src.ingestion.dedupe import Deduplicator
from src.ingestion.vectorstore import VectorStoreManager


@dataclass
class IngestionResult:
    source_file: str
    total_raw_docs: int
    total_chunks_generated: int
    inserted_chunks: int
    skipped_exact_duplicates: int
    skipped_near_duplicates: int
    chunking_strategy: str


class IngestionPipeline:
    def __init__(
        self,
        loader: Optional[DocumentLoader] = None,
        embedder: Optional[EmbeddingModelWrapper] = None,
        vector_store: Optional[VectorStoreManager] = None,
        deduplicator: Optional[Deduplicator] = None,
    ):
        self.loader = loader or DocumentLoader()
        self.embedder = embedder or shared_embedder
        self.vector_store = vector_store or VectorStoreManager()
        self.deduplicator = deduplicator or Deduplicator(vector_store=self.vector_store)

    def process_file(self, file_path: str, strategy: Optional[str] = None) -> IngestionResult:
        selected_strategy = strategy or settings.CHUNK_STRATEGY
        logger.info(f"Starting ingestion pipeline for file: {file_path} using strategy: '{selected_strategy}'")

        # 1. Load document
        raw_docs: List[RawDocument] = self.loader.load(file_path)
        logger.info(f"Loaded {len(raw_docs)} raw document sections/pages from {file_path}")

        chunker: BaseChunker = get_chunker(
            strategy=selected_strategy,
            embedder=self.embedder
        )

        all_chunks: List[Chunk] = []
        for raw_doc in raw_docs:
            chunks = chunker.chunk_document(raw_doc)
            all_chunks.extend(chunks)

        logger.info(f"Generated {len(all_chunks)} chunks across raw documents.")

        valid_chunks: List[Chunk] = []
        valid_embeddings: List[List[float]] = []
        skipped_exact = 0
        skipped_near = 0

        # Batch embed all candidate chunk texts for efficiency
        chunk_texts = [c.text for c in all_chunks]
        if chunk_texts:
            chunk_embeddings = self.embedder.embed_documents(chunk_texts)
        else:
            chunk_embeddings = []

        # 3. Deduplication check before insertion
        dedupe_results = self.deduplicator.check_near_duplicates_batch(all_chunks, chunk_embeddings)
        for chunk, embedding, dedupe_res in zip(all_chunks, chunk_embeddings, dedupe_results):
            if dedupe_res.is_duplicate:
                if dedupe_res.reason == "exact_hash":
                    skipped_exact += 1
                elif dedupe_res.reason == "cosine_similarity":
                    skipped_near += 1
                logger.info(f"Skipping duplicate chunk {chunk.chunk_id}: {dedupe_res.reason}")
            else:
                valid_chunks.append(chunk)
                valid_embeddings.append(embedding)

        # 4. Insert valid non-duplicate chunks into ChromaDB and BM25 index in parallel
        inserted_count = 0
        if valid_chunks:
            inserted_count = self.vector_store.add_chunks(valid_chunks, valid_embeddings)

        result = IngestionResult(
            source_file=file_path,
            total_raw_docs=len(raw_docs),
            total_chunks_generated=len(all_chunks),
            inserted_chunks=inserted_count,
            skipped_exact_duplicates=skipped_exact,
            skipped_near_duplicates=skipped_near,
            chunking_strategy=selected_strategy,
        )

        logger.info(
            f"Pipeline complete for {file_path}. Inserted: {inserted_count}, Exact Dupes Skipped: {skipped_exact}, Near Dupes Skipped: {skipped_near}"
        )
        return result
