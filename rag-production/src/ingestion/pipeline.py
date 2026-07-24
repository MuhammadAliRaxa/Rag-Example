"""Orchestrates: parse -> chunk -> embed -> store."""
from src.ingestion.loaders import DocumentLoader
from src.ingestion.chunkers import SemanticChunker
from src.ingestion.embedder import EmbeddingModelWrapper
from src.ingestion.dedupe import HashDeduplicator
from src.config.logging_config import logger

class IngestionPipeline:
    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = SemanticChunker()
        self.embedder = EmbeddingModelWrapper()
        self.deduper = HashDeduplicator()

    def process_file(self, file_path: str):
        logger.info(f"Starting ingestion pipeline for: {file_path}")
        raw_docs = self.loader.load_pdf(file_path)
        
        all_chunks = []
        for doc in raw_docs:
            if self.deduper.is_duplicate(doc["content"]):
                logger.info(f"Skipping duplicate document: {file_path}")
                continue
            chunks = self.chunker.chunk_document(doc["content"], doc["metadata"])
            all_chunks.extend(chunks)

        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.embed_documents(texts)

        logger.info(f"Successfully processed {len(all_chunks)} chunks with {len(embeddings)} embeddings.")
        return {"processed_chunks": len(all_chunks)}
