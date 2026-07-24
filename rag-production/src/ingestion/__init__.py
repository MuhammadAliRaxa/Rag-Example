"""Ingestion module exports."""
from src.ingestion.loaders import DocumentLoader, RawDocument
from src.ingestion.chunkers import get_chunker, BaseChunker, FixedSizeChunker, RecursiveChunker, SemanticChunker, Chunk
from src.ingestion.embedder import EmbeddingModelWrapper
from src.ingestion.dedupe import Deduplicator, DedupeResult
from src.ingestion.vectorstore import VectorStoreManager
from src.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "DocumentLoader",
    "RawDocument",
    "get_chunker",
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "Chunk",
    "EmbeddingModelWrapper",
    "Deduplicator",
    "DedupeResult",
    "VectorStoreManager",
    "IngestionPipeline",
    "IngestionResult",
]
