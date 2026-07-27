"""Dual Vector Store & BM25 Index manager to keep dense (ChromaDB) and sparse (BM25) indexes synchronized."""
import os
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import chromadb
from rank_bm25 import BM25Okapi

from src.config.settings import settings
from src.config.logging_config import logger
from src.ingestion.chunkers import Chunk


class VectorStoreManager:
    def __init__(
        self,
        chroma_dir: str = settings.CHROMA_PERSIST_DIR,
        bm25_path: str = settings.BM25_INDEX_PATH,
        collection_name: str = "rag_documents",
    ):
        self.chroma_dir = Path(chroma_dir)
        self.bm25_path = Path(bm25_path)
        self.collection_name = collection_name
        
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # BM25 state
        self.bm25_chunks: List[Dict[str, Any]] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self._load_bm25()

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace/alphanumeric tokenizer for BM25."""
        return [token.lower() for token in text.split() if token.strip()]

    def _load_bm25(self) -> None:
        """Load BM25 state from disk if exists."""
        if self.bm25_path.exists():
            try:
                with open(self.bm25_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25_chunks = data.get("chunks", [])
                    corpus_tokens = [self._tokenize(c["text"]) for c in self.bm25_chunks]
                    if corpus_tokens:
                        self.bm25_index = BM25Okapi(corpus_tokens)
                    else:
                        self.bm25_index = None
                logger.info(f"Loaded BM25 index with {len(self.bm25_chunks)} chunks from {self.bm25_path}")
            except Exception as e:
                logger.error(f"Error loading BM25 index from {self.bm25_path}: {e}")
                self.bm25_chunks = []
                self.bm25_index = None

    def _save_bm25(self) -> None:
        """Save BM25 state to disk."""
        try:
            with open(self.bm25_path, "wb") as f:
                pickle.dump({"chunks": self.bm25_chunks}, f)
            logger.info(f"Saved BM25 index ({len(self.bm25_chunks)} chunks) to {self.bm25_path}")
        except Exception as e:
            logger.error(f"Error saving BM25 index to {self.bm25_path}: {e}")

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """Atomically add chunks to ChromaDB (via upsert) and update BM25 index."""
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})")

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = []

        for c in chunks:
            # ChromaDB metadata must contain primitive types only
            meta = {
                "source": str(c.source),
                "chunk_index": int(c.chunk_index),
                "section_heading": str(c.section_heading or ""),
                "chunking_strategy": str(c.chunk_strategy),
                "character_count": int(c.char_count),
                "page_number": int(c.page_number),
                "raw_hash": str(c.raw_hash),
            }
            metadatas.append(meta)

        # 1. Upsert into ChromaDB (handling dimension changes & duplicate IDs gracefully)
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            if "dimension" in str(e).lower() or "expecting embedding" in str(e).lower():
                logger.warning(f"Embedding dimension changed ({e}). Re-creating collection for new dimension model.")
                self.clear()
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            else:
                raise e

        # 2. Update BM25 index
        existing_ids = {c["chunk_id"] for c in self.bm25_chunks}
        for c in chunks:
            if c.chunk_id not in existing_ids:
                self.bm25_chunks.append({
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "metadata": {
                        "source": str(c.source),
                        "chunk_index": int(c.chunk_index),
                        "section_heading": str(c.section_heading or ""),
                        "chunking_strategy": str(c.chunk_strategy),
                        "character_count": int(c.char_count),
                        "page_number": int(c.page_number),
                        "raw_hash": str(c.raw_hash),
                    }
                })

        corpus_tokens = [self._tokenize(c["text"]) for c in self.bm25_chunks]
        self.bm25_index = BM25Okapi(corpus_tokens)
        self._save_bm25()

        logger.info(f"Successfully added {len(chunks)} chunks to both ChromaDB and BM25 indexes.")
        return len(chunks)

    def get_collection_count(self) -> int:
        """Returns total count of stored chunks in ChromaDB."""
        return self.collection.count()

    def query_similar_embeddings(self, query_embedding: List[float], top_k: int = 5) -> Dict[str, Any]:
        """Query ChromaDB by embedding vector."""
        if self.collection.count() == 0:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

        try:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            if "dimension" in str(e).lower() or "expecting embedding" in str(e).lower():
                logger.warning(f"Embedding dimension mismatch during query ({e}). Clearing obsolete index.")
                self.clear()
                return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}
            raise e

    def clear(self) -> None:
        """Clear both ChromaDB and BM25 indexes."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25_chunks = []
        self.bm25_index = None
        if self.bm25_path.exists():
            self.bm25_path.unlink()
        logger.info("Cleared both ChromaDB and BM25 indexes.")

# Shared singleton instance for process-wide index synchronization
shared_vector_store = VectorStoreManager()
