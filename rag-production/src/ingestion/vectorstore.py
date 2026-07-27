"""Dual Vector Store & BM25 Index manager – supports ChromaDB (local) and Pinecone (cloud)."""
import os
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import chromadb
from rank_bm25 import BM25Okapi
from pinecone import Pinecone, ServerlessSpec

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
        self.use_pinecone = settings.VECTOR_DB_TYPE == "pinecone"

        if self.use_pinecone:
            # Initialize Pinecone cloud vector store
            if not settings.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY must be set when VECTOR_DB_TYPE=pinecone")
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index_name = settings.PINECONE_INDEX_NAME

            if settings.PINECONE_HOST:
                # Direct host connection — fastest, no list/create roundtrip
                self.pinecone_index = pc.Index(host=settings.PINECONE_HOST)
                logger.info(f"Connected to Pinecone index via host: {settings.PINECONE_HOST}")
            else:
                # Fallback: look up by name and create if it doesn't exist
                existing_indexes = [i.name for i in pc.list_indexes()]
                if index_name not in existing_indexes:
                    logger.info(f"Creating Pinecone index '{index_name}'...")
                    pc.create_index(
                        name=index_name,
                        dimension=384,  # matches all-MiniLM-L6-v2 default embedder
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                self.pinecone_index = pc.Index(index_name)
                logger.info(f"Using Pinecone cloud index: '{index_name}'")
            self.client = None
            self.collection = None
        else:
            # Initialize ChromaDB persistent client
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.pinecone_index = None

        # BM25 state (in-memory sparse retrieval; disk write is best-effort)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
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

        if self.use_pinecone:
            # 1a. Upsert into Pinecone cloud index
            vectors = [
                {
                    "id": ids[i],
                    "values": embeddings[i],
                    "metadata": {**metadatas[i], "text": documents[i]}
                }
                for i in range(len(ids))
            ]
            # Pinecone recommends batches of 100
            batch_size = 100
            for batch_start in range(0, len(vectors), batch_size):
                self.pinecone_index.upsert(vectors=vectors[batch_start:batch_start + batch_size])
            logger.info(f"Upserted {len(vectors)} vectors to Pinecone index.")
        else:
            # 1b. Upsert into ChromaDB (handling dimension changes & duplicate IDs gracefully)
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

        # 2. Update BM25 index (in-memory)
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
        # Persist BM25 index to disk best-effort (may fail on read-only serverless filesystems)
        try:
            self._save_bm25()
        except Exception as e:
            logger.warning(f"BM25 index could not be saved to disk (read-only filesystem?): {e}")

        logger.info(f"Successfully added {len(chunks)} chunks to vector store and BM25 index.")
        return len(chunks)

    def get_collection_count(self) -> int:
        """Returns total count of stored chunks in the vector store."""
        if self.use_pinecone:
            try:
                stats = self.pinecone_index.describe_index_stats()
                return stats.get("total_vector_count", 0)
            except Exception:
                return 0
        return self.collection.count()

    def query_similar_embeddings(self, query_embedding: List[float], top_k: int = 5) -> Dict[str, Any]:
        """Query the vector store by embedding vector. Supports both ChromaDB and Pinecone."""
        if self.use_pinecone:
            # Query Pinecone and normalize results to match ChromaDB's response format
            result = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            matches = result.get("matches", [])
            if not matches:
                return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}
            # Pinecone returns scores (higher = more similar); convert to distance (1 - score)
            ids = [[m["id"] for m in matches]]
            distances = [[1.0 - m["score"] for m in matches]]
            metadatas = [[{k: v for k, v in m["metadata"].items() if k != "text"} for m in matches]]
            documents = [[m["metadata"].get("text", "") for m in matches]]
            return {"ids": ids, "distances": distances, "metadatas": metadatas, "documents": documents}

        # ChromaDB path
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
