"""Pinecone Vector Store & BM25 Index manager."""
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi
from pinecone import Pinecone, ServerlessSpec

from src.config.settings import settings
from src.config.logging_config import logger
from src.ingestion.chunkers import Chunk


class VectorStoreManager:
    def __init__(
        self,
        bm25_path: str = settings.BM25_INDEX_PATH,
    ):
        self.bm25_path = Path(bm25_path)

        # Initialize Pinecone cloud vector store
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY must be set")
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index_name = settings.PINECONE_INDEX_NAME

        if settings.PINECONE_HOST:
            # Direct host connection — fastest, no DNS lookup
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

        # BM25 state (in-memory sparse retrieval; disk write is best-effort)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        self.bm25_chunks: List[Dict[str, Any]] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self._load_bm25()

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer for BM25."""
        return [token.lower() for token in text.split() if token.strip()]

    def _load_bm25(self) -> None:
        """Load BM25 state from disk if exists."""
        if self.bm25_path.exists():
            try:
                with open(self.bm25_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25_chunks = data.get("chunks", [])
                    self.bm25_index = data.get("index", None)
                logger.info(f"Loaded BM25 index with {len(self.bm25_chunks)} chunks from {self.bm25_path}")
            except Exception as e:
                logger.error(f"Error loading BM25 index from {self.bm25_path}: {e}")
                self.bm25_chunks = []
                self.bm25_index = None

    def _save_bm25(self) -> None:
        """Save BM25 state to disk."""
        try:
            with open(self.bm25_path, "wb") as f:
                pickle.dump({"chunks": self.bm25_chunks, "index": self.bm25_index}, f)
            logger.info(f"Saved BM25 index ({len(self.bm25_chunks)} chunks) to {self.bm25_path}")
        except Exception as e:
            logger.error(f"Error saving BM25 index to {self.bm25_path}: {e}")

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """Add chunks to Pinecone (via upsert) and update BM25 index."""
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
            )

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": str(c.source),
                "chunk_index": int(c.chunk_index),
                "section_heading": str(c.section_heading or ""),
                "chunking_strategy": str(c.chunk_strategy),
                "character_count": int(c.char_count),
                "page_number": int(c.page_number),
                "raw_hash": str(c.raw_hash),
            }
            for c in chunks
        ]

        # Upsert into Pinecone in parallel batches of 100
        vectors = [
            {"id": ids[i], "values": embeddings[i], "metadata": {**metadatas[i], "text": documents[i]}}
            for i in range(len(ids))
        ]
        batch_size = 100
        batches = [vectors[j:j + batch_size] for j in range(0, len(vectors), batch_size)]

        def _upsert_batch(batch):
            self.pinecone_index.upsert(vectors=batch)

        with ThreadPoolExecutor(max_workers=min(10, len(batches) or 1)) as executor:
            list(executor.map(_upsert_batch, batches))
        logger.info(f"Upserted {len(vectors)} vectors to Pinecone index.")

        # Update BM25 index
        existing_ids = {c["chunk_id"] for c in self.bm25_chunks}
        for c, meta in zip(chunks, metadatas):
            if c.chunk_id not in existing_ids:
                self.bm25_chunks.append({"chunk_id": c.chunk_id, "text": c.text, "metadata": meta})

        corpus_tokens = [self._tokenize(c["text"]) for c in self.bm25_chunks]
        self.bm25_index = BM25Okapi(corpus_tokens)
        try:
            self._save_bm25()
        except Exception as e:
            logger.warning(f"BM25 index could not be saved to disk: {e}")

        logger.info(f"Successfully added {len(chunks)} chunks to Pinecone and BM25 index.")
        return len(chunks)

    def get_collection_count(self) -> int:
        """Returns total number of stored vectors in Pinecone."""
        try:
            stats = self.pinecone_index.describe_index_stats()
            return stats.get("total_vector_count", 0)
        except Exception:
            return 0

    def query_similar_embeddings(self, query_embedding: List[float], top_k: int = 5) -> Dict[str, Any]:
        """Query Pinecone by a single embedding vector."""
        result = self.pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        matches = result.get("matches", [])
        if not matches:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}
        # Pinecone returns scores (higher = more similar); convert to distance
        ids = [[m["id"] for m in matches]]
        distances = [[1.0 - m["score"] for m in matches]]
        metadatas = [[{k: v for k, v in m["metadata"].items() if k != "text"} for m in matches]]
        documents = [[m["metadata"].get("text", "") for m in matches]]
        return {"ids": ids, "distances": distances, "metadatas": metadatas, "documents": documents}

    def query_similar_embeddings_batch(self, query_embeddings: List[List[float]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query Pinecone by multiple embedding vectors concurrently."""
        if not query_embeddings:
            return []

        def _query_single(emb):
            return self.pinecone_index.query(
                vector=emb, top_k=top_k, include_metadata=True, include_values=False
            )

        with ThreadPoolExecutor(max_workers=min(10, len(query_embeddings))) as executor:
            results = list(executor.map(_query_single, query_embeddings))

        outputs = []
        for res in results:
            matches = res.get("matches", [])
            if not matches:
                outputs.append({"ids": [], "distances": [], "metadatas": [], "documents": []})
            else:
                outputs.append({
                    "ids": [m["id"] for m in matches],
                    "distances": [1.0 - m["score"] for m in matches],
                    "metadatas": [{k: v for k, v in m["metadata"].items() if k != "text"} for m in matches],
                    "documents": [m["metadata"].get("text", "") for m in matches],
                })
        return outputs

    def clear(self) -> None:
        """Clear Pinecone index and BM25 index."""
        try:
            self.pinecone_index.delete(delete_all=True)
            logger.info("Cleared Pinecone index.")
        except Exception as e:
            logger.error(f"Error clearing Pinecone index: {e}")
        self.bm25_chunks = []
        self.bm25_index = None
        if self.bm25_path.exists():
            self.bm25_path.unlink()
        logger.info("Cleared Pinecone and BM25 indexes.")


# Shared singleton instance for process-wide index synchronization
shared_vector_store = VectorStoreManager()
