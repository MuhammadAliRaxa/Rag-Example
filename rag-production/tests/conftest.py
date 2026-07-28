"""
Root conftest.py — patches Pinecone before any module is imported so all tests
run fully offline without hitting the live Pinecone cloud.
"""
from unittest.mock import MagicMock
import numpy as np
import pinecone as _pinecone


class MockPineconeIndex:
    """In-memory Pinecone index for offline testing. Supports cosine similarity."""

    def __init__(self):
        self._vectors: dict = {}

    def upsert(self, vectors):
        for v in vectors:
            self._vectors[v["id"]] = {
                "values": v["values"],
                "metadata": v.get("metadata", {}),
            }

    def query(self, vector, top_k=5, include_metadata=True, include_values=False):
        if not self._vectors:
            return {"matches": []}
        q = np.array(vector, dtype=np.float32)
        scored = []
        for vid, vdata in self._vectors.items():
            v = np.array(vdata["values"], dtype=np.float32)
            norm = np.linalg.norm(q) * np.linalg.norm(v)
            score = float(np.dot(q, v) / norm) if norm > 0 else 0.0
            scored.append({"id": vid, "score": score, "metadata": vdata["metadata"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"matches": scored[:top_k]}

    def describe_index_stats(self):
        return {"total_vector_count": len(self._vectors)}

    def delete(self, delete_all=False, ids=None):
        if delete_all:
            self._vectors.clear()


def _make_mock_pinecone_client(api_key):
    """Returns a mock Pinecone client whose .Index() method returns a fresh MockPineconeIndex."""
    mock_pc = MagicMock()
    mock_pc.Index.return_value = MockPineconeIndex()
    return mock_pc


# Patch pinecone.Pinecone at module level — this runs before any test file
# imports src.ingestion.vectorstore, so the mock is in place when
# shared_vector_store = VectorStoreManager() executes.
_pinecone.Pinecone = _make_mock_pinecone_client
