from typing import List, Dict, Any
from src.ingestion.vectorstore import VectorStoreManager, shared_vector_store

class VectorSearchEngine:
    def __init__(self, collection_name: str = "rag_documents", vector_store=None):
        self.vector_store = vector_store or shared_vector_store

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Executes vector similarity search against vector store."""
        results = self.vector_store.query_similar_embeddings(query_vector, top_k=top_k)
        output = []
        if results and "ids" in results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
            
            for i in range(len(ids)):
                # similarity = 1 - cosine distance
                score = 1.0 - distances[i] if distances[i] is not None else 0.0
                output.append({
                    "chunk_id": ids[i],
                    "text": docs[i],
                    "score": score,
                    "metadata": metadatas[i] or {}
                })
        return output
