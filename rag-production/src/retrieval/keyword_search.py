from src.ingestion.vectorstore import VectorStoreManager, shared_vector_store

class KeywordSearchEngine:
    def __init__(self, vector_store=None):
        self.vector_store = vector_store or shared_vector_store

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Executes BM25 keyword search over text corpus."""
        if not self.vector_store.bm25_index or not self.vector_store.bm25_chunks:
            return []
            
        tokens = self.vector_store._tokenize(query)
        scores = self.vector_store.bm25_index.get_scores(tokens)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        output = []
        for idx in top_indices:
            score = float(scores[idx])
            if score == 0.0:
                continue
            chunk = self.vector_store.bm25_chunks[idx]
            output.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": score,
                "metadata": chunk.get("metadata", {})
            })
        return output
