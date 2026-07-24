"""Semantic chunking strategies (beyond simple char splitting)."""
from typing import List, Dict, Any

class SemanticChunker:
    def __init__(self, target_chunk_size: int = 512, overlap: int = 64):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_document(self, doc_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Splits text based on semantic boundaries (headings, paragraphs, sentences)."""
        paragraphs = doc_text.split("\n\n")
        chunks = []
        for i, para in enumerate(paragraphs):
            chunks.append({
                "chunk_id": f"{metadata.get('source', 'doc')}_{i}",
                "text": para,
                "metadata": {**metadata, "chunk_index": i}
            })
        return chunks
