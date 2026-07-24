from src.ingestion.chunkers import SemanticChunker

def test_semantic_chunker():
    chunker = SemanticChunker()
    text = "Section 1\n\nFirst paragraph text.\n\nSection 2\n\nSecond paragraph text."
    chunks = chunker.chunk_document(text, {"source": "test.txt"})
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"].startswith("test.txt")
