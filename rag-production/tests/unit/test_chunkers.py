import pytest
from src.ingestion.loaders import RawDocument
from src.ingestion.chunkers import FixedSizeChunker, RecursiveChunker, SemanticChunker, get_chunker, Chunk


@pytest.fixture
def sample_raw_doc():
    return RawDocument(
        source_path="/tmp/test_doc.md",
        file_type="markdown",
        raw_hash="abc123hash",
        content="# Introduction\n\nThis is the introduction paragraph.\n\n## Deep Dive\n\nHere is a detailed deep dive into RAG production architectures and pipelines.",
        metadata={
            "source": "/tmp/test_doc.md",
            "section_heading": "Introduction",
            "page_number": 1
        }
    )


def test_fixed_size_chunker(sample_raw_doc):
    chunker = FixedSizeChunker(chunk_size=15, overlap=5)
    chunks = chunker.chunk_document(sample_raw_doc)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.chunk_strategy == "fixed"
        assert chunk.raw_hash == "abc123hash"
        assert chunk.source == "/tmp/test_doc.md"
        assert chunk.char_count > 0


def test_recursive_chunker(sample_raw_doc):
    chunker = RecursiveChunker(chunk_size=20, overlap=5)
    chunks = chunker.chunk_document(sample_raw_doc)
    
    assert len(chunks) >= 2
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.chunk_strategy == "recursive"
        assert chunk.raw_hash == "abc123hash"
        assert chunk.source == "/tmp/test_doc.md"


def test_semantic_chunker_fallback(sample_raw_doc):
    # Without embedder, falls back gracefully to recursive chunker
    chunker = SemanticChunker(chunk_size=20, overlap=5, embedder=None)
    chunks = chunker.chunk_document(sample_raw_doc)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Chunk)


def test_get_chunker_factory():
    c_fixed = get_chunker("fixed")
    assert isinstance(c_fixed, FixedSizeChunker)

    c_rec = get_chunker("recursive")
    assert isinstance(c_rec, RecursiveChunker)

    c_sem = get_chunker("semantic")
    assert isinstance(c_sem, SemanticChunker)

    c_default = get_chunker("unknown_strategy")
    assert isinstance(c_default, RecursiveChunker)
