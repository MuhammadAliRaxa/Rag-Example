from src.generation.citation_mapper import CitationMapper

def test_citation_mapping():
    mapper = CitationMapper()
    llm_output = "According to facts [Chunk chunk_0], RAG is robust."
    source_chunks = [{"chunk_id": "chunk_0", "text": "RAG text", "metadata": {}}]
    res = mapper.extract_citations(llm_output, source_chunks)
    assert len(res["citations"]) == 1
    assert res["citations"][0]["chunk_id"] == "chunk_0"
