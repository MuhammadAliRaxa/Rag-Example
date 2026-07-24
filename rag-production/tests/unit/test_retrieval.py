from src.retrieval.hybrid import HybridSearchEngine

def test_hybrid_search():
    engine = HybridSearchEngine()
    results = engine.search("test query", [0.0]*1536, top_k=3)
    assert len(results) <= 3
