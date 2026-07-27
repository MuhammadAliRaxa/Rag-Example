import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

class MockEmbedder:
    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 384

def mock_generate_response(message):
    from src.ingestion.vectorstore import shared_vector_store
    if shared_vector_store.bm25_chunks:
        cid = shared_vector_store.bm25_chunks[0]["chunk_id"]
        return f"This is a mock LLM response citing [Chunk {cid}]."
    return "This is a mock LLM response."

@pytest.fixture(autouse=True)
def mock_embedding_and_llm():
    with patch("src.api.routes.ingest.pipeline.embedder", new_callable=MagicMock) as mock_ingest_emb, \
         patch("src.api.routes.retrieve.embedder", new_callable=MagicMock) as mock_retrieve_emb, \
         patch("src.api.routes.chat.embedder", new_callable=MagicMock) as mock_chat_emb, \
         patch("src.generation.llm_client.LLMClient.generate_response", side_effect=mock_generate_response):
        
        mock_instance = MockEmbedder()
        mock_ingest_emb.embed_documents = mock_instance.embed_documents
        mock_ingest_emb.embed_query = mock_instance.embed_query
        
        mock_retrieve_emb.embed_documents = mock_instance.embed_documents
        mock_retrieve_emb.embed_query = mock_instance.embed_query
        
        mock_chat_emb.embed_documents = mock_instance.embed_documents
        mock_chat_emb.embed_query = mock_instance.embed_query
        
        yield

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ingestion_and_retrieval_e2e():
    # 1. Ingest a dummy file
    file_content = b"# Test Document\n\nThis is the content of the test document for the end-to-end RAG system flow."
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test_doc.md", file_content, "text/markdown")},
        params={"strategy": "recursive"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_doc.md"
    assert data["status"] == "completed"
    assert data["processed_chunks"] > 0
    
    # 2. Retrieve using hybrid strategy
    retrieve_response = client.post(
        "/api/v1/retrieve",
        json={"query": "RAG system flow", "top_k": 2, "strategy": "hybrid"}
    )
    assert retrieve_response.status_code == 200
    ret_data = retrieve_response.json()
    assert ret_data["query"] == "RAG system flow"
    assert len(ret_data["results"]) > 0
    assert "chunk_id" in ret_data["results"][0]
    assert "text" in ret_data["results"][0]
    assert "score" in ret_data["results"][0]

    # 3. Retrieve using vector strategy
    vec_response = client.post(
        "/api/v1/retrieve",
        json={"query": "RAG system flow", "top_k": 1, "strategy": "vector"}
    )
    assert vec_response.status_code == 200
    assert len(vec_response.json()["results"]) > 0

    # 4. Retrieve using keyword strategy
    kw_response = client.post(
        "/api/v1/retrieve",
        json={"query": "RAG system flow", "top_k": 1, "strategy": "keyword"}
    )
    assert kw_response.status_code == 200
    assert len(kw_response.json()["results"]) > 0
    
    # 5. Chat endpoint test
    chat_response = client.post(
        "/api/v1/chat",
        json={"session_id": "test_session_123", "message": "What is the RAG system flow?"}
    )
    assert chat_response.status_code == 200
    chat_data = chat_response.json()
    assert chat_data["session_id"] == "test_session_123"
    assert "answer" in chat_data
    assert len(chat_data["citations"]) > 0
