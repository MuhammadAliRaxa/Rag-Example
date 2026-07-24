from fastapi import APIRouter, HTTPException
from src.api.schemas import ChatRequest, ChatResponse
from src.retrieval.hybrid import HybridSearchEngine
from src.generation.llm_client import LLMClient
from src.generation.citation_mapper import CitationMapper
from src.memory.conversation_store import RedisConversationStore

router = APIRouter()
search_engine = HybridSearchEngine()
llm_client = LLMClient()
citation_mapper = CitationMapper()
memory_store = RedisConversationStore()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        memory_store.add_message(request.session_id, "user", request.message)
        chunks = search_engine.search(request.message, [0.0]*1536, top_k=3)
        raw_response = llm_client.generate_response(request.message)
        citation_data = citation_mapper.extract_citations(raw_response, chunks)
        memory_store.add_message(request.session_id, "assistant", raw_response)

        return ChatResponse(
            session_id=request.session_id,
            answer=citation_data["response"],
            citations=[
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "metadata": c.get("metadata", {})
                } for c in citation_data["citations"]
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
