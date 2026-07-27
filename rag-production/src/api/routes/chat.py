from fastapi import APIRouter, HTTPException
from src.api.schemas import ChatRequest, ChatResponse
from src.retrieval.hybrid import HybridSearchEngine
from src.generation.llm_client import LLMClient
from src.generation.citation_mapper import CitationMapper
from src.memory.conversation_store import RedisConversationStore
from src.ingestion.embedder import EmbeddingModelWrapper

from src.ingestion.vectorstore import shared_vector_store

from src.generation.prompt_templates import RAG_SYSTEM_PROMPT

router = APIRouter()
search_engine = HybridSearchEngine(vector_store=shared_vector_store)
llm_client = LLMClient()
citation_mapper = CitationMapper()
memory_store = RedisConversationStore()
embedder = EmbeddingModelWrapper()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        memory_store.add_message(request.session_id, "user", request.message)
        query_vector = embedder.embed_query(request.message)
        chunks = search_engine.search(request.message, query_vector, top_k=3)
        
        # Format context for RAG
        context_str = "\n\n".join([
            f"[Chunk {c['chunk_id']}]\nContent: {c['text']}"
            for c in chunks
        ])
        
        # Format prompt with context and user query
        formatted_prompt = RAG_SYSTEM_PROMPT.format(context_str=context_str, question=request.message)
        
        raw_response = llm_client.generate_response(formatted_prompt)
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
