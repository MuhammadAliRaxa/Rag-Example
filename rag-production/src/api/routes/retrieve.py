from fastapi import APIRouter, HTTPException
from src.api.schemas import RetrieveRequest, RetrieveResponse
from src.retrieval.hybrid import HybridSearchEngine
from src.ingestion.embedder import shared_embedder

from src.ingestion.vectorstore import shared_vector_store

router = APIRouter()
search_engine = HybridSearchEngine(vector_store=shared_vector_store)
embedder = shared_embedder

@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(request: RetrieveRequest):
    try:
        strategy = request.strategy.lower()
        
        # Embed query vector for vector and hybrid search strategies
        query_vector = []
        if strategy in ("hybrid", "vector"):
            query_vector = embedder.embed_query(request.query)

        # Execute search based on strategy
        if strategy == "vector":
            raw_results = search_engine.vector_engine.search(query_vector, top_k=request.top_k)
        elif strategy == "keyword":
            raw_results = search_engine.keyword_engine.search(request.query, top_k=request.top_k)
        elif strategy == "hybrid":
            raw_results = search_engine.search(request.query, query_vector, top_k=request.top_k)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid retrieval strategy '{request.strategy}'. Supported strategies: hybrid, vector, keyword"
            )

        results = [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "score": float(r["score"]),
                "metadata": r.get("metadata", {})
            }
            for r in raw_results
        ]

        return RetrieveResponse(
            query=request.query,
            strategy=request.strategy,
            results=results
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
