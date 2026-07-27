from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique conversation session identifier")
    message: str = Field(..., description="User query message")

class CitationSchema(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[CitationSchema]

class IngestResponse(BaseModel):
    filename: str
    status: str
    processed_chunks: int
    total_chunks: Optional[int] = None
    strategy: Optional[str] = None

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Query to search for")
    top_k: int = Field(5, description="Number of results to return")
    strategy: str = Field("hybrid", description="Retrieval strategy (hybrid, vector, keyword)")

class ChunkResultSchema(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class RetrieveResponse(BaseModel):
    query: str
    strategy: str
    results: List[ChunkResultSchema]
