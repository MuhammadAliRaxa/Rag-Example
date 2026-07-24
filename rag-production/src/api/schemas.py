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
