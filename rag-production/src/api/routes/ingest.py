import tempfile
import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from src.api.schemas import IngestResponse
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.vectorstore import shared_vector_store

router = APIRouter()
pipeline = IngestionPipeline(vector_store=shared_vector_store)

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    strategy: Optional[str] = Query(None, description="Chunking strategy to use (fixed, recursive, semantic)")
):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".txt", ".md", ".markdown", ".html", ".htm", ".pdf"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: .txt, .md, .markdown, .html, .htm, .pdf"
        )
        
    try:
        # Create a temp file preserving the extension so the loader handles it correctly
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            res = pipeline.process_file(temp_file_path, strategy=strategy)
            return IngestResponse(
                filename=file.filename,
                status="completed",
                processed_chunks=res.inserted_chunks,
                total_chunks=res.total_chunks_generated,
                strategy=res.chunking_strategy
            )
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
