from fastapi import APIRouter, UploadFile, File, HTTPException
from src.api.schemas import IngestResponse
from src.ingestion.pipeline import IngestionPipeline

router = APIRouter()
pipeline = IngestionPipeline()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    try:
        res = pipeline.process_file(file.filename)
        return IngestResponse(
            filename=file.filename,
            status="completed",
            processed_chunks=res.get("processed_chunks", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
