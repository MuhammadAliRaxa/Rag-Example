from fastapi import FastAPI
from src.api.routes import chat, ingest, health
from src.config.settings import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="Production Ready RAG API Service",
    version="1.0.0"
)

app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])

@app.get("/")
def read_root():
    return {"message": "Welcome to RAG Production API", "docs": "/docs"}
