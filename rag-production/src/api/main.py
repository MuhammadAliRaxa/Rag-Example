import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from src.api.routes import chat, ingest, health, retrieve
from src.config.settings import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="Production Ready RAG API Service",
    version="1.0.0"
)

app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(retrieve.router, prefix="/api/v1", tags=["Retrieval"])

@app.get("/")
def read_root():
    return {
        "message": "Welcome to RAG Production API",
        "docs": "/docs",
        "static_docs": "/docs-static"
    }

@app.get("/docs-static", response_class=HTMLResponse)
def read_docs_static():
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

