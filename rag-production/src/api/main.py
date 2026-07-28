from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from src.api.routes import chat, ingest, health, retrieve
from src.config.settings import settings, BASE_DIR

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-create necessary data storage folders
    settings.RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    settings.PROCESSED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    yield

app = FastAPI(
    title=settings.APP_NAME,
    description="Production Ready RAG API Service",
    version="1.0.0",
    lifespan=lifespan
)

# Production Middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(retrieve.router, prefix="/api/v1", tags=["Retrieval"])

@app.get("/")
async def read_root():
    return {
        "message": "Welcome to RAG Production API",
        "docs": "/docs",
        "static_docs": "/docs-static"
    }

@app.get("/docs-static")
async def read_docs_static():
    file_path = BASE_DIR / "docs.html"
    return FileResponse(file_path, media_type="text/html")