from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "RAG Production API"
    ENV: str = "development"
    DEBUG: bool = True
    
    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    DEFAULT_MODEL: str = "gpt-4o-mini"
    
    # Vector DB (Pinecone)
    VECTOR_DB_URL: Optional[str] = None
    BM25_INDEX_PATH: Path = BASE_DIR / "vectorstore" / "bm25_index.pkl"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "rag-index"
    PINECONE_HOST: Optional[str] = None

    # Redis / Memory
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Document Storage
    RAW_DOCS_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DOCS_DIR: Path = BASE_DIR / "data" / "processed"
    
    # Ingestion & Chunking
    CHUNK_STRATEGY: str = "semantic"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.80  # Lowered for better chunking behavior
    SEMANTIC_SPLIT_THRESHOLD: float = 0.75
    
    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()