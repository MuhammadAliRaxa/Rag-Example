from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "RAG Production API"
    ENV: str = "development"
    DEBUG: bool = True
    
    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    DEFAULT_MODEL: str = "gpt-4o-mini"
    
    # Vector DB
    VECTOR_DB_TYPE: str = "chroma"  # chroma, qdrant, pinecone
    VECTOR_DB_URL: Optional[str] = None
    CHROMA_PERSIST_DIR: str = "./vectorstore/chroma"
    BM25_INDEX_PATH: str = "./vectorstore/bm25_index.pkl"

    # Redis / Memory
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Document Storage
    RAW_DOCS_DIR: str = "./data/raw"
    PROCESSED_DOCS_DIR: str = "./data/processed"
    
    # Ingestion & Chunking
    CHUNK_STRATEGY: str = "recursive"  # fixed, recursive, semantic
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.95
    SEMANTIC_SPLIT_THRESHOLD: float = 0.75
    
    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
