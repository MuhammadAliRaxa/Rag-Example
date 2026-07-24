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
    
    # Redis / Memory
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
