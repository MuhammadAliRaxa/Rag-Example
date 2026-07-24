"""LangSmith / Arize Phoenix tracing setup."""
import os

def setup_tracing():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "rag-production"
    # Provide LANGCHAIN_API_KEY in .env
