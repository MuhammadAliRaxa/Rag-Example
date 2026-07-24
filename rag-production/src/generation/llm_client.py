"""Wrapper around ChatGroq / OpenAI / Anthropic (swappable)."""
from src.config.settings import settings

class LLMClient:
    def __init__(self, provider: str = "openai", model_name: str = settings.DEFAULT_MODEL):
        self.provider = provider
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        """Generates a response from the configured LLM API."""
        return f"Generated answer for: '{prompt[:30]}...' using {self.provider}:{self.model_name} [Chunk chunk_0]"
