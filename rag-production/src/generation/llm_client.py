import google.generativeai as genai
from src.config.settings import settings
from src.config.logging_config import logger

class LLMClient:
    def __init__(self, provider: str = "gemini", model_name: str = settings.DEFAULT_MODEL):
        self.provider = provider
        self.model_name = model_name
        
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            logger.info("Gemini API client configured successfully.")
        else:
            logger.warning("GEMINI_API_KEY not found in settings. Running in mock fallback mode.")

    def generate_response(self, prompt: str) -> str:
        """Generates a response from the configured LLM API (Gemini)."""
        if not settings.GEMINI_API_KEY:
            # Safe mock fallback for unit testing offline
            return f"Generated answer for: '{prompt[:30]}...' using mock:gemini [Chunk chunk_0]"
            
        try:
            logger.info(f"Generating content using Gemini model: {self.model_name}")
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response from Gemini API: {e}")
            raise e
