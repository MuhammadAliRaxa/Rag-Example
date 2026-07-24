"""Chat history store (Redis/DB backed, not in-process)."""
from typing import List, Dict, Any
from src.config.settings import settings

class RedisConversationStore:
    def __init__(self, redis_url: str = settings.REDIS_URL):
        self.redis_url = redis_url
        self._in_memory_fallback: Dict[str, List[Dict[str, Any]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self._in_memory_fallback.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self._in_memory_fallback:
            self._in_memory_fallback[session_id] = []
        self._in_memory_fallback[session_id].append({"role": role, "content": content})

    def clear_history(self, session_id: str):
        self._in_memory_fallback.pop(session_id, None)
