"""Follow-up question condensing and query expansion."""
from typing import List

class QueryRewriter:
    def condense_question(self, chat_history: List[dict], latest_query: str) -> str:
        """Reformulates a follow-up query into a standalone question using chat context."""
        if not chat_history:
            return latest_query
        return f"Condensed standalone query: {latest_query}"

    def expand_query(self, query: str) -> List[str]:
        """Generates multiple search variations of a query for multi-query retrieval."""
        return [
            query,
            f"Detailed context for {query}",
            f"Key facts about {query}"
        ]
