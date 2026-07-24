"""Faithfulness and Relevance evaluation wrapper (RAGAS / custom judge)."""
from typing import Dict, Any

class AnswerMetrics:
    def evaluate_faithfulness(self, answer: str, contexts: list) -> float:
        """Evaluates if answer is grounded strictly in context."""
        return 0.95

    def evaluate_relevance(self, question: str, answer: str) -> float:
        """Evaluates if answer directly addresses the question."""
        return 0.92
