"""Loads golden test set Q&A pairs (JSON/CSV)."""
import json
from typing import List, Dict, Any

class GoldenDatasetLoader:
    def load(self, file_path: str = "data/golden_eval/test_set.json") -> List[Dict[str, Any]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return [
                {
                    "question": "What is the return policy?",
                    "expected_answer": "Customers can return within 30 days.",
                    "golden_context": ["Return policy allows 30-day refunds."]
                }
            ]
