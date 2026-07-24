"""Hash-based duplicate document detection."""
import hashlib
from typing import Set

class HashDeduplicator:
    def __init__(self):
        self.seen_hashes: Set[str] = set()

    def compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def is_duplicate(self, content: str) -> bool:
        doc_hash = self.compute_hash(content)
        if doc_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(doc_hash)
        return False
