"""Maps LLM output back to source chunks."""
import re
from typing import List, Dict, Any

class CitationMapper:
    def extract_citations(self, llm_response: str, source_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Matches citation tags in generated text back to full metadata of source chunks."""
        chunk_lookup = {c.get("chunk_id"): c for c in source_chunks}
        found_citations = re.findall(r"\[Chunk\s+([a-zA-Z0-9_-]+)\]", llm_response)
        
        referenced_sources = [chunk_lookup[cid] for cid in found_citations if cid in chunk_lookup]
        
        # Strip the inline [Chunk <id>] citation tags from the generated text response
        cleaned_response = re.sub(r"\s*\[Chunk\s+[a-zA-Z0-9_-]+\]", "", llm_response)
        
        return {
            "response": cleaned_response,
            "citations": referenced_sources
        }
