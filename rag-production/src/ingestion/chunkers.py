"""Chunking strategies: Fixed-size with overlap, Recursive header-aware, and Semantic embedding boundary splitting."""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol

import tiktoken
from src.config.settings import settings
from src.config.logging_config import logger
from src.ingestion.loaders import RawDocument
from src.ingestion.embedder import EmbeddingModelWrapper


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    raw_hash: str
    chunk_index: int
    section_heading: str
    chunk_strategy: str
    char_count: int
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseChunker:
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4  # heuristic approximation

    def chunk_document(self, doc: RawDocument) -> List[Chunk]:
        raise NotImplementedError


class FixedSizeChunker(BaseChunker):
    """Fixed token-size chunking with sliding overlap."""

    def chunk_document(self, doc: RawDocument) -> List[Chunk]:
        text = doc.content
        if not text.strip():
            return []

        chunks: List[Chunk] = []
        page_num = doc.metadata.get("page_number", 1)
        
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            step = max(1, self.chunk_size - self.overlap)
            idx = 0
            chunk_seq = 0
            
            while idx < len(tokens):
                chunk_tokens = tokens[idx : idx + self.chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens).strip()
                
                if chunk_text:
                    c_id = f"{doc.raw_hash}_p{page_num}_fixed_{chunk_seq}"
                    chunk_meta = {
                        **doc.metadata,
                        "chunk_index": chunk_seq,
                        "chunk_strategy": "fixed",
                        "section_heading": doc.metadata.get("section_heading", ""),
                        "page_number": page_num,
                        "char_count": len(chunk_text),
                        "token_count": len(chunk_tokens),
                        "source": doc.source_path,
                        "raw_hash": doc.raw_hash,
                    }
                    chunks.append(
                        Chunk(
                            chunk_id=c_id,
                            text=chunk_text,
                            source=doc.source_path,
                            raw_hash=doc.raw_hash,
                            chunk_index=chunk_seq,
                            section_heading=doc.metadata.get("section_heading", ""),
                            chunk_strategy="fixed",
                            char_count=len(chunk_text),
                            page_number=page_num,
                            metadata=chunk_meta,
                        )
                    )
                    chunk_seq += 1
                
                idx += step
        else:
            # Character fallback
            char_chunk_size = self.chunk_size * 4
            char_overlap = self.overlap * 4
            step = max(1, char_chunk_size - char_overlap)
            chunk_seq = 0
            
            for idx in range(0, len(text), step):
                chunk_text = text[idx : idx + char_chunk_size].strip()
                if chunk_text:
                    c_id = f"{doc.raw_hash}_p{page_num}_fixed_{chunk_seq}"
                    chunk_meta = {
                        **doc.metadata,
                        "chunk_index": chunk_seq,
                        "chunk_strategy": "fixed",
                        "section_heading": doc.metadata.get("section_heading", ""),
                        "page_number": page_num,
                        "char_count": len(chunk_text),
                        "source": doc.source_path,
                        "raw_hash": doc.raw_hash,
                    }
                    chunks.append(
                        Chunk(
                            chunk_id=c_id,
                            text=chunk_text,
                            source=doc.source_path,
                            raw_hash=doc.raw_hash,
                            chunk_index=chunk_seq,
                            section_heading=doc.metadata.get("section_heading", ""),
                            chunk_strategy="fixed",
                            char_count=len(chunk_text),
                            page_number=page_num,
                            metadata=chunk_meta,
                        )
                    )
                    chunk_seq += 1

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursive character/header splitting based on section headers and paragraph breaks."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        overlap: int = settings.CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(chunk_size, overlap)
        self.separators = separators or [
            "\n# ",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text into pieces under chunk_size tokens."""
        if not text:
            return []

        if self.count_tokens(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Hard character split fallback
            split_at = self.chunk_size * 4
            return [text[:split_at]] + self._split_text(text[split_at:], [])

        sep = separators[0]
        next_seps = separators[1:]

        if sep == "":
            splits = list(text)
        elif sep.startswith("\n#"):
            # Header separator - capture headers cleanly
            pattern = f"(?={re.escape(sep.strip())})"
            splits = [s for s in re.split(pattern, text) if s]
        else:
            splits = text.split(sep)

        final_chunks = []
        current_chunk = ""

        for part in splits:
            if not part:
                continue
            
            candidate = f"{current_chunk}{sep}{part}" if current_chunk else part
            if self.count_tokens(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                
                if self.count_tokens(part) > self.chunk_size:
                    # Recursive split sub-part using next separators
                    sub_splits = self._split_text(part, next_seps)
                    final_chunks.extend(sub_splits)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks

    def _extract_section_heading(self, text: str, doc_default: str) -> str:
        """Extract nearest markdown section header from chunk text."""
        match = re.search(r"^(#+)\s+(.+)$", text, re.MULTILINE)
        if match:
            return match.group(2).strip()
        return doc_default

    def chunk_document(self, doc: RawDocument) -> List[Chunk]:
        raw_splits = self._split_text(doc.content, self.separators)
        chunks: List[Chunk] = []
        page_num = doc.metadata.get("page_number", 1)
        
        current_heading = doc.metadata.get("section_heading", "")

        for seq, split in enumerate(raw_splits):
            clean_split = split.strip()
            if not clean_split:
                continue
                
            heading = self._extract_section_heading(clean_split, current_heading)
            if heading != current_heading:
                current_heading = heading

            c_id = f"{doc.raw_hash}_p{page_num}_rec_{seq}"
            chunk_meta = {
                **doc.metadata,
                "chunk_index": seq,
                "chunk_strategy": "recursive",
                "section_heading": heading,
                "page_number": page_num,
                "char_count": len(clean_split),
                "token_count": self.count_tokens(clean_split),
                "source": doc.source_path,
                "raw_hash": doc.raw_hash,
            }
            chunks.append(
                Chunk(
                    chunk_id=c_id,
                    text=clean_split,
                    source=doc.source_path,
                    raw_hash=doc.raw_hash,
                    chunk_index=seq,
                    section_heading=heading,
                    chunk_strategy="recursive",
                    char_count=len(clean_split),
                    page_number=page_num,
                    metadata=chunk_meta,
                )
            )

        return chunks


class SemanticChunker(BaseChunker):
    """Splits document into sentences and merges them into chunks based on embedding similarity drop."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        overlap: int = settings.CHUNK_OVERLAP,
        split_threshold: float = settings.SEMANTIC_SPLIT_THRESHOLD,
        embedder: Optional[EmbeddingModelWrapper] = None,
    ):
        super().__init__(chunk_size, overlap)
        self.split_threshold = split_threshold
        self.embedder = embedder

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences cleanly using regex."""
        sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9#])")
        paragraphs = text.split("\n\n")
        sentences = []
        for p in paragraphs:
            p_strip = p.strip()
            if not p_strip:
                continue
            p_sentences = sentence_endings.split(p_strip)
            for s in p_sentences:
                s_strip = s.strip()
                if s_strip:
                    sentences.append(s_strip)
        return sentences

    def chunk_document(self, doc: RawDocument) -> List[Chunk]:
        sentences = self._split_into_sentences(doc.content)
        if not sentences:
            return []

        if len(sentences) == 1 or not self.embedder:
            # Fallback if single sentence or embedder absent
            return RecursiveChunker(self.chunk_size, self.overlap).chunk_document(doc)

        # Get embeddings for all sentences
        try:
            embeddings = self.embedder.embed_documents(sentences)
        except Exception as e:
            logger.warning(f"Semantic chunker failed to generate embeddings ({e}), falling back to recursive chunking.")
            return RecursiveChunker(self.chunk_size, self.overlap).chunk_document(doc)

        # Calculate similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = EmbeddingModelWrapper.cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Group sentences into topic chunks
        grouped_chunks_texts: List[str] = []
        current_group: List[str] = [sentences[0]]

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            candidate_text = " ".join(current_group + [next_sentence])
            
            # Split if similarity drops below threshold OR candidate exceeds target token size
            if sim < self.split_threshold or self.count_tokens(candidate_text) > self.chunk_size:
                grouped_chunks_texts.append(" ".join(current_group))
                current_group = [next_sentence]
            else:
                current_group.append(next_sentence)

        if current_group:
            grouped_chunks_texts.append(" ".join(current_group))

        chunks: List[Chunk] = []
        page_num = doc.metadata.get("page_number", 1)
        current_heading = doc.metadata.get("section_heading", "")

        for seq, text in enumerate(grouped_chunks_texts):
            clean_text = text.strip()
            if not clean_text:
                continue

            match = re.search(r"^(#+)\s+(.+)$", clean_text, re.MULTILINE)
            if match:
                current_heading = match.group(2).strip()

            c_id = f"{doc.raw_hash}_p{page_num}_sem_{seq}"
            chunk_meta = {
                **doc.metadata,
                "chunk_index": seq,
                "chunk_strategy": "semantic",
                "section_heading": current_heading,
                "page_number": page_num,
                "char_count": len(clean_text),
                "token_count": self.count_tokens(clean_text),
                "source": doc.source_path,
                "raw_hash": doc.raw_hash,
            }
            chunks.append(
                Chunk(
                    chunk_id=c_id,
                    text=clean_text,
                    source=doc.source_path,
                    raw_hash=doc.raw_hash,
                    chunk_index=seq,
                    section_heading=current_heading,
                    chunk_strategy="semantic",
                    char_count=len(clean_text),
                    page_number=page_num,
                    metadata=chunk_meta,
                )
            )

        return chunks


def get_chunker(
    strategy: str = settings.CHUNK_STRATEGY,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
    embedder: Optional[EmbeddingModelWrapper] = None,
) -> BaseChunker:
    """Factory method to get selected chunking strategy."""
    strat = strategy.lower()
    if strat == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    elif strat == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
    elif strat == "semantic":
        return SemanticChunker(chunk_size=chunk_size, overlap=overlap, embedder=embedder)
    else:
        logger.warning(f"Unknown chunk strategy '{strategy}', defaulting to 'recursive'")
        return RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
