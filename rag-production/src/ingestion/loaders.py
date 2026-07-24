"""Document Loaders for Markdown, Plain Text, HTML, and PDF formats."""
import os
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from bs4 import BeautifulSoup
import html2text
import markdown
import pypdf

from src.config.settings import settings
from src.config.logging_config import logger


@dataclass
class RawDocument:
    source_path: str
    file_type: str
    raw_hash: str
    content: str  # Clean normalized text
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_saved_path: Optional[str] = None
    processed_saved_path: Optional[str] = None


class DocumentLoader:
    def __init__(self, raw_dir: str = settings.RAW_DOCS_DIR, processed_dir: str = settings.PROCESSED_DOCS_DIR):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # HTML to markdown converter
        self._html_converter = html2text.HTML2Text()
        self._html_converter.ignore_links = False
        self._html_converter.ignore_images = True
        self._html_converter.body_width = 0

    def _normalize_text(self, text: str) -> str:
        """Clean and normalize raw text content."""
        if not text:
            return ""
        # Remove null bytes and non-printable control characters except standard whitespace
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Normalize whitespace while preserving line structure
        lines = [line.strip() for line in text.splitlines()]
        # Remove consecutive blank lines
        normalized_lines = []
        blank = False
        for line in lines:
            if not line:
                if not blank:
                    normalized_lines.append("")
                    blank = True
            else:
                normalized_lines.append(line)
                blank = False
        return "\n".join(normalized_lines).strip()

    def _save_raw_and_processed(self, file_path: str, raw_bytes: bytes, processed_text: str) -> tuple[str, str, str]:
        """Save raw bytes and processed text with content hash for fast re-indexing."""
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        ext = Path(file_path).suffix or ".txt"
        
        raw_target = self.raw_dir / f"{raw_hash}{ext}"
        processed_target = self.processed_dir / f"{raw_hash}.txt"
        
        if not raw_target.exists():
            with open(raw_target, "wb") as f:
                f.write(raw_bytes)
                
        with open(processed_target, "w", encoding="utf-8") as f:
            f.write(processed_text)
            
        return raw_hash, str(raw_target), str(processed_target)

    def load_txt(self, file_path: str) -> RawDocument:
        """Load and normalize plain text file."""
        path = Path(file_path)
        with open(path, "rb") as f:
            raw_bytes = f.read()
        
        text = raw_bytes.decode("utf-8", errors="replace")
        normalized = self._normalize_text(text)
        
        raw_hash, raw_saved, proc_saved = self._save_raw_and_processed(file_path, raw_bytes, normalized)
        
        return RawDocument(
            source_path=str(path.resolve()),
            file_type="txt",
            raw_hash=raw_hash,
            content=normalized,
            metadata={
                "source": str(path.resolve()),
                "file_type": "txt",
                "file_name": path.name,
                "file_size": path.stat().st_size if path.exists() else len(raw_bytes),
                "section_heading": "",
                "page_number": 1
            },
            raw_saved_path=raw_saved,
            processed_saved_path=proc_saved
        )

    def load_markdown(self, file_path: str) -> RawDocument:
        """Load and parse markdown document, capturing structural metadata."""
        path = Path(file_path)
        with open(path, "rb") as f:
            raw_bytes = f.read()
            
        text = raw_bytes.decode("utf-8", errors="replace")
        normalized = self._normalize_text(text)
        
        # Extract main header if present
        first_heading = ""
        heading_match = re.search(r"^#+\s+(.+)$", normalized, re.MULTILINE)
        if heading_match:
            first_heading = heading_match.group(1).strip()

        raw_hash, raw_saved, proc_saved = self._save_raw_and_processed(file_path, raw_bytes, normalized)

        return RawDocument(
            source_path=str(path.resolve()),
            file_type="markdown",
            raw_hash=raw_hash,
            content=normalized,
            metadata={
                "source": str(path.resolve()),
                "file_type": "markdown",
                "file_name": path.name,
                "file_size": path.stat().st_size if path.exists() else len(raw_bytes),
                "section_heading": first_heading,
                "page_number": 1
            },
            raw_saved_path=raw_saved,
            processed_saved_path=proc_saved
        )

    def load_html(self, file_path: str) -> RawDocument:
        """Load, clean and extract plaintext from HTML files."""
        path = Path(file_path)
        with open(path, "rb") as f:
            raw_bytes = f.read()

        raw_text = raw_bytes.decode("utf-8", errors="replace")
        soup = BeautifulSoup(raw_text, "html.parser")

        # Extract page title / top headings for metadata
        page_title = ""
        if soup.title and soup.title.string:
            page_title = soup.title.string.strip()
        elif soup.h1:
            page_title = soup.h1.get_text().strip()

        # Remove script and style elements
        for script_or_style in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            script_or_style.decompose()

        cleaned_html = str(soup)
        md_text = self._html_converter.handle(cleaned_html)
        normalized = self._normalize_text(md_text)

        raw_hash, raw_saved, proc_saved = self._save_raw_and_processed(file_path, raw_bytes, normalized)

        return RawDocument(
            source_path=str(path.resolve()),
            file_type="html",
            raw_hash=raw_hash,
            content=normalized,
            metadata={
                "source": str(path.resolve()),
                "file_type": "html",
                "file_name": path.name,
                "file_size": path.stat().st_size if path.exists() else len(raw_bytes),
                "section_heading": page_title,
                "page_number": 1
            },
            raw_saved_path=raw_saved,
            processed_saved_path=proc_saved
        )

    def load_pdf(self, file_path: str) -> List[RawDocument]:
        """Load and extract pages from a PDF file as individual RawDocuments or concatenated with page metadata."""
        path = Path(file_path)
        with open(path, "rb") as f:
            raw_bytes = f.read()

        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        reader = pypdf.PdfReader(path)
        
        pages_docs: List[RawDocument] = []
        all_text_parts = []

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_normalized = self._normalize_text(page_text)
            
            # Simple heuristic for section heading on page
            heading = ""
            heading_match = re.search(r"^(?:Chapter|Section|[0-9]+\.|\b[A-Z\s]{4,}\b)(.+)?$", page_normalized, re.MULTILINE)
            if heading_match:
                heading = heading_match.group(0).strip()

            all_text_parts.append(f"--- Page {page_idx} ---\n" + page_normalized)
            
            pages_docs.append(
                RawDocument(
                    source_path=str(path.resolve()),
                    file_type="pdf",
                    raw_hash=raw_hash,
                    content=page_normalized,
                    metadata={
                        "source": str(path.resolve()),
                        "file_type": "pdf",
                        "file_name": path.name,
                        "file_size": path.stat().st_size if path.exists() else len(raw_bytes),
                        "section_heading": heading,
                        "page_number": page_idx,
                        "total_pages": len(reader.pages)
                    }
                )
            )

        full_normalized = "\n\n".join(all_text_parts)
        raw_target = self.raw_dir / f"{raw_hash}.pdf"
        proc_target = self.processed_dir / f"{raw_hash}.txt"

        if not raw_target.exists():
            with open(raw_target, "wb") as f:
                f.write(raw_bytes)
        with open(proc_target, "w", encoding="utf-8") as f:
            f.write(full_normalized)

        for doc in pages_docs:
            doc.raw_saved_path = str(raw_target)
            doc.processed_saved_path = str(proc_target)

        return pages_docs

    def load(self, file_path: str) -> List[RawDocument]:
        """Dispatch document loading based on file extension."""
        ext = Path(file_path).suffix.lower()
        if ext in [".md", ".markdown"]:
            return [self.load_markdown(file_path)]
        elif ext in [".txt"]:
            return [self.load_txt(file_path)]
        elif ext in [".html", ".htm"]:
            return [self.load_html(file_path)]
        elif ext in [".pdf"]:
            return self.load_pdf(file_path)
        else:
            # Fallback to plain text loader for unknown extensions
            logger.warning(f"Unsupported extension '{ext}' for {file_path}, falling back to plain text loader.")
            return [self.load_txt(file_path)]
