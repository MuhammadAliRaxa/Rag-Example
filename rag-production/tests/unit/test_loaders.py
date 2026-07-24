import tempfile
from pathlib import Path
import pytest
from src.ingestion.loaders import DocumentLoader, RawDocument


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_load_txt(temp_dir):
    txt_file = temp_dir / "sample.txt"
    txt_file.write_text("Hello World!\n\nThis is a plain text file test.", encoding="utf-8")
    
    loader = DocumentLoader(raw_dir=str(temp_dir / "raw"), processed_dir=str(temp_dir / "processed"))
    docs = loader.load(str(txt_file))
    
    assert len(docs) == 1
    doc = docs[0]
    assert doc.file_type == "txt"
    assert "Hello World!" in doc.content
    assert doc.metadata["page_number"] == 1
    assert Path(doc.raw_saved_path).exists()
    assert Path(doc.processed_saved_path).exists()


def test_load_markdown(temp_dir):
    md_file = temp_dir / "sample.md"
    md_file.write_text("# Main Title\n\nSome introductory content.\n\n## Sub Section\n\nMore detailed info.", encoding="utf-8")
    
    loader = DocumentLoader(raw_dir=str(temp_dir / "raw"), processed_dir=str(temp_dir / "processed"))
    docs = loader.load(str(md_file))
    
    assert len(docs) == 1
    doc = docs[0]
    assert doc.file_type == "markdown"
    assert doc.metadata["section_heading"] == "Main Title"
    assert "Some introductory content." in doc.content


def test_load_html(temp_dir):
    html_file = temp_dir / "sample.html"
    html_content = "<html><head><title>HTML Doc Title</title></head><body><h1>Heading 1</h1><p>Paragraph content.</p></body></html>"
    html_file.write_text(html_content, encoding="utf-8")
    
    loader = DocumentLoader(raw_dir=str(temp_dir / "raw"), processed_dir=str(temp_dir / "processed"))
    docs = loader.load(str(html_file))
    
    assert len(docs) == 1
    doc = docs[0]
    assert doc.file_type == "html"
    assert doc.metadata["section_heading"] == "HTML Doc Title"
    assert "Heading 1" in doc.content
    assert "Paragraph content." in doc.content
