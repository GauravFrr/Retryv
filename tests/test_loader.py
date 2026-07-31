import pytest
from unittest.mock import MagicMock
from pathlib import Path
from app.ingestion.loader import DocLoader
from app.models.document import RawDocument


def test_strip_frontmatter():
    loader = DocLoader()
    
    # Text with frontmatter
    content_with_fm = "---\ntitle: Document Title\ncategory: tutorial\n---\n# Main Heading\nThis is content."
    cleaned = loader.strip_frontmatter(content_with_fm)
    assert cleaned == "# Main Heading\nThis is content."

    # Text without frontmatter
    content_no_fm = "# Main Heading\nThis is content."
    cleaned = loader.strip_frontmatter(content_no_fm)
    assert cleaned == "# Main Heading\nThis is content."


def test_extract_title():
    loader = DocLoader()
    
    # Standard header
    content = "# FastAPI Tutorial\nSome text here."
    title = loader.extract_title(content, "Fallback")
    assert title == "FastAPI Tutorial"

    # Header with extra space
    content_space = "#   Another Title   \nSome text."
    title = loader.extract_title(content_space, "Fallback")
    assert title == "Another Title"

    # No header
    content_none = "Just plain text without headings."
    title = loader.extract_title(content_none, "Fallback")
    assert title == "Fallback"


def test_get_url_from_path():
    loader = DocLoader()
    
    # Directory index
    url = loader.get_url_from_path("docs/en/docs/tutorial/dependencies/index.md")
    assert url == "https://fastapi.tiangolo.com/tutorial/dependencies/"

    # Single markdown file
    url = loader.get_url_from_path("docs/en/docs/features.md")
    assert url == "https://fastapi.tiangolo.com/features/"

    # Root index
    url = loader.get_url_from_path("docs/en/docs/index.md")
    assert url == "https://fastapi.tiangolo.com/"

    # Backslashes (Windows-style pathing normalization)
    url = loader.get_url_from_path("docs\\en\\docs\\advanced\\index.md")
    assert url == "https://fastapi.tiangolo.com/advanced/"


def test_loader_load(tmp_path):
    # Set up mock folder structure
    source_dir = tmp_path / "fastapi-src"
    docs_dir = source_dir / "docs" / "en" / "docs"
    docs_dir.mkdir(parents=True)

    # Create mock markdown files
    file1 = docs_dir / "index.md"
    file1.write_text(
        "---\nicon: material/home\n---\n# Welcome to FastAPI\nFastAPI is a modern framework.",
        encoding="utf-8"
    )

    file2 = docs_dir / "tutorial" / "first-steps.md"
    file2.parent.mkdir()
    file2.write_text(
        "---\ntype: guide\n---\n# First Steps\nLet's write some code.",
        encoding="utf-8"
    )

    # Initialize loader pointing to tmp_path
    loader = DocLoader(source_path=str(tmp_path))
    
    # Mock fetch_docs to avoid real git clone
    loader.fetch_docs = MagicMock()

    # Load documents
    documents = loader.load()

    # Verify documents
    loader.fetch_docs.assert_called_once()
    assert len(documents) == 2

    # Sort documents by path for deterministic checks
    documents.sort(key=lambda d: d.source_file)

    # Check doc1
    doc1 = documents[0]
    assert doc1.source_file == "docs/en/docs/index.md"
    assert doc1.text == "# Welcome to FastAPI\nFastAPI is a modern framework."
    assert doc1.metadata["title"] == "Welcome to FastAPI"
    assert doc1.metadata["url"] == "https://fastapi.tiangolo.com/"

    # Check doc2
    doc2 = documents[1]
    assert doc2.source_file == "docs/en/docs/tutorial/first-steps.md"
    assert doc2.text == "# First Steps\nLet's write some code."
    assert doc2.metadata["title"] == "First Steps"
    assert doc2.metadata["url"] == "https://fastapi.tiangolo.com/tutorial/first-steps/"
