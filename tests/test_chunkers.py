import pytest
from unittest.mock import MagicMock, PropertyMock
import numpy as np
from app.models.document import RawDocument
from app.ingestion.chunkers.fixed_size import FixedSizeChunker
from app.ingestion.chunkers.structure_aware import StructureAwareChunker
from app.ingestion.chunkers.semantic import SemanticChunker


@pytest.fixture
def sample_doc():
    return RawDocument(
        id="doc1",
        text="# FastAPI Overview\nFastAPI is a modern web framework. It is fast and easy to use.\n\n## Sub Section\nHere is some detailed explanation about dependencies.",
        source_file="tutorial/index.md",
        metadata={"title": "FastAPI Overview", "url": "https://fastapi.tiangolo.com/tutorial/"}
    )


def test_fixed_size_chunker(sample_doc):
    # Set chunk size small enough to force split
    chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(sample_doc)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 50
        assert c.chunking_strategy == "fixed_size"
        assert c.source_file == sample_doc.source_file
        assert c.section_heading == "FastAPI Overview"
        assert c.metadata["url"] == sample_doc.metadata["url"]


def test_structure_aware_chunker(sample_doc):
    chunker = StructureAwareChunker(max_chunk_size=1000)
    chunks = chunker.chunk(sample_doc)

    assert len(chunks) == 2
    
    # First chunk under main heading
    assert chunks[0].section_heading == "FastAPI Overview"
    assert "FastAPI is a modern web framework" in chunks[0].text
    
    # Second chunk under Sub Section
    assert chunks[1].section_heading == "Sub Section"
    assert "detailed explanation about dependencies" in chunks[1].text


def test_structure_aware_chunker_ignores_code_blocks():
    doc = RawDocument(
        id="doc2",
        text="# Python Guide\nHere is code:\n```python\n# This is a comment, not a markdown heading\ndef test():\n    pass\n```\n## Next Part\nOut of code block.",
        source_file="guide.md",
        metadata={"title": "Python Guide"}
    )
    chunker = StructureAwareChunker(max_chunk_size=1000)
    chunks = chunker.chunk(doc)

    # Should only split on "## Next Part", not on the python comment "# This is a comment"
    assert len(chunks) == 2
    assert chunks[0].section_heading == "Python Guide"
    assert "This is a comment" in chunks[0].text
    assert chunks[1].section_heading == "Next Part"


def test_semantic_chunker_mocked():
    doc = RawDocument(
        id="doc3",
        text="Sentence one. Sentence two. Sentence three.",
        source_file="semantic.md",
        metadata={"title": "Semantic Title"}
    )
    
    chunker = SemanticChunker(similarity_threshold=0.5, min_chunk_size=5)
    
    # Mock the sentence-transformer model
    mock_model = MagicMock()
    
    # Mock encode to return 3 vectors
    # We want a split between sentence 1 & 2 (low similarity), and no split between 2 & 3 (high similarity)
    # Vector 1: [1, 0, 0]
    # Vector 2: [0, 1, 0] (orthogonal/similarity=0 to Vector 1 -> should split)
    # Vector 3: [0, 0.99, 0] (high similarity to Vector 2 -> should NOT split)
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    v3 = np.array([0.0, 0.99, 0.0])
    
    mock_model.encode.return_value = [v1, v2, v3]
    
    # Replace property model with mock
    type(chunker).model = PropertyMock(return_value=mock_model)
    
    chunks = chunker.chunk(doc)
    
    # We expect 2 chunks:
    # Chunk 1: "Sentence one."
    # Chunk 2: "Sentence two. Sentence three."
    assert len(chunks) == 2
    assert chunks[0].text == "Sentence one."
    assert chunks[1].text == "Sentence two. Sentence three."
    assert chunks[0].chunking_strategy == "semantic"
    assert chunks[0].section_heading == "Semantic Title"
