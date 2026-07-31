import pytest
import os
import shutil
import pickle
from unittest.mock import MagicMock, patch
from app.models.chunk import Chunk
from app.ingestion.indexer import ChromaIndexer, BM25Indexer, clean_tokenize
from app.core.config import settings


@pytest.fixture
def temp_dirs():
    # Setup temp paths inside workspace
    original_chroma = settings.CHROMA_DB_PATH
    original_data = settings.DATA_SOURCE_PATH
    
    settings.CHROMA_DB_PATH = "./data/test_chroma_db"
    settings.DATA_SOURCE_PATH = "./data/test_raw"
    
    yield
    
    # Cleanup
    shutil.rmtree(settings.absolute_chroma_path, ignore_errors=True)
    bm25_dir = settings.absolute_data_source_path.parent / "bm25"
    shutil.rmtree(bm25_dir, ignore_errors=True)
    
    settings.CHROMA_DB_PATH = original_chroma
    settings.DATA_SOURCE_PATH = original_data


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            id="chunk1",
            text="FastAPI is a modern web framework.",
            source_file="tutorial/index.md",
            section_heading="Overview",
            chunking_strategy="fixed_size",
            metadata={"title": "Overview", "url": "https://fastapi.tiangolo.com/"}
        ),
        Chunk(
            id="chunk2",
            text="It is fast and easy to use.",
            source_file="tutorial/index.md",
            section_heading="Overview",
            chunking_strategy="fixed_size",
            metadata={"title": "Overview", "url": "https://fastapi.tiangolo.com/"}
        ),
        Chunk(
            id="chunk3",
            text="Python type hints are awesome.",
            source_file="tutorial/index.md",
            section_heading="Overview",
            chunking_strategy="fixed_size",
            metadata={"title": "Overview", "url": "https://fastapi.tiangolo.com/"}
        ),
    ]


def test_clean_tokenize():
    text = "FastAPI's features! Are they fast, easy-to-use, and cool?"
    tokens = clean_tokenize(text)
    
    assert "fastapis" in tokens
    assert "features" in tokens
    assert "easy-to-use" in tokens
    assert "are" in tokens
    assert "cool" in tokens
    # Verifies punctuation like '!', '?', ',' are stripped
    for token in tokens:
        assert not any(char in token for char in ["!", "?", ",", "'"])


@patch("app.ingestion.indexer.GeminiEmbedder")
@patch("app.ingestion.indexer.chromadb.PersistentClient")
def test_chroma_indexer(mock_client_class, mock_embedder_class, sample_chunks, temp_dirs):
    # Setup mocks
    mock_embedder = MagicMock()
    # Return distinct mock vectors
    mock_embedder.embed_batch.return_value = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    mock_embedder_class.return_value = mock_embedder
    
    mock_collection = MagicMock()
    # Mock query to return empty results (no duplicates)
    mock_collection.query.return_value = {"distances": [[]]}
    
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client
    
    # Run Indexer
    indexer = ChromaIndexer(strategy="fixed_size")
    inserted, skipped = indexer.index_chunks(sample_chunks)
    
    assert inserted == 3
    assert skipped == 0
    
    # Verify collection add was called
    mock_collection.add.assert_called_once()
    call_args = mock_collection.add.call_args[1]
    assert call_args["ids"] == ["chunk1", "chunk2", "chunk3"]
    assert len(call_args["embeddings"]) == 3
    assert call_args["documents"] == [sample_chunks[0].text, sample_chunks[1].text, sample_chunks[2].text]
    assert call_args["metadatas"][0]["source_file"] == "tutorial/index.md"
    assert call_args["metadatas"][0]["section_heading"] == "Overview"


@patch("app.ingestion.indexer.GeminiEmbedder")
@patch("app.ingestion.indexer.chromadb.PersistentClient")
def test_chroma_indexer_deduplicates(mock_client_class, mock_embedder_class, sample_chunks, temp_dirs):
    # Setup mocks
    mock_embedder = MagicMock()
    # Return two identical/near-identical vectors to force in-memory duplicate check
    mock_embedder.embed_batch.return_value = [
        [1.0, 0.0, 0.0],
        [0.99, 0.0, 0.0],  # Cosine similarity is 0.99 (> 0.95) -> duplicate!
        [0.0, 0.0, 1.0]
    ]
    mock_embedder_class.return_value = mock_embedder
    
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"distances": [[]]}
    
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client
    
    # Run Indexer
    indexer = ChromaIndexer(strategy="fixed_size")
    inserted, skipped = indexer.index_chunks(sample_chunks)
    
    # The second chunk should be skipped as it's a duplicate of the first in-memory chunk
    assert inserted == 2
    assert skipped == 1


def test_bm25_indexer(sample_chunks, temp_dirs):
    indexer = BM25Indexer(strategy="fixed_size")
    
    # Build
    indexer.build_index(sample_chunks)
    
    # Verify file created
    assert indexer.index_path.exists()
    
    # Load and check
    bm25, loaded_chunks = indexer.load_index()
    assert len(loaded_chunks) == 3
    assert loaded_chunks[0].text == sample_chunks[0].text
    
    # Verify search query tokenizes and scores
    doc_scores = bm25.get_scores(clean_tokenize("web framework"))
    assert doc_scores[0] > 0  # First doc has "web framework"
    assert doc_scores[1] == 0  # Second doc doesn't
    assert doc_scores[2] == 0  # Third doc doesn't

