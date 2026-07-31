import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.models.chunk import Chunk
from app.models.search import SearchResult
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.fusion import HybridRetriever


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


@patch("app.retrieval.dense.GeminiEmbedder")
@patch("app.retrieval.dense.chromadb.PersistentClient")
def test_dense_retriever(mock_client_class, mock_embedder_class, sample_chunks):
    # Mock Embedder
    mock_embedder = MagicMock()
    mock_embedder.embed_text.return_value = [0.1, 0.2, 0.3]
    mock_embedder_class.return_value = mock_embedder

    # Mock Chroma Collection
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["chunk1", "chunk2"]],
        "documents": [["FastAPI is a modern web framework.", "It is fast and easy to use."]],
        "metadatas": [[
            {
                "source_file": "tutorial/index.md",
                "section_heading": "Overview",
                "chunking_strategy": "fixed_size",
                "title": "Overview",
                "url": "https://fastapi.tiangolo.com/"
            },
            {
                "source_file": "tutorial/index.md",
                "section_heading": "Overview",
                "chunking_strategy": "fixed_size",
                "title": "Overview",
                "url": "https://fastapi.tiangolo.com/"
            }
        ]],
        "distances": [[0.1, 0.3]]
    }

    # Mock Client
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    retriever = DenseRetriever()
    results = retriever.retrieve("test query", strategy="fixed_size", top_k=2)

    assert len(results) == 2
    # Verify score calculation: score = 1.0 - distance
    assert results[0].chunk.id == "chunk1"
    assert abs(results[0].score - 0.9) < 1e-5
    assert results[1].chunk.id == "chunk2"
    assert abs(results[1].score - 0.7) < 1e-5


def test_sparse_retriever(sample_chunks):
    # Mock BM25 model
    mock_bm25 = MagicMock()
    # Scores for chunks: chunk1=0.0, chunk2=1.5, chunk3=2.5
    mock_bm25.get_scores.return_value = [0.0, 1.5, 2.5]

    retriever = SparseRetriever()
    
    with patch.object(retriever, "_get_index", return_value=(mock_bm25, sample_chunks)):
        results = retriever.retrieve("test query", strategy="fixed_size", top_k=3)
        
        # Verify filtering of 0.0 scores
        assert len(results) == 2
        # Verify sorting: chunk3 (score 2.5) first, chunk2 (score 1.5) second
        assert results[0].chunk.id == "chunk3"
        assert results[0].score == 2.5
        assert results[1].chunk.id == "chunk2"
        assert results[1].score == 1.5


def test_hybrid_retriever_rrf(sample_chunks):
    # Setup Dense results: chunk1 (rank 1), chunk2 (rank 2)
    dense_results = [
        SearchResult(chunk=sample_chunks[0], score=0.9),
        SearchResult(chunk=sample_chunks[1], score=0.7)
    ]
    # Setup Sparse results: chunk3 (rank 1), chunk1 (rank 2)
    sparse_results = [
        SearchResult(chunk=sample_chunks[2], score=2.5),
        SearchResult(chunk=sample_chunks[0], score=1.5)
    ]

    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = dense_results

    mock_sparse = MagicMock()
    mock_sparse.retrieve.return_value = sparse_results

    retriever = HybridRetriever(dense_retriever=mock_dense, sparse_retriever=mock_sparse)
    results = retriever.retrieve("test query", strategy="fixed_size", top_k=3, rrf_k=60, rerank=False)

    # RRF Calculations:
    # chunk1: rank 1 in dense (1/(60+1)), rank 2 in sparse (1/(60+2))
    #         score = 1/61 + 1/62 = 0.0163934426 + 0.0161290322 = 0.0325224748
    # chunk2: rank 2 in dense (1/(60+2))
    #         score = 1/62 = 0.0161290322
    # chunk3: rank 1 in sparse (1/(60+1))
    #         score = 1/61 = 0.0163934426
    # Sorted order: chunk1 (0.0325), chunk3 (0.01639), chunk2 (0.016129)
    assert len(results) == 3
    assert results[0].chunk.id == "chunk1"
    assert abs(results[0].score - (1/61 + 1/62)) < 1e-7
    assert results[1].chunk.id == "chunk3"
    assert abs(results[1].score - (1/61)) < 1e-7
    assert results[2].chunk.id == "chunk2"
    assert abs(results[2].score - (1/62)) < 1e-7


@patch("app.api.v1.search.dense_retriever")
@patch("app.api.v1.search.sparse_retriever")
@patch("app.api.v1.search.hybrid_retriever")
def test_search_api_endpoint(mock_hybrid, mock_sparse, mock_dense, sample_chunks):
    client = TestClient(app)

    # Mock retriever response
    mock_results = [
        SearchResult(chunk=sample_chunks[0], score=0.95)
    ]
    mock_hybrid.retrieve.return_value = mock_results
    mock_dense.retrieve.return_value = mock_results
    mock_sparse.retrieve.return_value = mock_results

    # 1. Test hybrid search request
    response = client.post(
        "/api/v1/search/",
        json={
            "query": "lifespan details",
            "strategy": "structure_aware",
            "method": "hybrid",
            "top_k": 2
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["chunk"]["id"] == "chunk1"
    assert data[0]["score"] == 0.95

    # 2. Test invalid strategy validation
    response = client.post(
        "/api/v1/search/",
        json={
            "query": "lifespan details",
            "strategy": "invalid_strategy_name",
            "method": "hybrid"
        }
    )
    assert response.status_code == 400
    assert "Invalid strategy" in response.json()["detail"]

    # 3. Test invalid method validation
    response = client.post(
        "/api/v1/search/",
        json={
            "query": "lifespan details",
            "strategy": "fixed_size",
            "method": "invalid_method_name"
        }
    )
    assert response.status_code == 400
    assert "Invalid retrieval method" in response.json()["detail"]
