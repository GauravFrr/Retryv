from unittest.mock import MagicMock, patch
import pytest
from app.models.chunk import Chunk
from app.models.search import SearchResult
from app.retrieval.reranker import Reranker
from app.retrieval.fusion import HybridRetriever


def test_reranker_lazy_initialization():
    """Verify that CrossEncoder is not loaded during __init__ but is loaded on property access."""
    with patch("app.retrieval.reranker.CrossEncoder") as mock_encoder_cls:
        reranker = Reranker()
        assert reranker._model is None
        mock_encoder_cls.assert_not_called()

        # Access property
        model = reranker.model
        assert model is not None
        mock_encoder_cls.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_reranker_rerank_empty():
    """Verify that rerank() returns empty list immediately for empty candidates."""
    reranker = Reranker()
    assert reranker.rerank("query", []) == []


def test_reranker_rerank_logic():
    """Verify reranking scores are calculated and sorted correctly, and fields populated."""
    # Create sample chunks
    chunk1 = Chunk(id="chunk1", text="FastAPI is a modern web framework.", source_file="docs/main.md", section_heading="Intro", chunking_strategy="fixed_size", metadata={"url": ""})
    chunk2 = Chunk(id="chunk2", text="Lifespan events help manage startup events.", source_file="docs/events.md", section_heading="Events", chunking_strategy="fixed_size", metadata={"url": ""})
    chunk3 = Chunk(id="chunk3", text="ChromaDB stores embeddings.", source_file="docs/db.md", section_heading="DB", chunking_strategy="fixed_size", metadata={"url": ""})

    candidates = [
        SearchResult(chunk=chunk1, score=0.03, rrf_score=0.03),
        SearchResult(chunk=chunk2, score=0.02, rrf_score=0.02),
        SearchResult(chunk=chunk3, score=0.01, rrf_score=0.01),
    ]

    reranker = Reranker()
    mock_model = MagicMock()
    # Mock predict: index 1 (chunk2) gets highest score, index 0 (chunk1) second, index 2 (chunk3) lowest
    mock_model.predict.return_value = [0.1, 0.9, -0.5]
    
    with patch.object(Reranker, "model", new=mock_model):
        results = reranker.rerank("lifespan", candidates, top_k=2)

        # Assert predict was called with correct pairs
        mock_model.predict.assert_called_once_with([
            ("lifespan", "FastAPI is a modern web framework."),
            ("lifespan", "Lifespan events help manage startup events."),
            ("lifespan", "ChromaDB stores embeddings."),
        ])

        # Assert top_k limit and ordering (highest score first)
        assert len(results) == 2
        assert results[0].chunk.id == "chunk2"
        assert results[0].score == 0.9
        assert results[0].rrf_score == 0.02
        assert results[0].rerank_score == 0.9

        assert results[1].chunk.id == "chunk1"
        assert results[1].score == 0.1
        assert results[1].rrf_score == 0.03
        assert results[1].rerank_score == 0.1


def test_hybrid_retriever_calls_reranker():
    """Verify HybridRetriever invokes the reranker if enabled."""
    mock_dense = MagicMock()
    mock_sparse = MagicMock()
    mock_reranker = MagicMock()

    chunk = Chunk(id="c1", text="text", source_file="f", section_heading="h", chunking_strategy="fixed_size", metadata={"url": ""})
    mock_dense.retrieve.return_value = [SearchResult(chunk=chunk, score=0.8)]
    mock_sparse.retrieve.return_value = [SearchResult(chunk=chunk, score=12.0)]
    mock_reranker.rerank.return_value = [SearchResult(chunk=chunk, score=0.9, rrf_score=0.03, rerank_score=0.9)]

    retriever = HybridRetriever(dense_retriever=mock_dense, sparse_retriever=mock_sparse, reranker=mock_reranker)

    # Test with rerank=True
    results = retriever.retrieve("query", "fixed_size", top_k=5, rerank=True)
    assert len(results) == 1
    mock_reranker.rerank.assert_called_once()

    # Reset mock and test with rerank=False
    mock_reranker.reset_mock()
    results_no_rerank = retriever.retrieve("query", "fixed_size", top_k=5, rerank=False)
    assert len(results_no_rerank) == 1
    mock_reranker.rerank.assert_not_called()
