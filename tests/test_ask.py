from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.models.generation import GenerationResult, CitedChunk, VerifiedGenerationResult, VerificationResult
from app.models.search import SearchResult
from app.models.chunk import Chunk

client = TestClient(app)


@patch("app.api.v1.ask.hybrid_retriever")
@patch("app.api.v1.ask.generator")
@patch("app.api.v1.ask.verifier")
def test_ask_endpoint_happy_path(mock_verifier, mock_generator, mock_hybrid):
    # Setup mocks
    chunk = Chunk(
        id="test-chunk-1",
        text="FastAPI lifespan events work with async context managers.",
        source_file="docs/en/docs/advanced/events.md",
        section_heading="Lifespan Events",
        chunking_strategy="fixed_size"
    )
    search_result = SearchResult(chunk=chunk, score=0.04, rrf_score=0.04)
    mock_hybrid.retrieve.return_value = [search_result]

    cited = CitedChunk(
        index=1,
        source_file="docs/en/docs/advanced/events.md",
        section_heading="Lifespan Events",
        text="FastAPI lifespan events work with async context managers."
    )
    gen_result = GenerationResult(
        answer="FastAPI lifespan events work with async context managers [1].",
        cited_chunks=[cited],
        is_grounded=True,
        model="gemini-3.1-flash-lite"
    )
    mock_generator.generate.return_value = gen_result

    ver_result = VerificationResult(
        chunk_index=1,
        source_file="docs/en/docs/advanced/events.md",
        verdict="SUPPORTED",
        supported=True,
        raw_response="SUPPORTED"
    )
    mock_verifier.verify.return_value = VerifiedGenerationResult(
        generation=gen_result,
        verifications=[ver_result],
        all_supported=True,
        support_ratio=1.0
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "query": "How do lifespan events work?",
            "strategy": "fixed_size",
            "method": "hybrid",
            "rerank": True,
            "verify_citations": True
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["generation"]["answer"] == "FastAPI lifespan events work with async context managers [1]."
    assert data["all_supported"] is True
    assert data["support_ratio"] == 1.0
    assert len(data["verifications"]) == 1
    assert data["verifications"][0]["verdict"] == "SUPPORTED"


@patch("app.api.v1.ask.hybrid_retriever")
@patch("app.api.v1.ask.generator")
def test_ask_endpoint_skip_verification(mock_generator, mock_hybrid):
    # Setup mocks
    chunk = Chunk(
        id="test-chunk-1",
        text="FastAPI lifespan events work with async context managers.",
        source_file="docs/en/docs/advanced/events.md",
        section_heading="Lifespan Events",
        chunking_strategy="fixed_size"
    )
    search_result = SearchResult(chunk=chunk, score=0.04, rrf_score=0.04)
    mock_hybrid.retrieve.return_value = [search_result]

    cited = CitedChunk(
        index=1,
        source_file="docs/en/docs/advanced/events.md",
        section_heading="Lifespan Events",
        text="FastAPI lifespan events work with async context managers."
    )
    gen_result = GenerationResult(
        answer="FastAPI lifespan events work with async context managers [1].",
        cited_chunks=[cited],
        is_grounded=True,
        model="gemini-3.1-flash-lite"
    )
    mock_generator.generate.return_value = gen_result

    response = client.post(
        "/api/v1/ask",
        json={
            "query": "How do lifespan events work?",
            "strategy": "fixed_size",
            "method": "hybrid",
            "rerank": True,
            "verify_citations": False
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["generation"]["answer"] == "FastAPI lifespan events work with async context managers [1]."
    assert data["verifications"] == []
    assert data["all_supported"] is True
    assert data["support_ratio"] == 0.0


def test_ask_endpoint_invalid_strategy():
    response = client.post(
        "/api/v1/ask",
        json={
            "query": "How do lifespan events work?",
            "strategy": "invalid_strategy_name",
            "method": "hybrid"
        }
    )
    assert response.status_code == 400
    assert "Invalid strategy" in response.json()["detail"]
