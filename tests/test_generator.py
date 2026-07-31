"""
Unit tests for Generator (Phase 3 Unit 3.2).

All tests mock google.genai.Client — no real API calls are made.
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from google.genai.errors import APIError

from app.models.chunk import Chunk
from app.models.search import SearchResult
from app.models.generation import CitedChunk, GenerationResult
from app.generation.generator import Generator, GenerationError, _NO_CONTEXT_PHRASE


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chunk_a() -> Chunk:
    return Chunk(
        id="chunk-a",
        text="You can use the lifespan parameter to register startup and shutdown logic.",
        source_file="docs/en/docs/advanced/events.md",
        section_heading="Lifespan Events",
        chunking_strategy="fixed_size",
        metadata={"url": "https://fastapi.tiangolo.com/advanced/events/"},
    )


@pytest.fixture
def chunk_b() -> Chunk:
    return Chunk(
        id="chunk-b",
        text="Dependency Injection lets you declare dependencies in path operations.",
        source_file="docs/en/docs/tutorial/dependencies/index.md",
        section_heading="Dependencies",
        chunking_strategy="fixed_size",
        metadata={"url": "https://fastapi.tiangolo.com/tutorial/dependencies/"},
    )


@pytest.fixture
def results_two(chunk_a, chunk_b) -> list[SearchResult]:
    return [
        SearchResult(chunk=chunk_a, score=0.9),
        SearchResult(chunk=chunk_b, score=0.7),
    ]


def _mock_response(text: str) -> MagicMock:
    """Build a minimal mock that mimics the Gemini generate_content response."""
    response = MagicMock()
    response.text = text
    return response


def _make_generator_with_mock_client(mock_client: MagicMock) -> Generator:
    """Return a Generator whose internal client is replaced by mock_client."""
    gen = Generator(model_name="gemini-2.5-flash")
    gen._client = mock_client
    return gen


# ---------------------------------------------------------------------------
# generate() — happy-path
# ---------------------------------------------------------------------------

class TestGenerateHappyPath:
    def test_returns_generation_result_type(self, results_two):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            "FastAPI lifespan lets you run startup code [1]."
        )
        gen = _make_generator_with_mock_client(mock_client)
        result = gen.generate("How do lifespan events work?", results_two)
        assert isinstance(result, GenerationResult)

    def test_answer_text_matches_response(self, results_two):
        answer_text = "Use the lifespan context manager [1]."
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(answer_text)
        gen = _make_generator_with_mock_client(mock_client)
        result = gen.generate("Lifespan?", results_two)
        assert result.answer == answer_text

    def test_model_name_propagated(self, results_two):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response("Answer [1].")
        gen = _make_generator_with_mock_client(mock_client)
        result = gen.generate("?", results_two)
        assert result.model == "gemini-2.5-flash"

    def test_is_grounded_true_on_normal_answer(self, results_two):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            "Lifespan events use an async context manager [1]."
        )
        gen = _make_generator_with_mock_client(mock_client)
        result = gen.generate("Lifespan?", results_two)
        assert result.is_grounded is True

    def test_generate_calls_api_once_on_success(self, results_two):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response("Answer [1].")
        gen = _make_generator_with_mock_client(mock_client)
        gen.generate("?", results_two)
        assert mock_client.models.generate_content.call_count == 1


# ---------------------------------------------------------------------------
# _parse_citations
# ---------------------------------------------------------------------------

class TestParseCitations:
    def _gen(self) -> Generator:
        """Return a bare Generator (no client needed for pure-function tests)."""
        g = Generator.__new__(Generator)
        return g

    def test_single_citation_returns_one_cited_chunk(self, results_two):
        gen = self._gen()
        cited = gen._parse_citations("Answer [1].", results_two)
        assert len(cited) == 1
        assert cited[0].index == 1

    def test_two_citations_returned_in_order(self, results_two):
        gen = self._gen()
        cited = gen._parse_citations("First [1]. Second [2].", results_two)
        assert [c.index for c in cited] == [1, 2]

    def test_duplicate_citations_deduplicated(self, results_two):
        gen = self._gen()
        cited = gen._parse_citations("See [1]. Also [1]. And [1].", results_two)
        assert len(cited) == 1

    def test_out_of_range_index_skipped(self, results_two):
        gen = self._gen()
        # Only 2 chunks available; [5] is out of range
        cited = gen._parse_citations("See [5].", results_two)
        assert cited == []

    def test_zero_index_skipped(self, results_two):
        gen = self._gen()
        cited = gen._parse_citations("See [0].", results_two)
        assert cited == []

    def test_cited_chunk_fields_populated(self, results_two, chunk_a):
        gen = self._gen()
        cited = gen._parse_citations("Answer [1].", results_two)
        assert cited[0].source_file == chunk_a.source_file
        assert cited[0].section_heading == chunk_a.section_heading
        assert cited[0].url == chunk_a.metadata["url"]
        assert cited[0].text == chunk_a.text

    def test_no_citations_returns_empty_list(self, results_two):
        gen = self._gen()
        cited = gen._parse_citations("An answer with no citation markers.", results_two)
        assert cited == []

    def test_empty_search_results_out_of_range_graceful(self):
        gen = self._gen()
        cited = gen._parse_citations("See [1].", [])
        assert cited == []


# ---------------------------------------------------------------------------
# _is_grounded
# ---------------------------------------------------------------------------

class TestIsGrounded:
    def _gen(self) -> Generator:
        return Generator.__new__(Generator)

    def test_returns_false_on_exact_sentinel(self):
        gen = self._gen()
        assert gen._is_grounded(_NO_CONTEXT_PHRASE) is False

    def test_returns_false_when_sentinel_embedded_in_text(self):
        gen = self._gen()
        text = f"Sorry, {_NO_CONTEXT_PHRASE} Please try rephrasing."
        assert gen._is_grounded(text) is False

    def test_returns_true_on_normal_answer(self):
        gen = self._gen()
        assert gen._is_grounded("FastAPI uses Starlette under the hood [1].") is True

    def test_returns_true_on_empty_string(self):
        gen = self._gen()
        assert gen._is_grounded("") is True


# ---------------------------------------------------------------------------
# Insufficient context path
# ---------------------------------------------------------------------------

class TestInsufficientContext:
    def test_is_grounded_false_when_model_returns_no_context(self, results_two):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            _NO_CONTEXT_PHRASE
        )
        gen = _make_generator_with_mock_client(mock_client)
        result = gen.generate("Something obscure?", results_two)
        assert result.is_grounded is False
        assert result.cited_chunks == []

    def test_empty_search_results_blocked_by_confidence_guard(self):
        """With the confidence guard in place, empty search_results are blocked
        BEFORE the Gemini API is called — the guard short-circuits immediately.
        The result is still a valid GenerationResult with is_grounded=False."""
        mock_client = MagicMock()
        gen = _make_generator_with_mock_client(mock_client)
        result = gen.generate("Anything?", [])
        assert isinstance(result, GenerationResult)
        assert result.is_grounded is False
        # API must NOT have been called — guard blocked before reaching it
        assert mock_client.models.generate_content.call_count == 0


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_generate_retries_once_on_transient_failure(self, results_two):
        """Client raises APIError on attempt 1, succeeds on attempt 2.
        generate() must return a valid result — confirming the retry fired.
        """
        mock_client = MagicMock()

        # Build a minimal fake APIError with the real constructor signature:
        # APIError(code: int, response_json: Any)
        api_error = APIError(503, {"error": {"message": "Service Unavailable"}})

        success_response = _mock_response("Lifespan uses a context manager [1].")

        mock_client.models.generate_content.side_effect = [api_error, success_response]

        gen = _make_generator_with_mock_client(mock_client)

        # Patch time.sleep so the test doesn't actually wait
        with patch("app.generation.generator.time.sleep"):
            result = gen.generate("How do lifespan events work?", results_two)

        assert isinstance(result, GenerationResult)
        assert result.answer == "Lifespan uses a context manager [1]."
        # Client must have been called exactly twice (1 failure + 1 retry)
        assert mock_client.models.generate_content.call_count == 2

    def test_generate_raises_generation_error_after_all_retries_exhausted(self, results_two):
        """Client fails on every attempt — GenerationError must be raised."""
        mock_client = MagicMock()
        api_error = APIError(429, {"error": {"message": "Rate Limited"}})
        mock_client.models.generate_content.side_effect = api_error

        gen = _make_generator_with_mock_client(mock_client)

        with patch("app.generation.generator.time.sleep"):
            with pytest.raises(GenerationError):
                gen.generate("?", results_two)

        # All 3 attempts (_MAX_ATTEMPTS) must have been tried
        assert mock_client.models.generate_content.call_count == 3

    def test_non_api_error_not_retried(self, results_two):
        """A plain RuntimeError (not APIError) should surface immediately as GenerationError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Unexpected crash")

        gen = _make_generator_with_mock_client(mock_client)

        with patch("app.generation.generator.time.sleep"):
            with pytest.raises(GenerationError):
                gen.generate("?", results_two)

        # Must NOT have retried — only 1 call
        assert mock_client.models.generate_content.call_count == 1


# ---------------------------------------------------------------------------
# Confidence guard integration (Unit 3.3)
# ---------------------------------------------------------------------------

class TestConfidenceGuardIntegration:
    """Verify the guard is wired into Generator.generate() correctly."""

    def test_generate_skips_api_call_when_confidence_too_low(self):
        """Results below the threshold must short-circuit BEFORE the Gemini call."""
        mock_client = MagicMock()
        low_score_chunk = Chunk(
            id="low",
            text="Irrelevant content.",
            source_file="docs/en/docs/index.md",
            section_heading="Overview",
            chunking_strategy="fixed_size",
            metadata={},
        )
        low_results = [SearchResult(chunk=low_score_chunk, score=0.010)]
        gen = _make_generator_with_mock_client(mock_client)

        result = gen.generate("Tell me about medieval castles?", low_results)

        # Gemini must NOT have been called
        assert mock_client.models.generate_content.call_count == 0
        # Result must be the canonical insufficient-context response
        assert result.is_grounded is False
        assert _NO_CONTEXT_PHRASE in result.answer
        assert result.cited_chunks == []

    def test_generate_proceeds_when_confidence_sufficient(self):
        """Results above the threshold must reach the Gemini API call."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            "FastAPI lifespan lets you run startup code [1]."
        )
        high_score_chunk = Chunk(
            id="high",
            text="Lifespan events let you run code on startup and shutdown.",
            source_file="docs/en/docs/advanced/events.md",
            section_heading="Lifespan Events",
            chunking_strategy="fixed_size",
            metadata={"url": "https://fastapi.tiangolo.com/advanced/events/"},
        )
        high_results = [SearchResult(chunk=high_score_chunk, score=0.031)]
        gen = _make_generator_with_mock_client(mock_client)

        result = gen.generate("How do lifespan events work?", high_results)

        # Gemini must have been called exactly once
        assert mock_client.models.generate_content.call_count == 1
        assert isinstance(result, GenerationResult)
        assert result.is_grounded is True
