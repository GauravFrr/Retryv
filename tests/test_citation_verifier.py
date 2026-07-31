"""
Unit tests for CitationVerifier (Phase 3 Unit 3.4).

All tests mock google.genai.Client — no real API calls.
inter_call_sleep=0.0 is passed to CitationVerifier() in every test
so no real sleeps occur.
"""
import pytest
from unittest.mock import MagicMock, patch

from google.genai.errors import APIError

from app.models.chunk import Chunk
from app.models.search import SearchResult
from app.models.generation import (
    CitedChunk,
    GenerationResult,
    VerificationResult,
    VerifiedGenerationResult,
)
from app.generation.citation_verifier import (
    CitationVerifier,
    VerificationError,
    _SUPPORTED,
    _NOT_SUPPORTED,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cited_chunk_a() -> CitedChunk:
    return CitedChunk(
        index=1,
        source_file="docs/en/docs/advanced/events.md",
        section_heading="Lifespan Events",
        url="https://fastapi.tiangolo.com/advanced/events/",
        text="You can use the lifespan parameter to register startup and shutdown logic.",
    )


@pytest.fixture
def cited_chunk_b() -> CitedChunk:
    return CitedChunk(
        index=2,
        source_file="docs/en/docs/tutorial/dependencies/index.md",
        section_heading="Dependencies",
        url="https://fastapi.tiangolo.com/tutorial/dependencies/",
        text="Dependency Injection lets you declare dependencies in path operations.",
    )


@pytest.fixture
def gen_result_two(cited_chunk_a, cited_chunk_b) -> GenerationResult:
    return GenerationResult(
        answer="FastAPI lifespan uses a context manager [1]. Dependencies use Depends() [2].",
        cited_chunks=[cited_chunk_a, cited_chunk_b],
        is_grounded=True,
        model="gemini-2.5-flash",
    )


@pytest.fixture
def gen_result_empty() -> GenerationResult:
    return GenerationResult(
        answer="I do not have enough context to answer this question.",
        cited_chunks=[],
        is_grounded=False,
        model="gemini-2.5-flash",
    )


def _mock_judge(text: str) -> MagicMock:
    """Build a minimal mock response for the judge."""
    r = MagicMock()
    r.text = text
    return r


def _make_verifier(mock_client: MagicMock) -> CitationVerifier:
    """Return a CitationVerifier whose client is replaced by mock_client.
    inter_call_sleep=0.0 so tests don't actually wait."""
    v = CitationVerifier(model_name="gemini-2.5-flash", inter_call_sleep=0.0)
    v._client = mock_client
    return v


# ---------------------------------------------------------------------------
# _verify_one / _parse_verdict tests
# ---------------------------------------------------------------------------

class TestVerifyOne:
    def test_supported_verdict_parsed_correctly(self, cited_chunk_a):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("SUPPORTED")
        v = _make_verifier(mock_client)
        result = v._verify_one("Some answer [1].", cited_chunk_a)
        assert result.supported is True
        assert result.verdict == _SUPPORTED
        assert result.parse_error is False

    def test_not_supported_verdict_parsed_correctly(self, cited_chunk_a):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("NOT_SUPPORTED")
        v = _make_verifier(mock_client)
        result = v._verify_one("Some answer [1].", cited_chunk_a)
        assert result.supported is False
        assert result.verdict == _NOT_SUPPORTED
        assert result.parse_error is False

    def test_verdict_lowercase_normalised_to_supported(self, cited_chunk_a):
        """Judge returning lowercase should still be parsed as SUPPORTED."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("supported")
        v = _make_verifier(mock_client)
        result = v._verify_one("Some answer.", cited_chunk_a)
        assert result.supported is True
        assert result.parse_error is False

    def test_verdict_with_trailing_whitespace_normalised(self, cited_chunk_a):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("  SUPPORTED  ")
        v = _make_verifier(mock_client)
        result = v._verify_one("Some answer.", cited_chunk_a)
        assert result.supported is True
        assert result.parse_error is False

    def test_unexpected_response_sets_parse_error(self, cited_chunk_a):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("Maybe it does.")
        v = _make_verifier(mock_client)
        result = v._verify_one("Some answer.", cited_chunk_a)
        assert result.supported is False
        assert result.verdict == _NOT_SUPPORTED
        assert result.parse_error is True

    def test_raw_response_preserved_on_unexpected(self, cited_chunk_a):
        raw = "I think so?"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge(raw)
        v = _make_verifier(mock_client)
        result = v._verify_one("Some answer.", cited_chunk_a)
        assert result.raw_response == raw

    def test_raw_response_preserved_on_valid_verdict(self, cited_chunk_a):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("SUPPORTED")
        v = _make_verifier(mock_client)
        result = v._verify_one("Answer.", cited_chunk_a)
        assert result.raw_response == "SUPPORTED"

    def test_chunk_index_propagated_to_result(self, cited_chunk_a):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("SUPPORTED")
        v = _make_verifier(mock_client)
        result = v._verify_one("Answer.", cited_chunk_a)
        assert result.chunk_index == cited_chunk_a.index

    def test_source_file_propagated_to_result(self, cited_chunk_a):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("SUPPORTED")
        v = _make_verifier(mock_client)
        result = v._verify_one("Answer.", cited_chunk_a)
        assert result.source_file == cited_chunk_a.source_file


# ---------------------------------------------------------------------------
# verify() tests
# ---------------------------------------------------------------------------

class TestVerify:
    def test_empty_cited_chunks_returns_immediately_without_api_call(self, gen_result_empty):
        mock_client = MagicMock()
        v = _make_verifier(mock_client)
        result = v.verify(gen_result_empty)
        assert mock_client.models.generate_content.call_count == 0
        assert isinstance(result, VerifiedGenerationResult)

    def test_empty_cited_chunks_all_supported_true_by_convention(self, gen_result_empty):
        mock_client = MagicMock()
        v = _make_verifier(mock_client)
        result = v.verify(gen_result_empty)
        assert result.all_supported is True

    def test_empty_cited_chunks_support_ratio_one(self, gen_result_empty):
        mock_client = MagicMock()
        v = _make_verifier(mock_client)
        result = v.verify(gen_result_empty)
        assert result.support_ratio == 1.0

    def test_empty_cited_chunks_verifications_list_empty(self, gen_result_empty):
        mock_client = MagicMock()
        v = _make_verifier(mock_client)
        result = v.verify(gen_result_empty)
        assert result.verifications == []

    def test_all_supported_sets_all_supported_true(self, gen_result_two):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("SUPPORTED")
        v = _make_verifier(mock_client)
        result = v.verify(gen_result_two)
        assert result.all_supported is True
        assert result.support_ratio == 1.0

    def test_one_not_supported_sets_all_supported_false(self, gen_result_two):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _mock_judge("SUPPORTED"),
            _mock_judge("NOT_SUPPORTED"),
        ]
        v = _make_verifier(mock_client)
        result = v.verify(gen_result_two)
        assert result.all_supported is False
        assert result.support_ratio == pytest.approx(0.5)

    def test_support_ratio_two_of_three_supported(self):
        """Three citations, two SUPPORTED → ratio ≈ 0.667."""
        chunk_c = CitedChunk(
            index=3,
            source_file="docs/en/docs/tutorial/index.md",
            section_heading="Overview",
            url="",
            text="FastAPI is fast.",
        )
        gen = GenerationResult(
            answer="Answer [1] [2] [3].",
            cited_chunks=[
                CitedChunk(index=1, source_file="a.md", section_heading="", url="", text="A"),
                CitedChunk(index=2, source_file="b.md", section_heading="", url="", text="B"),
                chunk_c,
            ],
            is_grounded=True,
            model="gemini-2.5-flash",
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _mock_judge("SUPPORTED"),
            _mock_judge("NOT_SUPPORTED"),
            _mock_judge("SUPPORTED"),
        ]
        v = _make_verifier(mock_client)
        result = v.verify(gen)
        assert result.support_ratio == pytest.approx(2 / 3)

    def test_verifications_count_matches_cited_chunks(self, gen_result_two):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("SUPPORTED")
        v = _make_verifier(mock_client)
        result = v.verify(gen_result_two)
        assert len(result.verifications) == len(gen_result_two.cited_chunks)

    def test_verification_chunk_index_matches_cited_chunk(self, gen_result_two):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("SUPPORTED")
        v = _make_verifier(mock_client)
        result = v.verify(gen_result_two)
        for v_result, cited in zip(result.verifications, gen_result_two.cited_chunks):
            assert v_result.chunk_index == cited.index

    def test_generation_result_preserved_in_output(self, gen_result_two):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("SUPPORTED")
        v = _make_verifier(mock_client)
        result = v.verify(gen_result_two)
        assert result.generation is gen_result_two


# ---------------------------------------------------------------------------
# Inter-call sleep tests
# ---------------------------------------------------------------------------

class TestInterCallSleep:
    def test_sleep_called_between_calls_not_after_last(self, gen_result_two):
        """With 2 citations, sleep must be called exactly once (between calls,
        not after the last one)."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("SUPPORTED")
        v = _make_verifier(mock_client)
        v.inter_call_sleep = 0.1  # small but non-zero for this test

        with patch("app.generation.citation_verifier.time.sleep") as mock_sleep:
            v.verify(gen_result_two)

        assert mock_sleep.call_count == 1  # N-1 = 2-1 = 1

    def test_no_sleep_when_single_citation(self, gen_result_two):
        """Single citation → zero sleeps (nothing to space out)."""
        gen_single = GenerationResult(
            answer="Answer [1].",
            cited_chunks=[gen_result_two.cited_chunks[0]],
            is_grounded=True,
            model="gemini-2.5-flash",
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("SUPPORTED")
        v = _make_verifier(mock_client)
        v.inter_call_sleep = 0.1

        with patch("app.generation.citation_verifier.time.sleep") as mock_sleep:
            v.verify(gen_single)

        assert mock_sleep.call_count == 0

    def test_no_sleep_when_inter_call_sleep_zero(self, gen_result_two):
        """inter_call_sleep=0.0 → time.sleep never called."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_judge("SUPPORTED")
        v = _make_verifier(mock_client)  # already 0.0 from _make_verifier

        with patch("app.generation.citation_verifier.time.sleep") as mock_sleep:
            v.verify(gen_result_two)

        assert mock_sleep.call_count == 0


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_verify_one_retries_once_on_transient_failure(self, cited_chunk_a):
        """_verify_one retries on APIError then returns valid result."""
        mock_client = MagicMock()
        api_error = APIError(503, {"error": {"message": "Service Unavailable"}})
        mock_client.models.generate_content.side_effect = [
            api_error,
            _mock_judge("SUPPORTED"),
        ]
        v = _make_verifier(mock_client)

        with patch("app.generation.citation_verifier.time.sleep"):
            result = v._verify_one("Answer.", cited_chunk_a)

        assert result.supported is True
        assert mock_client.models.generate_content.call_count == 2

    def test_verify_one_raises_verification_error_after_all_retries_exhausted(
        self, cited_chunk_a
    ):
        """All 3 attempts fail → VerificationError raised."""
        mock_client = MagicMock()
        api_error = APIError(429, {"error": {"message": "Rate Limited"}})
        mock_client.models.generate_content.side_effect = api_error
        v = _make_verifier(mock_client)

        with patch("app.generation.citation_verifier.time.sleep"):
            with pytest.raises(VerificationError):
                v._verify_one("Answer.", cited_chunk_a)

        assert mock_client.models.generate_content.call_count == 3

    def test_non_api_error_raises_verification_error_immediately(self, cited_chunk_a):
        """Non-APIError surfaces as VerificationError without retrying."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("crash")
        v = _make_verifier(mock_client)

        with patch("app.generation.citation_verifier.time.sleep"):
            with pytest.raises(VerificationError):
                v._verify_one("Answer.", cited_chunk_a)

        assert mock_client.models.generate_content.call_count == 1
