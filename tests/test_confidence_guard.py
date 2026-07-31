"""
Unit tests for ConfidenceGuard (Phase 3 Unit 3.3).

All tests are pure Python — no I/O, no Gemini calls, no ChromaDB.
Score values are chosen to match the real RRF clusters observed in the
"lifespan events" manual verification (real matches ≈ 0.031, noise ≈ 0.016).
"""
import pytest

from app.models.chunk import Chunk
from app.models.search import SearchResult
from app.generation.confidence_guard import ConfidenceGuard

# Default threshold from settings (0.025); tests that depend on a specific
# value construct ConfidenceGuard with an explicit threshold instead of
# relying on the settings singleton so they stay deterministic.
DEFAULT_THRESHOLD = 0.025


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(score: float) -> SearchResult:
    """Build a minimal SearchResult with the given score."""
    chunk = Chunk(
        id=f"chunk-{score}",
        text="Some chunk text.",
        source_file="docs/en/docs/index.md",
        section_heading="Overview",
        chunking_strategy="fixed_size",
        metadata={},
    )
    return SearchResult(chunk=chunk, score=score)


def _results(*scores: float) -> list[SearchResult]:
    """Build an ordered list of SearchResults from a sequence of scores."""
    return [_make_result(s) for s in scores]


# ---------------------------------------------------------------------------
# Core threshold tests
# ---------------------------------------------------------------------------

class TestConfidenceGuard:
    def test_is_sufficient_above_threshold(self):
        guard = ConfidenceGuard(threshold=DEFAULT_THRESHOLD)
        # Real-match score cluster (both dense+sparse agreed at rank 1)
        assert guard.is_retrieval_sufficient(_results(0.031)) is True

    def test_is_not_sufficient_below_threshold(self):
        guard = ConfidenceGuard(threshold=DEFAULT_THRESHOLD)
        # Noise score cluster (single system, rank 1)
        assert guard.is_retrieval_sufficient(_results(0.016)) is False

    def test_is_sufficient_at_exact_threshold_boundary(self):
        # Boundary is inclusive: score == threshold → passes
        guard = ConfidenceGuard(threshold=DEFAULT_THRESHOLD)
        assert guard.is_retrieval_sufficient(_results(DEFAULT_THRESHOLD)) is True

    def test_is_not_sufficient_just_below_boundary(self):
        guard = ConfidenceGuard(threshold=DEFAULT_THRESHOLD)
        just_below = DEFAULT_THRESHOLD - 1e-6
        assert guard.is_retrieval_sufficient(_results(just_below)) is False

    def test_empty_results_returns_false(self):
        guard = ConfidenceGuard(threshold=DEFAULT_THRESHOLD)
        assert guard.is_retrieval_sufficient([]) is False

    def test_custom_threshold_respected(self):
        # A stricter threshold of 0.05 should block a score of 0.031
        guard = ConfidenceGuard(threshold=0.05)
        assert guard.is_retrieval_sufficient(_results(0.031)) is False

    def test_custom_threshold_passes_when_score_sufficient(self):
        guard = ConfidenceGuard(threshold=0.05)
        assert guard.is_retrieval_sufficient(_results(0.06)) is True

    def test_uses_top_score_only(self):
        # Top result is above threshold; second result is below.
        # Guard should pass — only top-1 is checked.
        guard = ConfidenceGuard(threshold=DEFAULT_THRESHOLD)
        results = _results(0.031, 0.016)
        assert guard.is_retrieval_sufficient(results) is True

    def test_top_score_below_threshold_even_with_multiple_results(self):
        # Even if there are multiple results, it's the top score that matters.
        guard = ConfidenceGuard(threshold=DEFAULT_THRESHOLD)
        # Reversed: first result is the worst (scores should be descending in real
        # usage, but guard must still use index 0 regardless)
        results = _results(0.010, 0.031, 0.031)
        assert guard.is_retrieval_sufficient(results) is False


# ---------------------------------------------------------------------------
# Single-result edge case
# ---------------------------------------------------------------------------

class TestSingleResultEdgeCase:
    def test_single_result_above_threshold_is_not_blocked(self):
        """A single high-scoring result clears the threshold and proceeds.

        Single results cannot reflect dual-system RRF agreement, so a WARNING
        is logged, but generation is NOT blocked — the threshold check still
        applies.  This preserves valid narrow queries (e.g. highly specific
        FastAPI questions that only one system ranked at position 1).
        """
        guard = ConfidenceGuard(threshold=DEFAULT_THRESHOLD)
        assert guard.is_retrieval_sufficient(_results(0.031)) is True

    def test_single_result_below_threshold_is_blocked(self):
        """A single low-scoring result is blocked by the threshold — not by
        a special single-result rule."""
        guard = ConfidenceGuard(threshold=DEFAULT_THRESHOLD)
        assert guard.is_retrieval_sufficient(_results(0.016)) is False

    def test_single_result_emits_warning(self, caplog):
        """is_retrieval_sufficient() must log a WARNING when len == 1, regardless
        of whether the result passes the threshold."""
        import logging
        guard = ConfidenceGuard(threshold=DEFAULT_THRESHOLD)
        with caplog.at_level(logging.WARNING, logger="app.generation.confidence_guard"):
            guard.is_retrieval_sufficient(_results(0.031))
        assert any("only 1 chunk" in record.message for record in caplog.records)
