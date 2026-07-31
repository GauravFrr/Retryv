"""
Unit tests for Evaluation Metrics (Phase 4 Unit 4.2).

All tests use hand-computed expected values and mock client calls. No live Gemini API
calls are made in these unit tests.
"""
import pytest
from unittest.mock import MagicMock, patch

from app.eval.dataset import GoldenQuery
from app.eval.metrics import (
    RetrievalMetrics,
    CitationAccuracyMetric,
    FaithfulnessMetric,
    AnswerCorrectnessMetric,
)
from app.models.chunk import Chunk
from app.models.generation import (
    CitedChunk,
    GenerationResult,
    VerifiedGenerationResult,
    VerificationResult,
)
from app.models.search import SearchResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chunk_a() -> Chunk:
    return Chunk(
        id="c1",
        text="Lifespan events handle startup and shutdown.",
        source_file="docs/en/docs/advanced/events.md",
        section_heading="Lifespan",
        chunking_strategy="fixed_size",
        metadata={},
    )


@pytest.fixture
def search_results(chunk_a) -> list[SearchResult]:
    return [SearchResult(chunk=chunk_a, score=0.9)]


@pytest.fixture
def cited_chunk() -> CitedChunk:
    return CitedChunk(
        index=1,
        source_file="docs/en/docs/advanced/events.md",
        section_heading="Lifespan",
        url="",
        text="Lifespan events handle startup and shutdown.",
    )


# ---------------------------------------------------------------------------
# 1. Retrieval Metrics Tests (Hand-Computed Math)
# ---------------------------------------------------------------------------

class TestRetrievalMetrics:

    def test_retrieval_recall_hand_computed(self):
        """Expected: ['a.md', 'b.md'], Retrieved: ['a.md', 'c.md'] -> Recall = 1 / 2 = 0.5."""
        expected = ["a.md", "b.md"]
        retrieved = ["a.md", "c.md"]
        metric_score = RetrievalMetrics.compute_recall(retrieved, expected)
        assert metric_score.score == 0.5
        assert metric_score.metadata["matched_sources"] == ["a.md"]

    def test_retrieval_recall_full_match(self):
        """Expected: ['a.md', 'b.md'], Retrieved: ['a.md', 'b.md', 'c.md'] -> Recall = 1.0."""
        expected = ["a.md", "b.md"]
        retrieved = ["a.md", "b.md", "c.md"]
        metric_score = RetrievalMetrics.compute_recall(retrieved, expected)
        assert metric_score.score == 1.0

    def test_retrieval_precision_hand_computed(self):
        """Expected: ['a.md', 'b.md'], Retrieved: ['a.md', 'c.md', 'd.md', 'e.md'] -> Precision = 1 / 4 = 0.25."""
        expected = ["a.md", "b.md"]
        retrieved = ["a.md", "c.md", "d.md", "e.md"]
        metric_score = RetrievalMetrics.compute_precision(retrieved, expected)
        assert metric_score.score == 0.25

    def test_retrieval_precision_empty_retrieved(self):
        metric_score = RetrievalMetrics.compute_precision([], ["a.md"])
        assert metric_score.score == 0.0


# ---------------------------------------------------------------------------
# 2. Citation Accuracy Metric Tests (Reuses CitationVerifier)
# ---------------------------------------------------------------------------

class TestCitationAccuracyMetric:

    def test_citation_accuracy_metric_reuses_verifier(self, cited_chunk):
        gen_result = GenerationResult(
            answer="FastAPI lifespan events handle startup [1].",
            cited_chunks=[cited_chunk],
            is_grounded=True,
            model="gemini-2.5-flash",
        )

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = VerifiedGenerationResult(
            generation=gen_result,
            verifications=[
                VerificationResult(
                    chunk_index=1,
                    source_file="docs/en/docs/advanced/events.md",
                    verdict="SUPPORTED",
                    supported=True,
                    raw_response="SUPPORTED",
                )
            ],
            all_supported=True,
            support_ratio=1.0,
        )

        metric = CitationAccuracyMetric(verifier=mock_verifier)
        score = metric.compute(gen_result)
        assert score.score == 1.0
        mock_verifier.verify.assert_called_once_with(gen_result)

    def test_citation_accuracy_unanswerable_is_grounded_false(self):
        gen_result = GenerationResult(
            answer="I do not have enough context from the documentation.",
            cited_chunks=[],
            is_grounded=False,
            model="gemini-2.5-flash",
        )
        metric = CitationAccuracyMetric()
        score = metric.compute(gen_result)
        assert score.score is None
        assert "N/A" in score.reasoning


# ---------------------------------------------------------------------------
# 3. Faithfulness Metric Tests (Sc * Su)
# ---------------------------------------------------------------------------

class TestFaithfulnessMetric:

    def test_faithfulness_fully_cited_and_supported(self, cited_chunk, search_results):
        """Answer is 100% covered by explicit citation [1]. Sc=1.0, Su=1.0 -> Faithfulness=1.0."""
        gen_result = GenerationResult(
            answer="Lifespan events handle startup and shutdown logic [1].",
            cited_chunks=[cited_chunk],
            is_grounded=True,
            model="gemini-2.5-flash",
        )

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = VerifiedGenerationResult(
            generation=gen_result,
            verifications=[
                VerificationResult(
                    chunk_index=1,
                    source_file="docs/en/docs/advanced/events.md",
                    verdict="SUPPORTED",
                    supported=True,
                    raw_response="SUPPORTED",
                )
            ],
            all_supported=True,
            support_ratio=1.0,
        )

        metric = FaithfulnessMetric(verifier=mock_verifier)
        score = metric.compute(gen_result, search_results)
        assert score.score == 1.0
        assert score.metadata["Sc_citation_support"] == 1.0
        assert score.metadata["Su_uncited_groundedness"] == 1.0

    def test_faithfulness_uncited_hallucination_penalty(self, cited_chunk, search_results):
        """Cited sentence is supported (Sc=1.0), but uncited sentence is UNGROUNDED (Su=0.0) -> Score = 0.0."""
        gen_result = GenerationResult(
            answer="FastAPI handles lifespan events [1]. Sebastian Ramirez created FastAPI in 2018.",
            cited_chunks=[cited_chunk],
            is_grounded=True,
            model="gemini-2.5-flash",
        )

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = VerifiedGenerationResult(
            generation=gen_result,
            verifications=[
                VerificationResult(
                    chunk_index=1,
                    source_file="events.md",
                    verdict="SUPPORTED",
                    supported=True,
                    raw_response="SUPPORTED",
                )
            ],
            all_supported=True,
            support_ratio=1.0,
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "UNGROUNDED"

        metric = FaithfulnessMetric(verifier=mock_verifier)
        metric._client = mock_client

        score = metric.compute(gen_result, search_results)
        assert score.score == 0.0
        assert score.metadata["Sc_citation_support"] == 1.0
        assert score.metadata["Su_uncited_groundedness"] == 0.0

    def test_faithfulness_partial_citation_support(self, cited_chunk, search_results):
        """Citations are 50% supported (Sc=0.5), uncited sentences grounded (Su=1.0) -> Score = 0.5."""
        gen_result = GenerationResult(
            answer="Lifespan events run startup logic [1]. Also shuts down [2].",
            cited_chunks=[cited_chunk, cited_chunk],
            is_grounded=True,
            model="gemini-2.5-flash",
        )

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = VerifiedGenerationResult(
            generation=gen_result,
            verifications=[
                VerificationResult(
                    chunk_index=1,
                    source_file="events.md",
                    verdict="SUPPORTED",
                    supported=True,
                    raw_response="SUPPORTED",
                ),
                VerificationResult(
                    chunk_index=2,
                    source_file="events.md",
                    verdict="NOT_SUPPORTED",
                    supported=False,
                    raw_response="NOT_SUPPORTED",
                ),
            ],
            all_supported=False,
            support_ratio=0.5,
        )

        metric = FaithfulnessMetric(verifier=mock_verifier)
        score = metric.compute(gen_result, search_results)
        assert score.score == 0.5

    def test_faithfulness_unanswerable_is_grounded_false(self, search_results):
        gen_result = GenerationResult(
            answer="I do not have enough context from the documentation.",
            cited_chunks=[],
            is_grounded=False,
            model="gemini-2.5-flash",
        )
        metric = FaithfulnessMetric()
        score = metric.compute(gen_result, search_results)
        assert score.score is None
        assert "N/A" in score.reasoning


# ---------------------------------------------------------------------------
# 4. Answer Correctness Metric Tests (LLM Judge & Ambiguous Logic)
# ---------------------------------------------------------------------------

class TestAnswerCorrectnessMetric:

    def test_correctness_ambiguous_category_prompt_formatting(self):
        """Ambiguous query must format judge prompt instructing at least ONE valid interpretation."""
        query = GoldenQuery(
            id="val-ambiguous",
            query="how to do validation in FastAPI",
            category="ambiguous",
            retrieval_sufficient=True,
            relevant_sources=["body.md"],
            expected_answer="FastAPI validates JSON body using Pydantic.",
            answer_quality_notes="(1) Pydantic body validation, (2) Path/Query param validation.",
            phase4_ready=True,
        )
        gen_result = GenerationResult(
            answer="You can validate request parameters using Path() and Query().",
            cited_chunks=[],
            is_grounded=True,
            model="gemini-2.5-flash",
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "SCORE: 1.0\nREASON: Answers Path/Query validation."

        metric = AnswerCorrectnessMetric()
        metric._client = mock_client

        score = metric.compute(query, gen_result)
        assert score.score == 1.0

        # Assert prompt contains the ambiguous instruction
        call_args = mock_client.models.generate_content.call_args[1]["contents"]
        assert "AT LEAST ONE of the valid interpretations" in call_args

    def test_correctness_standard_query_scoring(self):
        """Standard lookup query matches expected_answer via judge."""
        query = GoldenQuery(
            id="lifespan-std",
            query="how do lifespan events work",
            category="lookup",
            retrieval_sufficient=True,
            relevant_sources=["events.md"],
            expected_answer="Lifespan events run code before and after yield.",
            answer_quality_notes="Must mention yield.",
            phase4_ready=True,
        )
        gen_result = GenerationResult(
            answer="Lifespan events execute code prior to yield on startup.",
            cited_chunks=[],
            is_grounded=True,
            model="gemini-2.5-flash",
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "SCORE: 0.9\nREASON: Accurately explains startup yield."

        metric = AnswerCorrectnessMetric()
        metric._client = mock_client

        score = metric.compute(query, gen_result)
        assert score.score == 0.9
        assert score.reasoning == "Accurately explains startup yield."
