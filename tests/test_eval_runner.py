"""
Unit tests for EvalRunner (Phase 4 Unit 4.3).

All tests use mocked retrievers, generators, verifiers, and metrics.
No live API calls or ChromaDB operations occur during these tests.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.eval.dataset import GoldenQuery
from app.eval.models import EvaluationReport, MetricScore
from app.eval.runner import EvalRunner
from app.models.chunk import Chunk
from app.models.generation import GenerationResult, VerifiedGenerationResult, VerificationResult
from app.models.search import SearchResult


# ---------------------------------------------------------------------------
# Fixtures & Mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    chunk = Chunk(
        id="c1",
        text="Lifespan events description.",
        source_file="docs/en/docs/advanced/events.md",
        section_heading="Lifespan",
        chunking_strategy="fixed_size",
        metadata={},
    )
    retriever.retrieve.return_value = [SearchResult(chunk=chunk, score=0.9)]
    return retriever


@pytest.fixture
def mock_generator():
    generator = MagicMock()
    generator.generate.return_value = GenerationResult(
        answer="FastAPI lifespan events handle startup [1].",
        cited_chunks=[],
        is_grounded=True,
        model="gemini-2.5-flash",
    )
    return generator


@pytest.fixture
def mock_verifier():
    verifier = MagicMock()
    verifier.verify.return_value = VerifiedGenerationResult(
        generation=GenerationResult(
            answer="FastAPI lifespan events handle startup [1].",
            cited_chunks=[],
            is_grounded=True,
            model="gemini-2.5-flash",
        ),
        verifications=[],
        all_supported=True,
        support_ratio=1.0,
    )
    return verifier


@pytest.fixture
def sample_golden_queries():
    return [
        GoldenQuery(
            id="query-1-lookup",
            query="how do lifespan events work",
            category="lookup",
            retrieval_sufficient=True,
            relevant_sources=["docs/en/docs/advanced/events.md"],
            expected_answer="Lifespan events handle startup.",
            answer_quality_notes="Must mention startup.",
            phase4_ready=True,
        ),
        GoldenQuery(
            id="query-2-unanswerable",
            query="explain cricket rules",
            category="unanswerable",
            retrieval_sufficient=False,
            relevant_sources=None,
            expected_answer=None,
            answer_quality_notes=None,
            phase4_ready=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestEvalRunner:

    @patch("app.eval.runner.load_golden_dataset")
    def test_eval_runner_executes_specified_limit(
        self, mock_load, mock_retriever, mock_generator, mock_verifier, sample_golden_queries
    ):
        mock_load.return_value = sample_golden_queries
        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            verifier=mock_verifier,
            inter_query_sleep=0.0,
        )

        report = runner.run_evaluation(strategy="fixed_size", limit=1, save_report=False)

        assert isinstance(report, EvaluationReport)
        assert report.total_queries == 1
        assert len(report.results) == 1
        assert report.results[0].query_id == "query-1-lookup"

    @patch("app.eval.runner.load_golden_dataset")
    def test_eval_runner_aggregate_metrics_math(
        self, mock_load, mock_retriever, mock_generator, mock_verifier, sample_golden_queries
    ):
        mock_load.return_value = sample_golden_queries
        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            verifier=mock_verifier,
            inter_query_sleep=0.0,
        )

        report = runner.run_evaluation(strategy="fixed_size", limit=2, save_report=False)

        # 2 queries evaluated
        assert report.total_queries == 2
        assert "retrieval_recall" in report.aggregate_metrics
        assert "correctness" in report.aggregate_metrics
        # Recall for query 1 is 1.0 (matched events.md), Recall for query 2 is 1.0 (expected_sources=None)
        # Mean recall = (1.0 + 1.0) / 2 = 1.0
        assert report.aggregate_metrics["retrieval_recall"] == 1.0

    @patch("app.eval.runner.load_golden_dataset")
    def test_eval_runner_category_breakdown_math(
        self, mock_load, mock_retriever, mock_generator, mock_verifier, sample_golden_queries
    ):
        mock_load.return_value = sample_golden_queries
        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            verifier=mock_verifier,
            inter_query_sleep=0.0,
        )

        report = runner.run_evaluation(strategy="fixed_size", limit=2, save_report=False)

        assert "lookup" in report.category_breakdown
        assert "unanswerable" in report.category_breakdown
        assert report.category_breakdown["lookup"].count == 1
        assert report.category_breakdown["unanswerable"].count == 1

    @patch("app.eval.runner.load_golden_dataset")
    def test_eval_runner_unanswerable_correctness_handling(
        self, mock_load, mock_retriever, mock_generator, mock_verifier, sample_golden_queries
    ):
        """Unanswerable query with is_grounded=False must receive correctness score 1.0."""
        unanswerable_query = sample_golden_queries[1]
        mock_load.return_value = [unanswerable_query]

        # Generator returns is_grounded=False (correctly declined)
        mock_generator.generate.return_value = GenerationResult(
            answer="I do not have enough context to answer this question.",
            cited_chunks=[],
            is_grounded=False,
            model="gemini-2.5-flash",
        )

        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            verifier=mock_verifier,
            inter_query_sleep=0.0,
        )

        report = runner.run_evaluation(strategy="fixed_size", save_report=False)
        correctness_score = report.results[0].scores["correctness"].score
        assert correctness_score == 1.0
        assert "correctly returned insufficient-context" in report.results[0].scores["correctness"].reasoning

    @patch("app.eval.runner.load_golden_dataset")
    def test_eval_runner_report_file_saved(
        self, mock_load, mock_retriever, mock_generator, mock_verifier, sample_golden_queries, tmp_path
    ):
        mock_load.return_value = sample_golden_queries[:1]
        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            verifier=mock_verifier,
            inter_query_sleep=0.0,
        )

        report = runner.run_evaluation(
            strategy="fixed_size", limit=1, save_report=True, reports_dir=tmp_path
        )

        report_file = tmp_path / f"{report.id}.json"
        assert report_file.exists()
        assert "fixed_size" in report_file.name
