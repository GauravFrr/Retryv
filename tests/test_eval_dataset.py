"""
Unit tests for Golden Evaluation Dataset and Loader (Phase 4 Unit 4.1).

Validates schema integrity, query counts (>= 50), category coverage across
all 4 categories (lookup, multi_hop, ambiguous, unanswerable), and specific
domain rules (e.g. multi_hop >= 2 sources, ambiguous quality notes).
"""
import pytest
from pathlib import Path

from app.eval.dataset import (
    GoldenQuery,
    GoldenDataset,
    load_golden_dataset,
    DEFAULT_GOLDEN_DATASET_PATH,
)


@pytest.fixture
def dataset_queries() -> list[GoldenQuery]:
    return load_golden_dataset(ready_only=False)


class TestGoldenDatasetIntegrity:

    def test_golden_dataset_file_exists(self):
        assert DEFAULT_GOLDEN_DATASET_PATH.exists()

    def test_total_queries_count_at_least_50(self, dataset_queries):
        assert len(dataset_queries) >= 50

    def test_all_four_categories_represented(self, dataset_queries):
        categories = {q.category for q in dataset_queries}
        expected_categories = {"lookup", "multi_hop", "ambiguous", "unanswerable"}
        assert categories == expected_categories

    def test_lookup_category_count(self, dataset_queries):
        lookups = [q for q in dataset_queries if q.category == "lookup"]
        assert len(lookups) >= 15

    def test_multi_hop_category_count_and_multi_sources(self, dataset_queries):
        multi_hops = [q for q in dataset_queries if q.category == "multi_hop"]
        assert len(multi_hops) >= 10

        # Validate that EVERY multi_hop query has at least 2 relevant source files
        for q in multi_hops:
            assert q.relevant_sources is not None, f"Query '{q.id}' missing relevant_sources"
            assert len(q.relevant_sources) >= 2, (
                f"Multi-hop query '{q.id}' must have >= 2 sources, found: {q.relevant_sources}"
            )

    def test_ambiguous_category_properties(self, dataset_queries):
        ambiguous = [q for q in dataset_queries if q.category == "ambiguous"]
        assert len(ambiguous) >= 5

        for q in ambiguous:
            assert q.retrieval_sufficient is True
            assert q.answer_quality_notes is not None and len(q.answer_quality_notes) > 10, (
                f"Ambiguous query '{q.id}' must have detailed answer_quality_notes listing valid interpretations"
            )

    def test_unanswerable_category_properties(self, dataset_queries):
        unanswerable = [q for q in dataset_queries if q.category == "unanswerable"]
        assert len(unanswerable) >= 10

        for q in unanswerable:
            assert q.retrieval_sufficient is False
            assert q.relevant_sources is None
            assert q.expected_answer is None

    def test_answerable_queries_have_expected_answers(self, dataset_queries):
        answerable = [q for q in dataset_queries if q.retrieval_sufficient]
        for q in answerable:
            assert q.expected_answer is not None and len(q.expected_answer) > 5, (
                f"Answerable query '{q.id}' missing expected_answer"
            )
            assert q.relevant_sources is not None and len(q.relevant_sources) >= 1

    def test_phase4_ready_flag_set(self, dataset_queries):
        for q in dataset_queries:
            assert q.phase4_ready is True
