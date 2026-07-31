"""
Evaluation Runner Service for Retryv (Phase 4 Unit 4.3).

Executes queries from data/eval/golden_dataset.json through the complete RAG
pipeline (Retrieval -> Confidence Guard -> Generation -> Citation Verification -> Metrics),
and computes overall aggregate metrics and per-category performance breakdowns.
Persists report JSON files to data/eval/reports/.
"""
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.core.config import settings
from app.eval.dataset import GoldenQuery, load_golden_dataset
from app.eval.metrics import (
    AnswerCorrectnessMetric,
    CitationAccuracyMetric,
    FaithfulnessMetric,
    RetrievalMetrics,
)
from app.eval.models import CategoryMetrics, EvaluationReport, EvaluationResult, MetricScore
from app.generation.citation_verifier import CitationVerifier
from app.generation.generator import Generator
from app.models.search import SearchResult
from app.retrieval.fusion import HybridRetriever

logger = logging.getLogger(__name__)

# Default directory for report persistence
DEFAULT_REPORTS_DIR = settings.BASE_DIR / "data" / "eval" / "reports"


class EvalRunner:
    """Orchestrates end-to-end evaluation runs over the Golden Dataset."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[Generator] = None,
        verifier: Optional[CitationVerifier] = None,
        inter_query_sleep: float = 2.0,
    ):
        """
        Args:
            retriever: Optional HybridRetriever instance (created lazily if None).
            generator: Optional Generator instance (created lazily if None).
            verifier: Optional CitationVerifier instance (created lazily if None).
            inter_query_sleep: Courtesy sleep seconds between sequential query runs
                               to stay under free-tier API rate limits (default: 2.0s).
        """
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or Generator()
        self.verifier = verifier or CitationVerifier()
        self.inter_query_sleep = inter_query_sleep

        # Metric calculators
        self.citation_accuracy_metric = CitationAccuracyMetric(verifier=self.verifier)
        self.faithfulness_metric = FaithfulnessMetric(verifier=self.verifier)
        self.correctness_metric = AnswerCorrectnessMetric()

    def run_evaluation(
        self,
        strategy: str = "fixed_size",
        limit: Optional[int] = None,
        save_report: bool = True,
        reports_dir: Optional[Path] = None,
    ) -> EvaluationReport:
        """Runs the full evaluation benchmark over the Golden Dataset.

        Args:
            strategy: Chunking strategy to evaluate ('fixed_size', 'structure_aware', 'semantic').
            limit: Optional maximum number of queries to evaluate (useful for smoke tests).
            save_report: Whether to persist the report JSON to data/eval/reports/.
            reports_dir: Optional custom output directory for report files.

        Returns:
            Structured EvaluationReport containing aggregate & category breakdown metrics.
        """
        queries = load_golden_dataset(ready_only=True)
        if limit is not None and limit > 0:
            queries = queries[:limit]

        total = len(queries)
        logger.info(
            "Starting evaluation run — strategy=%s, total_queries=%d, inter_query_sleep=%.1fs",
            strategy,
            total,
            self.inter_query_sleep,
        )

        results: List[EvaluationResult] = []

        for i, query in enumerate(queries):
            logger.info(
                "Evaluating query %d/%d [%s | category=%s]: %r",
                i + 1,
                total,
                query.id,
                query.category,
                query.query[:50],
            )

            eval_result = self._evaluate_single_query(query, strategy)
            results.append(eval_result)

            # Courtesy sleep between API calls to protect rate limits
            if i < total - 1 and self.inter_query_sleep > 0:
                time.sleep(self.inter_query_sleep)

        # Compute aggregate & category breakdown metrics
        aggregate_metrics = self._compute_aggregate_metrics(results)
        category_breakdown = self._compute_category_breakdown(results)

        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        report_id = f"eval_{strategy}_{timestamp_str}"

        report = EvaluationReport(
            id=report_id,
            strategy=strategy,
            timestamp=now.isoformat(),
            total_queries=total,
            aggregate_metrics=aggregate_metrics,
            category_breakdown=category_breakdown,
            results=results,
        )

        if save_report:
            out_dir = reports_dir or DEFAULT_REPORTS_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{report_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))
            logger.info("Evaluation report successfully saved to %s", out_path)

        return report

    def _evaluate_single_query(self, query: GoldenQuery, strategy: str) -> EvaluationResult:
        """Evaluates a single GoldenQuery through the pipeline and all metric calculators."""
        scores: Dict[str, MetricScore] = {}

        try:
            # 1. Retrieval Pass
            search_results = self.retriever.retrieve(query.query, strategy=strategy, top_k=5)
            retrieved_sources = [r.chunk.source_file for r in search_results]

            # Compute Retrieval Recall & Precision
            scores["retrieval_recall"] = RetrievalMetrics.compute_recall(
                retrieved_sources, query.relevant_sources
            )
            scores["retrieval_precision"] = RetrievalMetrics.compute_precision(
                retrieved_sources, query.relevant_sources
            )

            # 2. Generation Pass (includes Confidence Guard check)
            gen_result = self.generator.generate(query.query, search_results)

            # 3. Citation Accuracy Metric
            scores["citation_accuracy"] = self.citation_accuracy_metric.compute(gen_result)

            # 4. Faithfulness Metric (Sc * Su)
            scores["faithfulness"] = self.faithfulness_metric.compute(gen_result, search_results)

            # 5. Answer Correctness Metric (category-aware)
            scores["correctness"] = self.correctness_metric.compute(query, gen_result)

        except Exception as e:
            logger.error("Error evaluating query [%s]: %s", query.id, e, exc_info=True)
            # Fallback scores on error
            err_msg = f"Evaluation failed: {e}"
            for m in ["retrieval_recall", "retrieval_precision", "citation_accuracy", "faithfulness", "correctness"]:
                if m not in scores:
                    scores[m] = MetricScore(metric_name=m, score=0.0, reasoning=err_msg)

        return EvaluationResult(
            query_id=query.id,
            category=query.category,
            scores=scores,
        )

    def _compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculates mean score per metric across all evaluated query results, excluding N/A (None) scores."""
        if not results:
            return {}

        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        for res in results:
            for m_name, m_score in res.scores.items():
                if m_score.score is not None:
                    metric_sums[m_name] = metric_sums.get(m_name, 0.0) + m_score.score
                    metric_counts[m_name] = metric_counts.get(m_name, 0) + 1

        all_metric_names = [
            "retrieval_recall",
            "retrieval_precision",
            "citation_accuracy",
            "faithfulness",
            "correctness",
        ]
        return {
            m_name: round(metric_sums[m_name] / metric_counts[m_name], 4)
            if metric_counts.get(m_name, 0) > 0 else 0.0
            for m_name in all_metric_names
        }

    def _compute_category_breakdown(
        self, results: List[EvaluationResult]
    ) -> Dict[str, CategoryMetrics]:
        """Calculates mean scores per metric grouped by query category."""
        category_groups: Dict[str, List[EvaluationResult]] = {}
        for res in results:
            category_groups.setdefault(res.category, []).append(res)

        breakdown: Dict[str, CategoryMetrics] = {}
        for cat, cat_results in category_groups.items():
            mean_scores = self._compute_aggregate_metrics(cat_results)
            breakdown[cat] = CategoryMetrics(
                category=cat,
                count=len(cat_results),
                mean_scores=mean_scores,
            )

        return breakdown
