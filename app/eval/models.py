"""
Pydantic Data Models for Evaluation Metrics (Phase 4 Unit 4.2).

Defines structured containers for individual metric scores and consolidated
per-query evaluation results.
"""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class MetricScore(BaseModel):
    """Container for a single calculated evaluation metric score."""

    metric_name: str = Field(
        description="Metric name: 'retrieval_recall', 'retrieval_precision', 'citation_accuracy', 'faithfulness', or 'correctness'"
    )
    score: Optional[float] = Field(
        default=None,
        description="Normalized score bounded between 0.0 and 1.0, or None if N/A for this query"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief natural-language explanation or judge feedback for the score",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional detailed metadata (e.g. support_ratio, Sc, Su, uncited_sentences_count)",
    )


class EvaluationResult(BaseModel):
    """Consolidated evaluation result for a single query across all metrics."""

    query_id: str = Field(description="Unique query ID from the GoldenDataset")
    category: str = Field(description="Query category: 'lookup', 'multi_hop', 'ambiguous', or 'unanswerable'")
    scores: Dict[str, MetricScore] = Field(
        default_factory=dict,
        description="Dictionary mapping metric_name -> MetricScore",
    )


class CategoryMetrics(BaseModel):
    """Aggregated mean metric scores for a specific query category."""

    category: str = Field(description="Category name: 'lookup', 'multi_hop', 'ambiguous', or 'unanswerable'")
    count: int = Field(description="Number of queries evaluated in this category")
    mean_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Mean score per metric for this category",
    )


class EvaluationReport(BaseModel):
    """Full benchmark evaluation report for a given chunking strategy."""

    id: str = Field(description="Unique report ID (e.g. eval_fixed_size_20260722_153000)")
    strategy: str = Field(description="Chunking strategy evaluated ('fixed_size', 'structure_aware', 'semantic')")
    timestamp: str = Field(description="ISO 8601 creation timestamp")
    total_queries: int = Field(description="Total queries evaluated in this run")
    aggregate_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Overall mean score per metric across all evaluated queries",
    )
    category_breakdown: Dict[str, CategoryMetrics] = Field(
        default_factory=dict,
        description="Category-specific mean metric breakdown",
    )
    results: list[EvaluationResult] = Field(
        default_factory=list,
        description="Detailed per-query evaluation results",
    )
