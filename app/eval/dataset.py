"""
Golden Dataset Loader & Pydantic Models for Retryv Evaluation Pipeline.

Provides typed data models and loader utility functions for data/eval/golden_dataset.json,
used across Phase 3 (confidence guard) and Phase 4 (retrieval recall, faithfulness,
correctness, and evaluation runners).
"""
import json
import logging
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings

logger = logging.getLogger(__name__)

# Default path to the golden dataset file
DEFAULT_GOLDEN_DATASET_PATH = settings.BASE_DIR / "data" / "eval" / "golden_dataset.json"

CategoryType = Literal["lookup", "multi_hop", "ambiguous", "unanswerable"]


class GoldenQuery(BaseModel):
    """A single evaluation query item in the golden dataset."""

    id: str = Field(description="Unique slug identifier for the query")
    query: str = Field(description="Natural-language user query text")
    category: CategoryType = Field(
        description="Query classification: 'lookup', 'multi_hop', 'ambiguous', or 'unanswerable'"
    )
    retrieval_sufficient: bool = Field(
        description="True if the FastAPI docs corpus can answer this query; False if out-of-domain"
    )
    relevant_sources: Optional[List[str]] = Field(
        default=None,
        description="List of relative file paths in FastAPI docs containing relevant context",
    )
    expected_answer: Optional[str] = Field(
        default=None,
        description="Ground-truth reference answer for correctness scoring (null for unanswerable)",
    )
    answer_quality_notes: Optional[str] = Field(
        default=None,
        description="Criteria / acceptable interpretations for scoring",
    )
    phase4_ready: bool = Field(
        default=True,
        description="True when fully validated and ready for Phase 4 evaluation runs",
    )

    @field_validator("relevant_sources")
    @classmethod
    def validate_multi_hop_sources(cls, v: Optional[List[str]], info) -> Optional[List[str]]:
        category = info.data.get("category")
        if category == "multi_hop" and (not v or len(v) < 2):
            raise ValueError(
                f"Multi-hop query '{info.data.get('id')}' must have at least 2 relevant_sources, got {v}"
            )
        return v


class GoldenDataset(BaseModel):
    """Container model for the complete golden evaluation dataset."""

    schema_version: str = Field(alias="_schema_version")
    description: str = Field(alias="_description")
    queries: List[GoldenQuery]


def load_golden_dataset(
    path: Optional[Path] = None,
    ready_only: bool = True,
) -> List[GoldenQuery]:
    """Load and validate queries from the golden dataset JSON file.

    Args:
        path: Optional Path to golden_dataset.json. Defaults to data/eval/golden_dataset.json.
        ready_only: If True, filters queries to only those with phase4_ready=True.

    Returns:
        List of validated GoldenQuery Pydantic objects.
    """
    target_path = path or DEFAULT_GOLDEN_DATASET_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Golden dataset file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = GoldenDataset.model_validate(data)
    queries = dataset.queries

    if ready_only:
        queries = [q for q in queries if q.phase4_ready]

    logger.info(
        "Loaded %d queries (ready_only=%s) from %s",
        len(queries),
        ready_only,
        target_path.name,
    )
    return queries
