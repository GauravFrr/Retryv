from typing import Optional
from pydantic import BaseModel, Field
from app.models.chunk import Chunk


class SearchResult(BaseModel):
    """Represents a single search result containing the retrieved chunk and its score."""

    chunk: Chunk = Field(description="The retrieved document chunk")
    score: float = Field(description="The match score (similarity, BM25, or rerank score)")
    rrf_score: Optional[float] = Field(default=None, description="The Reciprocal Rank Fusion score")
    rerank_score: Optional[float] = Field(default=None, description="The Cross-Encoder rerank score")
