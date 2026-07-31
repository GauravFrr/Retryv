from typing import Any, Dict
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a normalized document chunk stored in ChromaDB or BM25 index."""

    id: str = Field(description="Unique chunk ID (hash of source_file + text)")
    text: str = Field(description="Normalized text content of the chunk")
    source_file: str = Field(
        description="Relative path of the source file (e.g. docs/en/docs/tutorial/index.md)"
    )
    section_heading: str = Field(
        description="Section heading (e.g. h1, h2, h3) this chunk belongs to"
    )
    chunking_strategy: str = Field(
        description="Strategy used: 'fixed_size', 'structure_aware', or 'semantic'"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata dictionary containing url, parent title, etc.",
    )
