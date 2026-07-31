from typing import Any, Dict
from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """Represents a raw loaded document before chunking is applied."""

    id: str = Field(description="Unique document ID (hash of source path)")
    text: str = Field(description="Raw markdown/text content of the file")
    source_file: str = Field(
        description="Relative path of the source file (e.g. docs/en/docs/tutorial/index.md)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata dictionary containing url, title, and other properties",
    )
