from abc import ABC, abstractmethod
from typing import List
from app.models.chunk import Chunk
from app.models.document import RawDocument


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def chunk(self, doc: RawDocument) -> List[Chunk]:
        """Splits a RawDocument into a list of parsed Chunks.

        Args:
            doc: The raw document to split.

        Returns:
            A list of Chunk objects.
        """
        pass
