from abc import ABC, abstractmethod
from typing import List
from app.models.search import SearchResult


class BaseRetriever(ABC):
    """Abstract base class for all retrieval methods (dense, sparse, hybrid)."""

    @abstractmethod
    def retrieve(self, query: str, strategy: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieves top-K search results for the given query under a chunking strategy.

        Args:
            query: The search query string.
            strategy: The chunking strategy ('fixed_size', 'structure_aware', or 'semantic').
            top_k: Number of search results to return.

        Returns:
            List of SearchResult objects sorted in descending order of match score.
        """
        pass
