import logging
from typing import List
from sentence_transformers import CrossEncoder
from app.models.search import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker service that utilizes a local Cross-Encoder model to score document chunks against a query."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        """Lazily initialize the CrossEncoder model."""
        if self._model is None:
            logger.info("Loading CrossEncoder model: %s...", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("CrossEncoder model loaded successfully.")
        return self._model

    def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        """Reranks search results using the CrossEncoder model.

        Args:
            query: The user query.
            candidates: List of SearchResult candidates to rerank.
            top_k: Number of top reranked candidates to return.

        Returns:
            List of SearchResult objects with populated rerank_score, sorted by rerank_score descending.
        """
        if not candidates:
            return []

        # Prepare inputs for CrossEncoder: list of (query, chunk_text)
        pairs = [(query, res.chunk.text) for res in candidates]
        
        # Compute scores
        scores = self.model.predict(pairs)

        # Create new SearchResult objects with updated score fields
        reranked_results = []
        for res, score in zip(candidates, scores):
            rrf_val = res.rrf_score if res.rrf_score is not None else res.score
            reranked_results.append(
                SearchResult(
                    chunk=res.chunk,
                    score=float(score),
                    rrf_score=rrf_val,
                    rerank_score=float(score)
                )
            )

        # Sort descending by rerank_score
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
        return reranked_results[:top_k]
