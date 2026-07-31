from typing import Dict, List, Optional
from app.models.chunk import Chunk
from app.models.search import SearchResult
from app.retrieval.base import BaseRetriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.reranker import Reranker


class HybridRetriever(BaseRetriever):
    """Retrieves document chunks using Hybrid Search (Dense + Sparse) fused via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: DenseRetriever = None,
        sparse_retriever: SparseRetriever = None,
        reranker: Reranker = None,
    ):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or SparseRetriever()
        self.reranker = reranker or Reranker()

    def retrieve(
        self,
        query: str,
        strategy: str,
        top_k: int = 5,
        rrf_k: int = 60,
        rerank: bool = True,
    ) -> List[SearchResult]:
        """Retrieves top-K search results by merging Dense and Sparse rankings using RRF.

        Args:
            query: The search query string.
            strategy: The chunking strategy ('fixed_size', 'structure_aware', or 'semantic').
            top_k: Number of search results to return.
            rrf_k: The RRF constant parameter (default 60) that controls rank smoothing.
            rerank: Whether to apply the Cross-Encoder reranker.

        Returns:
            List of SearchResult objects sorted by score.
        """
        if not query:
            return []

        # Retrieve a deeper candidate pool from both retrievers to ensure robust fusion
        depth = 20 if rerank else max(top_k * 3, 30)

        # 1. Run Dense and Sparse search
        dense_results = self.dense_retriever.retrieve(query, strategy, top_k=depth)
        sparse_results = self.sparse_retriever.retrieve(query, strategy, top_k=depth)

        # 2. Apply Reciprocal Rank Fusion (RRF)
        # RRF_Score(d) = sum( 1 / (rrf_k + rank(d)) )
        rrf_scores: Dict[str, float] = {}
        chunk_registry: Dict[str, Chunk] = {}

        # Merge Dense rankings (ranks are 1-indexed)
        for rank, res in enumerate(dense_results):
            chunk_id = res.chunk.id
            chunk_registry[chunk_id] = res.chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))

        # Merge Sparse rankings (ranks are 1-indexed)
        for rank, res in enumerate(sparse_results):
            chunk_id = res.chunk.id
            chunk_registry[chunk_id] = res.chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))

        # 3. Create SearchResult objects and sort by RRF score descending
        hybrid_results: List[SearchResult] = []
        for chunk_id, score in rrf_scores.items():
            hybrid_results.append(
                SearchResult(
                    chunk=chunk_registry[chunk_id],
                    score=float(score),
                    rrf_score=float(score)
                )
            )

        hybrid_results.sort(key=lambda x: x.rrf_score, reverse=True)

        # 4. Rerank top candidates if enabled
        if rerank and hybrid_results:
            return self.reranker.rerank(query, hybrid_results[:depth], top_k=top_k)

        return hybrid_results[:top_k]
