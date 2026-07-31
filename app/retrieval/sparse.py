import os
import pickle
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from rank_bm25 import BM25Okapi
from app.core.config import settings
from app.models.chunk import Chunk
from app.models.search import SearchResult
from app.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


def clean_tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25 (lowercase, strips punctuation)."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.split()


class SparseRetriever(BaseRetriever):
    """Retrieves document chunks using BM25 keyword search."""

    def __init__(self):
        self.index_dir = settings.absolute_data_source_path.parent / "bm25"
        self._cached_indexes: Dict[str, Tuple[BM25Okapi, List[Chunk]]] = {}

    def _get_index(self, strategy: str) -> Tuple[BM25Okapi, List[Chunk]]:
        """Loads and caches the BM25 index and chunk list for the given strategy."""
        if strategy in self._cached_indexes:
            return self._cached_indexes[strategy]

        index_path = self.index_dir / f"bm25_{strategy}.pkl"
        if not index_path.exists():
            logger.info(f"BM25 index for strategy '{strategy}' missing on disk. Auto-rebuilding from ChromaDB...")
            try:
                from app.ingestion.indexer import ChromaIndexer, BM25Indexer
                chroma_indexer = ChromaIndexer(strategy=strategy)
                data = chroma_indexer.collection.get()
                chunks: List[Chunk] = []
                if data and data.get("ids"):
                    for cid, doc_text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
                        chunks.append(
                            Chunk(
                                id=cid,
                                text=doc_text,
                                source_file=meta.get("source_file", ""),
                                section_heading=meta.get("section_heading", ""),
                                chunking_strategy=meta.get("chunking_strategy", ""),
                                metadata={
                                    k: v for k, v in meta.items()
                                    if k not in ["source_file", "section_heading", "chunking_strategy"]
                                },
                            )
                        )
                if not chunks:
                    raise ValueError(f"ChromaDB collection retryv_{strategy} is empty.")
                bm25_indexer = BM25Indexer(strategy=strategy)
                bm25_indexer.build_index(chunks)
                logger.info(f"BM25 index successfully rebuilt for '{strategy}' on-the-fly.")
            except Exception as e:
                logger.error(f"Failed to auto-rebuild BM25 index for strategy '{strategy}': {e}", exc_info=True)
                raise FileNotFoundError(f"BM25 index file not found and auto-rebuild failed: {e}")

        with open(index_path, "rb") as f:
            data = pickle.load(f)
            bm25 = data["bm25"]
            chunks = data["chunks"]
            self._cached_indexes[strategy] = (bm25, chunks)
            return bm25, chunks

    def retrieve(self, query: str, strategy: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieves top-K search results using BM25 keyword matching.

        Args:
            query: The search query string.
            strategy: The chunking strategy ('fixed_size', 'structure_aware', or 'semantic').
            top_k: Number of search results to return.

        Returns:
            List of SearchResult objects sorted by descending BM25 score.
        """
        if not query:
            return []

        try:
            bm25, chunks = self._get_index(strategy)
        except FileNotFoundError as e:
            logger.warning(
                f"SparseRetriever: BM25 index file not found for strategy '{strategy}': {e}. "
                "Returning empty results. Run IngestionPipeline to build BM25 indexes."
            )
            return []
        except Exception as e:
            logger.error(
                f"SparseRetriever: Unexpected error loading BM25 index for strategy '{strategy}': {e}"
            )
            return []

        # 1. Tokenize query
        tokenized_query = clean_tokenize(query)
        if not tokenized_query:
            return []

        # 2. Get BM25 scores for all chunks
        scores = bm25.get_scores(tokenized_query)

        # 3. Pair chunks with scores, filter out non-positive scores, and sort
        search_results: List[SearchResult] = []
        for chunk, score in zip(chunks, scores):
            if score > 0.0:
                search_results.append(SearchResult(chunk=chunk, score=float(score)))

        # Sort descending by BM25 score
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:top_k]
