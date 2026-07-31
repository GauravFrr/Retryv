import os
import logging
from typing import List
import chromadb
from app.core.config import settings
from app.core.embeddings import GeminiEmbedder
from app.models.chunk import Chunk
from app.models.search import SearchResult
from app.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """Retrieves document chunks using dense vector search (cosine similarity) in ChromaDB."""

    def __init__(self):
        self.embedder = GeminiEmbedder()
        self.client = chromadb.PersistentClient(path=str(settings.absolute_chroma_path))

    def retrieve(self, query: str, strategy: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieves top-K search results using cosine similarity on Gemini Embeddings.

        Args:
            query: The search query string.
            strategy: The chunking strategy ('fixed_size', 'structure_aware', or 'semantic').
            top_k: Number of search results to return.

        Returns:
            List of SearchResult objects sorted by descending cosine similarity.
        """
        if not query:
            return []

        collection_name = f"retryv_{strategy}"
        
        try:
            # Check if collection exists
            collection = self.client.get_collection(name=collection_name)
        except Exception as e:
            logger.warning(
                f"DenseRetriever: ChromaDB collection '{collection_name}' not found or empty: {e}. "
                "Returning empty results."
            )
            return []

        # 1. Embed query
        query_vector = self.embedder.embed_text(query)

        # 2. Query ChromaDB
        # Since collection uses "cosine" space, ChromaDB returns distance:
        # Cosine Distance = 1.0 - Cosine Similarity
        # Thus, Cosine Similarity = 1.0 - Cosine Distance
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        search_results: List[SearchResult] = []
        
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            # Reconstruct Chunk
            source_file = meta.get("source_file", "")
            section_heading = meta.get("section_heading", "")
            chunking_strategy = meta.get("chunking_strategy", strategy)
            
            # Reconstruct original metadata dict
            other_meta = {
                k: v for k, v in meta.items()
                if k not in ("source_file", "section_heading", "chunking_strategy")
            }

            chunk = Chunk(
                id=chunk_id,
                text=doc_text,
                source_file=source_file,
                section_heading=section_heading,
                chunking_strategy=chunking_strategy,
                metadata=other_meta
            )

            # Cosine similarity score
            score = 1.0 - float(dist)
            search_results.append(SearchResult(chunk=chunk, score=score))

        # Sort descending by score (cosine similarity)
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results
