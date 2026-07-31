import os
import re
import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np

from app.core.config import settings
from app.core.embeddings import GeminiEmbedder
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)


def clean_tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25 indexing (lowercase, strips punctuation)."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.split()


class ChromaIndexer:
    """Handles embedding generation and indexing of document chunks into ChromaDB."""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.collection_name = f"retryv_{strategy}"
        self.embedder = GeminiEmbedder()
        
        # Ensure directories exist
        os.makedirs(settings.absolute_chroma_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(settings.absolute_chroma_path))
        # Use cosine distance space natively in Chroma
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Computes cosine similarity between two float vectors."""
        a = np.array(v1)
        b = np.array(v2)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def index_chunks(self, chunks: List[Chunk], sub_batch_size: int = 100) -> Tuple[int, int]:
        """Embeds, deduplicates, and indexes a list of Chunks in sub-batches.

        Args:
            chunks: List of Chunk objects to index.
            sub_batch_size: Number of chunks to embed and insert per iteration.

        Returns:
            A tuple of (inserted_count, skipped_count).
        """
        if not chunks:
            return 0, 0

        # Verify strategy matches
        for chunk in chunks:
            if chunk.chunking_strategy != self.strategy:
                raise ValueError(
                    f"Chunk strategy '{chunk.chunking_strategy}' does not match indexer strategy '{self.strategy}'"
                )

        total_chunks = len(chunks)
        logger.info(f"Processing {total_chunks} chunks in sub-batches of {sub_batch_size} for {self.strategy}...")

        total_inserted = 0
        total_skipped = 0

        for i in range(0, total_chunks, sub_batch_size):
            batch_chunks = chunks[i : i + sub_batch_size]
            batch_num = (i // sub_batch_size) + 1
            total_batches = (total_chunks + sub_batch_size - 1) // sub_batch_size

            # Check which chunks already exist in ChromaDB collection by ID to avoid embedding them
            try:
                existing_data = self.collection.get(ids=[c.id for c in batch_chunks])
                existing_ids = set(existing_data.get("ids", []))
            except Exception as e:
                logger.debug(f"Failed to check existing IDs in collection: {e}")
                existing_ids = set()

            chunks_to_embed = [c for c in batch_chunks if c.id not in existing_ids]
            skipped_existing = len(batch_chunks) - len(chunks_to_embed)
            total_skipped += skipped_existing

            if skipped_existing > 0:
                logger.info(f"Sub-batch {batch_num}/{total_batches}: Skipping {skipped_existing} chunks already in ChromaDB.")

            if not chunks_to_embed:
                logger.info(f"Sub-batch {batch_num}/{total_batches} already fully indexed.")
                continue

            logger.info(f"Embedding sub-batch {batch_num}/{total_batches} ({len(chunks_to_embed)} chunks)...")
            batch_texts = [c.text for c in chunks_to_embed]
            batch_embeddings = self.embedder.embed_batch(batch_texts)

            accepted_chunks: List[Chunk] = []
            accepted_embeddings: List[List[float]] = []

            for chunk, emb in zip(chunks_to_embed, batch_embeddings):
                is_duplicate = False

                # A. Check against already accepted chunks in this sub-batch
                for prev_emb in accepted_embeddings:
                    sim = self._cosine_similarity(emb, prev_emb)
                    if sim > 0.95:
                        is_duplicate = True
                        break

                if is_duplicate:
                    total_skipped += 1
                    continue

                # B. Check against existing chunks in the ChromaDB collection
                try:
                    results = self.collection.query(
                        query_embeddings=[emb],
                        n_results=1
                    )
                    if results and results.get("distances") and results["distances"][0]:
                        distance = results["distances"][0][0]
                        if distance < 0.05:
                            is_duplicate = True
                except Exception as e:
                    logger.debug(f"Collection query check skipped or empty: {e}")

                if is_duplicate:
                    total_skipped += 1
                    continue

                accepted_chunks.append(chunk)
                accepted_embeddings.append(emb)

            # Insert sub-batch into ChromaDB
            if accepted_chunks:
                ids = [c.id for c in accepted_chunks]
                documents = [c.text for c in accepted_chunks]
                metadatas = []
                for c in accepted_chunks:
                    meta = {
                        "source_file": c.source_file,
                        "section_heading": c.section_heading,
                        "chunking_strategy": c.chunking_strategy,
                    }
                    for k, v in c.metadata.items():
                        if isinstance(v, (str, int, float, bool)):
                            meta[k] = v
                    metadatas.append(meta)

                self.collection.add(
                    ids=ids,
                    embeddings=accepted_embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                total_inserted += len(accepted_chunks)

            logger.info(
                f"Sub-batch {batch_num}/{total_batches} complete: "
                f"{len(accepted_chunks)} inserted, cumulative collection total = {self.collection.count()}"
            )

        return total_inserted, total_skipped


class BM25Indexer:
    """Handles tokenizing and building a BM25 keyword search index for document chunks."""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.index_dir = settings.absolute_data_source_path.parent / "bm25"
        self.index_path = self.index_dir / f"bm25_{strategy}.pkl"
        os.makedirs(self.index_dir, exist_ok=True)

    def build_index(self, all_chunks: List[Chunk]):
        """Rebuilds the BM25 index from scratch with the entire corpus for this strategy."""
        # Filter chunks that belong to this strategy
        strategy_chunks = [c for c in all_chunks if c.chunking_strategy == self.strategy]
        if not strategy_chunks:
            logger.warning(f"No chunks to index for BM25 strategy: {self.strategy}")
            return

        logger.info(f"Building BM25 index for {len(strategy_chunks)} chunks using {self.strategy}...")

        # Tokenize corpus
        corpus = [clean_tokenize(c.text) for c in strategy_chunks]
        
        # Build BM25 Okapi model
        bm25 = BM25Okapi(corpus)

        # Save model and chunk registry to disk using pickle
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "bm25": bm25,
                "chunks": strategy_chunks
            }, f)
            
        logger.info(f"BM25 index successfully saved to {self.index_path}")

    def load_index(self) -> Tuple[BM25Okapi, List[Chunk]]:
        """Loads BM25 index and corresponding chunks from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 index file not found at {self.index_path}")

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            return data["bm25"], data["chunks"]
