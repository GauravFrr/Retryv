"""
Rebuild BM25 indexes from existing ChromaDB data — no API calls required.

Reads all chunks from each ChromaDB collection and writes the BM25 pkl files
to data/bm25/.  Run this whenever bm25_*.pkl files are missing.
"""
import sys
sys.path.insert(0, ".")

import logging
from app.core.config import settings
from app.ingestion.indexer import ChromaIndexer, BM25Indexer
from app.models.chunk import Chunk
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_chunks_from_chroma(chroma_indexer: ChromaIndexer) -> List[Chunk]:
    data = chroma_indexer.collection.get()
    chunks: List[Chunk] = []
    if not data or not data.get("ids"):
        return chunks
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
    return chunks


def main():
    strategies = ["fixed_size", "structure_aware", "semantic"]
    for strategy in strategies:
        logger.info(f"Rebuilding BM25 for '{strategy}'...")
        try:
            chroma_indexer = ChromaIndexer(strategy=strategy)
            count = chroma_indexer.collection.count()
            logger.info(f"  ChromaDB has {count} chunks for '{strategy}'")

            if count == 0:
                logger.warning(f"  No chunks in ChromaDB for '{strategy}' — skipping.")
                continue

            chunks = get_chunks_from_chroma(chroma_indexer)
            bm25_indexer = BM25Indexer(strategy=strategy)
            bm25_indexer.build_index(chunks)
            logger.info(f"  BM25 index built with {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"  Failed for '{strategy}': {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
