"""
Re-indexes all 3 chunking strategies over the complete raw FastAPI documentation.

Clears existing ChromaDB collections and BM25 index files, then runs
IngestionPipeline to ensure complete, 100% data coverage across fixed_size,
structure_aware, and semantic strategies.
"""
import sys
sys.path.insert(0, ".")

import logging
from app.core.config import settings
from app.ingestion.pipeline import IngestionPipeline
import chromadb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume indexing without clearing collections")
    args = parser.parse_args()

    if args.resume:
        logger.info("Resuming indexing. Keeping existing ChromaDB collections...")
    else:
        logger.info("Starting full re-indexing for all 3 chunking strategies...")
        # 1. Clear existing collections in ChromaDB to guarantee clean state
        chroma_client = chromadb.PersistentClient(path=str(settings.absolute_chroma_path))
        for strat in ["fixed_size", "structure_aware", "semantic"]:
            col_name = f"retryv_{strat}"
            try:
                chroma_client.delete_collection(name=col_name)
                logger.info(f"Deleted old ChromaDB collection: {col_name}")
            except Exception as e:
                logger.info(f"Collection {col_name} did not exist or could not be deleted: {e}")

    # 2. Run IngestionPipeline
    pipeline = IngestionPipeline()
    summary = pipeline.run()

    logger.info("=" * 70)
    logger.info("FULL RE-INDEXING SUMMARY")
    logger.info("=" * 70)
    for strat, data in summary.get("strategies", {}).items():
        logger.info(
            f"Strategy {strat:<16}: Chunks Generated={data['chunks_generated']}, "
            f"Inserted={data['chunks_inserted']}, Skipped Dupes={data['chunks_skipped_duplicate']}, "
            f"Total in DB={data['total_indexed_in_db']}"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
