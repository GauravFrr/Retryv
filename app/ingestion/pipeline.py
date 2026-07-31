import logging
from typing import Dict, Any, List
from app.ingestion.loader import DocLoader
from app.ingestion.chunkers.fixed_size import FixedSizeChunker
from app.ingestion.chunkers.structure_aware import StructureAwareChunker
from app.ingestion.chunkers.semantic import SemanticChunker
from app.ingestion.indexer import ChromaIndexer, BM25Indexer
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates loading, chunking, embedding, and indexing documents."""

    def __init__(self):
        self.loader = DocLoader()
        self.chunkers = {
            "fixed_size": FixedSizeChunker(),
            "structure_aware": StructureAwareChunker(),
            "semantic": SemanticChunker(),
        }

    def _get_all_chunks_from_chroma(self, chroma_indexer: ChromaIndexer) -> List[Chunk]:
        """Retrieves all chunks currently stored in the ChromaDB collection to align BM25."""
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
                        k: v
                        for k, v in meta.items()
                        if k not in ["source_file", "section_heading", "chunking_strategy"]
                    },
                )
            )
        return chunks

    def run(self) -> Dict[str, Any]:
        """Runs the full ingestion pipeline.

        Returns:
            A summary dictionary containing chunk counts and statistics.
        """
        logger.info("Starting document ingestion pipeline...")

        # 1. Load documents sparsely from git clone/docs folder
        raw_docs = self.loader.load()
        logger.info(f"Loaded {len(raw_docs)} raw documents.")

        summary: Dict[str, Any] = {
            "raw_documents_loaded": len(raw_docs),
            "strategies": {},
        }

        # 2. Process each chunking strategy
        for strategy, chunker in self.chunkers.items():
            logger.info(f"Processing strategy: {strategy}...")
            
            # A. Split docs into chunks
            strategy_chunks: List[Chunk] = []
            for doc in raw_docs:
                try:
                    chunks = chunker.chunk(doc)
                    strategy_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to chunk document {doc.source_file} using {strategy}: {e}")

            logger.info(f"Generated {len(strategy_chunks)} chunks for {strategy}.")

            # B. Index chunks into ChromaDB (with embedding & deduplication)
            try:
                chroma_indexer = ChromaIndexer(strategy=strategy)
                inserted, skipped = chroma_indexer.index_chunks(strategy_chunks)

                # C. Fetch all chunks currently in ChromaDB to build complete BM25 index
                all_db_chunks = self._get_all_chunks_from_chroma(chroma_indexer)

                # D. Build BM25 index
                bm25_indexer = BM25Indexer(strategy=strategy)
                bm25_indexer.build_index(all_db_chunks)

                # E. Save summary
                summary["strategies"][strategy] = {
                    "chunks_generated": len(strategy_chunks),
                    "chunks_inserted": inserted,
                    "chunks_skipped_duplicate": skipped,
                    "total_indexed_in_db": chroma_indexer.collection.count(),
                }
            except Exception as e:
                logger.error(f"Strategy {strategy} failed during indexing/BM25 build: {e}")
                summary["strategies"][strategy] = {
                    "chunks_generated": len(strategy_chunks),
                    "chunks_inserted": 0,
                    "chunks_skipped_duplicate": 0,
                    "total_indexed_in_db": 0,
                    "error": str(e),
                }

        logger.info("Ingestion pipeline run complete.")
        return summary
