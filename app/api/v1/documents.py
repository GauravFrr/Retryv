import logging
from fastapi import APIRouter, HTTPException
import chromadb
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("", summary="Get document chunking and indexing statistics")
def get_documents_stats():
    """Returns statistics for each of the three chunking strategy indexes in ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=str(settings.absolute_chroma_path))
        stats = {}
        
        # Check counts in ChromaDB
        strategies = ["fixed_size", "structure_aware", "semantic"]
        for strategy in strategies:
            collection_name = f"retryv_{strategy}"
            try:
                collection = client.get_collection(name=collection_name)
                count = collection.count()
                
                # Fetch a small sample of source files in the index
                sample_data = collection.get(limit=10)
                source_files = set()
                if sample_data and sample_data.get("metadatas"):
                    for meta in sample_data["metadatas"]:
                        if meta and "source_file" in meta:
                            source_files.add(meta["source_file"])
                
                stats[strategy] = {
                    "collection_name": collection_name,
                    "chunk_count": count,
                    "sample_files": list(source_files),
                }
            except Exception as e:
                # Collection might not exist yet if ingestion hasn't run
                stats[strategy] = {
                    "collection_name": collection_name,
                    "chunk_count": 0,
                    "sample_files": [],
                    "status": "not_initialized",
                }

        # Check BM25 file status
        bm25_stats = {}
        bm25_dir = settings.absolute_data_source_path.parent / "bm25"
        for strategy in strategies:
            pkl_path = bm25_dir / f"bm25_{strategy}.pkl"
            
            # Self-healing: if file is missing but ChromaDB is not empty, trigger auto-rebuild
            exists = pkl_path.exists()
            if not exists and stats.get(strategy, {}).get("chunk_count", 0) > 0:
                logger.info(f"Documents API: BM25 index for '{strategy}' missing. Triggering auto-rebuild...")
                try:
                    from app.retrieval.sparse import SparseRetriever
                    # Loading the index triggers the auto-rebuild logic we built in SparseRetriever
                    retriever = SparseRetriever()
                    retriever._get_index(strategy)
                    exists = pkl_path.exists()
                except Exception as e:
                    logger.error(f"Documents API: Failed to trigger auto-rebuild for '{strategy}': {e}")
            
            bm25_stats[strategy] = {
                "exists": exists,
                "path": str(pkl_path.relative_to(settings.BASE_DIR)) if exists else None
            }

        return {
            "status": "success",
            "chromadb_indexes": stats,
            "bm25_indexes": bm25_stats
        }
    except Exception as e:
        logger.exception("Failed to fetch documents stats.")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch documents stats: {str(e)}"
        )
