import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("", summary="Trigger document ingestion pipeline")
def trigger_ingestion(background_tasks: BackgroundTasks = None):
    """Triggers the document loader, chunkers, embedding, and indexing pipeline.

    Runs synchronously to return the ingestion summary results directly to the client.
    """
    try:
        pipeline = IngestionPipeline()
        summary = pipeline.run()
        return {
            "status": "success",
            "message": "Ingestion pipeline completed successfully.",
            "summary": summary
        }
    except Exception as e:
        logger.exception("Ingestion pipeline failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion pipeline failed with error: {str(e)}"
        )
