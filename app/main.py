from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle events."""
    setup_logging()
    logger.info("Starting Retryv API...")
    yield
    logger.info("Shutting down Retryv API...")


app = FastAPI(
    title="Retryv API",
    description="Production-grade RAG System over FastAPI documentation with citation verification & eval suite",
    version="0.1.0",
    lifespan=lifespan,
)

from app.api.v1 import ingest, documents, search, ask, eval

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingestion"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
app.include_router(ask.router, prefix="/api/v1/ask", tags=["Q&A"])
app.include_router(eval.router, prefix="/api/v1/eval", tags=["Evaluation"])



@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify backend service status."""
    return {
        "status": "ok",
        "app": "Retryv",
        "version": "0.1.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
