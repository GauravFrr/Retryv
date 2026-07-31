from typing import List
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from app.models.search import SearchResult
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.fusion import HybridRetriever

router = APIRouter()

# Initialize retrievers (cached globally at the router level)
dense_retriever = DenseRetriever()
sparse_retriever = SparseRetriever()
hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query string")
    strategy: str = Field(
        default="fixed_size",
        description="Chunking strategy: 'fixed_size', 'structure_aware', or 'semantic'"
    )
    method: str = Field(
        default="hybrid",
        description="Retrieval method: 'dense', 'sparse', or 'hybrid'"
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    rrf_k: int = Field(default=60, ge=1, le=200, description="RRF constant parameter")


@router.post("", response_model=List[SearchResult], status_code=status.HTTP_200_OK)
def search(request: SearchRequest):
    """Retrieves document chunks matching the query using dense, sparse, or hybrid search."""
    # 1. Validate strategy parameter
    valid_strategies = ("fixed_size", "structure_aware", "semantic")
    if request.strategy not in valid_strategies:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid strategy '{request.strategy}'. Must be one of {valid_strategies}"
        )

    # 2. Route request to appropriate retriever
    try:
        if request.method == "dense":
            return dense_retriever.retrieve(
                query=request.query,
                strategy=request.strategy,
                top_k=request.top_k
            )
        elif request.method == "sparse":
            return sparse_retriever.retrieve(
                query=request.query,
                strategy=request.strategy,
                top_k=request.top_k
            )
        elif request.method == "hybrid":
            return hybrid_retriever.retrieve(
                query=request.query,
                strategy=request.strategy,
                top_k=request.top_k,
                rrf_k=request.rrf_k
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid retrieval method '{request.method}'. Must be 'dense', 'sparse', or 'hybrid'."
            )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during retrieval: {str(e)}"
        )
