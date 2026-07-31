from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.models.generation import VerifiedGenerationResult, GenerationResult
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.fusion import HybridRetriever
from app.generation.generator import Generator
from app.generation.citation_verifier import CitationVerifier

router = APIRouter()

# Initialize services globally
dense_retriever = DenseRetriever()
sparse_retriever = SparseRetriever()
hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
generator = Generator()
verifier = CitationVerifier()


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user query/question")
    strategy: str = Field(
        default="fixed_size",
        description="Chunking strategy: 'fixed_size', 'structure_aware', or 'semantic'"
    )
    method: str = Field(
        default="hybrid",
        description="Retrieval method: 'dense', 'sparse', or 'hybrid'"
    )
    rerank: bool = Field(default=True, description="Whether to apply cross-encoder reranking")
    verify_citations: bool = Field(default=True, description="Whether to verify cited chunks")


@router.post("", response_model=VerifiedGenerationResult, status_code=status.HTTP_200_OK)
def ask(request: AskRequest):
    """Retrieves context chunks, generates a grounded answer, and optionally verifies citations."""
    valid_strategies = ("fixed_size", "structure_aware", "semantic")
    if request.strategy not in valid_strategies:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid strategy '{request.strategy}'. Must be one of {valid_strategies}"
        )

    try:
        # 1. Retrieve
        if request.method == "dense":
            search_results = dense_retriever.retrieve(
                query=request.query,
                strategy=request.strategy,
                top_k=5
            )
        elif request.method == "sparse":
            search_results = sparse_retriever.retrieve(
                query=request.query,
                strategy=request.strategy,
                top_k=5
            )
        elif request.method == "hybrid":
            search_results = hybrid_retriever.retrieve(
                query=request.query,
                strategy=request.strategy,
                top_k=5,
                rerank=request.rerank
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid retrieval method '{request.method}'. Must be 'dense', 'sparse', or 'hybrid'."
            )

        # 2. Generate
        generation_result = generator.generate(request.query, search_results)

        # 3. Verify
        if request.verify_citations and generation_result.is_grounded:
            return verifier.verify(generation_result)
        else:
            return VerifiedGenerationResult(
                generation=generation_result,
                verifications=[],
                all_supported=True,
                support_ratio=1.0 if not generation_result.cited_chunks else 0.0
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during Q&A processing: {str(e)}"
        )
