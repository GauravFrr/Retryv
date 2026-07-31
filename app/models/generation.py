from typing import List

from pydantic import BaseModel, Field


class CitedChunk(BaseModel):
    """A single chunk that was actually cited in the generated answer."""

    index: int = Field(description="1-based citation index matching [N] in the answer text")
    source_file: str = Field(
        description="Relative path of the source file (e.g. docs/en/docs/advanced/events.md)"
    )
    section_heading: str = Field(description="Section heading the chunk belongs to")
    url: str = Field(default="", description="Canonical URL for the chunk; empty if not available")
    text: str = Field(description="Raw chunk text that was cited")


class GenerationResult(BaseModel):
    """Structured output from a single grounded generation call."""

    answer: str = Field(description="Full answer text with inline [N] citation markers")
    cited_chunks: List[CitedChunk] = Field(
        default_factory=list,
        description="Chunks whose index actually appears in the answer text (de-duplicated)",
    )
    is_grounded: bool = Field(
        description=(
            "False if the model returned the insufficient-context sentinel phrase, "
            "True for all other answers"
        )
    )
    model: str = Field(description="Gemini model name used for generation")


class VerificationResult(BaseModel):
    """LLM-as-judge verdict for a single cited chunk."""

    chunk_index: int = Field(
        description="1-based citation index — matches CitedChunk.index"
    )
    source_file: str = Field(
        description="Source file path from the cited chunk (for traceability)"
    )
    verdict: str = Field(
        description="'SUPPORTED' or 'NOT_SUPPORTED' as returned by the judge model"
    )
    supported: bool = Field(
        description="True iff verdict == 'SUPPORTED'"
    )
    parse_error: bool = Field(
        default=False,
        description="True if the judge returned unexpected text that could not be parsed",
    )
    raw_response: str = Field(
        description="Raw text returned by the judge model (for debugging)"
    )


class VerifiedGenerationResult(BaseModel):
    """GenerationResult enriched with per-citation verification verdicts.

    Wraps GenerationResult rather than extending it so generation and verification
    remain independently testable and the /ask endpoint can skip verification
    when not needed.
    """

    generation: GenerationResult = Field(
        description="The original generation result containing the answer and cited chunks"
    )
    verifications: List[VerificationResult] = Field(
        default_factory=list,
        description="One VerificationResult per CitedChunk in generation.cited_chunks",
    )
    all_supported: bool = Field(
        description=(
            "True iff every cited chunk received a SUPPORTED verdict. "
            "True by convention when cited_chunks is empty (nothing to falsify)."
        )
    )
    support_ratio: float = Field(
        description=(
            "Fraction of cited chunks that are SUPPORTED. "
            "0.0 when cited_chunks is empty."
        )
    )
