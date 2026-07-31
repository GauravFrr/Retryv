"""
Gemini Generation Service for Retryv's RAG pipeline.

Wraps the Gemini generate_content API call with:
  - ConfidenceGuard pre-check (skips Gemini when retrieval scores are too low)
  - PromptBuilder integration (system instruction + numbered context block)
  - Exponential backoff retry (max 2 retries, matching code-standards.md §2)
  - Structured GenerationResult output with citation parsing
  - Explicit GenerationError on exhausted retries (caller can distinguish
    "generation failed" from "generation succeeded but ungrounded")
"""
import re
import time
import logging
from typing import List

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.core.generation_client import get_generation_client, rotate_generation_key

from app.core.config import settings
from app.generation.confidence_guard import ConfidenceGuard
from app.generation.prompt_builder import PromptBuilder
from app.models.generation import CitedChunk, GenerationResult
from app.models.search import SearchResult

logger = logging.getLogger(__name__)

# Exact phrase baked into the system instruction that signals insufficient context.
# Must stay in sync with prompt_builder._SYSTEM_INSTRUCTION.
_NO_CONTEXT_PHRASE = "I do not have enough context to answer this question."

# Retry configuration — matches code-standards.md §2 (max 2 retries, exponential backoff).
_MAX_ATTEMPTS = 3        # 1 initial attempt + 2 retries
_BACKOFF_FACTOR = 2.0   # sleep = backoff_factor ** attempt + 1.0  →  1 s, 3 s


class GenerationError(RuntimeError):
    """Raised when the Gemini generation call fails after all retry attempts.

    The caller (e.g. the /ask endpoint) should catch this to return an HTTP 502,
    keeping it distinct from a successful-but-ungrounded response (is_grounded=False).
    """


class Generator:
    """Calls Gemini to produce a grounded, cited answer from retrieved chunks.

    Usage::

        results = HybridRetriever().retrieve("lifespan events", top_k=5)
        result  = Generator().generate("How do lifespan events work?", results)
        print(result.answer)
        print(result.cited_chunks)
        print(result.is_grounded)
    """

    def __init__(self, model_name: str = None):
        self._model_name = model_name
        self._client: genai.Client | None = None
        self._prompt_builder = PromptBuilder()
        self._guard = ConfidenceGuard()

    @property
    def model_name(self) -> str:
        return self._model_name or settings.GEMINI_GEN_MODEL


    # ------------------------------------------------------------------
    # Lazy client initialisation (mirrors GeminiEmbedder pattern)
    # ------------------------------------------------------------------

    @property
    def client(self) -> genai.Client:
        """Return instance-level override if set (e.g. in tests), else rotating singleton."""
        if self._client is not None:
            return self._client
        return get_generation_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        search_results: List[SearchResult],
    ) -> GenerationResult:
        """Generate a grounded answer from retrieved chunks.

        Args:
            question: The user's natural-language question.
            search_results: Ranked list of retrieved chunks (most relevant first).

        Returns:
            A structured ``GenerationResult`` containing the answer, cited chunks,
            groundedness flag, and model name.

        Raises:
            GenerationError: If the Gemini API call fails after all retry attempts.
        """
        # --- Pre-generation confidence gate -----------------------------------
        # Check BEFORE building the prompt or touching the Gemini API.
        # If the best retrieved chunk scores below the threshold, retrieval already
        # signals "nothing relevant found" — skip the API call entirely.
        if not self._guard.is_retrieval_sufficient(search_results):
            logger.info(
                "Confidence guard blocked generation (top score=%.4f, threshold=%.4f).",
                search_results[0].score if search_results else 0.0,
                self._guard.threshold,
            )
            return GenerationResult(
                answer=_NO_CONTEXT_PHRASE,
                cited_chunks=[],
                is_grounded=False,
                model=self.model_name,
            )
        # ----------------------------------------------------------------------

        prompt = self._prompt_builder.build_prompt(question, search_results)
        system_instruction = prompt["system"]
        user_content = prompt["user"]

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,  # maximum determinism for grounded RAG
        )

        response_text = self._call_with_retry(user_content, config)

        cited_chunks = self._parse_citations(response_text, search_results)
        is_grounded = self._is_grounded(response_text)

        logger.info(
            "Generation complete — model=%s  cited=%d  grounded=%s",
            self.model_name,
            len(cited_chunks),
            is_grounded,
        )

        return GenerationResult(
            answer=response_text,
            cited_chunks=cited_chunks,
            is_grounded=is_grounded,
            model=self.model_name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        user_content: str,
        config: types.GenerateContentConfig,
    ) -> str:
        """Call generate_content with exponential-backoff retry on APIError.

        Returns:
            The raw response text string.

        Raises:
            GenerationError: After ``_MAX_ATTEMPTS`` failed attempts.
        """
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_content,
                    config=config,
                )
                return response.text

            except APIError as e:
                if attempt == _MAX_ATTEMPTS - 1:
                    logger.error(
                        "Gemini generation failed after %d attempts: %s",
                        _MAX_ATTEMPTS,
                        e,
                    )
                    raise GenerationError(
                        f"Gemini generation failed after {_MAX_ATTEMPTS} attempts: {e}"
                    ) from e

                is_429 = getattr(e, "code", None) == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if is_429:
                    rotate_generation_key()
                sleep_time = 2.0 if is_429 else (_BACKOFF_FACTOR ** attempt + 1.0)
                logger.warning(
                    "Gemini generation APIError on attempt %d/%d (is_429=%s) — retrying in %.1fs. Error: %s",
                    attempt + 1,
                    _MAX_ATTEMPTS,
                    is_429,
                    sleep_time,
                    e,
                )
                time.sleep(sleep_time)

            except Exception as e:
                # Non-APIError exceptions (e.g. network, serialisation) are not retried.
                logger.error("Unexpected error during Gemini generation: %s", e)
                raise GenerationError(f"Unexpected generation error: {e}") from e

        # Should be unreachable, but satisfies type-checkers.
        raise GenerationError("Generation loop exited without a response.")  # pragma: no cover

    def _parse_citations(
        self,
        text: str,
        search_results: List[SearchResult],
    ) -> List[CitedChunk]:
        """Extract [N] citation markers from the answer and map them to CitedChunk objects.

        Only indices that:
          1. Actually appear in the answer text, AND
          2. Are within the valid range [1, len(search_results)]
        are included in the output.  Indices are de-duplicated (first-seen order preserved).

        Args:
            text: Raw answer text from Gemini.
            search_results: The ordered list of chunks that was passed to the prompt.

        Returns:
            De-duplicated list of ``CitedChunk`` objects in first-seen citation order.
        """
        raw_indices = re.findall(r"\[(\d+)\]", text)
        seen: set[int] = set()
        cited: List[CitedChunk] = []

        for raw in raw_indices:
            idx = int(raw)

            if idx in seen:
                continue
            seen.add(idx)

            if idx < 1 or idx > len(search_results):
                logger.warning(
                    "Answer contains out-of-range citation [%d] (only %d chunks available) — skipping.",
                    idx,
                    len(search_results),
                )
                continue

            chunk = search_results[idx - 1].chunk  # 1-based → 0-based
            cited.append(
                CitedChunk(
                    index=idx,
                    source_file=chunk.source_file,
                    section_heading=chunk.section_heading,
                    url=chunk.metadata.get("url", ""),
                    text=chunk.text,
                )
            )

        return cited

    def _is_grounded(self, text: str) -> bool:
        """Return False if the answer is the insufficient-context sentinel phrase."""
        return _NO_CONTEXT_PHRASE not in text
