"""
Citation Verification Service for Retryv's RAG pipeline.

Uses Gemini as an LLM-as-judge to verify whether each cited chunk in a
GenerationResult actually supports the claims made in the answer.

Design decisions:
  - Chunk-level (not sentence-level): passes the full answer + full chunk to the
    judge, matching the standard RAGAS pattern and avoiding fragile sentence
    segmentation.
  - Standalone service: Generator.generate() is unaware of this class; the /ask
    endpoint decides whether to call verify().
  - Inter-call sleep: a small courtesy delay between sequential _verify_one calls
    prevents rapid-fire bursts from hitting the free-tier RPM limit (same principle
    as GeminiEmbedder's adaptive sleep during ingestion).
  - Same retry pattern as Generator: max 2 retries, exponential backoff,
    VerificationError on exhaustion.
"""
import time
import logging
from typing import List

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.core.config import settings
from app.core.generation_client import get_generation_client, rotate_generation_key
from app.models.generation import (
    CitedChunk,
    GenerationResult,
    VerificationResult,
    VerifiedGenerationResult,
)

logger = logging.getLogger(__name__)

# Retry configuration — matches code-standards.md §2 and Generator pattern.
_MAX_ATTEMPTS = 3       # 1 initial attempt + 2 retries
_BACKOFF_FACTOR = 2.0  # sleep = backoff_factor ** attempt + 1.0  →  1 s, 3 s

# Accepted verdicts from the judge (after strip + upper).
_SUPPORTED = "SUPPORTED"
_NOT_SUPPORTED = "NOT_SUPPORTED"
_VALID_VERDICTS = {_SUPPORTED, _NOT_SUPPORTED}

# System instruction for the judge — kept terse; one-word response only.
_JUDGE_SYSTEM = (
    "You are a citation fact-checker for a RAG (Retrieval-Augmented Generation) system.\n"
    "You will be given a generated answer and a reference chunk of text.\n"
    "Your task: determine whether the reference chunk directly supports the claims "
    "made in the answer.\n\n"
    "Respond with EXACTLY one word — no punctuation, no explanation:\n"
    "  SUPPORTED     — the chunk provides clear textual evidence for the answer's claims\n"
    "  NOT_SUPPORTED — the chunk does not support the claims, is irrelevant, or the "
    "answer makes claims not present in the chunk"
)


class VerificationError(RuntimeError):
    """Raised when the Gemini judge call fails after all retry attempts.

    The caller can decide whether to treat unverified citations as NOT_SUPPORTED
    or surface an error to the user.
    """


class CitationVerifier:
    """LLM-as-judge service that verifies each cited chunk in a GenerationResult.

    Usage::

        gen_result = Generator().generate("How do lifespan events work?", results)
        verified   = CitationVerifier().verify(gen_result)

        print(verified.all_supported)     # True / False
        print(verified.support_ratio)     # 0.0 – 1.0
        for v in verified.verifications:
            print(v.chunk_index, v.verdict)
    """

    def __init__(
        self,
        model_name: str = None,
        inter_call_sleep: float = None,
    ):
        """
        Args:
            model_name: Gemini model to use as judge. Defaults to
                ``settings.GEMINI_GEN_MODEL`` (gemini-2.5-flash).
            inter_call_sleep: Seconds to sleep between sequential _verify_one
                calls in a single verify() pass.  Defaults to
                ``settings.VERIFICATION_INTER_CALL_SLEEP`` (0.5 s).
                Pass 0.0 in tests to avoid real sleeps.
        """
        self._model_name = model_name
        self.inter_call_sleep = (
            inter_call_sleep if inter_call_sleep is not None
            else settings.VERIFICATION_INTER_CALL_SLEEP
        )
        self._client: genai.Client | None = None

    @property
    def model_name(self) -> str:
        return self._model_name or settings.GEMINI_GEN_MODEL

    # ------------------------------------------------------------------
    # Lazy client initialisation (mirrors GeminiEmbedder / Generator)
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

    def verify(self, generation_result: GenerationResult) -> VerifiedGenerationResult:
        """Verify every cited chunk in generation_result against the answer text.

        Args:
            generation_result: Output from ``Generator.generate()``.

        Returns:
            ``VerifiedGenerationResult`` with per-citation verdicts, ``all_supported``
            flag, and ``support_ratio``.

        Raises:
            VerificationError: If any individual _verify_one call exhausts all retries.
        """
        cited_chunks: List[CitedChunk] = generation_result.cited_chunks

        if not cited_chunks:
            logger.info("CitationVerifier: no cited chunks to verify — returning immediately.")
            return VerifiedGenerationResult(
                generation=generation_result,
                verifications=[],
                all_supported=True,
                support_ratio=1.0,
            )

        verifications: List[VerificationResult] = []

        for i, cited_chunk in enumerate(cited_chunks):
            result = self._verify_one(generation_result.answer, cited_chunk)
            verifications.append(result)

            # Courtesy inter-call sleep — spaces out 1–5 rapid calls to avoid
            # hitting the free-tier RPM ceiling (gemini-2.5-flash ~10 RPM).
            # Applied after every call EXCEPT the last one in the batch.
            if i < len(cited_chunks) - 1 and self.inter_call_sleep > 0:
                logger.debug(
                    "CitationVerifier: sleeping %.2fs between verification calls.",
                    self.inter_call_sleep,
                )
                time.sleep(self.inter_call_sleep)

        supported_count = sum(1 for v in verifications if v.supported)
        all_supported = supported_count == len(verifications)
        support_ratio = supported_count / len(verifications)

        logger.info(
            "CitationVerifier: %d/%d citations supported (ratio=%.2f).",
            supported_count,
            len(verifications),
            support_ratio,
        )

        return VerifiedGenerationResult(
            generation=generation_result,
            verifications=verifications,
            all_supported=all_supported,
            support_ratio=support_ratio,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verify_one(
        self, answer: str, cited_chunk: CitedChunk
    ) -> VerificationResult:
        """Call the judge for a single chunk.

        Args:
            answer: Full answer text from Gemini.
            cited_chunk: The cited chunk to verify against the answer.

        Returns:
            ``VerificationResult`` with verdict and metadata.

        Raises:
            VerificationError: After ``_MAX_ATTEMPTS`` failed API calls.
        """
        user_content = (
            f"Reference Chunk [{cited_chunk.index}]:\n"
            f"{cited_chunk.text.strip()}\n\n"
            f"Generated Answer:\n"
            f"{answer.strip()}\n\n"
            f"Does the reference chunk support the answer? "
            f"Reply SUPPORTED or NOT_SUPPORTED."
        )

        config = types.GenerateContentConfig(
            system_instruction=_JUDGE_SYSTEM,
            temperature=0.0,  # deterministic judge output
        )

        raw = self._call_with_retry(user_content, config)
        return self._parse_verdict(raw, cited_chunk)

    def _parse_verdict(
        self, raw: str, cited_chunk: CitedChunk
    ) -> VerificationResult:
        """Parse raw judge response into a VerificationResult.

        Args:
            raw: Raw text from the judge model.
            cited_chunk: The chunk being verified (for index/source_file).

        Returns:
            ``VerificationResult`` with ``parse_error=True`` if the response was
            not one of the expected one-word verdicts.
        """
        normalised = raw.strip().upper()
        parse_error = normalised not in _VALID_VERDICTS

        if parse_error:
            logger.warning(
                "CitationVerifier: unexpected judge response for chunk [%d]: %r — "
                "treating as NOT_SUPPORTED.",
                cited_chunk.index,
                raw,
            )
            verdict = _NOT_SUPPORTED
        else:
            verdict = normalised

        return VerificationResult(
            chunk_index=cited_chunk.index,
            source_file=cited_chunk.source_file,
            verdict=verdict,
            supported=(verdict == _SUPPORTED),
            parse_error=parse_error,
            raw_response=raw,
        )

    def _call_with_retry(
        self,
        user_content: str,
        config: types.GenerateContentConfig,
    ) -> str:
        """Call generate_content with exponential-backoff retry on APIError.

        Returns:
            Raw response text.

        Raises:
            VerificationError: After ``_MAX_ATTEMPTS`` failed attempts.
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
                        "CitationVerifier: judge call failed after %d attempts: %s",
                        _MAX_ATTEMPTS,
                        e,
                    )
                    raise VerificationError(
                        f"Citation verification failed after {_MAX_ATTEMPTS} attempts: {e}"
                    ) from e

                is_429 = getattr(e, "code", None) == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if is_429:
                    rotate_generation_key()
                sleep_time = 2.0 if is_429 else (_BACKOFF_FACTOR ** attempt + 1.0)
                logger.warning(
                    "CitationVerifier: APIError on attempt %d/%d (is_429=%s) — retrying in %.1fs. %s",
                    attempt + 1,
                    _MAX_ATTEMPTS,
                    is_429,
                    sleep_time,
                    e,
                )
                time.sleep(sleep_time)

            except Exception as e:
                logger.error("CitationVerifier: unexpected error during judge call: %s", e)
                raise VerificationError(
                    f"Unexpected citation verification error: {e}"
                ) from e

        raise VerificationError("Verification loop exited without a response.")  # pragma: no cover
