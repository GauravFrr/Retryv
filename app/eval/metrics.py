"""
Evaluation Metrics Implementation for Retryv (Phase 4 Unit 4.2).

Implements four core RAG evaluation metrics:
  1. Retrieval Metrics (Recall & Precision — pure mathematical set overlap)
  2. Citation Accuracy Metric (reuses CitationVerifier from Unit 3.4)
  3. Faithfulness Metric (Sc * Su: Citation support ratio * Uncited sentence groundedness)
  4. Answer Correctness Metric (LLM-as-judge with category-specific prompt for ambiguous queries)
"""
import re
import time
import logging
from typing import List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.core.config import settings
from app.core.generation_client import get_generation_client, rotate_generation_key
from app.eval.dataset import GoldenQuery
from app.eval.models import MetricScore
from app.generation.citation_verifier import CitationVerifier
from app.models.generation import GenerationResult
from app.models.search import SearchResult

logger = logging.getLogger(__name__)

# Retry configuration for LLM judge calls
_MAX_ATTEMPTS = 3
_BACKOFF_FACTOR = 2.0


# ---------------------------------------------------------------------------
# 1. Retrieval Metrics (Pure Math)
# ---------------------------------------------------------------------------

class RetrievalMetrics:
    """Calculates set-overlap retrieval metrics (Recall and Precision) over document source paths."""

    @staticmethod
    def compute_recall(
        retrieved_sources: List[str], expected_sources: Optional[List[str]]
    ) -> MetricScore:
        """Compute Retrieval Recall: |Retrieved ∩ Expected| / |Expected|.

        If expected_sources is None or empty (e.g. unanswerable queries), Recall = 1.0.
        """
        if not expected_sources:
            return MetricScore(
                metric_name="retrieval_recall",
                score=1.0,
                reasoning="No expected sources specified (unanswerable query) — score set to 1.0.",
                metadata={"retrieved_count": len(retrieved_sources), "expected_count": 0},
            )

        retrieved_set = set(retrieved_sources)
        expected_set = set(expected_sources)
        intersection = retrieved_set.intersection(expected_set)

        recall_score = len(intersection) / len(expected_set)
        return MetricScore(
            metric_name="retrieval_recall",
            score=recall_score,
            reasoning=f"Matched {len(intersection)} of {len(expected_set)} expected sources.",
            metadata={
                "retrieved_sources": list(retrieved_set),
                "expected_sources": list(expected_set),
                "matched_sources": list(intersection),
            },
        )

    @staticmethod
    def compute_precision(
        retrieved_sources: List[str], expected_sources: Optional[List[str]]
    ) -> MetricScore:
        """Compute Retrieval Precision: |Retrieved ∩ Expected| / |Retrieved|.

        If retrieved_sources is empty, Precision = 1.0 if expected is empty, 0.0 otherwise.
        """
        if not retrieved_sources:
            score = 1.0 if not expected_sources else 0.0
            return MetricScore(
                metric_name="retrieval_precision",
                score=score,
                reasoning="Retrieved sources list is empty.",
                metadata={"retrieved_count": 0, "expected_count": len(expected_sources or [])},
            )

        expected_set = set(expected_sources or [])
        retrieved_set = set(retrieved_sources)
        intersection = retrieved_set.intersection(expected_set)

        precision_score = len(intersection) / len(retrieved_set)
        return MetricScore(
            metric_name="retrieval_precision",
            score=precision_score,
            reasoning=f"{len(intersection)} of {len(retrieved_set)} retrieved sources were relevant.",
            metadata={
                "retrieved_sources": list(retrieved_set),
                "expected_sources": list(expected_set),
                "matched_sources": list(intersection),
            },
        )


# ---------------------------------------------------------------------------
# 2. Citation Accuracy Metric (Reuses CitationVerifier)
# ---------------------------------------------------------------------------

class CitationAccuracyMetric:
    """Evaluates citation truthfulness by reusing CitationVerifier from Unit 3.4."""

    def __init__(self, verifier: Optional[CitationVerifier] = None):
        self.verifier = verifier or CitationVerifier()

    def compute(self, generation_result: GenerationResult) -> MetricScore:
        """Compute citation accuracy score (support_ratio of cited chunks)."""
        if not generation_result.is_grounded:
            return MetricScore(
                metric_name="citation_accuracy",
                score=None,
                reasoning="N/A: Query is unanswerable or generation not grounded (refusal response).",
                metadata={"all_supported": None, "citations_count": 0},
            )

        verified = self.verifier.verify(generation_result)
        return MetricScore(
            metric_name="citation_accuracy",
            score=verified.support_ratio,
            reasoning=(
                f"{sum(1 for v in verified.verifications if v.supported)} of "
                f"{len(verified.verifications)} citations supported."
            ),
            metadata={
                "all_supported": verified.all_supported,
                "citations_count": len(verified.verifications),
            },
        )


# ---------------------------------------------------------------------------
# 3. Faithfulness Metric (Sc * Su)
# ---------------------------------------------------------------------------

class FaithfulnessMetric:
    """Evaluates overall faithfulness / groundedness: Sc * Su.

    Where:
      - Sc (Citation Support Ratio): VerifiedGenerationResult.support_ratio (from Unit 3.4)
      - Su (Uncited Sentence Groundedness): 1.0 if no uncited sentences or if uncited
        sentences contain ONLY facts directly supported by the retrieved context chunks;
        0.0 if uncited sentences introduce ungrounded outside facts.
    """

    def __init__(
        self,
        verifier: Optional[CitationVerifier] = None,
        model_name: Optional[str] = None,
    ):
        self.verifier = verifier or CitationVerifier()
        self._model_name = model_name
        self._client: Optional[genai.Client] = None

    @property
    def model_name(self) -> str:
        return self._model_name or settings.GEMINI_GEN_MODEL

    @property
    def client(self) -> genai.Client:
        """Return instance-level override if set (e.g. in tests), else rotating singleton."""
        if self._client is not None:
            return self._client
        return get_generation_client()

    def compute(
        self,
        generation_result: GenerationResult,
        search_results: List[SearchResult],
    ) -> MetricScore:
        """Compute Faithfulness = Sc * Su."""
        if not generation_result.is_grounded:
            return MetricScore(
                metric_name="faithfulness",
                score=None,
                reasoning="N/A: Query is unanswerable or generation not grounded (refusal response).",
                metadata={"Sc_citation_support": None, "Su_uncited_groundedness": None},
            )

        # 1. Sc: Citation support ratio (reuses CitationVerifier)
        verified = self.verifier.verify(generation_result)
        sc = verified.support_ratio if generation_result.cited_chunks else 1.0

        # 2. Extract uncited sentences from answer
        uncited_sentences = self._extract_uncited_sentences(generation_result.answer)

        # 3. Su: Uncited sentence groundedness
        if not uncited_sentences:
            su = 1.0
            uncited_reason = "All sentences contain explicit inline citations."
        else:
            su, uncited_reason = self._evaluate_uncited_sentences(
                uncited_sentences, search_results
            )

        faithfulness_score = sc * su
        return MetricScore(
            metric_name="faithfulness",
            score=faithfulness_score,
            reasoning=f"Citation support Sc={sc:.2f}, Uncited groundedness Su={su:.2f}. {uncited_reason}",
            metadata={
                "Sc_citation_support": sc,
                "Su_uncited_groundedness": su,
                "uncited_sentences": uncited_sentences,
            },
        )

    def _extract_uncited_sentences(self, answer: str) -> List[str]:
        """Splits answer into sentences and returns those without [N] citation tags."""
        raw_sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()
        ]
        uncited = []
        for sentence in raw_sentences:
            if not re.search(r"\[\d+\]", sentence):
                uncited.append(sentence)
        return uncited

    def _evaluate_uncited_sentences(
        self, uncited_sentences: List[str], search_results: List[SearchResult]
    ) -> Tuple[float, str]:
        """Runs LLM judge pass on uncited sentences against retrieved context."""
        context_text = "\n\n".join(
            f"Chunk {i+1}:\n{res.chunk.text}" for i, res in enumerate(search_results)
        )
        uncited_text = "\n".join(f"- {s}" for s in uncited_sentences)

        prompt = (
            f"Reference Context Chunks:\n{context_text}\n\n"
            f"Uncited Sentences from Answer:\n{uncited_text}\n\n"
            f"Task: Do the uncited sentences above contain ONLY facts directly supported "
            f"by the reference context chunks?\n"
            f"Reply GROUNDED if every statement is supported by the chunks.\n"
            f"Reply UNGROUNDED if any statement introduces outside facts or ungrounded claims.\n"
            f"Reply with EXACTLY one word: GROUNDED or UNGROUNDED."
        )

        config = types.GenerateContentConfig(
            system_instruction="You are a strict hallucination auditor for a RAG system. One word response only.",
            temperature=0.0,
        )

        try:
            raw_response = self._call_with_retry(prompt, config).strip().upper()
            if "GROUNDED" in raw_response and "UNGROUNDED" not in raw_response:
                return 1.0, "Uncited sentences are grounded in context."
            else:
                return 0.0, f"Uncited sentences contain ungrounded facts (judge output: {raw_response})."
        except Exception as e:
            logger.error(f"Faithfulness LLM judge failed: {e}")
            return 0.0, f"Uncited sentence evaluation failed: {e}"

    def _call_with_retry(self, prompt: str, config: types.GenerateContentConfig) -> str:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=prompt, config=config
                )
                return response.text
            except APIError as e:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise e
                is_429 = getattr(e, "code", None) == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                sleep_time = 15.0 if is_429 else (_BACKOFF_FACTOR ** attempt + 1.0)
                logger.warning("FaithfulnessMetric APIError (is_429=%s) — sleeping %.1fs: %s", is_429, sleep_time, e)
                time.sleep(sleep_time)
        raise RuntimeError("Faithfulness retry loop exhausted.")


# ---------------------------------------------------------------------------
# 4. Answer Correctness Metric (LLM Judge with Ambiguous Query Support)
# ---------------------------------------------------------------------------

class AnswerCorrectnessMetric:
    """Evaluates answer correctness (0.0 to 1.0) against ground truth expected_answer.

    For queries with category == 'ambiguous', the judge prompt is specifically
    instructed to award a full score (1.0) if the answer clearly addresses AT LEAST
    ONE of the valid interpretations/topics listed in answer_quality_notes.
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name
        self._client: Optional[genai.Client] = None

    @property
    def model_name(self) -> str:
        return self._model_name or settings.GEMINI_GEN_MODEL

    @property
    def client(self) -> genai.Client:
        """Return instance-level override if set (e.g. in tests), else rotating singleton."""
        if self._client is not None:
            return self._client
        return get_generation_client()

    def compute(
        self,
        golden_query: GoldenQuery,
        generation_result: GenerationResult,
    ) -> MetricScore:
        """Compute correctness score (0.0 to 1.0)."""
        # Unanswerable query handling
        if not golden_query.retrieval_sufficient:
            is_correct_fallback = not generation_result.is_grounded
            score = 1.0 if is_correct_fallback else 0.0
            return MetricScore(
                metric_name="correctness",
                score=score,
                reasoning=(
                    "Unanswerable query correctly returned insufficient-context response."
                    if is_correct_fallback
                    else "Unanswerable query incorrectly attempted an answer."
                ),
                metadata={"is_grounded": generation_result.is_grounded},
            )

        # Build category-specific prompt
        if golden_query.category == "ambiguous":
            user_prompt = (
                f"Query (Ambiguous): {golden_query.query}\n"
                f"Generated Answer: {generation_result.answer}\n"
                f"Reference Notes (Acceptable Interpretations):\n{golden_query.answer_quality_notes}\n"
                f"Reference Expected Answer Summary:\n{golden_query.expected_answer}\n\n"
                f"Evaluation Task: This query is ambiguous with multiple valid interpretations. "
                f"Do NOT enforce an exact match against the summary expected_answer. "
                f"Check if the generated answer clearly and accurately addresses AT LEAST ONE of the valid "
                f"interpretations listed in the Reference Notes.\n"
                f"Provide a numerical score between 0.0 and 1.0 (1.0 = clearly answers a valid interpretation, "
                f"0.5 = partially answers, 0.0 = incorrect or unhelpful) followed by a brief reason.\n"
                f"Format: SCORE: <float>\nREASON: <text>"
            )
        else:
            user_prompt = (
                f"Query: {golden_query.query}\n"
                f"Generated Answer: {generation_result.answer}\n"
                f"Expected Reference Answer: {golden_query.expected_answer}\n"
                f"Quality Criteria Notes: {golden_query.answer_quality_notes}\n\n"
                f"Evaluation Task: Compare the generated answer against the expected reference answer. "
                f"Provide a numerical score between 0.0 and 1.0 (1.0 = fully correct and accurate, "
                f"0.5 = partially correct, 0.0 = incorrect or unhelpful) followed by a brief reason.\n"
                f"Format: SCORE: <float>\nREASON: <text>"
            )

        config = types.GenerateContentConfig(
            system_instruction="You are an expert technical evaluation judge for FastAPI documentation RAG answers.",
            temperature=0.0,
        )

        try:
            raw_response = self._call_with_retry(user_prompt, config)
            score, reasoning = self._parse_judge_response(raw_response)
            return MetricScore(
                metric_name="correctness",
                score=score,
                reasoning=reasoning,
                metadata={"category": golden_query.category, "raw_response": raw_response},
            )
        except Exception as e:
            logger.error(f"AnswerCorrectnessMetric LLM judge failed: {e}")
            return MetricScore(
                metric_name="correctness",
                score=0.0,
                reasoning=f"LLM judge evaluation failed: {e}",
                metadata={"category": golden_query.category},
            )

    def _parse_judge_response(self, text: str) -> Tuple[float, str]:
        """Parses SCORE: <float> and REASON: <text> from judge response."""
        score_match = re.search(r"SCORE:\s*([0-1](?:\.\d+)?)", text, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.0
        reasoning = reason_match.group(1).strip() if reason_match else text.strip()
        return min(max(score, 0.0), 1.0), reasoning

    def _call_with_retry(self, prompt: str, config: types.GenerateContentConfig) -> str:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=prompt, config=config
                )
                return response.text
            except APIError as e:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise e
                is_429 = getattr(e, "code", None) == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if is_429:
                    rotate_generation_key()
                sleep_time = 2.0 if is_429 else (_BACKOFF_FACTOR ** attempt + 1.0)
                logger.warning("AnswerCorrectnessMetric APIError (is_429=%s) — sleeping %.1fs: %s", is_429, sleep_time, e)
                time.sleep(sleep_time)
        raise RuntimeError("Correctness retry loop exhausted.")
