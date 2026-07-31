"""
Confidence Guard — pre-generation retrieval quality gate.

Inspects the top RRF score from hybrid retrieval and short-circuits the Gemini
call when scores are too low to justify generation, saving an API call on queries
where the retrieval layer already signals "nothing relevant found."

Design decisions
----------------
* Top-1 check only: if the best result clears the threshold, the full ranked list
  is accepted.  Filtering every chunk would strip valid supporting evidence.
* Single-result warning: a list of length 1 may indicate a narrow query rather
  than genuine dual-system agreement (RRF score is highest when both dense AND
  sparse agree).  A warning is logged but the result is NOT blocked — blocking
  would be too aggressive for narrow but valid queries (e.g. "what is the
  default port for uvicorn?").  The threshold guard itself is the primary filter.
* Threshold is configurable via RETRIEVAL_CONFIDENCE_THRESHOLD in settings and
  can be overridden at construction time for testing or corpus-specific tuning.
"""
import logging
from typing import List

from app.core.config import settings
from app.models.search import SearchResult

logger = logging.getLogger(__name__)


class ConfidenceGuard:
    """Pre-generation gate that short-circuits the Gemini call when retrieval
    scores indicate nothing relevant was found.

    Usage::

        guard = ConfidenceGuard()
        if not guard.is_retrieval_sufficient(search_results):
            return GenerationResult(answer=_NO_CONTEXT_PHRASE, is_grounded=False, ...)
    """

    def __init__(self, threshold: float = None):
        """
        Args:
            threshold: Minimum top-1 RRF score required to proceed with generation.
                       Defaults to ``settings.RETRIEVAL_CONFIDENCE_THRESHOLD`` (0.025).
        """
        self.threshold = (
            threshold if threshold is not None
            else settings.RETRIEVAL_CONFIDENCE_THRESHOLD
        )

    def is_retrieval_sufficient(self, search_results: List[SearchResult]) -> bool:
        """Return True if the top result clears the confidence threshold.

        Short-circuits to False immediately when:
          - ``search_results`` is empty, OR
          - the top score < ``self.threshold``

        Emits a WARNING (but does NOT block) when only one chunk was retrieved,
        since a single result cannot reflect genuine dense+sparse agreement.

        Args:
            search_results: Ranked list from the retriever (highest score first).

        Returns:
            ``True`` → proceed with generation.
            ``False`` → skip the Gemini call, return the insufficient-context response.
        """
        if not search_results:
            logger.warning(
                "ConfidenceGuard received empty search_results — blocking generation."
            )
            return False

        top_score = (
            search_results[0].rrf_score 
            if search_results[0].rrf_score is not None 
            else search_results[0].score
        )

        if len(search_results) == 1:
            # Single result cannot reflect dual-system RRF agreement.
            # Log a warning so this case is visible in production logs,
            # but still defer to the threshold check — don't add a second
            # hard block that would suppress valid narrow queries.
            logger.warning(
                "ConfidenceGuard: only 1 chunk retrieved (score=%.4f). "
                "Single results cannot reflect dense+sparse agreement; "
                "treat this result with caution. Threshold check still applies.",
                top_score,
            )

        if top_score < self.threshold:
            logger.info(
                "ConfidenceGuard: top score %.4f < threshold %.4f — blocking generation.",
                top_score,
                self.threshold,
            )
            return False

        logger.debug(
            "ConfidenceGuard: top score %.4f >= threshold %.4f — proceeding.",
            top_score,
            self.threshold,
        )
        return True
