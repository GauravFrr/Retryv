"""
Prompt Builder for Retryv's grounded generation pipeline.

Constructs the system instruction and user prompt sent to Gemini for RAG generation.
The builder enforces strict grounding constraints — the model must only use the
provided context chunks and must cite sources inline using [N] notation.
"""
import logging
from typing import List

from app.models.search import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System instruction (reused across every request)
# ---------------------------------------------------------------------------
_SYSTEM_INSTRUCTION = """You are a precise technical assistant for the FastAPI web framework.

Your job is to answer the user's question **strictly** based on the reference documentation
provided below. Follow these rules without exception:

1. **Ground every claim** — only state facts that are directly supported by the provided
   reference chunks. Do not infer, speculate, or add knowledge from your own training.
2. **Cite sources inline** — every sentence (or group of closely related sentences) that
   draws on a reference chunk must end with an inline citation in the format [N], where N
   is the chunk index shown in the reference list. Example: "You can use the lifespan
   parameter to register startup logic [1]."
3. **Insufficient context** — if the provided reference chunks do not contain enough
   information to answer the question, respond with exactly:
       "I do not have enough context to answer this question."
   Do not attempt a partial answer or guess.
4. **Format** — use clear, concise prose. Use markdown code blocks for any code examples
   you quote from the references. Do not add headers or bullet points unless they appear
   in the source material.
""".strip()


class PromptBuilder:
    """Constructs the grounded generation prompt from a question and retrieved chunks.

    The prompt is returned as a dict with two keys:
      - "system": the system instruction string (passed as Gemini's system_instruction)
      - "user":   the user-turn content string (contains formatted context + question)

    Keeping system and user content separate allows callers to pass each correctly to
    the Gemini API without any string-munging in the generator layer.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        question: str,
        search_results: List[SearchResult],
    ) -> dict[str, str]:
        """Build the full generation prompt.

        Args:
            question: The user's natural-language question.
            search_results: Ranked list of retrieved chunks (most relevant first).

        Returns:
            Dict with keys ``"system"`` and ``"user"``.
        """
        context_block = self.format_context(search_results)

        if search_results:
            user_content = (
                "Reference Documentation:\n"
                "---\n"
                f"{context_block}\n"
                "---\n\n"
                f"User Question: {question}"
            )
        else:
            # No chunks retrieved — surface the empty-context signal to the model so
            # it triggers the "insufficient context" response path.
            logger.warning("PromptBuilder received empty search_results list.")
            user_content = (
                "Reference Documentation:\n"
                "---\n"
                "(No relevant documentation chunks were retrieved for this query.)\n"
                "---\n\n"
                f"User Question: {question}"
            )

        logger.debug(
            "Built prompt for question=%r  num_chunks=%d",
            question,
            len(search_results),
        )

        return {
            "system": _SYSTEM_INSTRUCTION,
            "user": user_content,
        }

    def format_context(self, search_results: List[SearchResult]) -> str:
        """Format retrieved chunks into a numbered reference block.

        Each chunk is rendered as:
            [N] Source: <source_file>
                Section: <section_heading>
                Content:
                <chunk text>

        Args:
            search_results: Ranked list of retrieved chunks.

        Returns:
            A multi-line string ready for insertion into the user prompt.
        """
        if not search_results:
            return ""

        parts: List[str] = []
        for idx, result in enumerate(search_results, start=1):
            chunk = result.chunk

            # Pull the canonical URL from metadata if present, otherwise omit it.
            url: str = chunk.metadata.get("url", "")
            url_line = f"    URL: {url}\n" if url else ""

            # Indent chunk text for visual separation from the metadata header.
            indented_text = "\n".join(
                f"    {line}" for line in chunk.text.strip().splitlines()
            )

            block = (
                f"[{idx}] Source: {chunk.source_file}\n"
                f"    Section: {chunk.section_heading}\n"
                f"{url_line}"
                f"    Content:\n"
                f"{indented_text}"
            )
            parts.append(block)

        return "\n\n".join(parts)
