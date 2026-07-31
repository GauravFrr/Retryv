"""
Unit tests for PromptBuilder (Phase 3 Unit 3.1).

All tests are pure-Python — no Gemini API calls, no ChromaDB, no BM25.
"""
import pytest

from app.models.chunk import Chunk
from app.models.search import SearchResult
from app.generation.prompt_builder import PromptBuilder, _SYSTEM_INSTRUCTION


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


@pytest.fixture
def chunk_a() -> Chunk:
    return Chunk(
        id="chunk-a",
        text="You can use the lifespan parameter to register startup and shutdown logic.",
        source_file="docs/en/docs/advanced/events.md",
        section_heading="Lifespan Events",
        chunking_strategy="fixed_size",
        metadata={"url": "https://fastapi.tiangolo.com/advanced/events/"},
    )


@pytest.fixture
def chunk_b() -> Chunk:
    return Chunk(
        id="chunk-b",
        text="Dependency Injection lets you declare dependencies in path operation functions.",
        source_file="docs/en/docs/tutorial/dependencies/index.md",
        section_heading="Dependencies",
        chunking_strategy="fixed_size",
        metadata={"url": "https://fastapi.tiangolo.com/tutorial/dependencies/"},
    )


@pytest.fixture
def chunk_no_url() -> Chunk:
    return Chunk(
        id="chunk-c",
        text="FastAPI is built on top of Starlette for the web parts.",
        source_file="docs/en/docs/index.md",
        section_heading="Overview",
        chunking_strategy="structure_aware",
        metadata={},  # deliberately empty — no URL key
    )


@pytest.fixture
def results_two(chunk_a, chunk_b) -> list[SearchResult]:
    return [
        SearchResult(chunk=chunk_a, score=0.9),
        SearchResult(chunk=chunk_b, score=0.7),
    ]


# ---------------------------------------------------------------------------
# format_context tests
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_empty_list_returns_empty_string(self, builder: PromptBuilder):
        assert builder.format_context([]) == ""

    def test_single_chunk_contains_index_1(self, builder: PromptBuilder, chunk_a: Chunk):
        result = builder.format_context([SearchResult(chunk=chunk_a, score=0.9)])
        assert "[1]" in result
        assert "[2]" not in result

    def test_two_chunks_numbered_sequentially(self, builder: PromptBuilder, results_two):
        result = builder.format_context(results_two)
        assert "[1]" in result
        assert "[2]" in result

    def test_chunk_source_file_present(self, builder: PromptBuilder, results_two):
        result = builder.format_context(results_two)
        assert "docs/en/docs/advanced/events.md" in result
        assert "docs/en/docs/tutorial/dependencies/index.md" in result

    def test_chunk_section_heading_present(self, builder: PromptBuilder, results_two):
        result = builder.format_context(results_two)
        assert "Lifespan Events" in result
        assert "Dependencies" in result

    def test_chunk_text_present(self, builder: PromptBuilder, results_two):
        result = builder.format_context(results_two)
        assert "lifespan parameter" in result
        assert "Dependency Injection" in result

    def test_url_included_when_present(self, builder: PromptBuilder, chunk_a: Chunk):
        result = builder.format_context([SearchResult(chunk=chunk_a, score=0.9)])
        assert "fastapi.tiangolo.com/advanced/events/" in result

    def test_url_line_omitted_when_missing(self, builder: PromptBuilder, chunk_no_url: Chunk):
        result = builder.format_context([SearchResult(chunk=chunk_no_url, score=0.5)])
        assert "URL:" not in result


# ---------------------------------------------------------------------------
# build_prompt tests
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_returns_system_and_user_keys(self, builder: PromptBuilder, results_two):
        prompt = builder.build_prompt("What is FastAPI?", results_two)
        assert "system" in prompt
        assert "user" in prompt

    def test_system_instruction_matches_module_constant(self, builder: PromptBuilder, results_two):
        prompt = builder.build_prompt("What is FastAPI?", results_two)
        assert prompt["system"] == _SYSTEM_INSTRUCTION

    def test_system_instruction_contains_grounding_rules(self, builder: PromptBuilder, results_two):
        prompt = builder.build_prompt("What is FastAPI?", results_two)
        system = prompt["system"]
        assert "Ground every claim" in system
        assert "Cite sources inline" in system
        assert "Insufficient context" in system
        assert "I do not have enough context to answer this question." in system

    def test_user_prompt_contains_question(self, builder: PromptBuilder, results_two):
        question = "How do lifespan events work?"
        prompt = builder.build_prompt(question, results_two)
        assert question in prompt["user"]

    def test_user_prompt_contains_numbered_chunks(self, builder: PromptBuilder, results_two):
        prompt = builder.build_prompt("What is FastAPI?", results_two)
        assert "[1]" in prompt["user"]
        assert "[2]" in prompt["user"]

    def test_user_prompt_contains_reference_header(self, builder: PromptBuilder, results_two):
        prompt = builder.build_prompt("What is FastAPI?", results_two)
        assert "Reference Documentation:" in prompt["user"]

    def test_empty_results_surfaces_no_context_signal(self, builder: PromptBuilder):
        prompt = builder.build_prompt("What is FastAPI?", [])
        assert "No relevant documentation chunks were retrieved" in prompt["user"]

    def test_empty_results_still_contains_question(self, builder: PromptBuilder):
        question = "Tell me about background tasks."
        prompt = builder.build_prompt(question, [])
        assert question in prompt["user"]

    def test_prompt_preserves_chunk_text(self, builder: PromptBuilder, chunk_a: Chunk):
        prompt = builder.build_prompt("Lifespan?", [SearchResult(chunk=chunk_a, score=0.9)])
        assert "lifespan parameter" in prompt["user"]
