import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.models.document import RawDocument


class DocLoader:
    """Handles fetching, loading, and normalizing FastAPI official documentation."""

    def __init__(self, source_path: Optional[str] = None):
        # Use provided source path or default from settings
        self.raw_dir = Path(source_path or settings.DATA_SOURCE_PATH)
        self.source_dir = self.raw_dir / "fastapi-src"
        self.docs_dir = self.source_dir / "docs" / "en" / "docs"

    def fetch_docs(self) -> None:
        """Clones the FastAPI repository sparsely to download English documentation files."""
        if self.docs_dir.exists() and any(self.docs_dir.iterdir()):
            logger.info(f"FastAPI documentation already exists at {self.docs_dir}")
            return

        logger.info("FastAPI documentation source not found. Starting sparse shallow clone...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Initialize sparse clone of FastAPI repo
            clone_cmd = [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "https://github.com/fastapi/fastapi.git",
                "fastapi-src",
            ]
            logger.info(f"Running command: {' '.join(clone_cmd)} in {self.raw_dir}")
            subprocess.run(clone_cmd, cwd=self.raw_dir, check=True, capture_output=True)

            # 2. Set sparse checkout to docs/en/docs
            sparse_cmd = ["git", "sparse-checkout", "set", "docs/en/docs"]
            logger.info(f"Running command: {' '.join(sparse_cmd)} in {self.source_dir}")
            subprocess.run(sparse_cmd, cwd=self.source_dir, check=True, capture_output=True)

            logger.info("FastAPI sparse documentation clone completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e.stderr.decode('utf-8', errors='ignore')}")
            raise RuntimeError(f"Failed to fetch FastAPI documentation via git: {e}")
        except Exception as e:
            logger.error(f"Error occurred while fetching docs: {e}")
            raise

    def strip_frontmatter(self, content: str) -> str:
        """Removes markdown frontmatter starting and ending with '---' at the top of the file."""
        # Strips frontmatter block (YAML metadata)
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if match:
            return content[match.end() :]
        return content

    def extract_title(self, content: str, default_title: str) -> str:
        """Extracts the first '# Heading' title from markdown content."""
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return default_title

    def get_url_from_path(self, relative_path: str) -> str:
        """Constructs the official FastAPI docs website URL from relative file path."""
        normalized = relative_path.replace("\\", "/")
        prefix = "docs/en/docs/"
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]

        if normalized == "index.md":
            return "https://fastapi.tiangolo.com/"

        if normalized.endswith("index.md"):
            path = normalized[: -len("index.md")]
        elif normalized.endswith(".md"):
            path = normalized[: -len(".md")] + "/"
        else:
            path = normalized

        return f"https://fastapi.tiangolo.com/{path}"

    def load(self) -> List[RawDocument]:
        """Loads and normalizes all markdown files under docs/en/docs."""
        # Ensure docs are fetched first
        self.fetch_docs()

        documents: List[RawDocument] = []
        logger.info(f"Loading markdown files from {self.docs_dir}")

        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if not file.endswith(".md") or file == "release-notes.md":
                    continue

                full_path = Path(root) / file
                # Compute relative path from the fastapi-src root
                rel_path_from_src = full_path.relative_to(self.source_dir)
                rel_path_str = str(rel_path_from_src).replace("\\", "/")

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        raw_content = f.read()

                    # Normalize content: strip frontmatter
                    clean_content = self.strip_frontmatter(raw_content)

                    # Extract metadata
                    title = self.extract_title(clean_content, default_title=full_path.stem)
                    url = self.get_url_from_path(rel_path_str)

                    # Calculate document ID (hash of source path)
                    doc_id = hashlib.md5(rel_path_str.encode("utf-8")).hexdigest()

                    doc = RawDocument(
                        id=doc_id,
                        text=clean_content,
                        source_file=rel_path_str,
                        metadata={
                            "url": url,
                            "title": title,
                            "filename": file,
                        },
                    )
                    documents.append(doc)

                except Exception as e:
                    logger.error(f"Error loading document {full_path}: {e}")
                    # Continue loading other documents

        # Optional: Limit documents loaded for dev/testing to save Gemini quota
        limit = int(os.environ.get("LIMIT_INGEST_FILES", "0"))
        if limit > 0:
            logger.info(f"LIMIT_INGEST_FILES={limit} is set. Limiting load to first {limit} documents.")
            documents = documents[:limit]

        logger.info(f"Loaded {len(documents)} documents successfully.")
        return documents
