import hashlib
import re
from typing import List, Optional
from app.ingestion.chunkers.base import BaseChunker
from app.models.chunk import Chunk
from app.models.document import RawDocument


class SemanticChunker(BaseChunker):
    """Splits raw documents into chunks based on semantic shifts between sentences.

    Uses a local lightweight SentenceTransformer model to compute cosine similarity
    between adjacent sentences, splitting where similarity falls below a threshold.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.65,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 150,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self._model = None

    @property
    def model(self):
        """Lazily load the SentenceTransformer model to speed up startup times."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger_name = "sentence_transformers"
                import logging
                # Mute verbose logs during model loading
                logging.getLogger(logger_name).setLevel(logging.WARNING)
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for SemanticChunker. "
                    "Install it via requirements.txt"
                )
        return self._model

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences using simple regex boundaries, keeping markdown elements intact."""
        # Split on sentence terminals followed by space or newline
        sentence_ends = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_ends.split(text)
        # Filter empty sentences and strip
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, v1, v2) -> float:
        """Calculates cosine similarity between two 1D vectors."""
        import numpy as np
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def chunk(self, doc: RawDocument) -> List[Chunk]:
        """Chunks a document by comparing semantic distance of consecutive sentences."""
        text = doc.text
        if not text:
            return []

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        # If we only have one sentence, return it as a single chunk
        if len(sentences) == 1:
            chunk_text = sentences[0]
            chunk_id = hashlib.md5(
                f"semantic:{doc.source_file}:{chunk_text}".encode("utf-8")
            ).hexdigest()
            return [
                Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    source_file=doc.source_file,
                    section_heading=doc.metadata.get("title", "Introduction"),
                    chunking_strategy="semantic",
                    metadata=doc.metadata.copy(),
                )
            ]

        # 1. Compute embeddings for all sentences
        embeddings = self.model.encode(sentences, convert_to_numpy=True)

        # 2. Compute similarities between adjacent sentences
        similarities: List[float] = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
            similarities.append(sim)

        # 3. Walk through sentences and group them
        chunks: List[Chunk] = []
        current_sentences: List[str] = [sentences[0]]
        chunk_idx = 0

        for i in range(len(sentences) - 1):
            next_sentence = sentences[i+1]
            similarity = similarities[i]
            
            # Check length of current group if we add the next sentence
            current_text_len = sum(len(s) for s in current_sentences) + len(next_sentence)

            # Split if:
            # - Similarity falls below threshold, AND current chunk is large enough
            # - OR adding the next sentence exceeds max chunk size
            should_split = (
                (similarity < self.similarity_threshold and current_text_len >= self.min_chunk_size) or
                (current_text_len > self.max_chunk_size)
            )

            if should_split:
                # Emit current group
                chunk_text = " ".join(current_sentences).strip()
                if chunk_text:
                    chunk_id = hashlib.md5(
                        f"semantic:{doc.source_file}:{chunk_idx}:{chunk_text}".encode("utf-8")
                    ).hexdigest()
                    
                    chunk_meta = doc.metadata.copy()
                    chunk_meta["chunk_index"] = chunk_idx

                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=chunk_text,
                            source_file=doc.source_file,
                            section_heading=doc.metadata.get("title", "Introduction"),
                            chunking_strategy="semantic",
                            metadata=chunk_meta,
                        )
                    )
                    chunk_idx += 1
                
                # Start new group
                current_sentences = [next_sentence]
            else:
                current_sentences.append(next_sentence)

        # Append last group
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunk_id = hashlib.md5(
                    f"semantic:{doc.source_file}:{chunk_idx}:{chunk_text}".encode("utf-8")
                ).hexdigest()
                
                chunk_meta = doc.metadata.copy()
                chunk_meta["chunk_index"] = chunk_idx

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=chunk_text,
                        source_file=doc.source_file,
                        section_heading=doc.metadata.get("title", "Introduction"),
                        chunking_strategy="semantic",
                        metadata=chunk_meta,
                    )
                )

        return chunks
