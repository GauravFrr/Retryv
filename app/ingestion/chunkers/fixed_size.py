import hashlib
from typing import List
from app.ingestion.chunkers.base import BaseChunker
from app.models.chunk import Chunk
from app.models.document import RawDocument


class FixedSizeChunker(BaseChunker):
    """Splits raw documents into overlapping chunks of a fixed character size."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc: RawDocument) -> List[Chunk]:
        """Splits doc text using character windows."""
        text = doc.text
        if not text:
            return []

        chunks: List[Chunk] = []
        text_len = len(text)
        
        # If the text is smaller than the chunk size, return it as a single chunk
        if text_len <= self.chunk_size:
            chunk_text = text.strip()
            chunk_id = hashlib.md5(
                f"fixed_size:{doc.source_file}:{chunk_text}".encode("utf-8")
            ).hexdigest()
            return [
                Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    source_file=doc.source_file,
                    section_heading=doc.metadata.get("title", "Introduction"),
                    chunking_strategy="fixed_size",
                    metadata=doc.metadata.copy(),
                )
            ]

        start = 0
        chunk_idx = 0
        while start < text_len:
            end = start + self.chunk_size
            
            # Extract window
            chunk_text = text[start:end]
            
            # Avoid cutting off in the middle of a word if possible (only if there is space remaining)
            if end < text_len:
                last_space = chunk_text.rfind(" ")
                # Adjust end boundary to the last space to avoid cutting words
                # but only if it's within a reasonable distance (e.g., 50 characters)
                if last_space != -1 and (len(chunk_text) - last_space) < 50:
                    end = start + last_space
                    chunk_text = text[start:end]

            chunk_text = chunk_text.strip()
            
            # Only create chunk if it has actual content
            if chunk_text:
                chunk_id = hashlib.md5(
                    f"fixed_size:{doc.source_file}:{chunk_idx}:{chunk_text}".encode("utf-8")
                ).hexdigest()
                
                # Copy document metadata and add chunk index
                chunk_meta = doc.metadata.copy()
                chunk_meta["chunk_index"] = chunk_idx
                
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=chunk_text,
                        source_file=doc.source_file,
                        section_heading=doc.metadata.get("title", "Introduction"),
                        chunking_strategy="fixed_size",
                        metadata=chunk_meta,
                    )
                )
                chunk_idx += 1

            # Slide window by (chunk_size - chunk_overlap)
            start += self.chunk_size - self.chunk_overlap
            
            # Safety break if loop gets stuck (should not happen)
            if self.chunk_size <= self.chunk_overlap:
                break

        return chunks
