import hashlib
from typing import List, Dict, Any
from app.ingestion.chunkers.base import BaseChunker
from app.models.chunk import Chunk
from app.models.document import RawDocument


class StructureAwareChunker(BaseChunker):
    """Splits raw markdown documents into chunks preserving section headers and layout structure."""

    def __init__(self, max_chunk_size: int = 1200, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def _split_large_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Sub-splits a large text block into smaller overlapping pieces."""
        text_len = len(text)
        if text_len <= chunk_size:
            return [text]

        sub_chunks: List[str] = []
        start = 0
        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                last_space = text[start:end].rfind(" ")
                if last_space != -1 and (end - start - last_space) < 50:
                    end = start + last_space
            
            sub_text = text[start:end].strip()
            if sub_text:
                sub_chunks.append(sub_text)
            
            start += chunk_size - overlap
            if chunk_size <= overlap:
                break
        return sub_chunks

    def chunk(self, doc: RawDocument) -> List[Chunk]:
        """Splits markdown based on headers, tracking code block state to avoid false header splits."""
        text = doc.text
        if not text:
            return []

        lines = text.split("\n")
        sections: List[Dict[str, Any]] = []
        
        current_heading = doc.metadata.get("title", "Introduction")
        current_section_lines: List[str] = []
        
        in_code_block = False

        for line in lines:
            # Toggle code block flag
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_section_lines.append(line)
                continue

            # Check if this line is a heading (not inside code block)
            is_heading = False
            if not in_code_block:
                stripped = line.strip()
                if stripped.startswith("#"):
                    # Count '#' symbols to verify it is a valid heading structure
                    heading_match = stripped.split(" ", 1)
                    if len(heading_match) == 2 and all(char == "#" for char in heading_match[0]):
                        is_heading = True
                        new_heading = heading_match[1].strip()

            if is_heading:
                # Save previous section if it has content
                section_text = "\n".join(current_section_lines).strip()
                if section_text:
                    sections.append({
                        "heading": current_heading,
                        "text": section_text
                    })
                
                # Start new section
                current_heading = new_heading
                current_section_lines = [line]  # Keep the heading in the chunk context
            else:
                current_section_lines.append(line)

        # Append last section
        section_text = "\n".join(current_section_lines).strip()
        if section_text:
            sections.append({
                "heading": current_heading,
                "text": section_text
            })

        # Process and build Chunks
        chunks: List[Chunk] = []
        chunk_idx = 0

        for sect in sections:
            sect_text = sect["text"]
            sect_heading = sect["heading"]

            # If section is too large, split it sub-structurally
            if len(sect_text) > self.max_chunk_size:
                sub_texts = self._split_large_text(sect_text, chunk_size=800, overlap=150)
                for sub_text in sub_texts:
                    chunk_id = hashlib.md5(
                        f"structure_aware:{doc.source_file}:{chunk_idx}:{sub_text}".encode("utf-8")
                    ).hexdigest()
                    
                    chunk_meta = doc.metadata.copy()
                    chunk_meta["chunk_index"] = chunk_idx
                    
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=sub_text,
                            source_file=doc.source_file,
                            section_heading=sect_heading,
                            chunking_strategy="structure_aware",
                            metadata=chunk_meta
                        )
                    )
                    chunk_idx += 1
            else:
                # If section is too small, check if we can group it with previous or just emit
                # For simplicity, we emit it directly if it meets min size, or if it's the only chunk
                if len(sect_text) >= self.min_chunk_size or not chunks:
                    chunk_id = hashlib.md5(
                        f"structure_aware:{doc.source_file}:{chunk_idx}:{sect_text}".encode("utf-8")
                    ).hexdigest()
                    
                    chunk_meta = doc.metadata.copy()
                    chunk_meta["chunk_index"] = chunk_idx

                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=sect_text,
                            source_file=doc.source_file,
                            section_heading=sect_heading,
                            chunking_strategy="structure_aware",
                            metadata=chunk_meta
                        )
                    )
                    chunk_idx += 1
                elif chunks:
                    # Merge with the previous chunk if it's too small
                    prev_chunk = chunks[-1]
                    if prev_chunk.section_heading == sect_heading:
                        merged_text = prev_chunk.text + "\n\n" + sect_text
                        prev_chunk.text = merged_text
                        # Update ID of the merged chunk
                        prev_chunk.id = hashlib.md5(
                            f"structure_aware:{doc.source_file}:{prev_chunk.metadata['chunk_index']}:{merged_text}".encode("utf-8")
                        ).hexdigest()
                    else:
                        # Fallback: emit anyway to avoid cross-section leakage
                        chunk_id = hashlib.md5(
                            f"structure_aware:{doc.source_file}:{chunk_idx}:{sect_text}".encode("utf-8")
                        ).hexdigest()
                        
                        chunk_meta = doc.metadata.copy()
                        chunk_meta["chunk_index"] = chunk_idx

                        chunks.append(
                            Chunk(
                                id=chunk_id,
                                text=sect_text,
                                source_file=doc.source_file,
                                section_heading=sect_heading,
                                chunking_strategy="structure_aware",
                                metadata=chunk_meta
                            )
                        )
                        chunk_idx += 1

        return chunks
