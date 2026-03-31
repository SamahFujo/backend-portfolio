"""
Chunking service for document text.
"""

import re
from typing import List


class ChunkService:
    """
    Splits raw text into smaller chunks for retrieval.
    Prefers paragraph-aware chunking over blind character slicing.
    """

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize line endings and extra whitespace while preserving paragraph breaks.
        """
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return ""

        # remove excessive spaces/tabs inside lines
        text = re.sub(r"[ \t]+", " ", text)

        # collapse too many blank lines, but preserve paragraph separation
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """
        Split text into paragraphs using blank lines first.
        Falls back to line-based splitting if needed.
        """
        if not text:
            return []

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        # fallback: if everything is one giant paragraph, split by lines
        if len(paragraphs) <= 1:
            paragraphs = [line.strip() for line in text.split("\n") if line.strip()]

        return paragraphs

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 800,
        overlap: int = 100,
        min_chunk_size: int = 80,
    ) -> list[str]:
        """
        Split text into semantically better overlapping chunks.

        Strategy:
        - normalize text
        - split into paragraphs
        - accumulate paragraphs until chunk_size is reached
        - add lightweight overlap from the previous chunk tail

        Args:
            text (str): Raw extracted text
            chunk_size (int): Approximate max characters per chunk
            overlap (int): Approximate overlap between chunks
            min_chunk_size (int): Skip very tiny chunks unless they are the only content

        Returns:
            list[str]: List of chunk strings
        """
        text = ChunkService._normalize_text(text)
        if not text:
            return []

        paragraphs = ChunkService._split_paragraphs(text)
        if not paragraphs:
            return []

        chunks: List[str] = []
        current_parts: List[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If a single paragraph is too large, split it safely by sentences / hard slices
            if len(para) > chunk_size:
                if current_parts:
                    chunk = "\n\n".join(current_parts).strip()
                    if chunk and (len(chunk) >= min_chunk_size or not chunks):
                        chunks.append(chunk)
                    current_parts = []
                    current_len = 0

                sentence_parts = re.split(r"(?<=[.!?])\s+", para)
                temp = ""

                for part in sentence_parts:
                    part = part.strip()
                    if not part:
                        continue

                    candidate = f"{temp} {part}".strip() if temp else part
                    if len(candidate) <= chunk_size:
                        temp = candidate
                    else:
                        if temp:
                            chunks.append(temp.strip())
                        # if one sentence itself is still too long, hard split it
                        if len(part) > chunk_size:
                            start = 0
                            while start < len(part):
                                end = min(start + chunk_size, len(part))
                                sub = part[start:end].strip()
                                if sub:
                                    chunks.append(sub)
                                start = max(end - overlap, end)
                            temp = ""
                        else:
                            temp = part

                if temp:
                    chunks.append(temp.strip())

                continue

            candidate_len = current_len + (2 if current_parts else 0) + len(para)

            if candidate_len <= chunk_size:
                current_parts.append(para)
                current_len = candidate_len
            else:
                chunk = "\n\n".join(current_parts).strip()
                if chunk and (len(chunk) >= min_chunk_size or not chunks):
                    chunks.append(chunk)

                # create paragraph-aware overlap
                overlap_text = ""
                if chunks and overlap > 0:
                    prev_chunk = chunks[-1]
                    overlap_text = prev_chunk[-overlap:].strip()

                if overlap_text:
                    current_parts = [overlap_text, para]
                    current_len = len(overlap_text) + 2 + len(para)
                else:
                    current_parts = [para]
                    current_len = len(para)

        # flush last chunk
        if current_parts:
            chunk = "\n\n".join(current_parts).strip()
            if chunk and (len(chunk) >= min_chunk_size or not chunks):
                chunks.append(chunk)

        # final dedup / cleanup
        cleaned_chunks: List[str] = []
        seen = set()

        for chunk in chunks:
            normalized = chunk.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned_chunks.append(normalized)

        return cleaned_chunks