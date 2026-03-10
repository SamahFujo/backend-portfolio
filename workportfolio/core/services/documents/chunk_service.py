"""
Chunking service for document text.
"""


class ChunkService:
    """
    Splits raw text into smaller chunks for retrieval.
    """

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
        """
        Split text into overlapping chunks.

        Args:
            text (str): Raw extracted text
            chunk_size (int): Max characters per chunk
            overlap (int): Overlap between chunks

        Returns:
            list[str]: List of chunk strings
        """
        text = (text or "").strip()
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end == text_length:
                break

            start = max(end - overlap, 0)

        return chunks
