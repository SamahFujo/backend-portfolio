"""
Simple retrieval service.

This is the first version:
- retrieves document chunks using keyword matching
- later we will replace or enhance this with vector search
"""

from typing import List
from core.models import DocumentChunk


class SearchService:
    """
    Service responsible for retrieving relevant chunks from stored documents.
    """

    @staticmethod
    def retrieve_relevant_chunks(query: str, limit: int = 5) -> List[DocumentChunk]:
        """
        Retrieve chunks using simple case-insensitive keyword matching.

        Args:
            query (str): User's question
            limit (int): Maximum number of chunks to return

        Returns:
            List[DocumentChunk]: Matching chunks
        """
        query = query.strip()
        if not query:
            return []

        # Very simple initial matching strategy.
        # We will later replace this with semantic/vector search.
        return list(
            DocumentChunk.objects.filter(content__icontains=query)[:limit]
        )
