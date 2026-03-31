"""
Simple retrieval service.

Keyword-based retrieval fallback.
Useful for debugging or lightweight retrieval when vector search is unavailable.
"""

import re
from typing import List
from django.db.models import Q

from core.models import DocumentChunk


class SearchService:
    """
    Service responsible for retrieving relevant chunks from stored documents.
    """

    STOP_WORDS = {
        "what", "is", "are", "the", "a", "an", "of", "to", "and", "or",
        "in", "on", "for", "with", "does", "do", "did", "can", "could",
        "should", "has", "have", "had", "samah",
    }

    @classmethod
    def _extract_keywords(cls, query: str) -> List[str]:
        words = re.findall(r"\b\w+\b", (query or "").lower())
        keywords = [w for w in words if len(w) > 2 and w not in cls.STOP_WORDS]
        return keywords[:8]

    @classmethod
    def retrieve_relevant_chunks(cls, query: str, limit: int = 5) -> List[DocumentChunk]:
        """
        Retrieve chunks using simple keyword matching.

        Args:
            query (str): User's question
            limit (int): Maximum number of chunks to return

        Returns:
            List[DocumentChunk]: Matching chunks
        """
        query = (query or "").strip()
        if not query:
            return []

        keywords = cls._extract_keywords(query)

        if not keywords:
            return list(
                DocumentChunk.objects.filter(content__icontains=query)[:limit]
            )

        q_obj = Q()
        for kw in keywords:
            q_obj |= Q(content__icontains=kw)

        return list(
            DocumentChunk.objects
            .filter(q_obj)
            .select_related("document")[:limit]
        )
