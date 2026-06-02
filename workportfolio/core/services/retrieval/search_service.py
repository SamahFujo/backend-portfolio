"""
Simple retrieval service.

Keyword-based retrieval fallback.
Useful for debugging or lightweight retrieval when vector search is unavailable.

Important:
This service must only retrieve approved chatbot chunks.
"""

import re
from typing import List

from django.db.models import Q

from core.models import DocumentChunk
from core.services.retrieval.approved_chunks import get_chatbot_available_chunks


class SearchService:
    """
    Service responsible for retrieving relevant chunks from approved documents.
    """

    STOP_WORDS = {
        "what", "is", "are", "the", "a", "an", "of", "to", "and", "or",
        "in", "on", "for", "with", "does", "do", "did", "can", "could",
        "should", "has", "have", "had", "samah",
    }

    @classmethod
    def _extract_keywords(cls, query: str) -> List[str]:
        """
        Extract useful keywords from the user's query.
        """

        words = re.findall(r"\b\w+\b", (query or "").lower())
        keywords = [w for w in words if len(w) > 2 and w not in cls.STOP_WORDS]
        return keywords[:8]

    @classmethod
    def retrieve_relevant_chunks(cls, query: str, limit: int = 5) -> List[DocumentChunk]:
        """
        Retrieve chunks using simple keyword matching.

        This method only searches approved, active, chatbot-safe chunks.
        """

        query = (query or "").strip()

        if not query:
            return []

        limit = max(1, int(limit or 5))
        keywords = cls._extract_keywords(query)

        base_qs = get_chatbot_available_chunks()

        if not keywords:
            return list(
                base_qs.filter(content__icontains=query)[:limit]
            )

        q_obj = Q()

        for keyword in keywords:
            q_obj |= Q(content__icontains=keyword)

        return list(
            base_qs.filter(q_obj)[:limit]
        )
