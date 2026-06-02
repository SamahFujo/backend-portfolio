from typing import List, Optional, Dict, Any

from pgvector.django import CosineDistance

from core.models import DocumentChunk
from core.services.documents.embedding_service import EmbeddingService
from core.services.retrieval.approved_chunks import get_chatbot_available_chunks


class VectorSearchService:
    """
    Vector retrieval service.

    This service performs semantic search only over approved chatbot chunks.
    """

    @staticmethod
    def _build_queryset(
        query_embedding,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """
        Build the safe vector-search queryset.

        Important:
        The base queryset already enforces:
        - approved document
        - active document
        - chatbot-available document
        - active chunk
        - embedded chunk
        - passed/warning quality status
        """

        qs = get_chatbot_available_chunks()

        if filters:
            # This is kept for backward compatibility.
            # The approved-chunks helper already enforces active approved docs.
            if filters.get("only_active_docs"):
                qs = qs.filter(
                    document__is_active=True,
                    document__is_approved=True,
                    document__is_available_for_chatbot=True,
                )

            # Hard filter by document type.
            if filters.get("document_type"):
                qs = qs.filter(
                    document__document_type=filters["document_type"]
                )

            # Filter by title contains.
            if filters.get("document_title_contains"):
                qs = qs.filter(
                    document__title__icontains=filters["document_title_contains"]
                )

            # Filter by specific document IDs.
            if filters.get("document_ids"):
                qs = qs.filter(document_id__in=filters["document_ids"])

        return (
            qs.annotate(distance=CosineDistance("embedding", query_embedding))
            .order_by("distance")
        )

    @classmethod
    def retrieve_candidates(
        cls,
        query: str,
        candidate_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Retrieve vector candidates from approved chatbot-safe chunks only.
        """

        query = (query or "").strip()

        if not query:
            return []

        candidate_k = max(1, int(candidate_k or 20))

        query_embedding = EmbeddingService.generate_embedding(
            query,
            task="retrieval.query",
        )

        # First pass: strict filtered retrieval.
        qs = cls._build_queryset(
            query_embedding=query_embedding,
            filters=filters,
        )
        results = list(qs[:candidate_k])

        # Fallback: relax strict type/title filters if nothing found.
        # This fallback is still safe because _build_queryset always starts
        # from get_chatbot_available_chunks().
        if not results and filters:
            relaxed_filters = dict(filters)
            relaxed_filters.pop("document_type", None)
            relaxed_filters.pop("document_title_contains", None)

            qs = cls._build_queryset(
                query_embedding=query_embedding,
                filters=relaxed_filters,
            )
            results = list(qs[:candidate_k])

        return results
