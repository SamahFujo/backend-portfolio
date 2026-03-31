from typing import List, Optional, Dict, Any
from pgvector.django import CosineDistance
from core.models import DocumentChunk
from core.services.documents.embedding_service import EmbeddingService


class VectorSearchService:
    @staticmethod
    def _build_queryset(
        query_embedding,
        filters: Optional[Dict[str, Any]] = None,
    ):
        qs = (
            DocumentChunk.objects
            .exclude(embedding__isnull=True)
            .select_related("document")
        )

        if filters:
            # only active docs (only keep this if your model has is_active)
            if filters.get("only_active_docs"):
                qs = qs.filter(document__is_active=True)

            # filter by document_type
            if filters.get("document_type"):
                qs = qs.filter(
                    document__document_type=filters["document_type"])

            # filter by title contains
            if filters.get("document_title_contains"):
                qs = qs.filter(
                    document__title__icontains=filters["document_title_contains"]
                )

            # filter by specific document IDs
            if filters.get("document_ids"):
                qs = qs.filter(document_id__in=filters["document_ids"])

        return qs.annotate(distance=CosineDistance("embedding", query_embedding)).order_by("distance")

    @classmethod
    def retrieve_candidates(
        cls,
        query: str,
        candidate_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        query = (query or "").strip()
        if not query:
            return []

        candidate_k = max(1, int(candidate_k or 20))

        query_embedding = EmbeddingService.generate_embedding(
            query,
            task="retrieval.query",
        )

        # First pass: strict filtered retrieval
        qs = cls._build_queryset(
            query_embedding=query_embedding, filters=filters)
        results = list(qs[:candidate_k])

        # Fallback: relax strict type/title filters if nothing found
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
