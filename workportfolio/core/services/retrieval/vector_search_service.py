from typing import List, Optional, Dict, Any
from pgvector.django import CosineDistance
from core.models import DocumentChunk
from core.services.documents.embedding_service import EmbeddingService


class VectorSearchService:
    @staticmethod
    def retrieve_candidates(
        query: str,
        candidate_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        query = (query or "").strip()
        if not query:
            return []

        query_embedding = EmbeddingService.generate_embedding(query)

        qs = (
            DocumentChunk.objects
            .exclude(embedding__isnull=True)
            .select_related("document")
        )

        # ✅ Apply enterprise filters (optional)
        if filters:
            # only active docs (recommended if you add is_active)
            if filters.get("only_active_docs"):
                qs = qs.filter(document__is_active=True)

            # filter by document_type (if you use it)
            if filters.get("document_type"):
                qs = qs.filter(document__document_type=filters["document_type"])

            # filter by doc title contains (simple but very useful)
            if filters.get("document_title_contains"):
                qs = qs.filter(document__title__icontains=filters["document_title_contains"])

            # filter by specific doc IDs
            if filters.get("document_ids"):
                qs = qs.filter(document_id__in=filters["document_ids"])

        qs = (
            qs.annotate(distance=CosineDistance("embedding", query_embedding))
            .order_by("distance")[:candidate_k]
        )

        return list(qs)