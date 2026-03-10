from typing import List
from pgvector.django import CosineDistance
from core.models import DocumentChunk
from core.services.documents.embedding_service import EmbeddingService


class VectorSearchService:
    @staticmethod
    def retrieve_candidates(query: str, candidate_k: int = 20) -> List[DocumentChunk]:
        query = (query or "").strip()
        if not query:
            return []

        query_embedding = EmbeddingService.generate_embedding(query)

        qs = (
            DocumentChunk.objects
            .exclude(embedding__isnull=True)
            .select_related("document")
            .annotate(distance=CosineDistance("embedding", query_embedding))
            .order_by("distance")[:candidate_k]
        )

        return list(qs)