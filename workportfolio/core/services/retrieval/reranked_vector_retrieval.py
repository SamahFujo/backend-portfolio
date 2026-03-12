from __future__ import annotations

from typing import List, Tuple
from django.conf import settings

from core.models import DocumentChunk
from core.services.retrieval.vector_search_service import VectorSearchService
from core.services.retrieval.rerank_service import RerankService
from typing import List, Tuple, Optional, Dict, Any


class RerankedVectorRetrievalService:
    @staticmethod
    def _chunk_to_rerank_text(chunk: DocumentChunk, max_len: int = 1200) -> str:
        text = (chunk.content or "").strip()
        if len(text) > max_len:
            text = text[:max_len].rstrip() + "..."
        return text

    @classmethod
    def retrieve_relevant_chunks(
        cls,
        query: str,
        candidate_k: int | None = None,
        top_n: int | None = None,
        min_rerank_score: float | None = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[DocumentChunk], List[dict]]:
        """
        Returns:
        - chunks: reranked (and threshold-filtered) chunks
        - retrieval_debug: debug info including vector distance + rerank score
        """
        candidate_k = candidate_k if candidate_k is not None else settings.RERANK_CANDIDATE_K
        top_n = top_n if top_n is not None else settings.RERANK_TOP_N
        min_rerank_score = min_rerank_score if min_rerank_score is not None else settings.RERANK_MIN_SCORE

        candidates = VectorSearchService.retrieve_candidates(
            query=query,
            candidate_k=candidate_k,
            filters=filters
        )
        if not candidates:
            return [], []

        documents_for_rerank = [
            cls._chunk_to_rerank_text(c) for c in candidates]
        rerank_items = RerankService.rerank(
            query=query, documents=documents_for_rerank, top_n=top_n)

        # If reranker fails/returns empty, fallback to vector top_n
        if not rerank_items:
            fallback = candidates[:top_n]
            debug = [
                {
                    "rank": i + 1,
                    "chunk_id": str(c.id),
                    "chunk_index": c.chunk_index,
                    "doc_title": c.document.title,
                    "vector_distance": float(getattr(c, "distance", 0.0)) if getattr(c, "distance", None) is not None else None,
                    "rerank_score": None,
                    "source": "vector_fallback",
                }
                for i, c in enumerate(fallback)
            ]
            return fallback, debug

        reranked_chunks: List[DocumentChunk] = []
        debug: List[dict] = []

        for rank, item in enumerate(rerank_items, start=1):
            if item.score < min_rerank_score:
                continue

            idx = item.index
            if 0 <= idx < len(candidates):
                chunk = candidates[idx]
                reranked_chunks.append(chunk)
                debug.append(
                    {
                        "rank": rank,
                        "chunk_id": str(chunk.id),
                        "chunk_index": chunk.chunk_index,
                        "doc_title": chunk.document.title,
                        "vector_distance": float(getattr(chunk, "distance", 0.0)) if getattr(chunk, "distance", None) is not None else None,
                        "rerank_score": float(item.score),
                        "source": "reranked",
                    }
                )

        # If threshold removed everything, fallback to best reranked items (top_n) ignoring threshold
        if not reranked_chunks:
            reranked_chunks = []
            debug = []
            for rank, item in enumerate(rerank_items, start=1):
                idx = item.index
                if 0 <= idx < len(candidates):
                    chunk = candidates[idx]
                    reranked_chunks.append(chunk)
                    debug.append(
                        {
                            "rank": rank,
                            "chunk_id": str(chunk.id),
                            "chunk_index": chunk.chunk_index,
                            "doc_title": chunk.document.title,
                            "vector_distance": float(getattr(chunk, "distance", 0.0)) if getattr(chunk, "distance", None) is not None else None,
                            "rerank_score": float(item.score),
                            "source": "reranked_no_threshold_fallback",
                        }
                    )
                if len(reranked_chunks) >= top_n:
                    break

        return reranked_chunks[:top_n], debug[:top_n]
