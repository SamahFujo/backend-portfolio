from __future__ import annotations

import re
from typing import List, Tuple, Optional, Dict, Any
from django.conf import settings

from core.models import DocumentChunk
from core.services.retrieval.vector_search_service import VectorSearchService
from core.services.retrieval.rerank_service import RerankService


class RerankedVectorRetrievalService:
    @staticmethod
    def _chunk_to_rerank_text(chunk: DocumentChunk, max_len: int = 1200) -> str:
        """
        Prepare chunk text for reranking.
        """
        text = " ".join((chunk.content or "").split())
        if len(text) > max_len:
            text = text[:max_len].rstrip() + "..."
        return text

    @staticmethod
    def _build_debug_item(
        chunk: DocumentChunk,
        rank: int,
        rerank_score: Optional[float],
        source: str,
        filters: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "rank": rank,
            "chunk_id": str(chunk.id),
            "chunk_index": chunk.chunk_index,
            "doc_title": chunk.document.title,
            "document_type": getattr(chunk.document, "document_type", None),
            "vector_distance": float(getattr(chunk, "distance", 0.0))
            if getattr(chunk, "distance", None) is not None
            else None,
            "rerank_score": float(rerank_score) if rerank_score is not None else None,
            "source": source,
            "matched_filter_document_type": filters.get("document_type") if filters else None,
            "reason": reason,
        }

    @staticmethod
    def _is_broad_capability_query(query: str, filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Detect broad/open-ended capability questions that usually need a lower rerank threshold.
        """
        q = (query or "").strip().lower()

        broad_patterns = [
            r"\bwhat can samah do\b",
            r"\bwhat can samah help with\b",
            r"\bwhat can she do\b",
            r"\bwhat can she help with\b",
            r"\bwhat does samah do\b",
            r"\bwhat kind of work\b",
            r"\bwhat kind of projects\b",
            r"\bwhat are samah'?s strongest\b",
        ]

        if any(re.search(p, q) for p in broad_patterns):
            return True

        if filters and filters.get("document_type") in {"capabilities", "faq", "achievements", "career_timeline"}:
            # broad questions on these doc types often get lower rerank scores
            if len(q.split()) <= 8:
                return True

        return False

    @classmethod
    def _resolve_min_rerank_score(
        cls,
        query: str,
        min_rerank_score: float,
        filters: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Dynamically relax rerank threshold for broad capability/profile questions.
        """
        if cls._is_broad_capability_query(query, filters=filters):
            # Lower threshold for open-ended questions
            return min(min_rerank_score, 0.10)

        return min_rerank_score

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
        min_rerank_score = (
            min_rerank_score if min_rerank_score is not None else settings.RERANK_MIN_SCORE
        )

        adaptive_min_rerank_score = cls._resolve_min_rerank_score(
            query=query,
            min_rerank_score=min_rerank_score,
            filters=filters,
        )

        candidates = VectorSearchService.retrieve_candidates(
            query=query,
            candidate_k=candidate_k,
            filters=filters,
        )

        if not candidates:
            return [], []

        documents_for_rerank = [
            cls._chunk_to_rerank_text(c) for c in candidates
        ]

        rerank_items = RerankService.rerank(
            query=query,
            documents=documents_for_rerank,
            top_n=top_n,
        )

        # If reranker fails/returns empty, fallback to vector top_n
        if not rerank_items:
            fallback = candidates[:top_n]
            debug = [
                cls._build_debug_item(
                    chunk=c,
                    rank=i + 1,
                    rerank_score=None,
                    source="vector_fallback",
                    filters=filters,
                    reason="reranker_empty",
                )
                for i, c in enumerate(fallback)
            ]
            return fallback, debug

        reranked_chunks: List[DocumentChunk] = []
        debug: List[dict] = []
        seen_chunk_ids = set()

        for rank, item in enumerate(rerank_items, start=1):
            if item.score < adaptive_min_rerank_score:
                continue

            idx = item.index
            if 0 <= idx < len(candidates):
                chunk = candidates[idx]

                if chunk.id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk.id)

                reranked_chunks.append(chunk)
                debug.append(
                    cls._build_debug_item(
                        chunk=chunk,
                        rank=rank,
                        rerank_score=float(item.score),
                        source="reranked",
                        filters=filters,
                        reason=f"passed_threshold(min={adaptive_min_rerank_score:.2f})",
                    )
                )

                if len(reranked_chunks) >= top_n:
                    break

        # If threshold removed everything, fallback to best reranked items ignoring threshold
        if not reranked_chunks:
            reranked_chunks = []
            debug = []
            seen_chunk_ids = set()

            for rank, item in enumerate(rerank_items, start=1):
                idx = item.index
                if 0 <= idx < len(candidates):
                    chunk = candidates[idx]

                    if chunk.id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk.id)

                    reranked_chunks.append(chunk)
                    debug.append(
                        cls._build_debug_item(
                            chunk=chunk,
                            rank=rank,
                            rerank_score=float(item.score),
                            source="reranked_no_threshold_fallback",
                            filters=filters,
                            reason=f"threshold_removed_all(min={adaptive_min_rerank_score:.2f})",
                        )
                    )

                if len(reranked_chunks) >= top_n:
                    break

        return reranked_chunks[:top_n], debug[:top_n]
