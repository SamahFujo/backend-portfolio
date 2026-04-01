from __future__ import annotations

from typing import Dict, Any, List, Optional

from core.models import DocumentChunk
from core.services.retrieval.scope_resolver import ScopeResolver
from core.services.retrieval.reranked_vector_retrieval import RerankedVectorRetrievalService
from core.services.chatbot.extractors import (
    try_extract_contact,
    try_extract_preferences,
    try_extract_availability,
    try_extract_skills,
)
from core.services.chatbot.gemini_grounded_answerer import GeminiGroundedAnswerer


class ProfileQAService:
    """
    Main orchestration service for profile/document-grounded Q&A.

    Flow:
    1. Resolve scope filters from the ORIGINAL user question
    2. Retrieve relevant chunks using retrieval_query
    3. If retrieval is weak and scope is broad, retry without filters
    4. Try deterministic extractors first
    5. Fall back to Gemini grounded answering
    """

    STRICT_DOCUMENT_TYPES = {
        "cv",
        "preferences",
        "compensation",
        "capabilities",
        "achievements",
        "career_timeline",
    }

    @staticmethod
    def _build_used_sources(
        chunks: List[DocumentChunk],
        max_items: int = 3,
    ) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for c in chunks[:max_items]:
            sources.append({
                "doc_title": c.document.title,
                "document_type": getattr(c.document, "document_type", None),
                "chunk_index": c.chunk_index,
                "chunk_id": str(c.id),
                "document_id": str(c.document_id),
            })
        return sources

    @classmethod
    def _is_strict_scope(cls, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return False
        doc_type = filters.get("document_type")
        return doc_type in cls.STRICT_DOCUMENT_TYPES

    @classmethod
    def _augment_filters_for_precision(
        cls,
        question: str,
        filters: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Tighten certain filters for better retrieval precision.
        Especially useful for CV/contact queries.
        """
        if not filters:
            return filters

        updated = dict(filters)
        q = (question or "").lower()

        if updated.get("document_type") == "cv":
            # Help retrieval prefer actual CV/resume docs
            if any(k in q for k in ["contact", "email", "phone", "linkedin", "reach", "get in touch"]):
                updated["document_title_contains"] = "cv"

        return updated

    @classmethod
    def _try_deterministic_answer(
        cls,
        question: str,
        chunks: List[DocumentChunk],
    ) -> Optional[Dict[str, Any]]:
        extractors = [
            try_extract_contact,
            try_extract_preferences,
            try_extract_availability,
            try_extract_skills,
        ]

        for extractor in extractors:
            handled, answer, confidence_boost = extractor(question, chunks)
            if handled:
                return {
                    "verdict": "supported",
                    "answer": answer,
                    "bullets": [],
                    "used_chunk_indices": list(range(min(len(chunks), 3))),
                    "used_sources": cls._build_used_sources(chunks, max_items=3),
                    "meta": {
                        "model_used": None,
                        "tried_models": [],
                        "error": None,
                        "answer_source": "deterministic_extractor",
                        "extractor_used": extractor.__name__,
                        "confidence_boost": confidence_boost,
                    },
                }

        return None

    @classmethod
    def _retrieve_chunks(
        cls,
        question: str,
        retrieval_query: str,
    ) -> tuple[list[DocumentChunk], list[dict], Optional[Dict[str, Any]]]:
        """
        Retrieve evidence with scope-aware filtering first.

        Important rule:
        - For strict scopes like CV/contact, keep narrow retrieval if it returns anything.
        - Only broaden to all docs for broad/weak scopes.
        """
        filters = ScopeResolver.resolve_filters(question)
        filters = cls._augment_filters_for_precision(question, filters)

        chunks, retrieval_debug = RerankedVectorRetrievalService.retrieve_relevant_chunks(
            query=retrieval_query,
            filters=filters,
        )

        # If strict scope is already applied and we got any chunks, DO NOT broaden.
        if cls._is_strict_scope(filters) and chunks:
            return chunks, retrieval_debug, filters

        # For broad scope or weak/no scope, fallback to all docs if retrieval is too weak
        if not chunks or len(chunks) < 3:
            fallback_chunks, fallback_debug = RerankedVectorRetrievalService.retrieve_relevant_chunks(
                query=retrieval_query,
                filters=None,
            )

            if fallback_chunks:
                chunks = fallback_chunks
                retrieval_debug = fallback_debug

        return chunks, retrieval_debug, filters

    @classmethod
    def answer_question(
        cls,
        question: str,
        retrieval_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        question = (question or "").strip()
        retrieval_query = (retrieval_query or question).strip()

        chunks, retrieval_debug, filters = cls._retrieve_chunks(
            question=question,
            retrieval_query=retrieval_query,
        )

        deterministic_result = cls._try_deterministic_answer(question, chunks)
        if deterministic_result:
            deterministic_result["retrieval_query"] = retrieval_query
            deterministic_result["rewrite_notes"] = "local_fast_path"
            deterministic_result["retrieval_debug"] = retrieval_debug
            deterministic_result["applied_filters"] = filters
            return deterministic_result

        result = GeminiGroundedAnswerer.answer(
            question=question, evidence_chunks=chunks)
        result["retrieval_query"] = retrieval_query
        result["rewrite_notes"] = "local_fast_path"
        result["retrieval_debug"] = retrieval_debug
        result["applied_filters"] = filters
        return result
