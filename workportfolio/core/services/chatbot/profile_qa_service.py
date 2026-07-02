from __future__ import annotations

from typing import Dict, Any, List, Optional

from core.models import DocumentChunk
from core.services.retrieval.scope_resolver import ScopeResolver
from core.services.retrieval.reranked_vector_retrieval import RerankedVectorRetrievalService
from core.services.retrieval.approved_chunks import get_chatbot_available_chunks
from core.services.chatbot.extractors import (

    try_extract_contact,
    try_extract_preferences,
    try_extract_experience_duration,
    try_extract_strengths,
    try_extract_capability_with_tool,
    try_extract_project_fit,
    try_extract_projects_by_technology,
    try_extract_project_list,
    try_extract_skills,
    try_extract_availability,

)
from core.services.chatbot.grounded_answerer import GroundedAnswerer



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

    CHATBOT_PROFILE = {
        "name": "Samah.ai Assistant",
        "purpose": (
            "An AI portfolio assistant that helps visitors learn about "
            "Samah’s experience, projects, skills, and profile."
        ),
        "developer_hint": (
            "Built as part of the Samah.ai portfolio experience."
        ),
        "language_support": "English",
    }

    AGGREGATION_SHAPES = {"list", "timeline", "summary", "comparison"}

    @staticmethod
    def _is_experience_duration_question(question: str) -> bool:
        q = (question or "").strip().lower()

        direct_markers = [
            "how many years of experience",
            "how much experience",
            "how long has she worked",
            "how long did she work",
            "what is her total experience",
            "how many years has she worked",
            "years of experience",
            "total experience",
            "overall experience",
            "experience in total",
            "total years",
            "how many years she have",
            "how many years of experience she have",
        ]
        if any(marker in q for marker in direct_markers):
            return True

        # tolerate common misspellings / rough phrasing
        has_years = "year" in q or "years" in q
        has_experience_like = any(
            word in q for word in ["experience", "expirience", "experiance", "exp"]
        )
        has_work_like = any(
            phrase in q for phrase in ["she have", "she has", "she worked", "she work", "worked"]
        )

        return (has_years and has_experience_like) or (has_experience_like and has_work_like)

    @classmethod
    def _answer_identity_question(cls, question: str) -> Dict[str, Any]:
        """Answer identity-related questions about the chatbot.

        Args:
            question (str): The user's question.

        Returns:
            Dict[str, Any]: The chatbot's response.
        """
        q = (question or "").strip().lower()

        if "who developed you" in q or "who built you" in q or "who made you" in q or "who develop you" in q:
            answer = (
                "I am part of the Samah.ai portfolio experience and appear to be "
                "built as an interactive assistant for presenting Samah’s profile, "
                "projects, skills, and experience."
            )
        elif "what are you" in q or "who are you" in q:
            answer = (
                "I’m an AI portfolio assistant for Samah.ai. "
                "I help answer questions about Samah’s experience, projects, skills, "
                "and professional profile."
            )
        elif "what can you do" in q or "what do you do" in q:
            answer = (
                "I can answer questions about Samah’s background, technical skills, "
                "projects, experience, preferences, and contact-related details based "
                "on the available portfolio documents and assistant logic."
            )
        else:
            answer = (
                f"{cls.CHATBOT_PROFILE['name']} is {cls.CHATBOT_PROFILE['purpose']} "
                f"It appears to be {cls.CHATBOT_PROFILE['developer_hint']}"
            )

        return {
            "verdict": "supported",
            "answer": answer,
            "bullets": [],
            "used_sources": [],
            "meta": {
                "model_used": None,
                "tried_models": [],
                "provider_used": "chatbot_profile",
                "fallback_used": False,
                "generation_ok": True,
                "safe_fallback": False,
                "error": None,
                "answer_source": "chatbot_profile",
                "extractor_used": None,
                "confidence_boost": 0.0,
            },
            "retrieval_debug": [],
            "applied_filters": None,
            "debug_chunks_before_llm": [],
        }

    @staticmethod
    def _is_current_work_status_question(question: str) -> bool:
        q = (question or "").strip().lower()

        markers = [
            "working now",
            "not working now",
            "currently working",
            "currently employed",
            "current job",
            "current role",
            "still working",
            "is she working",
            "she is not working",
            "she is unemployed",
            "employment status",
            "work status",
            "available now",
            "open to work",
            "open to opportunities",
        ]

        return any(marker in q for marker in markers)

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
            if any(k in q for k in [
                "contact",
                "email",
                "phone",
                "linkedin",
                "reach",
                "get in touch",
                "communicate",
                "connect",
                "talk to her",
                "speak with her",
            ]):
                updated["document_title_contains"] = "cv"

        return updated

    @staticmethod
    def _is_project_list_question(question: str) -> bool:
        q = (question or "").strip().lower()

        markers = [
            "what projects",
            "which projects",
            "list projects",
            "projects samah worked",
            "what project samah worked",
            "what projects did she work on",
            "worked on",
            "projects has she worked on",
        ]

        return any(marker in q for marker in markers)

    @classmethod
    def _try_deterministic_answer(
        cls,
        question: str,
        chunks: List[DocumentChunk],
    ) -> Optional[Dict[str, Any]]:
        extractors = [
            try_extract_contact,
            try_extract_preferences,
            try_extract_experience_duration,
            try_extract_strengths,
            try_extract_availability,
            try_extract_capability_with_tool,
            try_extract_project_fit,
            try_extract_projects_by_technology,
            try_extract_project_list,
            try_extract_skills,
            try_extract_availability
        ]

        for extractor in extractors:
            handled, answer, confidence_boost = extractor(question, chunks)
            if handled and answer and confidence_boost > 0:
                return {
                    "verdict": "supported",
                    "answer": answer,
                    "bullets": [],
                    "used_chunk_indices": list(range(min(len(chunks), 3))),
                    "used_sources": cls._build_used_sources(chunks, max_items=3),
                    "meta": {
                        "model_used": None,
                        "tried_models": [],
                        "provider_used": "deterministic_extractor",
                        "fallback_used": False,
                        "generation_ok": True,
                        "safe_fallback": False,
                        "error": None,
                        "answer_source": "deterministic_extractor",
                        "extractor_used": extractor.__name__,
                        "confidence_boost": confidence_boost,
                    },
                }

        return None
    
    
    @staticmethod
    def _boost_and_sort_chunks(
        chunks: List[DocumentChunk],
        retrieval_debug: List[Dict[str, Any]],
        query_plan: Dict[str, Any],
    ) -> tuple[List[DocumentChunk], List[Dict[str, Any]]]:
        """
        Re-rank retrieved chunks using lightweight document-type boosting.

        This avoids repeated retrieval attempts while still guiding the system
        toward the most relevant document groups.
        """
        if not chunks or not retrieval_debug:
            return chunks, retrieval_debug

        preferred_list = list(query_plan.get("preferred_document_types") or [])
        preferred = set(preferred_list)
        avoid = set(query_plan.get("avoid_document_types") or [])
        question_shape = (query_plan.get("question_shape") or "fact").strip().lower()

        if not preferred and not avoid:
            return chunks, retrieval_debug

        chunk_by_id = {str(chunk.id): chunk for chunk in chunks}
        scored_items = []
        preferred_rank = {doc_type: idx for idx, doc_type in enumerate(preferred_list)}

        for row in retrieval_debug:
            chunk_id = str(row.get("chunk_id"))
            chunk = chunk_by_id.get(chunk_id)

            if not chunk:
                continue

            doc_type = getattr(chunk.document, "document_type", None)
            base_score = float(row.get("rerank_score") or 0.0)
            final_score = base_score

            boost = 0.0
            penalty = 0.0

            if doc_type in preferred:
                order_index = preferred_rank.get(doc_type, len(preferred_rank))
                boost = max(0.10, 0.28 - (order_index * 0.06))

                if question_shape in cls.AGGREGATION_SHAPES:
                    boost += 0.05

                final_score += boost

            if doc_type in avoid:
                penalty = 0.30
                final_score -= penalty

            updated_row = dict(row)
            updated_row["query_plan_answer_type"] = query_plan.get("answer_type")
            updated_row["base_rerank_score"] = base_score
            updated_row["final_score"] = final_score
            updated_row["doc_type_boost"] = boost
            updated_row["doc_type_penalty"] = penalty
            updated_row["preferred_document_types"] = list(preferred)
            updated_row["avoid_document_types"] = list(avoid)

            scored_items.append((final_score, chunk, updated_row))

        if not scored_items:
            return chunks, retrieval_debug

        scored_items.sort(key=lambda item: item[0], reverse=True)

        sorted_chunks = [item[1] for item in scored_items]
        sorted_debug = [item[2] for item in scored_items]

        return sorted_chunks, sorted_debug

    @staticmethod
    def _merge_chunk_results(
        retrieval_sets: List[tuple[List[DocumentChunk], List[Dict[str, Any]]]],
    ) -> tuple[List[DocumentChunk], List[Dict[str, Any]]]:
        merged_chunks: List[DocumentChunk] = []
        merged_debug: List[Dict[str, Any]] = []
        seen_chunk_ids = set()

        for chunks, debug_rows in retrieval_sets:
            debug_by_chunk_id = {
                str(row.get("chunk_id")): row for row in (debug_rows or [])
            }

            for chunk in chunks or []:
                chunk_id = str(chunk.id)
                if chunk_id in seen_chunk_ids:
                    continue

                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(chunk)

                if chunk_id in debug_by_chunk_id:
                    merged_debug.append(debug_by_chunk_id[chunk_id])

        return merged_chunks, merged_debug

    @classmethod
    def _should_expand_preferred_retrieval(
        cls,
        question: str,
        query_plan: Dict[str, Any],
    ) -> bool:
        shape = (query_plan.get("question_shape") or "").strip().lower()
        if shape in cls.AGGREGATION_SHAPES:
            return True

        q = (question or "").strip().lower()
        return any(marker in q for marker in [
            "which", "what are", "list", "all ", "timeline",
            "history", "companies", "projects", "roles",
        ])

    @classmethod
    def _retrieve_chunks(
        cls,
        question: str,
        retrieval_query: str,
        question_route: Optional[str] = None,
        query_plan: Optional[Dict[str, Any]] = None,
    ) -> tuple[list[DocumentChunk], list[dict], Optional[Dict[str, Any]]]:
        """
        Single-pass retrieval.

        Strategy:
        1. Use conservative ScopeResolver filters only for direct document-specific questions.
        2. Retrieve globally, then expand across preferred document families for broad questions.
        3. Apply query-plan document-type boosting/penalty.
        """
        query_plan = query_plan or {}

        filters = ScopeResolver.resolve_filters(question, route=question_route)
        filters = cls._augment_filters_for_precision(question, filters)

        retrieval_sets: List[tuple[List[DocumentChunk], List[dict]]] = []

        base_chunks, base_debug = RerankedVectorRetrievalService.retrieve_relevant_chunks(
            query=retrieval_query,
            filters=filters,
        )
        retrieval_sets.append((base_chunks, base_debug))

        preferred_doc_types = list(query_plan.get("preferred_document_types") or [])
        expand_preferred = cls._should_expand_preferred_retrieval(
            question=question,
            query_plan=query_plan,
        )

        if expand_preferred and preferred_doc_types:
            for doc_type in preferred_doc_types[:3]:
                scoped_filters = dict(filters or {})
                scoped_filters["document_type"] = doc_type

                scoped_chunks, scoped_debug = (
                    RerankedVectorRetrievalService.retrieve_relevant_chunks(
                        query=retrieval_query,
                        filters=scoped_filters,
                    )
                )

                if scoped_chunks:
                    retrieval_sets.append((scoped_chunks, scoped_debug))

        chunks, retrieval_debug = cls._merge_chunk_results(retrieval_sets)

        chunks, retrieval_debug = cls._boost_and_sort_chunks(
            chunks=chunks,
            retrieval_debug=retrieval_debug,
            query_plan=query_plan,
        )

        applied_filters = {
            **(filters or {"only_active_docs": True}),
            "query_plan": {
                "answer_type": query_plan.get("answer_type"),
                "question_shape": query_plan.get("question_shape"),
                "preferred_document_types": query_plan.get("preferred_document_types") or [],
                "avoid_document_types": query_plan.get("avoid_document_types") or [],
                "needs_document_retrieval": query_plan.get("needs_document_retrieval", True),
                "source": query_plan.get("source"),
                "expanded_preferred_retrieval": expand_preferred,
            },
        }

        return chunks, retrieval_debug, applied_filters

    @classmethod
    def _answer_compensation_question(
        cls,
        question: str,
        chunks: List[DocumentChunk],
    ) -> Dict[str, Any]:
        """
        Compensation questions should not be answered as capability inference.
        Prefer a grounded, direct, and cautious response.
        """
        used_sources = cls._build_used_sources(chunks, max_items=3)

        joined_text = "\n\n".join((c.content or "")
                                  for c in chunks[:5]).strip()
        low = (question or "").strip().lower()

        # Try to detect whether a fixed number is explicitly stated
        salary_markers = [
            "aed", "usd", "salary range", "expected salary", "monthly rate",
            "hourly rate", "daily rate", "package"
        ]
        has_explicit_comp = any(marker in joined_text.lower()
                                for marker in salary_markers)

        if has_explicit_comp:
            answer = (
                "Based on the compensation-related documents, there is salary or rate-related information available, "
                "but it should be presented in the exact wording supported by the retrieved evidence."
            )
        else:
            answer = (
                "Samah’s documents do not state a fixed salary number. "
                "They indicate that compensation depends on the role scope, seniority level, technical depth, "
                "leadership responsibility, and work arrangement."
            )

        if any(k in low for k in ["freelance", "contract", "project-based"]):
            answer += (
                " The documents also indicate openness to freelance or project-based work when the project is well-defined "
                "and aligned with her technical strengths."
            )

        return {
            "verdict": "supported",
            "answer": answer,
            "bullets": [],
            "used_sources": used_sources,
            "meta": {
                "model_used": None,
                "tried_models": [],
                "provider_used": "compensation_handler",
                "fallback_used": False,
                "generation_ok": True,
                "safe_fallback": False,
                "error": None,
                "answer_source": "compensation_handler",
                "extractor_used": None,
                "confidence_boost": 0.10,
                "primary_meta": None,
                "secondary_meta": None,
            },
        }

    @classmethod
    def _answer_conversation_followup_question(
        cls,
        question: str,
        chunks: List[DocumentChunk],
        history: Optional[List[Dict[str, Any]]] = None,
        retrieval_query: Optional[str] = None,
        retrieval_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Route follow-up questions into the unified grounded answerer.
        """
        history = history or []
        resolved_question = cls._resolve_question_from_history(
            question=question,
            retrieval_query=retrieval_query,
            history=history,
            question_route="conversation_followup_question",
        )

        return GroundedAnswerer.answer(
            current_message=question,
            resolved_question=resolved_question,
            conversation_history=history,
            evidence_chunks=chunks,
            retrieval_confidence=retrieval_confidence,
            preferred_source="hybrid",
        )

    @classmethod
    def _answer_clarification_question(
        cls,
        question: str,
        chunks: List[DocumentChunk],
        history: Optional[List[Dict[str, Any]]] = None,
        retrieval_query: Optional[str] = None,
        retrieval_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Route clarification questions into the unified grounded answerer.
        """
        history = history or []
        resolved_question = cls._resolve_question_from_history(
            question=question,
            retrieval_query=retrieval_query,
            history=history,
            question_route="clarification_question",
        )

        return GroundedAnswerer.answer(
            current_message=question,
            resolved_question=resolved_question,
            conversation_history=history,
            evidence_chunks=chunks,
            retrieval_confidence=retrieval_confidence,
            preferred_source="hybrid",
        )

    @classmethod
    def _answer_session_memory_question(
        cls,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Route session-memory questions into the grounded answerer in history mode.
        """
        history = history or []

        return GroundedAnswerer.answer(
            current_message=question,
            resolved_question=question,
            conversation_history=history,
            evidence_chunks=[],
            retrieval_confidence=1.0 if history else 0.0,
            preferred_source="history",
        )

    @classmethod
    def _normalize_retrieval_query(
        cls,
        question: str,
        retrieval_query: str,
    ) -> str:
        """
        Convert a natural user question into a retrieval-friendly query.

        Important:
        - This does NOT answer the user directly.
        - This only improves vector search quality.
        - The final answer still comes from retrieved evidence.
        """
        q = (retrieval_query or question or "").strip()
        lower_q = q.lower()

        # Keep project/tool-specific questions as-is.
        # These questions usually need exact project/tool wording.
        if any(k in lower_q for k in [
            "used to build",
            "built with",
            "used in",
            "technologies used in",
            "tools used in",
            "frameworks used in",
        ]):
            return q
        
        if cls._is_current_work_status_question(question) or cls._is_current_work_status_question(q):
            return (
                "Samah current employment status current role current job "
                "currently working currently employed not working now availability "
                "open to work open to opportunities employment ended last working date "
                "experience letter career timeline compensation availability"
            )

        # Contact / call / reach questions.
        # Example:
        # "if i want to call her to discuss further what should i do"
        # becomes a better retrieval query for contact-related chunks.
        if cls._is_contact_question(question) or cls._is_contact_question(q):
            return (
                "Samah contact details email phone mobile LinkedIn portfolio website "
                "how to contact Samah reach Samah call Samah discuss further "
                "schedule a call communicate with Samah"
            )

        if "tech stack" in lower_q:
            return (
                "Samah technical skills technologies frameworks tools "
                "backend frontend AI databases devops"
            )

        if "technologies does she use" in lower_q:
            return "Samah technical skills technologies frameworks tools"

        if "frameworks" in lower_q:
            return (
                "Samah frameworks Django Django REST Framework FastAPI Flask "
                "React Next.js Tailwind CSS LangChain"
            )

        return q

    @staticmethod
    def _is_contact_question(question: str) -> bool:
        q = (question or "").strip().lower()

        contact_markers = [
            "contact",
            "email",
            "phone",
            "mobile",
            "call",
            "whatsapp",
            "linkedin",
            "reach",
            "get in touch",
            "connect",
            "contact details",
            "how can i contact",
            "how do i contact",
            "how to contact",
            "how can we communicate",
            "can we communicate",
            "communicate with her",
            "reach her",
            "reach out",
            "message her",
            "talk to her",
            "speak with her",
            "know more about samah",
        ]

        return any(marker in q for marker in contact_markers)

    @staticmethod
    def _resolve_question_from_history(
        question: str,
        retrieval_query: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        question_route: Optional[str] = None,
    ) -> str:
        """
        Build a more explicit question for retrieval / final answering.
        For normal fact questions, use retrieval_query or question as-is.
        For follow-up / clarification questions, enrich using recent history.
        """
        question = (question or "").strip()
        retrieval_query = (retrieval_query or question).strip()
        history = history or []

        if question_route not in {"conversation_followup_question", "clarification_question"}:
            return retrieval_query or question

        recent_user_turns = [
            item.get("content", "").strip()
            for item in history
            if item.get("role") == "user" and item.get("content")
        ]
        recent_assistant_turns = [
            item.get("content", "").strip()
            for item in history
            if item.get("role") == "assistant" and item.get("content")
        ]

        previous_user = recent_user_turns[-2] if len(recent_user_turns) >= 2 else (
            recent_user_turns[-1] if recent_user_turns else ""
        )
        previous_answer = recent_assistant_turns[-1] if recent_assistant_turns else ""

        if not previous_user and not previous_answer:
            return retrieval_query or question

        q_low = question.lower()

        if question_route == "clarification_question":
            return (
                f"Clarify this question using earlier conversation context. "
                f"Current question: {question}. "
                f"Previous user question: {previous_user}. "
                f"Previous assistant answer: {previous_answer}."
            ).strip()

        if any(x in q_low for x in ["cost", "price", "salary", "payment", "rate", "budget"]):
            return (
                f"Answer this follow-up compensation-related question in context. "
                f"Current question: {question}. "
                f"Previous user question: {previous_user}. "
                f"Previous assistant answer: {previous_answer}."
            ).strip()

        return (
            f"Answer this follow-up question in context. "
            f"Current question: {question}. "
            f"Previous user question: {previous_user}. "
            f"Previous assistant answer: {previous_answer}."
        ).strip()

    @staticmethod
    def _estimate_retrieval_confidence(
        chunks: List[DocumentChunk],
        retrieval_debug: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Lightweight confidence estimate for downstream answering/debugging.
        """
        retrieval_debug = retrieval_debug or []

        if not chunks:
            return 0.0

        if len(chunks) >= 5:
            return 0.9
        if len(chunks) >= 3:
            return 0.75
        if len(chunks) == 2:
            return 0.6
        return 0.45

    @staticmethod
    def _is_project_technology_question(question: str) -> bool:
        q = (question or "").strip().lower()
        markers = [
            "which projects used",
            "what projects used",
            "which project used",
            "what project used",
            "projects using",
            "projects with",
        ]
        return any(marker in q for marker in markers)

    @staticmethod
    def _preferred_source_for_route(question_route: Optional[str]) -> str:
        """
        Decide which source should be preferred by the final answerer.
        """
        if question_route == "session_memory_question":
            return "history"

        if question_route in {"profile_docs_question", "capability_inference_question"}:
            return "documents"

        if question_route == "general_question":
            return "documents"

        return "documents"

    @classmethod
    def _get_experience_duration_chunks(cls) -> List[DocumentChunk]:
        """
        Fetch richer approved evidence for total-experience questions.

        This must only use approved chatbot-safe chunks.
        """

        qs = (
            get_chatbot_available_chunks()
            .filter(
                document__document_type__in=["cv", "career_timeline"],
            )
            .order_by(
                "document__document_type",
                "document__title",
                "chunk_index",
            )
        )

        return list(qs)

    @classmethod
    def _get_all_chunks_for_filters(
        cls,
        filters: Optional[Dict[str, Any]],
    ) -> List[DocumentChunk]:
        """
        Fetch all approved chatbot-safe chunks matching filters.

        This method must never return unapproved or inactive chunks.
        """

        if not filters:
            return []

        qs = get_chatbot_available_chunks()

        if filters.get("document_type"):
            qs = qs.filter(document__document_type=filters["document_type"])

        if filters.get("document_title_contains"):
            qs = qs.filter(
                document__title__icontains=filters["document_title_contains"]
            )

        return list(qs.order_by("document__title", "chunk_index"))

    @classmethod
    def answer_question(
        cls,
        question: str,
        retrieval_query: Optional[str] = None,
        question_route: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        query_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        question = (question or "").strip()
        retrieval_query = (retrieval_query or question).strip()
        history = history or []
        query_plan = query_plan or {}
        
        resolved_question = cls._resolve_question_from_history(
            question=question,
            retrieval_query=retrieval_query,
            history=history,
            question_route=question_route,
        )

        if query_plan.get("retrieval_query"):
            retrieval_query = query_plan["retrieval_query"]
        else:
            retrieval_query = cls._normalize_retrieval_query(
                question,
                resolved_question,
            )
        preferred_source = cls._preferred_source_for_route(question_route)

        # Step 1: Special direct handlers before general retrieval
        if question_route == "identity_question":
            result = cls._answer_identity_question(question)
            result["retrieval_query"] = None
            result["rewrite_notes"] = "route_identity_fast_path"
            result["retrieval_debug"] = []
            result["applied_filters"] = None
            result["debug_chunks_before_llm"] = []
            return result

        if question_route == "session_memory_question":
            result = cls._answer_session_memory_question(
                question=question,
                history=history,
            )
            result["retrieval_query"] = None
            result["rewrite_notes"] = "route_session_memory_history_only"
            result["retrieval_debug"] = []
            result["applied_filters"] = None
            result["debug_chunks_before_llm"] = []
            return result

        # Step 1.5: Special deterministic path for total-experience questions
        if (
            question_route == "profile_docs_question"
            and cls._is_experience_duration_question(question)
        ):
            duration_chunks = cls._get_experience_duration_chunks()

            deterministic_result = cls._try_deterministic_answer(
                question,
                duration_chunks,
            )
            if deterministic_result:
                deterministic_result["retrieval_query"] = retrieval_query
                deterministic_result["rewrite_notes"] = "route_profile_docs_experience_duration_direct"
                deterministic_result["retrieval_debug"] = [
                    {
                        "source": "deterministic_duration_chunk_pool",
                        "document_type": getattr(c.document, "document_type", None),
                        "doc_title": c.document.title,
                        "chunk_index": c.chunk_index,
                        "chunk_id": str(c.id),
                    }
                    for c in duration_chunks[:12]
                ]
                deterministic_result["applied_filters"] = {
                    "document_type__in": ["cv", "career_timeline"],
                    "only_active_docs": True,
                    "special_case": "experience_duration_question",
                }
                deterministic_result["debug_chunks_before_llm"] = [
                    {
                        "chunk_index": c.chunk_index,
                        "doc_title": c.document.title,
                        "document_type": getattr(c.document, "document_type", None),
                        "content_preview": (c.content or "")[:1200],
                    }
                    for c in duration_chunks[:8]
                ]
                return deterministic_result
            
        retrieval_query = cls._normalize_retrieval_query(
            question=question,
            retrieval_query=retrieval_query,
        )

        # Step 2: Route-aware retrieval
        chunks, retrieval_debug, filters = cls._retrieve_chunks(
            question=question,
            retrieval_query=retrieval_query,
            question_route=question_route,
            query_plan=query_plan,
        )

        debug_chunks = []
        for c in chunks[:5]:
            debug_chunks.append({
                "chunk_index": c.chunk_index,
                "doc_title": c.document.title,
                "document_type": getattr(c.document, "document_type", None),
                "content_preview": (c.content or "")[:1200],
            })

        retrieval_confidence = cls._estimate_retrieval_confidence(
            chunks=chunks,
            retrieval_debug=retrieval_debug,
        )

        # Step 3: Dedicated route handlers
        if question_route == "capability_inference_question":
            result = GroundedAnswerer.answer(
                current_message=question,
                resolved_question=resolved_question,
                conversation_history=history,
                evidence_chunks=chunks,
                retrieval_confidence=retrieval_confidence,
                preferred_source="documents",
            )

            result["retrieval_query"] = retrieval_query
            result["rewrite_notes"] = "route_capability_inference_grounded"
            result["retrieval_debug"] = retrieval_debug
            result["applied_filters"] = filters
            result["debug_chunks_before_llm"] = debug_chunks
            return result

        if question_route == "profile_docs_question":
            deterministic_result = cls._try_deterministic_answer(
                question, chunks)
            if deterministic_result:
                deterministic_result["retrieval_query"] = retrieval_query
                deterministic_result["rewrite_notes"] = "route_profile_docs_direct"
                deterministic_result["retrieval_debug"] = retrieval_debug
                deterministic_result["applied_filters"] = filters
                deterministic_result["debug_chunks_before_llm"] = debug_chunks
                return deterministic_result

            result = GroundedAnswerer.answer(
                current_message=question,
                resolved_question=resolved_question,
                conversation_history=history,
                evidence_chunks=chunks,
                retrieval_confidence=retrieval_confidence,
                preferred_source=preferred_source,
            )
            result["retrieval_query"] = retrieval_query
            result["rewrite_notes"] = "route_profile_docs_grounded"
            result["retrieval_debug"] = retrieval_debug
            result["applied_filters"] = filters
            result["debug_chunks_before_llm"] = debug_chunks
            return result

        # Step 4: General question path
        result = GroundedAnswerer.answer(
            current_message=question,
            resolved_question=resolved_question,
            conversation_history=history,
            evidence_chunks=chunks,
            retrieval_confidence=retrieval_confidence,
            preferred_source=preferred_source,
        )
        result["retrieval_query"] = retrieval_query
        result["rewrite_notes"] = "grounded_answer_unified"
        result["retrieval_debug"] = retrieval_debug
        result["applied_filters"] = filters
        result["debug_chunks_before_llm"] = debug_chunks
        return result
