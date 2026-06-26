from __future__ import annotations

import json
from typing import List, Dict, Any, Tuple, Optional
from django.conf import settings

from core.models import DocumentChunk
from core.services.llm.router import LLMRouter


class GroundedAnswerer:
    """
    Produces the final grounded answer ONLY from evidence chunks.

    Updated strategy:
    1) Primary provider: Gemini
    2) Retry Gemini for transient failures (503 / timeout / temporary provider issues)
    3) Secondary provider: DeepSeek
    4) If both fail: return a safe temporary fallback message
    5) Never expose raw retrieved chunk text as the final answer
    """

    SAFE_TEMPORARY_MESSAGE = (
        "I found relevant information about Samah’s experience, "
        "but I’m temporarily unable to generate a polished answer. "
        "Please try again in a moment."
    )

    HISTORY_NOT_ENOUGH_MESSAGE = (
        "I couldn’t find enough conversation history in this session to answer that clearly."
    )

    DOCUMENTS_NOT_ENOUGH_MESSAGE = (
        "I couldn’t find enough verified information in the available documents to answer that."
    )

    HYBRID_NOT_ENOUGH_MESSAGE = (
        "I could understand the question from the conversation, but I couldn’t verify the factual answer from the available documents."
    )

    @staticmethod
    def _filter_safe_evidence_chunks(
        evidence_chunks: Optional[List[DocumentChunk]],
    ) -> List[DocumentChunk]:
        """
        Final defensive safety filter.

        GroundedAnswerer should normally receive already-approved chunks from
        ProfileQAService and the retrieval layer. This method protects the system
        if another service accidentally passes unapproved chunks directly.
        """

        safe_chunks = []

        for chunk in evidence_chunks or []:
            document = getattr(chunk, "document", None)

            if document is None:
                continue

            if getattr(document, "status", None) != "approved":
                continue

            if not getattr(document, "is_active", False):
                continue

            if not getattr(document, "is_approved", False):
                continue

            if not getattr(document, "is_available_for_chatbot", False):
                continue

            if not getattr(chunk, "is_active", False):
                continue

            if not getattr(chunk, "has_embedding", False):
                continue

            if getattr(chunk, "quality_status", None) not in ["passed", "warning"]:
                continue

            safe_chunks.append(chunk)

        return safe_chunks

    @staticmethod
    def _format_history_for_prompt(
        history: Optional[List[Dict[str, Any]]],
        max_turns: int = 6,
        max_chars_per_turn: int = 500,
    ) -> str:
        """
        Prepare recent conversation history for prompt usage.
        Keeps only the most recent turns and trims very long content.
        """
        if not history:
            return ""

        selected = history[-max_turns:]
        lines = []

        for item in selected:
            role = (item.get("role") or "").strip().lower()
            content = (item.get("content") or "").strip()

            if not role or not content:
                continue

            if len(content) > max_chars_per_turn:
                content = content[:max_chars_per_turn].rstrip() + "..."

            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")

        return "\n".join(lines).strip()

    @staticmethod
    def _exclude_current_user_turn(
        history: Optional[List[Dict[str, Any]]],
        current_message: str,
    ) -> List[Dict[str, Any]]:
        """
        Remove the current user message from history if it is already stored.
        This prevents memory questions like 'what was my second question'
        from counting themselves.
        """
        history = history or []
        current_clean = (current_message or "").strip()

        if not history or not current_clean:
            return history

        last = history[-1]
        last_role = (last.get("role") or "").strip().lower()
        last_content = (last.get("content") or "").strip()

        if last_role == "user" and last_content == current_clean:
            return history[:-1]

        return history

    @classmethod
    def _resolve_history_question_with_llm(
        cls,
        *,
        current_message: str,
        conversation_history: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Use the LLM only to interpret session-memory questions.
        The final answer is still generated deterministically in Python.
        """
        history = conversation_history or []
        history_text = cls._format_history_for_prompt(
            history, max_turns=8, max_chars_per_turn=300
        )

        system_instruction = (
            "You are a parser for conversation-memory questions. "
            "Your task is to detect whether the user is asking about earlier messages "
            "in the current session and return raw JSON only."
        )

        prompt = f"""
Return raw JSON only.

Schema:
{{
  "is_memory_question": true or false,
  "target_role": "user" | "assistant",
  "target_type": "question" | "answer" | "message",
  "reference_mode": "first" | "last" | "previous" | "ordinal" | "recent_n" | "summary",
  "ordinal": integer or null,
  "count": integer or null,
  "anchor_text": string or null
}}

Examples:
- "what was my first question"
  => {{"is_memory_question": true, "target_role": "user", "target_type": "question", "reference_mode": "first", "ordinal": null, "count": 1, "anchor_text": null}}

- "what was my second question"
  => {{"is_memory_question": true, "target_role": "user", "target_type": "question", "reference_mode": "ordinal", "ordinal": 2, "count": 1, "anchor_text": null}}

- "what was my third question"
  => {{"is_memory_question": true, "target_role": "user", "target_type": "question", "reference_mode": "ordinal", "ordinal": 3, "count": 1, "anchor_text": null}}

- "what was your last answer"
  => {{"is_memory_question": true, "target_role": "assistant", "target_type": "answer", "reference_mode": "last", "ordinal": null, "count": 1, "anchor_text": null}}

- "what did i ask before"
  => {{"is_memory_question": true, "target_role": "user", "target_type": "question", "reference_mode": "previous", "ordinal": null, "count": 1, "anchor_text": null}}

- "show my last 3 questions"
  => {{"is_memory_question": true, "target_role": "user", "target_type": "question", "reference_mode": "recent_n", "ordinal": null, "count": 3, "anchor_text": null}}

- "summarize what we discussed"
  => {{"is_memory_question": true, "target_role": "user", "target_type": "message", "reference_mode": "summary", "ordinal": null, "count": 5, "anchor_text": null}}

If it is not a session-memory question, return:
{{"is_memory_question": false}}

Recent conversation:
{history_text or "None"}

Current message:
{current_message}
""".strip()

        json_schema = {
            "type": "object",
            "properties": {
                "is_memory_question": {"type": "boolean"},
                "target_role": {"type": "string"},
                "target_type": {"type": "string"},
                "reference_mode": {"type": "string"},
                "ordinal": {"type": ["integer", "null"]},
                "count": {"type": ["integer", "null"]},
                "anchor_text": {"type": ["string", "null"]},
            },
            "required": ["is_memory_question"],
        }

        ok, text, meta = LLMRouter.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.0,
            model_chain=[
                getattr(settings, "GROUNDED_PRIMARY_MODEL",
                        "gemini-2.5-flash-lite")
            ],
            json_schema=json_schema,
            task=LLMRouter.TASK_GROUNDED_ANSWER,
        )

        if not ok:
            return {"is_memory_question": False, "_meta": meta or {}, "_fallback": True}

        try:
            data = cls._parse_json_safely(text)
            if not isinstance(data, dict):
                return {"is_memory_question": False, "_meta": meta or {}, "_fallback": True}
            data["_meta"] = meta or {}
            return data
        except Exception:
            return {"is_memory_question": False, "_meta": meta or {}, "_fallback": True}

    @classmethod
    def _execute_history_question(
        cls,
        *,
        current_message: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        resolver_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a parsed memory question deterministically from history.
        """
        history = cls._exclude_current_user_turn(
            conversation_history, current_message)

        if not history:
            return {
                "verdict": "not_enough_evidence",
                "answer": cls.HISTORY_NOT_ENOUGH_MESSAGE,
                "bullets": [],
                "used_chunk_indices": [],
                "used_sources": [],
                "meta": {
                    "model_used": None,
                    "tried_models": [],
                    "provider_used": "history_only",
                    "fallback_used": False,
                    "generation_ok": True,
                    "safe_fallback": False,
                    "answer_source": "conversation_history",
                    "error": "no_history",
                },
            }

        target_role = (resolver_payload.get(
            "target_role") or "user").strip().lower()
        target_type = (resolver_payload.get("target_type")
                       or "message").strip().lower()
        reference_mode = (resolver_payload.get(
            "reference_mode") or "summary").strip().lower()
        ordinal = resolver_payload.get("ordinal")
        count = resolver_payload.get("count") or 3

        filtered_messages = [
            (item.get("content") or "").strip()
            for item in history
            if (item.get("role") or "").strip().lower() == target_role
            and (item.get("content") or "").strip()
        ]

        if not filtered_messages:
            return {
                "verdict": "not_enough_evidence",
                "answer": cls.HISTORY_NOT_ENOUGH_MESSAGE,
                "bullets": [],
                "used_chunk_indices": [],
                "used_sources": [],
                "meta": {
                    "model_used": None,
                    "tried_models": [],
                    "provider_used": "history_only",
                    "fallback_used": False,
                    "generation_ok": True,
                    "safe_fallback": False,
                    "answer_source": "conversation_history",
                    "error": "no_matching_history_messages",
                },
            }

        owner = "Your" if target_role == "user" else "My"

        def ordinal_label(n: int) -> str:
            mapping = {
                1: "first",
                2: "second",
                3: "third",
                4: "fourth",
                5: "fifth",
                6: "sixth",
                7: "seventh",
                8: "eighth",
                9: "ninth",
                10: "tenth",
            }
            return mapping.get(n, f"{n}th")

        if reference_mode == "first":
            return {
                "verdict": "supported",
                "answer": f'{owner} first {target_type} in this session was: "{filtered_messages[0]}"',
                "bullets": [],
                "used_chunk_indices": [],
                "used_sources": [],
                "meta": {
                    "model_used": None,
                    "tried_models": [],
                    "provider_used": "history_only",
                    "fallback_used": False,
                    "generation_ok": True,
                    "safe_fallback": False,
                    "answer_source": "conversation_history",
                    "error": None,
                },
            }

        if reference_mode == "last":
            return {
                "verdict": "supported",
                "answer": f'{owner} last {target_type} in this session was: "{filtered_messages[-1]}"',
                "bullets": [],
                "used_chunk_indices": [],
                "used_sources": [],
                "meta": {
                    "model_used": None,
                    "tried_models": [],
                    "provider_used": "history_only",
                    "fallback_used": False,
                    "generation_ok": True,
                    "safe_fallback": False,
                    "answer_source": "conversation_history",
                    "error": None,
                },
            }

        if reference_mode == "previous":
            if len(filtered_messages) >= 2:
                return {
                    "verdict": "supported",
                    "answer": f'{owner} previous {target_type} in this session was: "{filtered_messages[-2]}"',
                    "bullets": [],
                    "used_chunk_indices": [],
                    "used_sources": [],
                    "meta": {
                        "model_used": None,
                        "tried_models": [],
                        "provider_used": "history_only",
                        "fallback_used": False,
                        "generation_ok": True,
                        "safe_fallback": False,
                        "answer_source": "conversation_history",
                        "error": None,
                    },
                }

        if reference_mode == "ordinal":
            if isinstance(ordinal, int) and ordinal > 0:
                index = ordinal - 1
                if index < len(filtered_messages):
                    return {
                        "verdict": "supported",
                        "answer": f'{owner} {ordinal_label(ordinal)} {target_type} in this session was: "{filtered_messages[index]}"',
                        "bullets": [],
                        "used_chunk_indices": [],
                        "used_sources": [],
                        "meta": {
                            "model_used": None,
                            "tried_models": [],
                            "provider_used": "history_only",
                            "fallback_used": False,
                            "generation_ok": True,
                            "safe_fallback": False,
                            "answer_source": "conversation_history",
                            "error": None,
                        },
                    }
                return {
                    "verdict": "not_enough_evidence",
                    "answer": f"I could only find {len(filtered_messages)} matching messages in this session.",
                    "bullets": [],
                    "used_chunk_indices": [],
                    "used_sources": [],
                    "meta": {
                        "model_used": None,
                        "tried_models": [],
                        "provider_used": "history_only",
                        "fallback_used": False,
                        "generation_ok": True,
                        "safe_fallback": False,
                        "answer_source": "conversation_history",
                        "error": "ordinal_out_of_range",
                    },
                }

        if reference_mode == "recent_n":
            selected = filtered_messages[-count:]
            bullets = [f"{i + 1}. {msg}" for i, msg in enumerate(selected)]
            return {
                "verdict": "supported",
                "answer": f"Here are the last {len(selected)} {target_role} {target_type}s in this session.",
                "bullets": bullets,
                "used_chunk_indices": [],
                "used_sources": [],
                "meta": {
                    "model_used": None,
                    "tried_models": [],
                    "provider_used": "history_only",
                    "fallback_used": False,
                    "generation_ok": True,
                    "safe_fallback": False,
                    "answer_source": "conversation_history",
                    "error": None,
                },
            }

        summary_lines = [
            f"{i}. {msg}"
            for i, msg in enumerate(
                filtered_messages[-5:], start=max(1,
                                                  len(filtered_messages) - 4)
            )
        ]

        return {
            "verdict": "supported",
            "answer": "Here is a quick summary of the recent discussion.",
            "bullets": summary_lines,
            "used_chunk_indices": [],
            "used_sources": [],
            "meta": {
                "model_used": None,
                "tried_models": [],
                "provider_used": "history_only",
                "fallback_used": False,
                "generation_ok": True,
                "safe_fallback": False,
                "answer_source": "conversation_history",
                "error": None,
            },
        }

    @classmethod
    def _resolve_answer_mode(
        cls,
        preferred_source: Optional[str],
        conversation_history: Optional[List[Dict[str, Any]]],
        evidence_chunks: Optional[List[DocumentChunk]],
    ) -> str:
        """
        Decide which answering mode to use.
        Modes:
        - history_only
        - documents_only
        - hybrid
        - no_evidence
        """
        history_exists = bool(conversation_history)
        docs_exist = bool(evidence_chunks)

        if preferred_source == "history":
            return "history_only"

        if preferred_source == "hybrid":
            if history_exists and docs_exist:
                return "hybrid"
            if history_exists:
                return "history_only"
            if docs_exist:
                return "documents_only"
            return "no_evidence"

        if preferred_source == "documents":
            if docs_exist:
                return "documents_only"
            return "no_evidence"

        if history_exists and docs_exist:
            return "hybrid"
        if docs_exist:
            return "documents_only"
        if history_exists:
            return "history_only"

        return "no_evidence"

    @classmethod
    def _answer_from_history_only(
        cls,
        *,
        current_message: str,
        resolved_question: str,
        conversation_history: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Answer pure session-memory / conversation-reference questions
        using conversation history only.
        """
        history = cls._exclude_current_user_turn(
            conversation_history, current_message)
        question = (resolved_question or current_message or "").strip()

        if not history:
            return {
                "verdict": "not_enough_evidence",
                "answer": cls.HISTORY_NOT_ENOUGH_MESSAGE,
                "bullets": [],
                "used_chunk_indices": [],
                "used_sources": [],
                "meta": {
                    "model_used": None,
                    "tried_models": [],
                    "provider_used": "history_only",
                    "fallback_used": False,
                    "generation_ok": True,
                    "safe_fallback": False,
                    "answer_source": "conversation_history",
                    "error": "no_history",
                },
            }

        resolver_payload = cls._resolve_history_question_with_llm(
            current_message=question,
            conversation_history=history,
        )

        if not resolver_payload.get("is_memory_question"):
            low = question.lower()

            if "first question" in low:
                resolver_payload = {
                    "is_memory_question": True,
                    "target_role": "user",
                    "target_type": "question",
                    "reference_mode": "first",
                    "ordinal": None,
                    "count": 1,
                }
            elif "last question" in low:
                resolver_payload = {
                    "is_memory_question": True,
                    "target_role": "user",
                    "target_type": "question",
                    "reference_mode": "last",
                    "ordinal": None,
                    "count": 1,
                }
            elif "previous question" in low or "what did i ask before" in low:
                resolver_payload = {
                    "is_memory_question": True,
                    "target_role": "user",
                    "target_type": "question",
                    "reference_mode": "previous",
                    "ordinal": None,
                    "count": 1,
                }
            elif "last answer" in low or "what did you say" in low or "your last answer" in low:
                resolver_payload = {
                    "is_memory_question": True,
                    "target_role": "assistant",
                    "target_type": "answer",
                    "reference_mode": "last",
                    "ordinal": None,
                    "count": 1,
                }
            else:
                return {
                    "verdict": "not_enough_evidence",
                    "answer": cls.HISTORY_NOT_ENOUGH_MESSAGE,
                    "bullets": [],
                    "used_chunk_indices": [],
                    "used_sources": [],
                    "meta": {
                        "model_used": None,
                        "tried_models": [],
                        "provider_used": "history_only",
                        "fallback_used": False,
                        "generation_ok": True,
                        "safe_fallback": False,
                        "answer_source": "conversation_history",
                        "error": "history_question_not_resolved",
                    },
                }

        result = cls._execute_history_question(
            current_message=current_message,
            conversation_history=history,
            resolver_payload=resolver_payload,
        )

        result["meta"]["history_resolver"] = {
            "is_memory_question": resolver_payload.get("is_memory_question"),
            "target_role": resolver_payload.get("target_role"),
            "target_type": resolver_payload.get("target_type"),
            "reference_mode": resolver_payload.get("reference_mode"),
            "ordinal": resolver_payload.get("ordinal"),
            "count": resolver_payload.get("count"),
        }

        return result

    @staticmethod
    def _snippet(text: str, max_len: int = 320) -> str:
        """
        Normalizes and shortens chunk text for prompt usage only.
        Smaller default length reduces prompt tokens significantly.
        """
        t = (text or "").replace("\r", " ").strip()
        t = " ".join(t.split())
        return t if len(t) <= max_len else t[:max_len].rstrip() + "..."

    @staticmethod
    def _chunk_budget(question: str) -> int:
        """
        Decide how many chunks to include based on question complexity.

        Recommended:
        - 4 chunks by default to reduce missed evidence.
        - 5 chunks for broad/profile/status questions.
        - 6 chunks maximum for complex overview/comparison questions.
        """
        q = (question or "").strip().lower()

        high_value_markers = [
            "current",
            "currently",
            "working now",
            "not working now",
            "available",
            "availability",
            "open to work",
            "contact",
            "call",
            "email",
            "phone",
            "linkedin",
            "salary",
            "compensation",
            "notice period",
            "experience",
            "background",
            "career",
            "timeline",
        ]

        if any(marker in q for marker in high_value_markers):
            return 5

        if any(marker in q for marker in [
            "what certificates",
            "which certificates",
            "what certifications",
            "which certifications",
            "certificates does she have",
            "certifications does she have",
        ]):
            return 5

        if any(marker in q for marker in [
            "what tools",
            "which tools",
            "what technologies",
            "which technologies",
            "what framework",
            "which framework",
            "what frameworks",
            "which frameworks",
            "technology stack",
            "tech stack",
        ]):
            return 5

        complex_markers = [
            "compare",
            "difference",
            "across",
            "summarize",
            "summary",
            "overview",
            "background",
            "experience",
            "tell me about",
            "multiple",
            "all",
        ]

        medium_markers = [
            "project",
            "projects",
            "role fit",
            "strengths",
            "why",
            "how",
        ]

        if any(marker in q for marker in complex_markers):
            return 6

        if any(marker in q for marker in medium_markers):
            return 5

        return 4

    @staticmethod
    def _is_yes_no_question(question: str) -> bool:
        q = (question or "").strip().lower()

        starts_like_yes_no = q.startswith((
            "is ", "are ", "do ", "does ", "did ",
            "can ", "could ", "should ", "has ", "have ",
            "was ", "were ", "am ", "will ", "would ", "shall "
        ))

        if not starts_like_yes_no:
            return False

        choice_markers = [
            " or ",
            "which ",
            "what kind ",
            "what type ",
            "what role ",
            "what framework ",
            "what technologies ",
            "what tools ",
        ]

        if any(marker in q for marker in choice_markers):
            return False

        return True

    @staticmethod
    def _build_used_sources(
        evidence_chunks: List[DocumentChunk],
        used_indices: List[int],
    ) -> List[Dict[str, Any]]:
        used_sources: List[Dict[str, Any]] = []

        for idx in used_indices:
            if 0 <= idx < len(evidence_chunks):
                chunk = evidence_chunks[idx]
                used_sources.append({
                    "doc_title": chunk.document.title,
                    "document_type": getattr(chunk.document, "document_type", None),
                    "chunk_index": chunk.chunk_index,
                    "chunk_id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                })

        return used_sources

    @classmethod
    def _safe_failure_response(
        cls,
        evidence_chunks: List[DocumentChunk],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Returns a safe, controlled response when both providers fail
        or when the model output is unusable.
        """
        fallback_used = list(range(min(2, len(evidence_chunks))))

        return {
            "verdict": "not_enough_evidence",
            "answer": cls.SAFE_TEMPORARY_MESSAGE,
            "bullets": [],
            "used_chunk_indices": fallback_used,
            "used_sources": cls._build_used_sources(evidence_chunks, fallback_used),
            "meta": {
                **(meta or {}),
                "fallback_used": True,
                "generation_ok": False,
                "safe_fallback": True,
            },
        }

    @staticmethod
    def _is_transient_error(meta: Dict[str, Any]) -> bool:
        """
        Detect whether the provider failure is temporary and worth retrying.
        """
        error_text = json.dumps(meta or {}).lower()

        transient_markers = [
            "503",
            "service unavailable",
            "serviceunavailable",
            "timeout",
            "timed out",
            "deadline exceeded",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "try again later",
            "invalid_json_after_cleaning",
            "empty_json_after_cleaning",
            "unacceptable_answer",
            "json_parse",
            "schema",
        ]

        return any(marker in error_text for marker in transient_markers)

    @staticmethod
    def _is_answer_acceptable(answer: str) -> bool:
        """
        Reject weak / broken / fallback-looking answers.
        """
        if not answer:
            return False

        a = answer.strip()
        lower = a.lower()

        banned_patterns = [
            "i couldn’t generate the usual final answer",
            "i couldn't generate the usual final answer",
            "based on the retrieved documents",
            "here is the most relevant grounded evidence",
            "most relevant evidence",
        ]

        if len(a) < 20:
            return False

        if any(p in lower for p in banned_patterns):
            return False

        if a.endswith(":") or a.endswith("-"):
            return False

        return True

    @staticmethod
    def _extract_json_text(text: str) -> str:
        """
        Extract the most likely JSON object from model output.
        Handles:
        - ```json ... ```
        - extra prose before/after JSON
        - plain JSON responses
        """
        text = (text or "").strip()

        if not text:
            return ""

        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

            if text.lower().startswith("json"):
                text = text[4:].strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            return text[start:end + 1].strip()

        return text.strip()

    @staticmethod
    def _parse_json_safely(text: str) -> Dict[str, Any]:
        """
        Parse model output into JSON after cleaning.
        Raises ValueError if parsing still fails.
        """
        cleaned = GroundedAnswerer._extract_json_text(text)

        if not cleaned:
            raise ValueError("empty_json_after_cleaning")

        try:
            return json.loads(cleaned)
        except Exception as exc:
            raise ValueError(f"invalid_json_after_cleaning:{exc}") from exc

    @staticmethod
    def _plain_text_recovery(text: str) -> str:
        """
        Recover a usable plain-text answer when the model answered naturally
        instead of returning valid JSON.
        """
        cleaned = (text or "").strip()

        if not cleaned:
            return ""

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()

            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        lower = cleaned.lower()

        # If it still looks like broken JSON, do not expose it.
        if '"verdict"' in lower and '"answer"' in lower:
            return ""

        # Remove common model prefaces.
        prefixes = [
            "here is the answer:",
            "answer:",
            "final answer:",
        ]

        for prefix in prefixes:
            if lower.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        if len(cleaned) < 20:
            return ""

        return cleaned

    @classmethod
    def _prompt_mode(cls, question: str) -> str:
        """
        Decide response richness based on question complexity.
        Modes:
        - small: short factual / yes-no / direct tools-skills questions
        - standard: normal grounded answers
        - rich: summaries, comparisons, broad overviews
        """
        q = (question or "").strip().lower()

        rich_markers = [
            "compare",
            "difference",
            "across",
            "summary",
            "summarize",
            "overview",
            "background",
            "tell me about",
            "walk me through",
        ]

        small_markers = [
            "what tools",
            "which tools",
            "what technologies",
            "which technologies",
            "what frameworks",
            "which frameworks",
            "what database",
            "which database",
            "used to build",
            "built with",
            "does samah know",
            "has samah used",
            "is samah",
            "can samah",
        ]

        if any(marker in q for marker in rich_markers):
            return "rich"

        if cls._is_yes_no_question(question) or any(marker in q for marker in small_markers):
            return "small"

        return "standard"

    @classmethod
    def _build_hybrid_prompt(
        cls,
        *,
        current_message: str,
        resolved_question: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        evidence_chunks: List[DocumentChunk],
        mode: str,
        instruction_mode: str = "default",
    ) -> Tuple[str, str, bool]:
        """
        Build a prompt that uses both recent conversation history
        and retrieved document evidence.
        """
        history_text = cls._format_history_for_prompt(conversation_history)
        evidence_lines = []

        for i, c in enumerate(evidence_chunks):
            evidence_lines.append(
                f"[{i}] {c.document.title} (chunk {c.chunk_index}): {cls._snippet(c.content)}"
            )

        evidence_text = "\n".join(evidence_lines)
        question_for_reasoning = (
            resolved_question or current_message or "").strip()
        is_yes_no = cls._is_yes_no_question(question_for_reasoning)

        system_instruction = (
            "Answer questions about Samah using only the provided evidence. "
            "Do not invent facts. "
            "If evidence is incomplete, state only what is supported and avoid guessing. "
            "If evidence is insufficient, return not_enough_evidence. "
            "For current employment or work-status questions, if evidence includes an ended employment period, mention that ended role separately from current employment status. "
            "Do not conclude unemployment unless the evidence explicitly says so. "
            "Return raw JSON only. "
            "Do not mention retrieval, chunks, or internal processing."
        )

        if instruction_mode == "capability_inference":
            system_instruction += (
                " You are answering a capability or suitability question about Samah. "
                "Answer the exact capability asked by the user. "
                "Do not switch to another role, skill, or topic. "
                "Use only the provided evidence. "
                "If the evidence is partial, say 'based on the available evidence'. "
                "If evidence shows related skills but not direct proof, explain the relation carefully. "
                "Do not invent certifications, years, employers, or exact experience. "
                "Keep the answer helpful, natural, and concise."
            )

        if mode == "small":
            prompt = (
                "Return raw JSON with keys:\n"
                "- verdict: for yes/no questions -> one of ['yes','no','not_enough_evidence']\n"
                "          for non-yes/no questions -> one of ['supported','not_enough_evidence']\n"
                "- answer: concise polished answer in 1-2 sentences\n"
                "- used_chunk_indices: array of integers referencing only the evidence chunks actually used\n\n"
                "Rules:\n"
                "A) Use history only to understand what the user means.\n"
                "B) Use evidence for factual claims.\n"
                "C) If the fact is not verified in evidence, say that clearly.\n"
                "D) Do not mention retrieval or internal logic.\n\n"
                f"Current message: {current_message}\n"
                f"Resolved question: {question_for_reasoning}\n\n"
                f"Recent conversation:\n{history_text or 'None'}\n\n"
                f"Evidence:\n{evidence_text}\n"
            )
        elif mode == "rich":
            prompt = (
                "Return raw JSON with keys:\n"
                "- verdict: for yes/no questions -> one of ['yes','no','not_enough_evidence']\n"
                "          for non-yes/no questions -> one of ['supported','not_enough_evidence']\n"
                "- answer: concise polished answer in 2-3 sentences\n"
                "- bullets: 0-4 unique bullets adding new non-overlapping details\n"
                "- used_chunk_indices: array of integers referencing only the evidence chunks actually used\n\n"
                "Rules:\n"
                "A) Use history to resolve references like this/that/the first one.\n"
                "B) Use evidence for profile facts.\n"
                "C) If evidence is incomplete, separate what is known from what is uncertain.\n"
                "D) Do not mention retrieval or internal logic.\n\n"
                f"Current message: {current_message}\n"
                f"Resolved question: {question_for_reasoning}\n\n"
                f"Recent conversation:\n{history_text or 'None'}\n\n"
                f"Evidence:\n{evidence_text}\n"
            )
        else:
            prompt = (
                "Return raw JSON with keys:\n"
                "- verdict: for yes/no questions -> one of ['yes','no','not_enough_evidence']\n"
                "          for non-yes/no questions -> one of ['supported','not_enough_evidence']\n"
                "- answer: concise polished answer in 1-2 sentences\n"
                "- bullets: 0-2 unique bullets only if they add important new details\n"
                "- used_chunk_indices: array of integers referencing only the evidence chunks actually used\n\n"
                "Rules:\n"
                "A) Use conversation history only to resolve context.\n"
                "B) Use evidence for factual claims.\n"
                "C) If the answer is only partially supported, say so honestly.\n"
                "D) Do not mention retrieval or internal logic.\n\n"
                f"Current message: {current_message}\n"
                f"Resolved question: {question_for_reasoning}\n\n"
                f"Recent conversation:\n{history_text or 'None'}\n\n"
                f"Evidence:\n{evidence_text}\n"
            )

        return system_instruction, prompt, is_yes_no

    @staticmethod
    def _looks_uncertain_answer(answer: str) -> bool:
        """
        Detect fallback / uncertainty phrasing that should keep not_enough_evidence.
        """
        if not answer:
            return True

        lower = answer.strip().lower()

        uncertainty_markers = [
            "not enough evidence",
            "i don’t have enough evidence",
            "i don't have enough evidence",
            "cannot determine",
            "can't determine",
            "cannot answer",
            "can't answer",
            "does not contain enough information",
            "does not specify",
            "insufficient evidence",
            "not stated",
            "not explicitly stated",
        ]

        return any(marker in lower for marker in uncertainty_markers)

    @staticmethod
    def _normalize_verdict_from_answer(
        *,
        question: str,
        answer: str,
        verdict: str,
        used_chunk_indices: list[int],
    ) -> str:
        """
        Correct inconsistent verdicts returned by the model.
        """
        answer = (answer or "").strip()
        lower = answer.lower()

        if not answer:
            return "not_enough_evidence"

        if GroundedAnswerer._looks_uncertain_answer(answer):
            return "not_enough_evidence"

        is_yes_no = GroundedAnswerer._is_yes_no_question(question)

        if used_chunk_indices:
            if is_yes_no:
                if lower.startswith("yes"):
                    return "yes"
                if lower.startswith("no"):
                    return "no"

                if verdict == "not_enough_evidence":
                    if any(lower.startswith(prefix) for prefix in ["yes", "yes,", "yes."]):
                        return "yes"
                    if any(lower.startswith(prefix) for prefix in ["no", "no,", "no."]):
                        return "no"
                    return "not_enough_evidence"

            return "supported"

        return verdict

    @classmethod
    def _provider_strategy(cls, resolved_question: str, answer_mode: str) -> str:
        """
        Decide which provider to try first.
        - hybrid mode usually needs richer reasoning -> gemini_first
        - small/simple document questions can use deepseek_first
        """
        if answer_mode == "hybrid":
            return "gemini_first"

        mode = cls._prompt_mode(resolved_question)

        if mode == "small":
            return "deepseek_first"

        return "gemini_first"

    @classmethod
    def _call_deepseek_first_then_gemini(
        cls,
        *,
        current_message: str,
        resolved_question: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        evidence_chunks: List[DocumentChunk],
        answer_mode: str,
        instruction_mode: str = "default",
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        deepseek_success, deepseek_result, deepseek_meta = cls._call_deepseek(
            current_message=current_message,
            resolved_question=resolved_question,
            conversation_history=conversation_history,
            evidence_chunks=evidence_chunks,
            answer_mode=answer_mode,
            instruction_mode=instruction_mode,
        )
        if deepseek_success:
            return True, deepseek_result, deepseek_meta, {}

        gemini_success, gemini_result, gemini_meta = cls._call_gemini_with_retry(
            current_message=current_message,
            resolved_question=resolved_question,
            conversation_history=conversation_history,
            evidence_chunks=evidence_chunks,
            answer_mode=answer_mode,
            instruction_mode=instruction_mode,
        )
        if gemini_success:
            return True, gemini_result, deepseek_meta, gemini_meta

        return False, {}, deepseek_meta, gemini_meta

    @classmethod
    def _call_gemini_first_then_deepseek(
        cls,
        *,
        current_message: str,
        resolved_question: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        evidence_chunks: List[DocumentChunk],
        answer_mode: str,
        instruction_mode: str = "default",
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        gemini_success, gemini_result, gemini_meta = cls._call_gemini_with_retry(
            current_message=current_message,
            resolved_question=resolved_question,
            conversation_history=conversation_history,
            evidence_chunks=evidence_chunks,
            answer_mode=answer_mode,
            instruction_mode=instruction_mode,
        )
        if gemini_success:
            return True, gemini_result, gemini_meta, {}

        deepseek_success, deepseek_result, deepseek_meta = cls._call_deepseek(
            current_message=current_message,
            resolved_question=resolved_question,
            conversation_history=conversation_history,
            evidence_chunks=evidence_chunks,
            answer_mode=answer_mode,
            instruction_mode=instruction_mode,
        )
        if deepseek_success:
            return True, deepseek_result, gemini_meta, deepseek_meta

        return False, {}, gemini_meta, deepseek_meta

    @classmethod
    def _build_prompt(
        cls,
        question: str,
        evidence_chunks: List[DocumentChunk],
        mode: str,
        instruction_mode: str = "default",
    ) -> Tuple[str, str, bool]:
        """
        Build a prompt for document-grounded answering only.
        """
        evidence_lines = []
        for i, c in enumerate(evidence_chunks):
            evidence_lines.append(
                f"[{i}] {c.document.title} (chunk {c.chunk_index}): {cls._snippet(c.content)}"
            )

        evidence_text = "\n".join(evidence_lines)
        is_yes_no = cls._is_yes_no_question(question)

        system_instruction = (
            "Answer questions about Samah using only the provided evidence. "
            "Do not invent facts. "
            "If evidence is incomplete, state only what is supported and avoid guessing. "
            "If evidence is insufficient, return not_enough_evidence. "
            "For current employment or work-status questions, if evidence includes an ended employment period, mention that ended role separately from current employment status. "
            "Do not conclude unemployment unless the evidence explicitly says so. "
            "Return raw JSON only. "
            "Do not mention retrieval, chunks, or internal processing."
        )

        if instruction_mode == "capability_inference":
            system_instruction += (
                " You are answering a capability or suitability question about Samah. "
                "Answer the exact capability asked by the user. "
                "Do not switch to another role, skill, or topic. "
                "Use only the provided evidence. "
                "If the evidence is partial, say 'based on the available evidence'. "
                "If evidence shows related skills but not direct proof, explain the relation carefully. "
                "Do not invent certifications, years, employers, or exact experience. "
                "Keep the answer helpful, natural, and concise."
            )

        if mode == "small":
            prompt = (
                "Return raw JSON with keys:\n"
                "- verdict: for yes/no questions -> one of ['yes','no','not_enough_evidence']\n"
                "          for non-yes/no questions -> one of ['supported','not_enough_evidence']\n"
                "- answer: concise polished answer in 1-2 sentences\n"
                "- used_chunk_indices: array of integers referencing only the evidence chunks actually used\n\n"
                "Rules:\n"
                "A) Keep the answer short and direct.\n"
                "B) Do not add bullets.\n"
                "C) Do not mention retrieval or internal logic.\n"
                "D) If the evidence does not verify the fact, say so clearly.\n\n"
                f"Question (yes/no = {str(is_yes_no).lower()}): {question}\n\n"
                f"Evidence:\n{evidence_text}\n"
            )

        elif mode == "rich":
            prompt = (
                "Return raw JSON with keys:\n"
                "- verdict: for yes/no questions -> one of ['yes','no','not_enough_evidence']\n"
                "          for non-yes/no questions -> one of ['supported','not_enough_evidence']\n"
                "- answer: concise polished answer in 2-3 sentences\n"
                "- bullets: 0-4 unique bullets adding new non-overlapping details\n"
                "- used_chunk_indices: array of integers referencing only the evidence chunks actually used\n\n"
                "Rules:\n"
                "A) Keep the answer grounded and complete.\n"
                "B) Bullets must add new information and must not repeat the answer.\n"
                "C) If evidence is partial, separate what is known from what is uncertain.\n"
                "D) Do not mention retrieval or internal logic.\n\n"
                f"Question (yes/no = {str(is_yes_no).lower()}): {question}\n\n"
                f"Evidence:\n{evidence_text}\n"
            )

        else:
            prompt = (
                "Return raw JSON with keys:\n"
                "- verdict: for yes/no questions -> one of ['yes','no','not_enough_evidence']\n"
                "          for non-yes/no questions -> one of ['supported','not_enough_evidence']\n"
                "- answer: concise polished answer in 1-2 sentences\n"
                "- bullets: 0-2 unique bullets only if they add important new details\n"
                "- used_chunk_indices: array of integers referencing only the evidence chunks actually used\n\n"
                "Rules:\n"
                "A) Keep the answer focused and non-repetitive.\n"
                "B) Use bullets only when they add real value.\n"
                "C) If evidence is incomplete, be honest and precise.\n"
                "D) Do not mention retrieval or internal logic.\n\n"
                f"Question (yes/no = {str(is_yes_no).lower()}): {question}\n\n"
                f"Evidence:\n{evidence_text}\n"
            )

        return system_instruction, prompt, is_yes_no

    @classmethod
    def _call_provider(
        cls,
        *,
        current_message: str,
        resolved_question: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        evidence_chunks: List[DocumentChunk],
        answer_mode: str,
        instruction_mode: str = "default",
        model_chain: List[str],
        provider_name: str,
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Calls the LLM router with a specific model chain and parses the response.
        Supports:
        - documents_only
        - hybrid
        """
        question_for_reasoning = (
            resolved_question or current_message or "").strip()

        chunk_budget = cls._chunk_budget(question_for_reasoning)
        selected_chunks = evidence_chunks[:chunk_budget]
        mode = cls._prompt_mode(question_for_reasoning)

        if mode == "small":
            json_schema = {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string"},
                    "answer": {"type": "string"},
                    "used_chunk_indices": {
                        "type": "array",
                        "items": {"type": "integer"}
                    }
                },
                "required": ["verdict", "answer", "used_chunk_indices"]
            }
        else:
            json_schema = {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string"},
                    "answer": {"type": "string"},
                    "bullets": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "used_chunk_indices": {
                        "type": "array",
                        "items": {"type": "integer"}
                    }
                },
                "required": ["verdict", "answer", "bullets", "used_chunk_indices"]
            }
        if answer_mode == "hybrid":
            system_instruction, prompt, is_yes_no = cls._build_hybrid_prompt(
                current_message=current_message,
                resolved_question=resolved_question,
                conversation_history=conversation_history,
                evidence_chunks=selected_chunks,
                mode=mode,
                instruction_mode=instruction_mode,
            )
        else:
            system_instruction, prompt, is_yes_no = cls._build_prompt(
                question_for_reasoning,
                selected_chunks,
                mode,
                instruction_mode=instruction_mode,
            )

        ok, text, meta = LLMRouter.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2,
            model_chain=model_chain,
            json_schema=json_schema,
            task=LLMRouter.TASK_GROUNDED_ANSWER,
        )

        meta = meta or {}
        meta["provider_used"] = provider_name
        meta["instruction_mode"] = instruction_mode
        meta["answer_mode"] = answer_mode

        def recover_plain_text_result(reason: str):
            plain_answer = cls._plain_text_recovery(text or "")

            if not cls._is_answer_acceptable(plain_answer):
                return None

            used = list(range(min(3, len(selected_chunks))))

            initial_verdict = "not_enough_evidence" if is_yes_no else "supported"

            verdict = cls._normalize_verdict_from_answer(
                question=question_for_reasoning,
                answer=plain_answer,
                verdict=initial_verdict,
                used_chunk_indices=used,
            )

            recovered_meta = {
                **(meta or {}),
                "provider_used": meta.get("provider_used", provider_name),
                "answer_mode": answer_mode,
                "instruction_mode": instruction_mode,
                "fallback_used": False,
                "generation_ok": True,
                "safe_fallback": False,
                "answer_source": "model_plain_text_recovery",
                "recovered_from_unstructured_text": True,
                "recovery_reason": reason,
                "raw_text_preview": (text or "")[:500],
            }

            result = {
                "verdict": verdict,
                "answer": plain_answer,
                "bullets": [],
                "used_chunk_indices": used,
                "used_sources": cls._build_used_sources(selected_chunks, used),
                "meta": recovered_meta,
            }

            return result, recovered_meta

        if not ok:
            recovered = recover_plain_text_result("generate_json_not_ok")

            if recovered:
                recovered_result, recovered_meta = recovered
                return True, recovered_result, recovered_meta

            meta = {
                **(meta or {}),
                "error": meta.get("error") or "generate_json_failed",
                "raw_text_preview": (text or "")[:500],
                "provider_used": meta.get("provider_used", provider_name),
                "answer_mode": answer_mode,
                "instruction_mode": instruction_mode,
            }
            return False, {}, meta

        try:
            data = cls._parse_json_safely(text)
        except Exception as exc:
            recovered = recover_plain_text_result(str(exc))

            if recovered:
                recovered_result, recovered_meta = recovered
                return True, recovered_result, recovered_meta

            meta = {
                **(meta or {}),
                "error": str(exc),
                "raw_text_preview": (text or "")[:500],
                "provider_used": meta.get("provider_used", provider_name),
                "answer_mode": answer_mode,
                "instruction_mode": instruction_mode,
            }
            return False, {}, meta

        verdict = data.get("verdict", "not_enough_evidence")

        if is_yes_no:
            allowed = {"yes", "no", "not_enough_evidence"}
        else:
            allowed = {"supported", "not_enough_evidence"}

        if verdict not in allowed:
            verdict = "not_enough_evidence"

        used: List[int] = []
        seen = set()
        for idx in data.get("used_chunk_indices", []) or []:
            if isinstance(idx, int) and 0 <= idx < len(selected_chunks) and idx not in seen:
                seen.add(idx)
                used.append(idx)
            if len(used) >= 6:
                break

        answer = (data.get("answer") or "").strip()

        verdict = cls._normalize_verdict_from_answer(
            question=question_for_reasoning,
            answer=answer,
            verdict=verdict,
            used_chunk_indices=used,
        )

        bullets = data.get("bullets", []) or []
        clean_bullets = []
        seen_bullets = set()
        answer_lower = answer.lower() if answer else ""

        for b in bullets:
            b = str(b).strip()
            if not b:
                continue
            key = b.lower()
            if key in seen_bullets:
                continue
            if key == answer_lower:
                continue
            seen_bullets.add(key)
            clean_bullets.append(b)
            if len(clean_bullets) >= 6:
                break

        if not cls._is_answer_acceptable(answer):
            meta["error"] = meta.get("error") or "unacceptable_answer"
            return False, {}, meta

        debug_prompt_chunks = []
        for i, c in enumerate(selected_chunks):
            debug_prompt_chunks.append({
                "prompt_index": i,
                "chunk_index": c.chunk_index,
                "doc_title": c.document.title,
                "document_type": getattr(c.document, "document_type", None),
                "full_content_preview": (c.content or "")[:1200],
                "snippet_sent_to_llm": cls._snippet(c.content),
            })

        result = {
            "verdict": verdict,
            "answer": answer,
            "bullets": clean_bullets,
            "used_chunk_indices": used,
            "used_sources": cls._build_used_sources(selected_chunks, used),
            "meta": {
                **meta,
                "generation_ok": True,
                "fallback_used": provider_name != "gemini",
                "debug_prompt_chunks": debug_prompt_chunks,
                "prompt_mode": mode,
                "chunk_budget": chunk_budget,
                "answer_mode": answer_mode,
            },
        }
        return True, result, meta

    @classmethod
    def _call_gemini_with_retry(
        cls,
        *,
        current_message: str,
        resolved_question: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        evidence_chunks: List[DocumentChunk],
        answer_mode: str,
        instruction_mode: str = "default",
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Primary provider: Gemini
        Retry only on transient failures.
        """
        gemini_chain = [
            getattr(settings, "GROUNDED_PRIMARY_MODEL",
                    "gemini-2.5-flash-lite")
        ]

        max_attempts = getattr(settings, "GROUNDED_GEMINI_MAX_RETRIES", 3)
        last_meta: Dict[str, Any] = {}

        for attempt in range(1, max_attempts + 1):
            success, result, meta = cls._call_provider(
                current_message=current_message,
                resolved_question=resolved_question,
                conversation_history=conversation_history,
                evidence_chunks=evidence_chunks,
                answer_mode=answer_mode,
                instruction_mode=instruction_mode,
                model_chain=gemini_chain,
                provider_name="gemini",
            )

            meta["attempt"] = attempt
            last_meta = meta

            if success:
                return True, result, meta

            if not cls._is_transient_error(meta):
                break

        return False, {}, last_meta

    @classmethod
    def _call_deepseek(
        cls,
        *,
        current_message: str,
        resolved_question: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        evidence_chunks: List[DocumentChunk],
        answer_mode: str,
        instruction_mode: str = "default",
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Secondary provider: DeepSeek
        """
        deepseek_chain = [
            getattr(settings, "GROUNDED_SECONDARY_MODEL", "deepseek-chat")
        ]

        return cls._call_provider(
            current_message=current_message,
            resolved_question=resolved_question,
            conversation_history=conversation_history,
            evidence_chunks=evidence_chunks,
            answer_mode=answer_mode,
            instruction_mode=instruction_mode,
            model_chain=deepseek_chain,
            provider_name="deepseek",
        )

    @classmethod
    def answer(
        cls,
        *,
        current_message: str,
        resolved_question: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        evidence_chunks: Optional[List[DocumentChunk]] = None,
        retrieval_confidence: Optional[float] = None,
        preferred_source: Optional[str] = None,
        answer_mode: str = "default",
    ) -> Dict[str, Any]:

        evidence_chunks = cls._filter_safe_evidence_chunks(evidence_chunks)
        conversation_history = conversation_history or []
        resolved_question = (
            resolved_question or current_message or "").strip()

        instruction_mode = answer_mode or "default"

        answer_mode = cls._resolve_answer_mode(
            preferred_source=preferred_source,
            conversation_history=conversation_history,
            evidence_chunks=evidence_chunks,
        )

        if answer_mode == "history_only":
            return cls._answer_from_history_only(
                current_message=current_message,
                resolved_question=resolved_question,
                conversation_history=conversation_history,
            )

        if answer_mode == "no_evidence":
            fallback_message = (
                cls.HISTORY_NOT_ENOUGH_MESSAGE
                if preferred_source == "history"
                else cls.DOCUMENTS_NOT_ENOUGH_MESSAGE
            )

            return {
                "verdict": "not_enough_evidence",
                "answer": fallback_message,
                "bullets": [],
                "used_chunk_indices": [],
                "used_sources": [],
                "meta": {
                    "model_used": None,
                    "tried_models": [],
                    "provider_used": "no_evidence",
                    "fallback_used": False,
                    "generation_ok": False,
                    "safe_fallback": False,
                    "answer_source": "no_evidence",
                    "error": "no_available_context",
                    "answer_mode": answer_mode,
                    "retrieval_confidence": retrieval_confidence,
                },
            }

        if answer_mode in {"documents_only", "hybrid"} and not evidence_chunks:
            fallback_message = (
                cls.HYBRID_NOT_ENOUGH_MESSAGE
                if answer_mode == "hybrid"
                else cls.DOCUMENTS_NOT_ENOUGH_MESSAGE
            )

            return {
                "verdict": "not_enough_evidence",
                "answer": fallback_message,
                "bullets": [],
                "used_chunk_indices": [],
                "used_sources": [],
                "meta": {
                    "model_used": None,
                    "tried_models": [],
                    "provider_used": "documents_guard",
                    "fallback_used": False,
                    "generation_ok": False,
                    "safe_fallback": False,
                    "answer_source": "documents_guard",
                    "error": "no_evidence",
                    "answer_mode": answer_mode,
                    "retrieval_confidence": retrieval_confidence,
                },
            }

        strategy = cls._provider_strategy(resolved_question, answer_mode)

        if strategy == "deepseek_first":
            success, result, deepseek_meta, gemini_meta = cls._call_deepseek_first_then_gemini(
                current_message=current_message,
                resolved_question=resolved_question,
                conversation_history=conversation_history,
                evidence_chunks=evidence_chunks,
                answer_mode=answer_mode,
                instruction_mode=instruction_mode,
            )
            if success:
                result["meta"]["retrieval_confidence"] = retrieval_confidence
                return result
        else:
            success, result, gemini_meta, deepseek_meta = cls._call_gemini_first_then_deepseek(
                current_message=current_message,
                resolved_question=resolved_question,
                conversation_history=conversation_history,
                evidence_chunks=evidence_chunks,
                answer_mode=answer_mode,
                instruction_mode=instruction_mode,
            )
            if success:
                result["meta"]["retrieval_confidence"] = retrieval_confidence
                return result

        return cls._safe_failure_response(
            evidence_chunks=evidence_chunks,
            meta={
                "primary_meta": gemini_meta,
                "secondary_meta": deepseek_meta,
                "provider_used": "safe_fallback",
                "tried_models": [
                    getattr(settings, "GROUNDED_PRIMARY_MODEL",
                            "gemini-2.5-flash-lite"),
                    getattr(settings, "GROUNDED_SECONDARY_MODEL",
                            "deepseek-chat"),
                ],
                "error": "all_grounded_providers_failed",
                "routing_strategy": strategy,
                "answer_mode": answer_mode,
                "instruction_mode": instruction_mode,
                "retrieval_confidence": retrieval_confidence,
            },
        )
