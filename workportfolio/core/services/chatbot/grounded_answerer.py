from __future__ import annotations

import json
from typing import List, Dict, Any, Tuple
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
        Keep this conservative to reduce token usage without hurting quality.
        """
        q = (question or "").strip().lower()
        
        # Certification questions often need the exact "Certifications" chunk plus contribution/context chunks.
        if any(marker in q for marker in [
            "what certificates",
            "which certificates",
            "what certifications",
            "which certifications",
            "certificates does she have",
            "certifications does she have",
        ]):
            return 4

        # Project/tool/framework/technology questions often need the exact
        # "Technology Stack" chunk plus contribution/context chunks.
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
            return 4

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
            return 4

        if any(marker in q for marker in medium_markers):
            return 3

        return 2

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

        # Exclude choice / comparison style questions that are not really yes-no
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

        # Avoid abrupt / fragment-like endings
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

        # Remove fenced code block markers if present
        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

            if text.lower().startswith("json"):
                text = text[4:].strip()

        # Try to isolate the first JSON object
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
    def _build_prompt(
        cls,
        question: str,
        evidence_chunks: List[DocumentChunk],
        mode: str,
    ) -> Tuple[str, str, bool]:
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
            "If evidence is insufficient, return not_enough_evidence. "
            "Return raw JSON only. "
            "Do not mention retrieval, chunks, or internal processing."
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
                "C) Do not mention retrieval or documents.\n\n"
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
                "C) If the answer is already complete, bullets can be empty.\n"
                "D) Do not mention retrieval or documents.\n\n"
                f"Question (yes/no = {str(is_yes_no).lower()}): {question}\n\n"
                f"Evidence:\n{evidence_text}\n"
            )

        else:  # standard
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
                "C) Do not mention retrieval or documents.\n\n"
                f"Question (yes/no = {str(is_yes_no).lower()}): {question}\n\n"
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

        # Keep explicit uncertainty answers as not_enough_evidence
        if GroundedAnswerer._looks_uncertain_answer(answer):
            return "not_enough_evidence"

        is_yes_no = GroundedAnswerer._is_yes_no_question(question)

        # If we have grounded evidence and a real answer, do not keep
        # not_enough_evidence unless the text is actually uncertain.
        if used_chunk_indices:
            if is_yes_no:
                if lower.startswith("yes"):
                    return "yes"
                if lower.startswith("no"):
                    return "no"

                # If the model forgot the yes/no verdict but gave a direct answer,
                # keep the original unless it was not_enough_evidence.
                if verdict == "not_enough_evidence":
                    if any(lower.startswith(prefix) for prefix in ["yes", "yes,", "yes."]):
                        return "yes"
                    if any(lower.startswith(prefix) for prefix in ["no", "no,", "no."]):
                        return "no"
                    return "not_enough_evidence"

            # Non-yes/no question with grounded answer
            return "supported"

        return verdict
    
    @classmethod
    def _provider_strategy(cls, question: str) -> str:
        """
        Decide which provider to try first based on question complexity.
        Returns:
        - deepseek_first
        - gemini_first
        """
        mode = cls._prompt_mode(question)

        if mode == "small":
            return "deepseek_first"

        return "gemini_first"
    
    @classmethod
    def _call_deepseek_first_then_gemini(
        cls,
        question: str,
        evidence_chunks: List[DocumentChunk],
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Try DeepSeek first for low-cost/simple questions.
        If it fails or returns unusable output, fall back to Gemini.
        Returns:
            success, result, deepseek_meta, gemini_meta
        """
        deepseek_success, deepseek_result, deepseek_meta = cls._call_deepseek(
            question=question,
            evidence_chunks=evidence_chunks,
        )
        if deepseek_success:
            return True, deepseek_result, deepseek_meta, {}

        gemini_success, gemini_result, gemini_meta = cls._call_gemini_with_retry(
            question=question,
            evidence_chunks=evidence_chunks,
        )
        if gemini_success:
            return True, gemini_result, deepseek_meta, gemini_meta

        return False, {}, deepseek_meta, gemini_meta
    
    @classmethod
    def _call_gemini_first_then_deepseek(
        cls,
        question: str,
        evidence_chunks: List[DocumentChunk],
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Try Gemini first for richer questions.
        If it fails, fall back to DeepSeek.
        Returns:
            success, result, gemini_meta, deepseek_meta
        """
        gemini_success, gemini_result, gemini_meta = cls._call_gemini_with_retry(
            question=question,
            evidence_chunks=evidence_chunks,
        )
        if gemini_success:
            return True, gemini_result, gemini_meta, {}

        deepseek_success, deepseek_result, deepseek_meta = cls._call_deepseek(
            question=question,
            evidence_chunks=evidence_chunks,
        )
        if deepseek_success:
            return True, deepseek_result, gemini_meta, deepseek_meta

        return False, {}, gemini_meta, deepseek_meta

    @classmethod
    def _call_provider(
        cls,
        *,
        question: str,
        evidence_chunks: List[DocumentChunk],
        model_chain: List[str],
        provider_name: str,
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Calls the LLM router with a specific model chain and parses the response.
        Returns:
            success, parsed_result, meta
        """
        
        chunk_budget = cls._chunk_budget(question)
        selected_chunks = evidence_chunks[:chunk_budget]
        mode = cls._prompt_mode(question)
        
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
        
        
        system_instruction, prompt, is_yes_no = cls._build_prompt(
            question, selected_chunks, mode
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

        if not ok:
            return False, {}, meta

        try:
            data = cls._parse_json_safely(text)
        except Exception as exc:
            meta = {
                **(meta or {}),
                "error": str(exc),
                "raw_text_preview": (text or "")[:500],
                "provider_used": meta.get("provider_used", provider_name),
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
            question=question,
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
            },
        }
        return True, result, meta


    @classmethod
    def _call_gemini_with_retry(
        cls,
        question: str,
        evidence_chunks: List[DocumentChunk],
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Primary provider: Gemini
        Retry only on transient failures.
        """
        gemini_chain = [
            getattr(settings, "GROUNDED_PRIMARY_MODEL", "gemini-2.5-flash-lite")
        ]

        max_attempts = getattr(settings, "GROUNDED_GEMINI_MAX_RETRIES", 3)

        last_meta: Dict[str, Any] = {}

        for attempt in range(1, max_attempts + 1):
            success, result, meta = cls._call_provider(
                question=question,
                evidence_chunks=evidence_chunks,
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
        question: str,
        evidence_chunks: List[DocumentChunk],
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Secondary provider: DeepSeek
        """
        deepseek_chain = [
            getattr(settings, "GROUNDED_SECONDARY_MODEL", "deepseek-chat")
        ]

        return cls._call_provider(
            question=question,
            evidence_chunks=evidence_chunks,
            model_chain=deepseek_chain,
            provider_name="deepseek",
        )

    @classmethod
    def answer(cls, question: str, evidence_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        if not evidence_chunks:
            return {
                "verdict": "not_enough_evidence",
                "answer": "I don’t have enough evidence in the uploaded documents to answer that.",
                "bullets": [],
                "used_chunk_indices": [],
                "used_sources": [],
                "meta": {
                    "model_used": None,
                    "tried_models": [],
                    "error": "no_evidence",
                    "generation_ok": False,
                },
            }

        strategy = cls._provider_strategy(question)

        if strategy == "deepseek_first":
            success, result, deepseek_meta, gemini_meta = cls._call_deepseek_first_then_gemini(
                question=question,
                evidence_chunks=evidence_chunks,
            )
            if success:
                return result

        else:
            success, result, gemini_meta, deepseek_meta = cls._call_gemini_first_then_deepseek(
                question=question,
                evidence_chunks=evidence_chunks,
            )
            if success:
                return result

        # 3) Safe polite fallback only
        return cls._safe_failure_response(
            evidence_chunks=evidence_chunks,
            meta={
                "primary_meta": gemini_meta,
                "secondary_meta": deepseek_meta,
                "provider_used": "safe_fallback",
                "tried_models": [
                    getattr(settings, "GROUNDED_PRIMARY_MODEL", "gemini-2.5-flash-lite"),
                    getattr(settings, "GROUNDED_SECONDARY_MODEL", "deepseek-chat"),
                ],
                "error": "all_grounded_providers_failed",
                "routing_strategy": strategy,
            },
        )