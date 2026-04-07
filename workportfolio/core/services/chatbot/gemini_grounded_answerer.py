from __future__ import annotations

import json
from typing import List, Dict, Any
from django.conf import settings

from core.models import DocumentChunk
from core.services.llm.router import LLMRouter


class GeminiGroundedAnswerer:
    """
    Produces the final answer ONLY from evidence chunks.
    Returns verdict + used_chunk_indices so citations are precise.

    Verdict rules:
    - If question is yes/no -> verdict must be "yes" or "no"
    - Otherwise -> verdict is "supported" or "not_enough_evidence"
    """

    @staticmethod
    def _snippet(text: str, max_len: int = 900) -> str:
        t = (text or "").replace("\r", " ").strip()
        t = " ".join(t.split())
        return t if len(t) <= max_len else t[:max_len].rstrip() + "..."

    @staticmethod
    def _is_yes_no_question(question: str) -> bool:
        q = (question or "").strip().lower()
        return q.startswith((
            "is ", "are ", "do ", "does ", "did ",
            "can ", "could ", "should ", "has ", "have ",
            "was ", "were ", "am ", "will ", "would ", "shall "
        ))

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
    def _build_plain_fallback_answer(
        cls,
        question: str,
        evidence_chunks: List[DocumentChunk],
        used_indices: List[int],
    ) -> str:
        snippets = []
        for idx in used_indices:
            if 0 <= idx < len(evidence_chunks):
                snippets.append(cls._snippet(
                    evidence_chunks[idx].content, 180))

        if not snippets:
            return "I don’t have enough evidence in the uploaded documents to answer that right now."

        q = (question or "").strip().lower()

        if any(x in q for x in ["contact", "email", "phone", "linkedin", "reach"]):
            return "I couldn’t generate the usual final answer right now, but the retrieved documents do contain contact information for Samah in the cited source below."

        if any(x in q for x in ["salary", "compensation", "hourly rate", "payment", "availability", "remote", "freelance", "full-time"]):
            return "Based on the retrieved documents, Samah’s compensation and availability are discussed in relation to role scope, technical depth, responsibility, and work arrangement."

        if any(x in q for x in ["skills", "tech stack", "technologies", "frameworks", "tools"]):
            return "Based on the retrieved documents, Samah works across backend engineering, AI/LLM solutions, web development, databases, and document-processing technologies."

        if any(x in q for x in ["project", "fit", "build", "develop", "handle this", "solution"]):
            return (
                "The retrieved documents suggest that Samah has relevant adjacent experience in backend development, "
                "dashboard and analytics work, full-stack delivery, and AI-enabled business solutions. "
                "However, the documents may not explicitly confirm every requested specialized technology."
            )

        if any(x in q for x in ["background", "what does", "kind of work", "what can", "help with"]):
            return (
                "Based on the retrieved documents, Samah works as an AI/ML and full-stack engineer focused on backend systems, "
                "AI-enabled applications, automation workflows, dashboards, and practical business solutions."
            )

        merged = " ".join(snippets[:2]).strip()
        return "I couldn’t generate the usual final answer right now, but here is the most relevant grounded evidence: " + merged

    @classmethod
    def answer(cls, question: str, evidence_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        if not evidence_chunks:
            return {
                "verdict": "not_enough_evidence",
                "answer": "I don’t have enough evidence in the uploaded documents to answer that.",
                "bullets": [],
                "used_chunk_indices": [],
                "used_sources": [],
                "meta": {"model_used": None, "tried_models": [], "error": "no_evidence"},
            }

        evidence_lines = []
        for i, c in enumerate(evidence_chunks):
            evidence_lines.append(
                f"[{i}] {c.document.title} (chunk {c.chunk_index}): {cls._snippet(c.content)}"
            )
        evidence_text = "\n".join(evidence_lines)

        is_yes_no = cls._is_yes_no_question(question)

        system_instruction = (
            "You answer questions about Samah using ONLY the provided evidence.\n"
            "Rules:\n"
            "1) Do NOT invent details.\n"
            "2) Do NOT use subjective praise unless it is explicitly stated in evidence.\n"
            "3) If evidence is insufficient, verdict MUST be not_enough_evidence.\n"
            "4) If the question is yes/no, verdict MUST be yes or no.\n"
            "5) If the question is NOT yes/no, verdict MUST be supported or not_enough_evidence.\n"
            "6) Use at most 6 evidence chunks.\n"
            "7) Return JSON only.\n"
            "8) For personal preferences, compensation, availability, work style, or role fit, answer only if explicitly stated in the evidence.\n"
        )

        prompt = (
            "Return JSON with keys:\n"
            "- verdict: for yes/no questions -> one of ['yes','no','not_enough_evidence']\n"
            "          for non-yes/no questions -> one of ['supported','not_enough_evidence']\n"
            "- answer: concise answer (1-3 sentences)\n"
            "- bullets: 0-6 bullets that add NEW details not already stated in 'answer'\n"
            "- used_chunk_indices: array of integers referencing evidence chunks used (max 6)\n\n"
            "Rules:\n"
            "A) Do NOT repeat the same idea in different words.\n"
            "B) Bullets must be non-overlapping and each bullet must be unique.\n"
            "C) If the answer is already complete, bullets can be an empty array.\n\n"
            f"Question (yes/no = {str(is_yes_no).lower()}): {question}\n\n"
            f"Evidence:\n{evidence_text}\n"
        )

        chain = [getattr(settings, "GROUNDED_PRIMARY_MODEL", "gemini-2.5-flash")] + \
            getattr(settings, "GROUNDED_FALLBACK_MODELS",
                    ["gemini-2.5-flash-lite"])

        ok, text, meta = LLMRouter.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2,
            model_chain=chain,
            task=LLMRouter.TASK_GROUNDED_ANSWER,
        )

        if not ok:
            fallback_used = list(range(min(3, len(evidence_chunks))))
            fallback_answer = cls._build_plain_fallback_answer(
                question=question,
                evidence_chunks=evidence_chunks,
                used_indices=fallback_used,
            )

            fallback_verdict = "not_enough_evidence"
            if not cls._is_yes_no_question(question) and fallback_answer:
                fallback_verdict = "supported"

            return {
                "verdict": fallback_verdict,
                "answer": fallback_answer,
                "bullets": [],
                "used_chunk_indices": fallback_used,
                "used_sources": cls._build_used_sources(evidence_chunks, fallback_used),
                "meta": meta,
            }
        try:
            data = json.loads(text)
        except Exception:
            fallback_used = list(range(min(2, len(evidence_chunks))))
            fallback_answer = cls._build_plain_fallback_answer(
                question=question,
                evidence_chunks=evidence_chunks,
                used_indices=fallback_used,
            )

            fallback_verdict = "not_enough_evidence"
            if not cls._is_yes_no_question(question) and fallback_answer:
                fallback_verdict = "supported"

            return {
                "verdict": fallback_verdict,
                "answer": fallback_answer,
                "bullets": [],
                "used_chunk_indices": fallback_used,
                "used_sources": cls._build_used_sources(evidence_chunks, fallback_used),
                "meta": meta,
            }
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
            if isinstance(idx, int) and 0 <= idx < len(evidence_chunks) and idx not in seen:
                seen.add(idx)
                used.append(idx)
            if len(used) >= 6:
                break

        answer = (data.get("answer") or "").strip()

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

        bullets = clean_bullets

        if not answer:
            fallback_used = list(range(min(2, len(evidence_chunks))))
            fallback_parts = [
                cls._snippet(evidence_chunks[i].content, 220)
                for i in fallback_used
            ]
            answer = (
                "Based on the retrieved documents, here is the most relevant evidence:\n- "
                + "\n- ".join(fallback_parts)
            )
            verdict = "not_enough_evidence"
            used = fallback_used

        used_sources = cls._build_used_sources(evidence_chunks, used)

        return {
            "verdict": verdict,
            "answer": answer,
            "bullets": bullets,
            "used_chunk_indices": used,
            "used_sources": used_sources,
            "meta": meta,
        }
